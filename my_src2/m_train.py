"""
这个训练脚本可以在调试模式下在单个GPU上运行，
也可以在分布式数据并行（ddp）的大规模训练中运行。

要在单个GPU上运行，示例：
$ python train.py --batch_size=32 --compile=False

要在1个节点上的4个GPU上使用DDP运行，示例：
$ torchrun --standalone --nproc_per_node=4 train.py

要在2个节点上的4个GPU上使用DDP运行，示例：
- 在第一个（主）节点上运行，示例IP为123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果您的集群没有Infiniband互连，请在前面加上NCCL_IB_DISABLE=1）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext


import gc
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.checkpoint import checkpoint

from m_model import GPTConfig, GPT

# 数据配置
data_dir = '/home/gpt2_data_bin/'  # 直接使用绝对路径


out_dir = 'out'
eval_interval = 1000  # 每2000步评估一次
log_interval = 1   # 每2000步记录一次日志
eval_iters = 64
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
swan_log = True # disabled by default
swan_project = 'gpt2-124M-chinese-seq_monkey'
swan_run_name = 'gpt2-124M-chinese-seq_monkey' # 'run' + str(time.time())
# data
gradient_accumulation_steps = 32  # 梯度累积步数
batch_size = 24  # 每个设备的训练批次大小
eval_batch_size = 16  # 评估时的批次大小
block_size = 512
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 5e-5  # 学习率
max_iters = 10000  # 总训练迭代次数
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
# learning rate decay settings
decay_lr = True 
warmup_iters = 2500  # 预热步数
lr_decay_iters = 10000  # 学习率衰减的总步数
min_lr = 5e-5  # 最小学习率，约为初始学习率的1/10
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
# 添加版本检查
import sys
compile = False if sys.version_info >= (3, 12) else True  # 自动禁用3.12+的编译

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
device_type = 'cuda' 
# note: float16 data type will automatically use a GradScaler
ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 检查路径是否存在
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"数据目录 {data_dir} 不存在！请检查路径设置")
    
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'test.bin')
if not os.path.exists(train_path):
    raise FileNotFoundError(f"训练文件 {train_path} 不存在！请运行数据预处理脚本")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"验证文件 {val_path} 不存在！请运行数据预处理脚本")

def get_batch(split):
    # 添加文件路径定义
    file_path = os.path.join(data_dir, f'{split}.bin')  # 假设数据文件名为 train.bin 和 val.bin
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# 直接设置vocab_size为21128
meta_vocab_size = 21128

# model init
model_args = dict(n_layer=n_layer, 
                 n_head=n_head, 
                 n_embed=n_embd,
                 block_size=block_size,
                 bias=bias, 
                 vocab_size=21128,  # 保持直接设置
                 dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(f"初始化模型词汇表大小: {model.config.vocab_size}")
elif init_from == 'resume':
    # 优先尝试加载最佳检查点
    ckpt_path = os.path.join(out_dir, 'best_ckpt.pt')
    if not os.path.exists(ckpt_path):
        # 如果最佳检查点不存在，加载最新的常规检查点
        checkpoints = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            ckpt_path = os.path.join(out_dir, checkpoints[-1])
    print(f"从 {ckpt_path} 恢复训练")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # 在模型加载后添加显存优化
    model.to(device)
    torch.cuda.empty_cache()
    
    # 分阶段释放内存
    del state_dict
    del checkpoint_model_args
    _ = gc.collect()
    torch.cuda.empty_cache()
    print(f"从检查点恢复模型，词汇表大小: {model.config.vocab_size}")
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model
# if compile:
#     print("compiling the model... (takes a ~minute)")
#     unoptimized_model = model
#     # 添加动态形状配置
#     torch._dynamo.config.dynamic_shapes = True
#     torch._dynamo.config.assume_static_by_default = False
#     model = torch.compile(model, dynamic=True)

# 添加优化器初始化
# 创建AdamW优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if swan_log and master_process:
    import swanlab
    swanlab.init(project=swan_project,
                 name=swan_run_name,
                 config=config)
    
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

def print_memory_stats():
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats()
        print(f"已分配: {stats['allocated_bytes.all.current']/1024**3:.2f}GB")
        print(f"保留缓存: {stats['reserved_bytes.all.current']/1024**3:.2f}GB")
        print(f"活跃内存: {stats['active_bytes.all.current']/1024**3:.2f}GB")

# 在系统设置部分添加检查点保存数量限制
max_checkpoints = 10  # 最大保存的检查点数量

# ! 训练循环
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
        if swan_log:
            swanlab.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['test'],
                "lr": lr,
                #"mfu": running_mfu*100, # convert to percentage
            })
        if losses['test'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['test']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                
                # 保存常规检查点（带迭代编号）
                checkpoint_name = f'ckpt_{iter_num}.pt'
                checkpoint_path = os.path.join(out_dir, checkpoint_name)
                print(f"保存检查点到 {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
                
                # 始终保存最佳检查点（单独文件）
                if losses['test'] == best_val_loss:
                    best_checkpoint_path = os.path.join(out_dir, 'best_ckpt.pt')
                    torch.save(checkpoint, best_checkpoint_path)
                
                # 清理旧检查点（保留最近max_checkpoints个 + 最佳检查点）
                checkpoints = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                # 删除超出数量的旧检查点
                while len(checkpoints) > max_checkpoints:
                    oldest = checkpoints.pop(0)
                    os.remove(os.path.join(out_dir, oldest))
                    print(f"删除旧检查点: {oldest}")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # 每2个batch更新swanlab
        if swan_log:
            swanlab.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100*4,
            })

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
                    


