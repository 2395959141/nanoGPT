import math
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_embed: int = 768
    head_size: int = 64
    n_layer: int = 12
    n_head: int = 12
    bias: bool = False
    dropout: float = 0.0



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.key = nn.Linear(config.n_embed, config.head_size)
        self.query = nn.Linear(config.n_embed, config.head_size)
        self.value = nn.Linear(config.n_embed, config.head_size)
        self.dropout_p = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if (not self.flash):
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('attention_mask', torch.tril(
            torch.ones(config.block_size, config.block_size)
        ))
            
    def forward(self, x):
        batch_size, seq_len, n_embed = x.size()
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )
        else:
            weight = (query @ key.transpose(-2, -1)) 
            weight = weight.masked_fill(self.attention_mask[: seq_len, :seq_len] == 0, float('-inf'))
            weight = F.softmax(weight, dim=-1)
            weight = self.dropout(weight)
            out = weight @ value
        
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
                                    CausalSelfAttention(config)
                                    for _ in range(config.n_head)
                                    ])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
   
    def forward(self, x):
        out = torch.cat([
                 h(x) for h in self.heads], 
                 dim = -1)
        out = self.dropout(self.proj(out))
        return out
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiheadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存 config 对象
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(
                *(Block(config) for _ in range(config.n_layer))
        )
        self.ln_final = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    
    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(
            torch.arange(seq_len,device=idx.device)
        )
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch*seq_len, vocab_size)
            targets = targets.view(batch*seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob , num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding_table.weight.numel()
        return n_params
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.position_embedding_table.weight = nn.Parameter(self.position_embedding_table.weight[:block_size])
        for block in self.blocks:
            if hasattr(block.attn, 'attention_mask'):
                block.attn.attention_mask = block.attn.attention_mask[:block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"需要权重衰减的参数张量 ：{len(decay_params)}, 需要权重衰减的参数数量：{num_decay_params:,}")
        print(f"不需要权重衰减的参数张量 ：{len(nodecay_params)}, 不需要权重衰减的参数数量：{num_nodecay_params:,}")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用融合的AdamW：{use_fused}")

        return optimizer
    

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估计模型flops利用率（MFU），单位为4090的 bfloat16峰值flops """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embed//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 165.2e12  # RTX 4090的BF16理论峰值性能：165.2 TFLOPS
        
        mfu = flops_achieved / flops_promised
        return mfu 