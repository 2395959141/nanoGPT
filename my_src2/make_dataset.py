import random
import os
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig
from glob import glob

print("开始数据处理任务...")
random.seed(42) 

# 设置HuggingFace镜像
print("配置HuggingFace环境...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'  # 设置离线模式

print("正在扫描数据文件...")
all_file_list = glob("/root/autodl-fs/gpt2_data/*/**")
print(f"总共发现文件数量: {len(all_file_list)}")

# 修改开始：先打乱整个文件列表
print("打乱文件顺序...")
random.shuffle(all_file_list)  # 原地打乱文件顺序

# 划分训练集和测试集（假设仍要50个测试文件）
test_file_list = all_file_list[:50]  # 取前50个作为测试集
train_file_list = all_file_list[50:]  # 剩余作为训练集
# 修改结束

print(f"数据集划分完成:")
print(f"训练集数量: {len(train_file_list)}, 测试集数量: {len(test_file_list)}")

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/huggingface/transformers/bert_base_chinese_tokenizer")
print("tokenizer加载完成")

print("开始加载数据集...")
raw_dataset = load_dataset("csv", 
                          data_files={"train": train_file_list, "test": test_file_list},
                          cache_dir="/root/autodl-fs/cache",
                          column_names=["text"])  # 假设CSV只有文本内容一列
print("数据集加载完成")

def process(example):
    text = example["text"]
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_attention_mask=False
    )
    return {"ids": encoded["input_ids"], "len": len(encoded["input_ids"])}

print("开始tokenize数据...")
# 首先进行tokenize处理
tokenized_datasets = raw_dataset.map(
    process,
    remove_columns=raw_dataset["train"].column_names,
    batched=False,
    num_proc=8,
    desc="Tokenizing"
)

# 对每个数据集分别进行shuffle
print("打乱数据集...")
shuffled_datasets = DatasetDict({
    split: tokenized_datasets[split].shuffle(seed=42)
    for split in tokenized_datasets.keys()
})

print("tokenize和shuffle完成")

output_dir = "./gpt2_data_bin"
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录创建完成: {output_dir}")

for split in ["train", "test"]:
    print(f"\n处理{split}数据集...")
    total_len = sum(shuffled_datasets[split]['len'])
    print(f"{split}数据集总token数: {total_len}")
    
    print(f"创建内存映射文件: {split}.bin")
    arr = np.memmap(os.path.join(output_dir, f'{split}.bin'), 
                   dtype=np.uint16, mode='w+', shape=(total_len,))
    
    idx = 0
    for sample in tqdm(shuffled_datasets[split], desc=f'写入{split}数据'):
        arr[idx : idx + len(sample['ids'])] = sample['ids']
        idx += len(sample['ids'])
    arr.flush()
    print(f"{split}数据集处理完成")

print("\n所有数据处理任务完成!")