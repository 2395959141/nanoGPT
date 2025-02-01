"""
将arron666/seq_monkey数据集预处理为二进制格式
运行前请先安装依赖：
pip install datasets tiktoken tqdm numpy
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets


# 配置参数
num_proc = 8  # 并行处理进程数
enc = tiktoken.get_encoding("p50k_base")   # *使用GPT3编码器

if __name__ == '__main__':
    # 1. 加载数据集
    dataset = load_dataset("arron666/seq_monkey")
    
    # datase
    
    # 2. 划分训练集和验证集（如果数据集未自带验证集）
    if 'validation' not in dataset:
        print("数据集未包含验证集，正在从训练集划分5%作为验证集...")
        split_dataset = dataset["train"].train_test_split(
            test_size=0.05,  # 5%作为验证集
            seed=2023,
            shuffle=True
        )
        split_dataset['val'] = split_dataset.pop('test')   # *将'test'重命名为'val'
    else:
        split_dataset = dataset
        split_dataset['val'] = split_dataset.pop('validation')

    # 3. 定义分词处理函数
    def process(example):
        assert 'text' in example, "数据集必须包含 text 字段"
        text = example['text']
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    # 4. 并行处理所有数据
    tokenized = split_dataset.map(
        process,
        remove_columns=[col for col in split_dataset['train'].column_names],
        desc="分词处理",
        num_proc=num_proc,
    )

    # 5. 保存为二进制文件
    output_dir = os.path.join(os.path.dirname(__file__), 'seq_monkey')
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        filename = os.path.join(output_dir, f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+')

        # 合并所有样本
        total_len = sum(dset['len'])
        arr.resize(total_len)
        
        idx = 0
        for sample in tqdm(dset, desc=f'写入 {filename}'):
            arr[idx : idx + len(sample['ids'])] = sample['ids']
            idx += len(sample['ids'])
        arr.flush()
