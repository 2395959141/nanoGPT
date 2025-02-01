"""
将arron666/seq_monkey数据集预处理为二进制格式
运行前请先安装依赖：
pip install datasets tiktoken tqdm numpy
"""
import os
import logging
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets


# 配置参数
num_proc = 8  # 并行处理进程数
enc = tiktoken.get_encoding("p50k_base")   # *使用GPT3编码器

if __name__ == '__main__':
    # 1. 加载数据集
    #dataset = load_dataset("arron666/seq_monkey")
    
    try:
        dataset = load_dataset('json', 
                              data_files={
                                  'train': '/openbayes/input/input0/fixed_corpus.jsonl'
                              },
                              split='train')
        
        # 直接对dataset进行划分，因为它已经是一个Dataset对象
        print("正在从数据集划分5%作为验证集...")
        split_dataset = dataset.train_test_split(
            test_size=0.05,  # 5%作为验证集
            seed=2023,
            shuffle=True
        )
        # 将split_dataset转换为字典格式
        split_dataset = {
            'train': split_dataset['train'],
            'val': split_dataset['test']
        }
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        raise 
    
    # 3. 定义分词处理函数
    def process(example):
        assert 'text' in example, "数据集必须包含 text 字段"
        text = example['text']
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    # 4. 并行处理所有数据
    tokenized = {}
    for split_name, split_data in split_dataset.items():
        print(f"处理{split_name}数据集...")
        tokenized[split_name] = split_data.map(
            process,
            remove_columns=split_data.column_names,
            desc=f"处理{split_name}集",
            num_proc=num_proc,
        )

    # 5. 保存为二进制文件
    output_dir = os.path.join(os.path.dirname(__file__), 'seq_monkey')
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        filename = os.path.join(output_dir, f'{split}.bin')
        # 计算总长度
        total_len = sum(dset['len'])
        print(f"{split}数据集总token数: {total_len}")
        
        # 创建memmap时指定shape
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_len,))
        
        # 写入数据
        idx = 0
        for sample in tqdm(dset, desc=f'写入 {filename}'):
            arr[idx : idx + len(sample['ids'])] = sample['ids']
            idx += len(sample['ids'])
        
        # 确保数据写入磁盘
        arr.flush()
