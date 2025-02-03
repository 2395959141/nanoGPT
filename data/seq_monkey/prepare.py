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
enc = tiktoken.get_encoding("gpt2")   # *使用GPT3编码器

# 修改输出目录为绝对路径
output_dir = '/root/autodl-tmp/bin_data'
os.makedirs(output_dir, exist_ok=True)
train_bin_path = os.path.join(output_dir, 'train.bin')
val_bin_path = os.path.join(output_dir, 'val.bin')


# 新增：保存原始数据集到指定路径
dataset_save_path = '/root/autodl-fs/data'
os.makedirs(dataset_save_path, exist_ok=True)

if __name__ == '__main__':
    # 1. 加载数据集
    #dataset = load_dataset("arron666/seq_monkey")
    
    try:
        dataset = load_dataset('json', 
                              data_files={
                                  'train': '/root/autodl-fs/mobvoi_seq_monkey_general_open_corpus.jsonl'
                              },
                              split='train[:60%]')
        
        dataset.save_to_disk(dataset_save_path)
        print(f"原始数据集已保存到：{dataset_save_path}")

        # 直接对dataset进行划分，因为它已经是一个Dataset对象
        print("正在从数据集划分0.05%作为验证集...")
        split_dataset = dataset.train_test_split(
            test_size=0.0005,  # 5%作为验证集
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
    for split, dset in tokenized.items():
        # 计算总长度
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        # 根据split选择对应的文件路径
        filename = train_bin_path if split == 'train' else val_bin_path
        print(f"{split}数据集总token数: {arr_len}")
        print(f"保存路径: {filename}")
        
        # 创建memmap
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # 使用分批写入
        total_batches = 1024  # 总批次数
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'写入 {filename}'):
            # 将数据集分片处理
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            # 将批次的ids连接起来
            arr_batch = np.concatenate(batch['ids'])
            # 写入memmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        # 确保数据写入磁盘
        arr.flush()
