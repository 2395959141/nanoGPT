"""
将arron666/seq_monkey数据集预处理为二进制格式
运行前请先安装依赖：
pip install datasets transformers tqdm numpy
"""
import os
import logging
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer  # 替换tiktoken为BertTokenizer
from datasets import load_dataset, load_from_disk # huggingface datasets
import shutil
import gc

# # 在文件开头添加临时文件清理
# shutil.rmtree('/tmp/datasets_cache', ignore_errors=True)  # 清理临时缓存

# 修改数据集缓存路径到有空间的磁盘
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-fs/data/seq_monkey"  # 新路径

# ! 总Token数量：2,734,775,146 tokens

# 配置参数
num_proc = 1 # 并行处理进程数
# 使用本地BERT tokenizer替换GPT2 tokenizer
tokenizer = BertTokenizer.from_pretrained('/root/autodl-fs/huggingface/transformers/bert_base_chinese_tokenizer')

# 在配置参数部分预先定义路径
output_dir = '/root/autodl-tmp/bin_data'
train_bin_path = os.path.join(output_dir, 'train.bin')
val_bin_path = os.path.join(output_dir, 'val.bin')
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

# 新增：保存原始数据集到指定路径
dataset_save_path = '/root/autodl-tmp/seq_data'
os.makedirs(dataset_save_path, exist_ok=True)

if __name__ == '__main__':
    # 1. 加载数据集
    try:
        print("正在加载数据集的前60%...")
        dataset = load_dataset(
            'json', 
            data_files={'train': '/root/autodl-fs/data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl'},
            split='train[:50%]',  # 只加载前60%的数据
            num_proc=num_proc,
            cache_dir=dataset_save_path,
            #streaming=True  # 添加流式模式
        )
        
        # 划分验证集
        print("正在从数据集划分0.05%作为验证集...")
        split_dataset = dataset.train_test_split(
            test_size=0.0005,
            seed=2023,
            shuffle=True
        )
        split_dataset = {
            'train': split_dataset['train'],
            'val': split_dataset['test']
        }
        
        # 在加载数据后添加
        print(f"成功加载数据集，共包含 {len(dataset)} 条数据")
        
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        raise

    # 3. 定义分词处理函数
    def process(example):
        assert 'text' in example, "数据集必须包含 text 字段"
        text = example['text']
        # 使用BERT tokenizer进行编码
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]标记
            truncation=True,
            max_length=512  # 设置最大长度
        )
        return {'ids': encoded, 'len': len(encoded)}

    # 4. 并行处理所有数据
    tokenized = {}
    for split_name, split_data in split_dataset.items():
        print(f"处理{split_name}数据集...")
        tokenized[split_name] = split_data.map(
            process,
            remove_columns=split_data.column_names,
            desc=f"处理{split_name}集",
            num_proc=8,
            # batch_size=1000  # 添加批处理
        )

    # 5. 保存为二进制文件
    for split, dset in tokenized.items():
        filename = train_bin_path if split == 'train' else val_bin_path
        print(f"\n正在处理 {split} 数据集...")
        print(f"目标路径：{filename}")
        
        # 计算总长度
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"{split}数据集总token数: {arr_len}")
        
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

        # 添加保存完成提示
        print(f"{split} 数据集已成功保存至：{filename}")

    # 在保存为二进制文件的循环后添加验证代码
    print("\n=== 文件保存验证 ===")
    total_tokens = 0  # 添加总token计数器
    for path in [train_bin_path, val_bin_path]:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 ** 2)  # 转换为MB
            num_tokens = os.path.getsize(path) // 2  # 因为dtype是uint16，所以除以2得到token数
            total_tokens += num_tokens  # 累加token数量
            print(f"✓ 文件已生成：{path}")
            print(f"   文件大小：{file_size:.2f} MB")
            print(f"   Token数量：{num_tokens:,} tokens")  # 添加token数量显示
        else:
            print(f"× 文件缺失：{path}")
    
    print(f"\n=== 总计统计 ===")
    print(f"总Token数量：{total_tokens:,} tokens")
