import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
#from modelscope.msdatasets import MsDataset
from pathlib import Path  # 新增pathlib模块

# 配置基础路径
OUTPUT_BASE = Path('/root/autodl-tmp/mobile')
DATASET_DIR = OUTPUT_BASE / 'split_dataset'
BIN_OUTPUT_DIR = OUTPUT_BASE / 'processed_bins'

num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")


if __name__ == '__main__':

    os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/mobile/hf_cache'
        # 指定数据集名称和下载目录
    dataset_name = 'opencsg/chinese-cosmopedia'


    print("正在加载数据集...")
        
        # 首先获取数据集的配置信息
    dataset = load_dataset(
        'parquet',  # 这里是加载 parquet 文件
        data_files={
        'train': [
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00013.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00146.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00147.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00148.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00149.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00336.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00337.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00338.parquet'),
            # os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00339.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00340.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00341.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00342.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00343.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00344.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00345.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00346.parquet'),
            os.path.expanduser('~/autodl-fs/BAAI_IndustryCorpus2_automobil/rank_00347.parquet')
            ]
        },
         cache_dir='/root/autodl-tmp/mobile/hf_cache'  # 新增缓存路径指定  
    )
    # print("\n正在分割数据集...")
    
    # 获取训练集
    train_dataset = dataset['train']
    
    # 对训练集进行分割
    split_dataset = train_dataset.train_test_split(
        test_size=0.0005,
        seed=2357,
        shuffle=True
    )
    print("数据集分割完成！")
    print(f"训练集大小：{len(split_dataset['train'])}")
    print(f"验证集大小：{len(split_dataset['test'])}")

     # 将分割后的数据集保存到自定义目录
    split_dataset.save_to_disk(DATASET_DIR)
    print(f"已保存分割后的数据集到：{DATASET_DIR}")

    split_dataset["val"] = split_dataset.pop("test")
    print("已将测试集重命名为验证集")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    print("\n开始进行tokenization...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print("Tokenization完成！")

    for split, dset in tokenized.items():
        print(f"\n正在处理 {split} 数据集...")
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"总token数量：{arr_len}")
        output_dir = BIN_OUTPUT_DIR
        # 新增：确保目标目录存在且可写
        os.makedirs(BIN_OUTPUT_DIR, exist_ok=True)  # 自动创建目录
        if not os.access(BIN_OUTPUT_DIR, os.W_OK):
            raise PermissionError(f"无写入权限: {BIN_OUTPUT_DIR}")

        filename = output_dir / f'{split}.bin'
        print(f"将保存到文件：{filename}")
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{split} 数据集处理完成！")