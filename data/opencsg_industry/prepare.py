import os
from tqdm import tqdm
import numpy as np
import tiktoken
#from datasets import load_dataset
from modelscope.msdatasets import MsDataset

num_proc = 8
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")


if __name__ == '__main__':
    try:
       # 指定数据集名称
        dataset_name = 'opencsg/chinese-cosmopedia'

        # 加载数据集
        dataset = MsDataset.load(dataset_name, subset_name='default', split='train', namespace='modelscope')

        # 下载指定的parquet文件
        for i in range(10):
            file_name = f'0000{i}.parquet' if i < 10 else f'000{i}.parquet'
            print(f"正在下载文件: {file_name}")
            dataset.download(file_name)
            print(f"文件 {file_name} 下载完成！")
        
        # 从本地目录加载数据集
        dataset = MsDataset.load(
            'opencsg/chinese-cosmopedia',
            subset_name='default',
            split='train',
            namespace='modelscope',
            trust_remote_code=True    # 添加信任远程代码参数
        )

        print("数据集加载成功！")
        print(f"数据集信息：\n{dataset}")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请检查：")
        print("1. 本地目录是否存在")
        print("2. 数据集文件是否完整")
        print("3. 文件权限是否正确")
        exit(1)

    # print("\n正在分割数据集...")
    
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005,
        seed=2357,
        shuffle=True
    )
    print("数据集分割完成！")
    print(f"训练集大小：{len(split_dataset['train'])}")
    print(f"验证集大小：{len(split_dataset['test'])}")

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
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
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