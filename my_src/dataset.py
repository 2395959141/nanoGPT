import torch
from torch.utils.data import Dataset
import json
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, data_path, block_size=1024, max_lines=300000):
        print(f"初始化数据集，block_size={block_size}, max_lines={max_lines}")
        # 初始化 tiktoken 编码器
        self.enc = tiktoken.get_encoding("p50k_base")
        self.block_size = block_size
        
        # 获取结束符 token
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        print(f"结束符token ID: {self.eos_token}")
        
        # 读取并处理数据
        raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    print(f"处理第{i+1}行时出错: {str(e)}")
                    continue
        
        print(f"成功读取{len(raw_data)}条文本数据")
        
        # 编码所有文本
        full_encoded = []
        total_tokens = 0
        for i, text in enumerate(raw_data):
            encoded_text = self.enc.encode(text)
            total_tokens += len(encoded_text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        print(f"编码后总token数: {total_tokens}")
        print(f"添加结束符后的总token数: {len(full_encoded)}")
        
        # 将编码后的数据分割成固定大小的块
        self.encoded_data = []
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i + self.block_size + 1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
        
        print(f"数据已分割成{len(self.encoded_data)}个训练块")
        print(f"每个训练块大小: {self.block_size + 1} tokens")
                
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        """将文本编码为token IDs"""
        ids = self.enc.encode(text)
        print(f"编码文本，输入长度: {len(text)}字符，输出: {len(ids)} tokens")
        return ids

    def decode(self, ids):
        """将token IDs解码为文本"""
        text = self.enc.decode(ids)
        print(f"解码tokens，输入: {len(ids)} tokens，输出长度: {len(text)}字符")
        return text 