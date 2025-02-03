import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class BinaryDataset(Dataset):
    def __init__(self, split, block_size=512, data_dir='/home/gpt2_data_bin/'):
        self.data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return {'input_ids': x, 'labels': y}

def get_dataloaders(block_size=512, batch_size=32, data_dir='/home/gpt2_data_bin/'):
    train_dataset = BinaryDataset('train', block_size, data_dir)
    val_dataset = BinaryDataset('val', block_size, data_dir)
    
    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if torch.distributed.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    return train_loader, val_loader 