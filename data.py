import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib as plt

class mnist(Dataset):
    def __init__(self, train):
        if train:
            content = [ ]
            for i in range(5):
                dataset = np.load(os.path.join(os.getcwd(), 'corruptmnist',f'train_{i}.npz'), allow_pickle=True)
                # dataset.files to see keys
                content.append(dataset)
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(os.path.join(os.getcwd(), 'corruptmnist','test.npz'), allow_pickle=True)
            # dataset.files to see keys
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])

        self.data = data
        self.targets = targets
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]  
       
if __name__ == "__main__":
    train_data =  mnist(train=True)
    test_data =  mnist(train=False)    
    print(train_data.data.shape)
    print(train_data.targets.shape)
    print(test_data.data.shape)
    print(test_data.targets.shape)