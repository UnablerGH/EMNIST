import matplotlib.pyplot as plt
import torch
import torchvision
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

def preprocess():
    X_train = np.load("X_train.npy")
    X_val = np.load("X_val.npy")
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()


    class MyDataset(Dataset):
        def __init__(self, data, target, transform=None, target_transform=None):
            self.data = data
            self.target = target
            self.transform = transform
            self.target_transform = target_transform
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            x = self.data[index]
            if self.transform:
                x = self.transform(x)
            
            y = self.target[index]
            if self.target_transform:
                y = self.target_transform(y)
                
            return x, y


    from torchvision import transforms

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])


    train_dataset = MyDataset(X_train, y_train, transform=transform)
    val_dataset = MyDataset(X_val, y_val)


    train_loader = DataLoader(train_dataset, batch_size=32)

    return train_loader, val_dataset


