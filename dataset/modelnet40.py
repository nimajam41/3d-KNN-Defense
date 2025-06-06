import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(DATA_DIR, set='train', targeted=False):
    path = os.path.join(DATA_DIR, set)
    
    X_file = os.path.join(path, 'pointclouds.npy')
    y_file = os.path.join(path, 'labels.npy')
    
    X, y = np.load(X_file, allow_pickle=True), np.load(y_file).astype("int")

    if targeted:
        t_file = os.path.join(path, 'targets.npy')
        return X, y, np.load(t_file).astype("int")

    return X, y


class ModelNet40Dataset(Dataset):
    def __init__(self, X, y, transform, t=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.t = t

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype("float")
        x = self.transform(x)
        y = torch.tensor(self.y[idx])

        if self.t is not None:
            t = torch.tensor(self.t[idx])
            return x, y, t
        
        return x, y