import os
import time

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SleepEDF_Dataset(Dataset):
    def __init__(self, samples, labels):
        self.data_x = samples
        self.data_y = labels
        self.num_class = 5

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


def data_pipeline(data_path, flag='train', batch_size=6, drop_last=True, fine_tune_frac=0.2, scale=True):
    if flag == 'pre_train':
        data = torch.load(os.path.join(data_path, "train.pt"))
    elif flag == 'val':
        data = torch.load(os.path.join(data_path, "val.pt"))
    elif flag == 'test':
        data = torch.load(os.path.join(data_path, "test.pt"))
    elif flag == 'fine_tune':
        data = torch.load(os.path.join(data_path, "train.pt"))
        # Select random 20% of train data for fine-tuning
        num_samples = len(data["samples"])
        indices = torch.randperm(num_samples)[:int(num_samples * fine_tune_frac)]
        data = {
            "samples": data["samples"][indices],
            "labels": data["labels"][indices]
        }
    data["samples"] = data["samples"].permute(0, 2, 1)

    # Scaling the data
    if scale:
        num_features = data["samples"].size(2)
        scalers = [StandardScaler() for _ in range(num_features)]
        for j in range(num_features):
            feature_data = data["samples"][:,:,j].numpy()  # Extract data for feature j
            feature_data = feature_data.reshape(-1, 1)     # Reshape to 2D
            scalers[j].fit(feature_data)
            feature_data = scalers[j].transform(feature_data)
            data["samples"][:,:,j] = torch.tensor(feature_data).view(data["samples"][:,:,j].size())

    data_set = SleepEDF_Dataset(data["samples"], data["labels"])
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,   # Shuffle data for training
        drop_last=drop_last
    )

    return data_set, data_loader
