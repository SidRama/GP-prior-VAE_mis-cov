from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np


class PhysionetDataset(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    """

    def __init__(self, data_file, root_dir, transform=None):
        data = np.load(os.path.join(root_dir, data_file))
        self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
        self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
        self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
        self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx, :]
        patient_data = torch.Tensor(np.array(patient_data))

        mask = self.mask_source[idx, :]
        mask = np.array(mask, dtype='uint8')

        label = self.label_source[idx, :]
        label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx, :]

        label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class Physionet2019Dataset(Dataset):
    """
    Dataset definition for the Physionet2019Dataset
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        digit = self.data_source.iloc[idx, :]
        digit = np.array([digit], dtype='float32').squeeze()

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8').squeeze()

        label = self.label_source.iloc[idx, :]
        # TimeToSepsis, WhetherSepsisAtSomePoint, Id, Age, Gender, Unit, HospAdmTime, ICULOS
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([1, 2, 0, 3, 4, 5, 6, 7])], dtype='float32')))

        if self.transform:
            digit = self.transform(digit)
        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class HMNIST_dataset(Dataset):
    """
    Dataset definiton for the modified MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """
    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):
        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(36, 36)
        digit = digit[..., np.newaxis]

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, 0:6]
        label = torch.Tensor(np.nan_to_num(np.array(label)))

        label_mask = self.label_source.iloc[idx, 6:10]
        label_mask = torch.Tensor(np.array(label_mask))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'label_mask': label_mask}
        return sample
