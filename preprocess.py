# preprocessing
import torch
import random
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ObtainDataset(Dataset):
    '''
    Inherits functionality from Torch dataset.
    Required dict keys to load associated data.
    '''
    def __init__(self, dic, images, labels, transform=None, target_transform=None):
        self.img_labels = dic[labels]
        self.imgs = dic[images]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # the method returns a pair: given - label for the index number i
        label = self.img_labels[idx]
        image = torch.from_numpy(self.imgs[idx]).float()
        image = image.permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class ObtainDualDataset(Dataset):
    '''
    Inherits functionality from Torch dataset.
    Required dict keys to load associated data.
    '''
    def __init__(self, dic, imagesA, imagesB, labels, transform=None, target_transform=None):
        self.img_labels = dic[labels]
        self.imgsA = dic[imagesA]
        self.imgsB = dic[imagesB]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # the method returns a pair: given - label for the index number i
        label = self.img_labels[idx]

        imageA = torch.from_numpy(self.imgsA[idx]).float()
        imageA = imageA.permute(2, 0, 1)

        imageB = torch.from_numpy(self.imgsB[idx]).float()
        imageB = imageB.permute(2, 0, 1)

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
        if self.target_transform:
            label = self.target_transform(label)
        return imageA, imageB, label

def split_data(data, train_set_perc=0.8, shuffle=True):
    '''
    Splits dataset into training and test set.
    Returns training and test set.
    '''

    length = len(data['actions'])
    absolute_split = length*train_set_perc

    if shuffle:
        key_list = list(data.keys())
        seed = random.random()

        for i in key_list:
            data_i = data.pop(i)
            random.seed(seed)
            random.shuffle(data_i)
            data[i] = data_i

    train_data = {}
    test_data = {}

    for i in data.keys():
        col_train = []
        col_test = []
        for index, val in enumerate(data[i]):
            if index < absolute_split:
                col_train.append(val)
            else:
                col_test.append(val)
        train_data[str(i)] = col_train
        test_data[str(i)] = col_test

    return train_data, test_data

def split_n_steps_between(dataset, n):
    '''
    Have original observation (A) and observation (B) shifted by n.
    They length of the lists thus differs by n.
    We create action windows of length n that connect A and B.
    We lose n-1 values in the action list.
    However, the impact of the shif is larger then the window.
    '''

    # A (start obs.) --> B (goal obs.)
    dataset['observationsA'] = copy.deepcopy(dataset['observations'])
    dataset['observationsB'] = dataset.pop('observations')

    # sliding window to generate the target action sequence
    def window(seq, size):
        windows = []
        for i in range(len(seq) - size + 1):
            windows.append(seq[i: i+size])
        return windows

    dataset['actions'] = window(dataset['actions'], n)

     # delete left n values
    for i in range(n):
        dataset['observationsB'].pop(0)

    # delete exessive values in other sets (on the right!)
    for key in dataset.keys():
        while len(dataset[key]) != len(dataset['observationsB']):
            dataset[key].pop()

    return dataset
