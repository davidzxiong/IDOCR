from __future__ import division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class IDDataset(Dataset):
    """ID dataset."""

    def __init__(self, training):
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, 'data/id/')
        files = os.listdir(path)
        if training:
            self.files = files[:int(len(files)*0.9)]
        else:
            self.files = files[int(len(files)*0.9):]
        self.root_dir = path
        self.data = np.zeros([120, 560, len(self.files)])
        for idx in xrange(len(self.files)):
            img_file = self.files[idx]
            image = Image.open(os.path.join(path, img_file))
            image = np.array(image)
            mean = np.mean(image)
            std = np.std(image)
            self.data[:,:, idx] = (image - mean) / std

        # vocabulary 0~9 and X for ID number
        vocab = dict()
        for i in xrange(10):
            vocab[str(i)] = i
        vocab['X'] = 10
        vocab['<start>'] = 11
        self.vocab = vocab

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        id = [self.vocab[i] for i in self.files[idx][:-4]]
        id = [11] + id
        sample = {'id': id, 'image': self.data[:,:,idx]}
        return sample
