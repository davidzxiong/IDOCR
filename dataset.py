from __future__ import division
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# vocabulary 0~9 and X for ID number
vocab = dict()
idx_to_word = dict()
for i in xrange(10):
    vocab[str(i)] = i
    idx_to_word[i] = str(i)
vocab['X'] = 10
vocab['<start>'] = 11
idx_to_word[10] = 'X'
idx_to_word[11] = '<start>'




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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        id = [vocab[i] for i in self.files[idx][:-4]]
        id = [vocab['<start>']] + id
        id = torch.Tensor(id)
        image = torch.Tensor(self.data[:,:,idx])
        image_tensor = torch.zeros(1, image.shape[0], image.shape[1]).float()
        image_tensor[0,:,:] = image
        return image_tensor, id


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, id).
            - image: torch tensor of shape (1, 120, 560).
            - id: torch tensor of shape (19)
    Returns:
        images: torch tensor of shape (batch_size, 1, 120, 560).
        targets: torch tensor of shape (batch_size, 19).
    """
    # Sort a data list by caption length (descending order).
    images, captions = zip(*data)

    # Merge images (from tuple of 2D tensor to 3D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    targets = torch.zeros(len(captions), len(captions[0])).long()
    for i, cap in enumerate(captions):
        targets[i,:] = cap

    return images, targets