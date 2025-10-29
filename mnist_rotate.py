"""
mnist_rotate.py

MNIST dataset with optional 90° rotations.
- Supports 28x28 torchvision MNIST or 8x8 sklearn digits.
- Can be used as NumPy arrays or as a PyTorch Dataset.

Requires: torch, torchvision, sklearn, numpy, scipy
"""

import numpy as np
from sklearn.datasets import load_digits
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import Dataset
from scipy.ndimage import rotate as scipy_rotate


def load_mnist_numpy(size=28, normalize=True):
    """
    Load MNIST as NumPy arrays.
    
    Args:
        size: 8 (sklearn digits) or 28 (torchvision MNIST).
        normalize: scale to [0,1].
    """
    if size == 8:
        digits = load_digits()
        X, y = digits.images, digits.target
        if normalize:
            X = X / 16.0
    elif size == 28:
        dataset = MNIST(root="./data", train=True, download=True)
        testset = MNIST(root="./data", train=False, download=True)
        X = np.concatenate([dataset.data.numpy(), testset.data.numpy()], axis=0)
        y = np.concatenate([dataset.targets.numpy(), testset.targets.numpy()], axis=0)
        if normalize:
            X = X.astype(np.float32) / 255.0
    else:
        raise ValueError("Only size=8 or size=28 is supported")
    return X, y


def add_rotations(X, y, angles=(0, 90, 180, 270)):
    """
    Add rotated copies of dataset.
    Returns new (X, y, rot_labels).
    """
    X_out, y_out, rot_labels = [], [], []
    for angle in angles:
        X_rot = np.array([scipy_rotate(img, angle, reshape=False) for img in X])
        X_out.append(X_rot)
        y_out.append(y)
        rot_labels.extend([angle] * len(y))
    X_aug = np.concatenate(X_out, axis=0)
    y_aug = np.concatenate(y_out, axis=0)
    rot_labels = np.array(rot_labels)
    return X_aug, y_aug, rot_labels


class MNISTRotate(Dataset):
    """
    PyTorch Dataset wrapper for MNIST with optional rotations.
    """

    def __init__(self, size=28, rotate=True, angles=(0, 90, 180, 270), transform=None):
        super().__init__()
        self.size = size
        self.rotate = rotate
        self.angles = angles
        self.transform = transform

        X, y = load_mnist_numpy(size=size)
        if rotate:
            self.X, self.y, self.rot_labels = add_rotations(X, y, angles)
        else:
            self.X, self.y = X, y
            self.rot_labels = np.zeros(len(y), dtype=int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img, label, rot = self.X[idx], self.y[idx], self.rot_labels[idx]
        if self.size == 28:
            img = img.astype(np.float32)
        if self.transform:
            img = self.transform(img)
        return img, label, rot


def MNISTRotate90(size=28, transform=None):
    """
    Convenience function that returns a MNISTRotate dataset
    with only 0° and 90° rotations.
    Rotation label: 0 = original, 1 = rotated.
    """
    dataset = MNISTRotate(size=size, rotate=True, angles=(0, 90), transform=transform)
    # remap {0,90} -> {0,1}
    dataset.rot_labels = (dataset.rot_labels > 0).astype(int)
    return dataset
