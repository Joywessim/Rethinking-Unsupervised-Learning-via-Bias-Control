from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, ConcatDataset
import torch
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt

class MNIST_13Rot(Dataset):
    """
    MNIST subset with digits {1, 3} and their 90° rotated versions.
    Pixel values are kept in [0, 1] for Bernoulli likelihood.
    """
    def __init__(self, root="data", train=True, device="cpu"):
        self.device = device

        # Load raw MNIST data (no normalization)
        mnist = MNIST(root=root, train=train, download=True)

        # Keep only digits 1 and 3
        mask = (mnist.targets == 1) | (mnist.targets == 3)
        data = mnist.data[mask]          # (N, 28, 28)
        targets = mnist.targets[mask]    # (N,)

        # Add channel dimension -> (N, 1, 28, 28)
        data = data.unsqueeze(1).float()

        # Create rotated version
        rotated_data = torch.stack([TF.rotate(img, 90) for img in data])
        rotated_targets = targets.clone()  # same labels

        # Combine originals and rotated
        self.data = (torch.cat([data, rotated_data], dim=0) > 127).float()


        self.targets = torch.cat([targets, rotated_targets], dim=0)

        # Move to device
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)

        # Store metadata
        self.height = 28
        self.width = 28
        self.in_channels = 1
        self.n_classes = 2  # digits 1 and 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



# Test it
if __name__ == "__main__":
    dataset = MNIST_13Rot(train=True, device='cpu')
    print("Dataset size:", len(dataset))

    # choose first 8 originals and their rotated versions
    n = 8
    imgs = [dataset[i][0] for i in range(2*n)]  # first n originals + n rotated
    labels = [dataset[i][1].item() for i in range(2*n)]

    # plot grid 2 × n
    fig, axes = plt.subplots(2, n, figsize=(n*1.2, 2.4))
    for i in range(n):
        # original
        axes[0, i].imshow(imgs[i][0], cmap='gray')
        axes[0, i].set_title(f"orig {labels[i]}")
        # rotated
        axes[1, i].imshow(imgs[i+n][0], cmap='gray')
        axes[1, i].set_title(f"rot {labels[i+n]}")
        for ax in (axes[0, i], axes[1, i]):
            ax.axis('off')

    plt.tight_layout()
    plt.show()
