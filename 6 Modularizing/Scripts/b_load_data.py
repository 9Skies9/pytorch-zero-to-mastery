"""
Contains functions for creating Pytorch Datasets and DataLoaders on image classification data
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int
    ):


    """
    Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.
    """


    #target folder of images, and transforms to perform on data (images)
    train_data = datasets.ImageFolder(root=train_dir, transform=transform) 
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    #gets the class names, which contains the list of output labels
    class_names = train_data.classes

    #batch size is how many samples in a batch
    #num workers is how many cores you want the cpu to be running this dataloader on
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, test_dataloader, class_names