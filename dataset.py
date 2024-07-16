import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms


def get_data(args):
    if args.dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="data/CIFAR-10", train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(
            root="data/CIFAR-10", train=False,  download=True, transform=train_transform)
  
    elif args.dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])


        trainset = torchvision.datasets.CIFAR100(
            root="data/CIFAR-100", train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(
            root="data/CIFAR-100", train=False, download=True, transform=train_transform)


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    return trainloader, testloader

