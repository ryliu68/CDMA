import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class Adv_PTH_Dataset(data.Dataset):

    def __init__(self, data_root):

        self.data = torch.load(data_root)

        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])


    def __getitem__(self, index):
        ret = {}

        adv_img = self.data["adv_imgs"][index]
        img = self.data["imgs"][index]

        adv_img = torchvision.transforms.functional.to_pil_image(adv_img)
        img = torchvision.transforms.functional.to_pil_image(img)

        adv_img = self.tfs(adv_img)

        img = self.tfs(img)

        ret['gt_image'] = adv_img
        ret['cond_image'] = img
        ret['path'] = str(index)+".png"

        return ret

    def __len__(self):
        return len(self.data["adv_imgs"])

def get_data(args):
    if args.dataset == "cifar10":
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="data/CIFAR-10", train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(
            root="data/CIFAR-10", train=False,  download=True, transform=train_transform)
        
    elif args.dataset == "svhn":

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])


        trainset = torchvision.datasets.SVHN(
            root="data/SVHN", split='train', download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.SVHN(
            root="data/SVHN",  split='test', download=False, transform=test_transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=True, num_workers=8)

        
    elif args.dataset == 'stl10':
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.STL10(
            root="data/STL10", split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.STL10(
            root="data/STL10", split='test', download=True, transform=train_transform)

    elif args.dataset == "cifar100":
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root="data/CIFAR-100", train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(
            root="data/CIFAR-100", train=False, download=True, transform=train_transform)

    elif args.dataset == "tinyimagenet":
        data_dir = 'data/tiny-imagenet-200'
        train_transform = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = TinyImageNet(data_dir, train=True,
                                transform=train_transform)
        testset = TinyImageNet(data_dir, train=False,
                               transform=train_transform)
        
    elif args.dataset == "celeba":
        data_dir = 'data/CelebA'
        train_transform = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5]),
            transforms.Resize([64,64])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5]),
            transforms.Resize([64,64])
        ])
        trainset = torchvision.datasets.CelebA(data_dir, split='test',
                                transform=train_transform,download=True)
        testset = torchvision.datasets.CelebA(data_dir, split='test',
                               transform=test_transform,download=True)
        
        # split (string): One of {'train', 'valid', 'test', 'all'}.


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    return trainloader, testloader


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.test_dir = os.path.join(self.root_dir, "test")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (
                        words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        classes = [d.name for d in os.scandir(
            self.train_dir) if d.is_dir()]

        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")

        val_annotations_file = os.path.join(
            self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (
                                path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt




