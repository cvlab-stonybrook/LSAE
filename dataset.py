import os
import pdb
import numpy as np
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset

import cv2
from tqdm import tqdm

import pdb

class CXR14Dataset(Dataset):
    def __init__(self, data_root, listfile, transform, gray=False):
        self.image_list = []
        self.label_list = []
        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split()
                # pdb.set_trace()
                image_path = os.path.join(data_root, items[0])
                label = torch.tensor(list(map(int, items[1:])), dtype=torch.float32)
                self.image_list.append(image_path)
                self.label_list.append(label)

        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        label = self.label_list[index]
        image = self.transform(image)

        return image, label

class IM100Dataset(Dataset):
    def __init__(self, data_root, listfile, transform, gray=False):
        self.image_list = []
        self.label_list = []
        with open(listfile) as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split()
                image_path = os.path.join(data_root, items[0])
                label = int(items[1])
                self.image_list.append(image_path)
                self.label_list.append(label)

        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        label = self.label_list[index]
        image = self.transform(image)

        return image, label

class MaskedCXR14Dataset(Dataset):
    def __init__(self, data_root, mask_root, listfile, transform, gray=False, nolabel=False):
        self.image_list = []
        self.mask_list = []
        self.label_list = []
        self.nolabel = nolabel
        with open(listfile) as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i]
                items = line.strip().split()
                # pdb.set_trace()
                image_path = os.path.join(data_root, items[0])
                mask_path = os.path.join(mask_root, 'mask_'+items[0])
                if not nolabel:
                    label = torch.tensor(list(map(int, items[1:])), dtype=torch.float32)
                # mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # mask_ratio = np.count_nonzero(mask_array) / mask_array.size
                # # pdb.set_trace()
                # if mask_ratio > 0.1:
                self.image_list.append(image_path)
                self.mask_list.append(mask_path)
                if not nolabel:
                    self.label_list.append(label)

        self.transform = transform
        self.gray = gray
        print(f"Length of dataset is {len(self)}")

    def __len__(self):
        assert(len(self.image_list) == len(self.mask_list)), "Image, mask and label list should be of same length"
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        mask = Image.open(self.mask_list[index])
        mask = mask.convert('RGB')
        masked_img = ImageChops.multiply(image, mask)
        image = self.transform(masked_img)

        if not self.nolabel:
            label = self.label_list[index]
            return image, label
        else:
            return image

class CXR14maskDataset(Dataset):
    def __init__(self, data_root, mask_root, listfile, transform, gray=False, nolabel=False):
        self.image_list = []
        self.mask_list = []
        self.label_list = []
        self.nolabel = nolabel
        with open(listfile) as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i]
                items = line.strip().split()
                # pdb.set_trace()
                image_path = os.path.join(data_root, items[0])
                mask_path = os.path.join(mask_root, 'mask_'+items[0])
                if not nolabel:
                    label = torch.tensor(list(map(int, items[1:])), dtype=torch.float32)
                # mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # mask_ratio = np.count_nonzero(mask_array) / mask_array.size
                # # pdb.set_trace()
                # if mask_ratio > 0.1:
                self.image_list.append(image_path)
                self.mask_list.append(mask_path)
                if not nolabel:
                    self.label_list.append(label)

        self.transform = transform
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.gray = gray
        print(f"Length of dataset is {len(self)}")

    def __len__(self):
        assert(len(self.image_list) == len(self.mask_list)), "Image, mask and label list should be of same length"
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if self.gray:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        mask = Image.open(self.mask_list[index])
        mask = mask.convert('L')
        image = self.transform(image)
        mask = self.mask_transform(mask)

        if not self.nolabel:
            label = self.label_list[index]
            return image, mask, label
        else:
            return image, mask

class SeqDataset(Dataset):
    def __init__(self, data_root, transform, length=2, phase='train'):
        self.data_root = data_root
        self.transform = transform
        self.length = length
        self.phase = phase
        self.pid2dates = self.parse_dataset()
        self.all_keys = list(self.pid2dates.keys())
        tr_keys, ts_keys = train_test_split(self.all_keys, train_size=0.8, shuffle=False)
        if phase == 'train':
            self.keys = tr_keys
        else:
            self.keys = ts_keys

    def parse_dataset(self):
        pid2dates = {}
        files = os.listdir(self.data_root)
        for file in files:
            items = file.split('.')[0].split('_')
            pid, date = items[0], int(items[1])
            if pid not in pid2dates:
                pid2dates[pid] = []
            pid2dates[pid].append(date)
        # select sequences
        del_keys = []
        for key in pid2dates.keys():
            dates = pid2dates[key]
            if len(dates) == 1 or len(dates) < self.length:
                del_keys.append(key)
            else:
                pid2dates[key] = dates[-self.length:]
        for key in del_keys:
            pid2dates.pop(key)
        print(f"There are {len(pid2dates)} patients")
        return pid2dates
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        images = []
        dates = self.pid2dates[key]
        for date in dates:
            image_path = os.path.join(self.data_root, f"{key}_{date:03d}_0.png")
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = self.transform(image)
            images.append(image)
        images = torch.cat(images, dim=0)
        return images, key, dates

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    # dataset = CXR14Dataset('/home/leizhou/covid_proj/data/chestxray8/images', '/home/leizhou/covid_proj/data/chestxray8/trainval_list.txt', transform)
    dataset = SeqDataset('/home/leizhou/covid_proj/data/TemporalData/images', transform, phase='train')
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    for batch in loader:
        pdb.set_trace()
