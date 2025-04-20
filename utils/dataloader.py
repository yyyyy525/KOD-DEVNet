import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class CODataset(data.Dataset):
    """
    dataloader
    """
    def __init__(self, image_root, gtc_root, gts_root, gta_root, gtn_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.gts_c = [gtc_root + f for f in os.listdir(gtc_root) if f.endswith('.png')]
        self.gts_s = [gts_root + f for f in os.listdir(gts_root) if f.endswith('.png')]
        self.gts_a = [gta_root + f for f in os.listdir(gta_root) if f.endswith('.png')]
        self.gtns = [gtn_root + f for f in os.listdir(gtn_root) if f.endswith('.png')]
        #
        self.images = sorted(self.images)
        #print('images path :',self.images)
        self.gts_c = sorted(self.gts_c)
        self.gts_s = sorted(self.gts_s)
        self.gts_a = sorted(self.gts_a)
        self.gtns = sorted(self.gtns)
        #print('gts path :',self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gtc_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.gts_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.gta_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gtc_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.gts_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.gta_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            self.gtn_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt_c = self.binary_loader(self.gts_c[index])
        gt_s = self.binary_loader(self.gts_s[index])
        gt_a = self.binary_loader(self.gts_a[index])
        gtn = self.manyclass_loader(self.gtns[index])
        gtn = np.array(gtn, dtype=np.float64)
        gtn = gtn - 1
        gtn = Image.fromarray(gtn)
        # print(np.max(gtn))
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gtc_transform is not None:
            gt_c = self.gtc_transform(gt_c)
        if self.gts_transform is not None:
            gt_s = self.gts_transform(gt_s)
        if self.gts_transform is not None:
            gt_a = self.gta_transform(gt_a)
        if self.gtn_transform is not None:
            gtn = self.gtn_transform(gtn)
            # gtn = gtn.resize((self.trainsize, self.trainsize), Image.NEAREST)
            # # gtn = np.array(gtn).astype(np.float32)
            # gtn = torch.from_numpy(gtn).float()
        return image, gt_c, gt_s, gt_a, gtn, name

    def filter_files(self):
        assert len(self.images) == len(self.gts_c)
        images = []
        gts_c = []
        gts_s = []
        gts_a = []
        gtns = []
        for img_path, gtc_path, gts_path, gta_path, gtn_path in zip(self.images, self.gts_c, self.gts_s, self.gts_a, self.gtns):
            img = Image.open(img_path)
            gt_c = Image.open(gtc_path)
            if img.size == gt_c.size:
                images.append(img_path)
                gts_c.append(gtc_path)
                gts_s.append(gts_path)
                gts_a.append(gta_path)
                gtns.append(gtn_path)
        self.images = images
        self.gts_c = gts_c
        self.gts_s = gts_s
        self.gts_a = gts_a
        self.gtns = gtns

    def manyclass_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('P')

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt_c, gt_s, gt_a):
        assert img.size == gt_c.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt_c.resize((w, h), Image.NEAREST), gt_s.resize((w, h), Image.NEAREST), gt_a.resize((w, h), Image.NEAREST)
        else:
            return img, gt_c, gt_s, gt_a

    def __len__(self):
        return self.size


def get_loader(image_root, gtc_root, gts_root, gta_root, gtn_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = CODataset(image_root, gtc_root, gts_root, gta_root, gtn_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gtc_root, gts_root, gta_root, gtn_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts_c = [gtc_root + f for f in os.listdir(gtc_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts_s = [gts_root + f for f in os.listdir(gts_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts_a = [gta_root + f for f in os.listdir(gta_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts_n = [gtn_root + f for f in os.listdir(gtn_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts_c = sorted(self.gts_c)
        self.gts_s = sorted(self.gts_s)
        self.gts_a = sorted(self.gts_a)
        self.gts_n = sorted(self.gts_n)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gtc_transform = transforms.ToTensor()
        self.gts_transform = transforms.ToTensor()
        self.gta_transform = transforms.ToTensor()
        self.gtn_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt_c = self.binary_loader(self.gts_c[self.index])
        gt_s = self.binary_loader(self.gts_s[self.index])
        gt_a = self.binary_loader(self.gts_a[self.index])
        gt_n = self.manyclass_loader(self.gts_n[self.index])
        gt_n = self.gtn_transform(gt_n)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt_c, gt_s, gt_a, gt_n, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def manyclass_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('P')

    def __len__(self):
        return self.size

class My_test_dataset:
    def __init__(self, image_root, gtc_root, gts_root, gta_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts_c = [gtc_root + f for f in os.listdir(gtc_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts_s = [gts_root + f for f in os.listdir(gts_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts_a = [gta_root + f for f in os.listdir(gta_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts_c = sorted(self.gts_c)
        self.gts_s = sorted(self.gts_s)
        self.gts_a = sorted(self.gts_a)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gtc_transform = transforms.ToTensor()
        self.gts_transform = transforms.ToTensor()
        self.gta_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt_c = self.binary_loader(self.gts_c[self.index])
        gt_s = self.binary_loader(self.gts_s[self.index])
        gt_a = self.binary_loader(self.gts_a[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt_c, gt_s, gt_a, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')