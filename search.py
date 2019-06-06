#modules
import os
from pathlib import Path

import argparse

import pandas as pd
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torchvision as vision

import PIL
from PIL import Image

import matplotlib.pyplot as plt
import scipy.spatial.distance

#py .\search.py --images ./train2017 --image 000000340727.jpg --k 5

# prarse options
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images')
parser.add_argument('-m', '--image')
parser.add_argument('-c', '--k')
opts = parser.parse_args()

images = str(opts.images)
image = str(opts.image)
k = int(opts.k)

# init
torch.cuda.init()

train_files = pd.DataFrame({'images':os.listdir(images)})
#train_files = train_files[0:600]
print(train_files.shape)

# define helpers
class Img(object):
    def __init__(self, path:Path, transforms=None):
        self._path = path
        self._transform = transforms
    
    def show(self, image_file):
        file = self._path / image_file
        img = Image.open(file).convert('RGB')
        plt.imshow(img)
        
    def transform(self, image_file):
        file = self._path / image_file
        img = Image.open(file).convert('RGB')
        img = self._transform(img)
        return img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, root_dir: Path, transforms=None):
        self._root = root_dir
        self._transform = transforms
        self.data = data        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]        
        file_name = self._root / row.images
        img = Image.open(file_name).convert('RGB')        
        img = self._transform(img)        
        return (row.images, img)

data_transforms = vision.transforms.Compose([
    vision.transforms.Resize(32),
    vision.transforms.CenterCrop(32),
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
])

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        
    def forward(self, input):
        input = self.l(input)
        input = torch.relu(input)
        input = self.pool(input)
        return input

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(3, 8),
            Block(8, 16),
            Block(16, 32)
        ])    
        self.global_pool = nn.MaxPool2d(8,8)
        
    def forward(self, x):
        for l in self.blocks:
            x = l(x)
        x = self.global_pool(x)       
        x = x.view(x.shape[0], x.shape[1])
        return x

# x = torch.zeros((1, 3, 128, 128)) # one sample
# model = Model()
# y = model(x)
# print(y.shape)

def vectorize(model):
    if os.path.isfile('./vectors.csv'):
        vectors = pd.read_csv('./vectors.csv')
    else:
        vectors = pd.DataFrame(columns=['image', 'vector'])

        train_dataset = Dataset(
                train_files,
                root_dir = Path(images),
                transforms = data_transforms)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle = False, batch_size=128, drop_last=False)
        b = 0
        with torch.no_grad():
            model.eval()
            for ims, items in train_loader:
                items = items.cuda()              
                y = model(items)
                for i, l in enumerate(y):                
                    vectors.loc[len(vectors)] = {'image':ims[i], 'vector':' '.join([str(i) for i in l.tolist()])}
                if (not(b % 5)):
                    print('batch = {}'.format(b))
                b = b + 1
        vectors.to_csv('./vectors.csv', index=False)
        
    for i, l in enumerate(vectors.vector.values):
        vectors.vector[i] =  [float(c) for c in l.split(' ')]
        
    return vectors

im = Img(path = Path(images), transforms = data_transforms)

model = Model().cuda()
vectors = vectorize(model)

print('Test image ', image)
im.show(image)

img = im.transform(image)
img = img.unsqueeze(0)
img = img.cuda()
y = model(img) 
y = y.cpu().detach().numpy()[0]

print('Similar imagees ', image)
dist = pd.DataFrame(columns=['image', 'dist']) 
for i, v in enumerate(vectors.vector):
    _dist = scipy.spatial.distance.cosine(y, v) 
    dist.loc[len(dist)] = {'image':vectors.image[i], 'dist':_dist}

indexes = np.argsort(dist.dist.values)
similar = [dist.image[i] for i in indexes[0:k]]
print(similar)
for i in similar:
    im.show(i)
    plt.show()