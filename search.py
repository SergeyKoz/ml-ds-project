#modules
import os
import json
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

#py .\search.py --images ./train2017 --image ./train2017/000000340727.jpg --k 5

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
#train_files = train_files[0:100]
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
        plt.show()
        
    def transform(self, image_file):
        file = self._path / image_file
        img = Image.open(file).convert('RGB')
        img = self._transform(img)
        return img

    def showTest(self, image_file):
        img = Image.open(image_file).convert('RGB')
        plt.imshow(img)
        plt.show()
        
    def transformTest(self, image_file):
        img = Image.open(image_file).convert('RGB')
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
    vision.transforms.Resize(128),
    vision.transforms.CenterCrop(128),
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
])

def vectorize(model):
    if os.path.isfile('./vectors.df'):
        vectors = pd.read_pickle('./vectors.df')
    else:
        vectors = pd.DataFrame(columns=['image', 'vector'])

        train_dataset = Dataset(
                train_files,
                root_dir = Path(images),
                transforms = data_transforms)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle = False, batch_size=256, drop_last=False)
        b = 0
        with torch.no_grad():
            model.eval()
            for ims, items in train_loader:
                items = items.cuda()              
                y = model(items)
                for i, l in enumerate(y):                
                    vectors.loc[len(vectors)] = {'image':ims[i], 'vector':l.cpu().numpy()}
                if (not(b % 5)):
                    print('batch = {}'.format(b))
                b = b + 1
        vectors.to_pickle('./vectors.df')
    return vectors

im = Img(path = Path(images), transforms = data_transforms)

model = vision.models.resnet50(pretrained = True)
model.fc =nn.Sequential()
model.cuda()
model.eval()

# x = torch.zeros((1, 3, 32, 32)) # one sample
# # model = Model()
# y = model(x)
# print(y.shape)

vectors = vectorize(model)

print('Test image ', image)
im.showTest(image)

img = im.transformTest(image)
img = img.unsqueeze(0)
img = img.cuda()
y = model(img) 
y = y.cpu().detach().numpy()[0]

dist = pd.DataFrame(columns=['image', 'dist']) 
for i, v in enumerate(vectors.vector):
    _dist = scipy.spatial.distance.cosine(y, v)
    dist.loc[len(dist)] = {'image':vectors.image[i], 'dist':_dist}

indexes = np.argsort(dist.dist.values)
similar = [dist.image[i] for i in indexes[0:k]]
for i in similar:
    im.show(i)
    plt.show()