import numpy as np 
import pandas as pd

import os
from pathlib import Path

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision as vision

import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score

from sklearn.model_selection import StratifiedKFold, KFold

# load data
root_dir = '../input/'

print(os.listdir(root_dir))

labels = pd.read_csv(root_dir+'labels.csv')
df_train = pd.read_csv(root_dir+'train.csv')
df_sub = pd.read_csv(root_dir+'sample_submission.csv')


#df_train = df_train[0: 10000]

print(labels.head())
print(df_train.head())
print(df_sub.head())
print(labels.shape[0])


threshold = 0.1
lr = 2e-4
classes = labels.shape[0]
batch = 128
kfold = 5

def attributes_prepare(x):
    x = x.split()
    z = np.zeros((classes), dtype=int)
    for i in x:
        z[int(i)] = 1
    return z
df_train.attribute_ids = df_train.attribute_ids.map(attributes_prepare).values

DL = 1000
L = 10000
def load_data(data, k):
    k = round(len(data) * k)
    return data[:k], data[k:]

# kf = KFold(n_splits = kfold)

# skf = StratifiedKFold(n_splits=kfold)

# for train_index, test_index in skf.split(df_train):
#     print(train_index.shape)
#     print(test_index.shape)

# classes defining

# statistics collector
class Mean(object):
    def __init__(self):
        self.values = []

    def compute(self):
        return sum(self.values) / len(self.values)

    def update(self, value):
        self.values.extend(np.reshape(value, [-1]))

    def reset(self):
        self.values = []

    def compute_and_reset(self):
        value = self.compute()
        self.reset()

        return value

# data loader for reading dataset
class ImageDataLoader(data.DataLoader) : 
    def __init__(self, root_dir: Path, df: pd.DataFrame, mode="train", transforms=None):
        self._root = root_dir
        self.transform = transforms[mode]
        self._id = df.id.values
        self._img_id = (df["id"] + ".png").values
        self._targets = df.attribute_ids.values
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        
        if self.transform:
            img = self.transform(img)
          
        return (self._targets[idx], img, self._id[idx])

# model defining
model = vision.models.resnet50() #pretrained=True
#model = vision.models.densenet121()
#model = vision.models.resnet101()


model.fc = torch.nn.Sequential(
    torch.nn.Dropout2d(0.1),
    torch.nn.Linear(
        in_features=2048,
        out_features=classes
    ),
    torch.nn.Sigmoid()
)
#model = Model()
# x = torch.zeros((1, 3, 224, 224)) # one sample
# y = model(x)
# print(y.shape)

# transforms defining
data_transforms = {
    'train': vision.transforms.Compose([
        vision.transforms.RandomResizedCrop(224),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'val': vision.transforms.Compose([
        vision.transforms.Resize(256),
        vision.transforms.CenterCrop(224),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
}
data_transforms["test"] = data_transforms["val"]

model = model.cuda()

# optim
opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
#opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
#opt = torch.optim.SGD(model.parameters(), 1e-2, weight_decay=1e-4, momentum=0.9)

sched = torch.optim.lr_scheduler.StepLR(opt, [30, 80])
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 5)


def f2_score(y_true, y_pred, threshold):
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def show_batch(targets, images):
    for i in images:
        plt.imshow(i.cpu().permute(1, 2, 0))
        plt.title('title')
        plt.show()

# loss function
BCELoss = nn.BCELoss(reduction='mean').cuda()
#BCELoss = torch.nn.BCEWithLogitsLoss().cuda()

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
# BCELoss = FocalLoss()


# training
for i in range(11):
    print('split {}'.format(i))
        
    train = df_train[i * L: (i + 1) * L]
    
    #train, test = load_data(train_data, 0.99)
    
    train_dataset = ImageDataLoader(
        root_dir = Path(root_dir + "train/"),
        df = train,
        mode = "train",
        transforms = data_transforms)
    train_loader = data.DataLoader(dataset=train_dataset, shuffle = True, batch_size=batch, drop_last=True)
    
    # test_dataset = ImageDataLoader(
    #     root_dir = Path(root_dir + "train/"),
    #     df = test,
    #     mode = "test",
    #     transforms = data_transforms)
    # test_loader = data.DataLoader(dataset=test_dataset, shuffle = False, batch_size=128, drop_last=True)
    
    # stats = {
    #     'train_loss': [],
    #     'test_loss': [],
    #     'train_acc': []#,
    #     'test_acc': []
    # }
    mean_loss = Mean()
    mean_acc = Mean()
    for epoch in range(25):
        model.train()
        for targets, images, ids in train_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            
            #train_loss = F.binary_cross_entropy(input = logits, target = targets.float())
            train_loss = BCELoss(logits, targets.float())
            
            y_hat = logits.detach().cpu().numpy()
            y_hat = (y_hat > threshold).astype(int)
            
            train_acc = f2_score(targets.detach().cpu().numpy(), y_hat, threshold)
            
            mean_loss.update(train_loss.data.cpu().numpy())
            mean_acc.update(train_acc)
            
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
        sched.step()
            
            #show_batch(targets, images)
            
        train_loss = mean_loss.compute_and_reset()
        train_acc = mean_acc.compute_and_reset()
        
        # with torch.no_grad():
        #     model.eval()
        #     for targets, images, ids in test_loader:
        #         images, targets = images.cuda(), targets.cuda()
        #         logits = model(images).detach()
                
        #         #test_loss = F.binary_cross_entropy(input = logits, target = targets.float())
        #         test_loss = BCELoss(logits, targets.float())
                
        #         y_hat = logits.cpu().numpy()
        #         y_hat = (y_hat > threshold).astype(int)
        #         test_acc = f2_score(targets.cpu().numpy(), y_hat, threshold)
                
        #         mean_loss.update(test_loss.data.cpu().numpy())
        #         mean_acc.update(test_acc)
                
        #     test_loss = mean_loss.compute_and_reset()
        #     test_acc = mean_acc.compute_and_reset()
    
        # stats['train_loss'].append(train_loss)
        # stats['train_acc'].append(train_acc)
        # stats['test_loss'].append(test_loss)
        # stats['test_acc'].append(test_acc)
        
        if (not(epoch % 5)):
            # print('epoch {}, train Loss train:{:.2f} test:{:.2f} Acc train:{:.2f} test:{:.2f}'.format(epoch, float(train_loss), float(test_loss), float(train_acc), float(test_acc)))
            print('epoch {}, train Loss train:{:.2f} Acc train:{:.2f}'.format(epoch, float(train_loss), float(train_acc)))
        
    
    # plt.plot(stats['train_loss'], label='train')
    # plt.plot(stats['test_loss'], label='test')
    # plt.title('loss')
    # plt.legend()
    # plt.show()
    
    # plt.plot(stats['train_acc'], label='train')
    # plt.plot(stats['test_acc'], label='test')
    # plt.title('acc')
    # plt.legend()
    # plt.show()

print('write results')

# result processing
sub_dataset = ImageDataLoader(
    root_dir = Path(root_dir + "test/"),
    df = df_sub,
    mode = "val",
    transforms = data_transforms)
    
sub_loader = data.DataLoader(dataset=sub_dataset, shuffle = False, batch_size=batch, drop_last=False)
sub = pd.DataFrame(columns=['id', 'attribute_ids'])
with torch.no_grad():
    model.eval()
    for targets, images, ids in sub_loader:
        images = images.cuda()
        logits = model(images)
        y_hat = logits.clone().detach().cpu().numpy()
        y_hat = (y_hat > threshold).astype(int)
        for i, l in enumerate(y_hat):
            sub.loc[len(sub)] = {'id':ids[i], 'attribute_ids':' '.join([str(k) for k, v in enumerate(l) if v == 1])}

sub.to_csv('submission.csv', index=False)
print('done')