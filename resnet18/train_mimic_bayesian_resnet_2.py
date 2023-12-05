# Train model with MIMICIII, 224x224, MC Dropout = 0.5

################ experiment environment ################

# environment needs to be set before assigning device to torch
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
torch.cuda.set_device(1)
print(torch.cuda.is_available())
print(os.environ["CUDA_VISIBLE_DEVICES"])
print("Current device:", torch.cuda.current_device())
print("Device count:", torch.cuda.device_count())

import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import glob as glob
from PIL import Image
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

def is_cuda_available():
    if torch.cuda.is_available():
        print("CUDA available. Training on GPU!")
        return torch.device('cuda')
    else:
        print("CUDA not available. Training on CPU!")
        return torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    
device = is_cuda_available()

################ dataset ################

def prepare_dataset(dataframe):
    x = dataframe['img_paths'].tolist()
    y = dataframe['labels'].tolist()
    # split dataset into train and test
    return x[:int(len(x) * 3/4)], y[:int(len(y) * 3/4)], x[int(len(x)* 3/4):], y[int(len(y) * 3/4):]

class mimic_dataset(Dataset):
    def __init__(self, image_paths, image_labels, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_res = Image.fromarray(image)
        
        label = self.image_labels[idx]
        
        if(self.transform is not None):
            image_res = self.transform(image_res)
        
        return image_res, label

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_x, train_y, val_x, val_y = prepare_dataset(pd.read_csv('mimic.csv'))

train_ds = mimic_dataset(train_x, train_y, transform_train)
val_ds = mimic_dataset(val_x, val_y, transform_val)

dataloader_train = DataLoader(train_ds, batch_size=750, shuffle=True, num_workers=4)
dataloader_valid = DataLoader(val_ds, batch_size=750, shuffle=True, num_workers=4)

################ train ################

import bayesian_resnet_2

model = bayesian_resnet_2.BayesianResNet2(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                             betas=(0.9, 0.999),
                             weight_decay=1e-8)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)
criterion = nn.CrossEntropyLoss()

# train bayesian ResNet18 model with a dropout probability of 0.5
model = bayesian_resnet_2.train_model(model, optimizer, lr_scheduler, criterion, device, "bayesian2_mimic",
                                      dataloader_train, dataloader_valid, bayesian_dropout_p=0.5, epochs=100)