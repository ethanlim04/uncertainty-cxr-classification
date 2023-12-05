# Train model with CheXpert + MIMICIII, 512x512, CLAHE, MC Dropout = 0.5

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
    return x[:int(len(x) * 9/10)], y[:int(len(y) * 9/10)], x[int(len(x)* 9/10):], y[int(len(y) * 9/10):]

train_x, train_y, val_x, val_y = prepare_dataset(pd.read_csv('train.csv'))

class cxr_dataset(Dataset):
    def __init__(self, image_paths, image_labels, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]

        if("CheXpert-v1.0" in image_filepath):
            image_filepath = "/media/Seagate Hub/CheXpert/chexpertchestxrays-u20210408/" + image_filepath

        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        # CLAHE instead of normal histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        image = clahe.apply(image)#.astype('uint8')
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image_res = Image.fromarray(image)
        
        label = self.image_labels[idx]
        
        if(self.transform is not None):
            image_res = self.transform(image_res)
        
        return image_res, label

transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_dataset = cxr_dataset(train_x, train_y, transform_train)
train_loader = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, num_workers=4, drop_last=False)
test_dataset = cxr_dataset(val_x, val_y, transform_val)
test_loader = DataLoader(test_dataset, batch_size=32,
                        shuffle=False, num_workers=4, drop_last=False)

################ train ################

import densenet

net = densenet.densenet121(pretrained=True, drop_rate=0.5).to(device)
net.classifier = nn.Linear(net.classifier.in_features, out_features=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

disp_freq = 500
val_freq = 500

net = train(net, criterion, optimizer, train_loader, test_loader, 100, device, disp_freq, val_freq, save_dir="/media/ethan/model_saves/densenet_combined/")