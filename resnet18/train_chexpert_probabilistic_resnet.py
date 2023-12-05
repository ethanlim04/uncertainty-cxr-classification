# Train model with CheXpert, 224x224, CLAHE, MC Dropout = 0.5

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
    dataset_df = dataframe[dataframe['Frontal/Lateral'] == 'Frontal'] # take frontal x-rays only
    df = dataset_df.sample(frac=1., random_state=1)
    df.fillna(0, inplace=True) # replace NaNs with zeros
    x = []
    res = []
    for i, (index, row) in enumerate(df.iterrows()):
        if(row['No Finding'] == 1):
            label = 0
        else:
            label = 1
        res.append(label)
        x.append("/media/CheXpert/chexpertchestxrays-u20210408/" + row['Path'])
    return x, res

class chexpert_dataset(Dataset):
    def __init__(self, image_paths, image_labels, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # apply CLAHE & preprocess image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        image = clahe.apply(image)#.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image_res = Image.fromarray(image)
        
        label = self.image_labels[idx]
        
        if(self.transform is not None):
            image_res = self.transform(image_res)
        
        return image_path, image_res, label

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_x, train_y = prepare_dataset(pd.read_csv('/media/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv'))
val_x, val_y = prepare_dataset(pd.read_csv('/media/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/valid.csv'))

train_ds = chexpert_dataset(train_x, train_y, transform_train)
val_ds = chexpert_dataset(val_x, val_y, transform_val)

dataloader_train = DataLoader(train_ds, batch_size=750, shuffle=True, num_workers=4)
dataloader_valid = DataLoader(val_ds, batch_size=750, shuffle=True, num_workers=4)

################ train ################

import probabilistic_resnet

model = probabilistic_resnet.ProbabilisticResNet(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                             betas=(0.9, 0.999),
                             weight_decay=1e-8)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)
criterion = nn.CrossEntropyLoss()

model = probabilistic_resnet.train_model(model, optimizer, lr_scheduler, criterion, device, "probabilistic_chexpert",
                                        dataloader_train, dataloader_valid, lambda_prop=0.01, epochs=100)