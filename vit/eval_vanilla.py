# Evaluate Vanilla ViT model with MIMICIII

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
    return dataframe["img_paths"].tolist(), dataframe["labels"].tolist()

val_x, val_y = prepare_dataset(pd.read_csv("test.csv"))

class cxr_dataset(Dataset):
    def __init__(self, image_paths, image_labels, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        image = clahe.apply(image)#.astype('uint8')

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
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

test_dataset = cxr_dataset(val_x, val_y, transform_val)
test_loader = DataLoader(test_dataset, batch_size=1500,
                        shuffle=False, num_workers=4, drop_last=False)

################ functions for evaluation ################

import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    model.activate_dropout()
    test_loss = 0
    targets, outputs, probs = [], [], []

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = F.softmax(output)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            probs += [prob]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
    
    return torch.cat(probs).unsqueeze(0).cpu().numpy(), F.one_hot(torch.cat(targets), 2).cpu().numpy(), test_loss

################ prepare models ################

from vanilla_vit import ViT
model = ViT(num_classes=2, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model = model.to(device)

################ run tests ################


iter_probs, iter_targets, iter_loss = evaluate(model, test_loader, criterion, device)

true_positive = []
true_negative = []
false_positive = []
false_negative = []


tp, tn, fp, fn = {}, {}, {}, {}

for i in range(len(iter_probs)):
    if(iter_probs.argmax(axis=1)[i] == iter_targets.argmax(axis=1)[i]):
        if(iter_targets.argmax(axis=1)[i] == 1):
            true_positive.append(iter_probs[i])
        else:
            true_negative.append(iter_probs[i])
    else:
        if(iter_targets.argmax(axis=1)[i] == 1):
            false_negative.append(iter_probs[i])
        else:
            false_positive.append(iter_probs[i])
correct = true_positive + true_negative

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist([true_negative, false_positive, false_negative, true_positive], bins=10, alpha=1.0, label=['true negative', 'false positive', 'false negative', 'true_positive'])
axs.set_xlabel("ViT-B Vanilla")
axs.legend()
print(true_positive)
try:
    plt.savefig("/media/ethan/figure_saves/vit_vanilla.png")
except:
    pass
try:
    plt.savefig("figure_saves/vit_vanilla.png")
except:
    pass