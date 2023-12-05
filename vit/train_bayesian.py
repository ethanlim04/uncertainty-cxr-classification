# Train model with MIMICIII, 224x224, CLAHE, MC Dropout = 0.2

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
    zeros_df = dataframe[dataframe["labels"] == 0]
    ones_df = dataframe[dataframe["labels"] == 1]


    count = min(len(zeros_df), len(ones_df))
    train_df = pd.concat([zeros_df.iloc[:int(count * 9/10)], ones_df.iloc[:int(count * 9/10)]], ignore_index=True)
    test_df = pd.concat([zeros_df.iloc[int(count * 9/10):], ones_df.iloc[int(count * 9/10):]], ignore_index=True)

    x_train = train_df['img_paths'].tolist()
    y_train = train_df['labels'].tolist()

    x_test = test_df['img_paths'].tolist()
    y_test = test_df['labels'].tolist()

    print(len(x_train), len(x_test))
    print(len(zeros_df), len(ones_df), 9/5*len(ones_df), len(dataframe) - len(x_train))
     
    return x_train, y_train, x_test, y_test

train_x, train_y, val_x, val_y = prepare_dataset(pd.read_csv('~/bayesian/densenet_cxr/mimic.csv'))

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

train_dataset = cxr_dataset(train_x, train_y, transform_train)
train_loader = DataLoader(train_dataset, batch_size=100,
                        shuffle=True, num_workers=4, drop_last=False)
test_dataset = cxr_dataset(val_x, val_y, transform_val)
test_loader = DataLoader(test_dataset, batch_size=100,
                        shuffle=False, num_workers=4, drop_last=False)

################ train ################

from bayesian_vit import MCViT

model = MCViT(num_classes=2, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 250

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Validation Accuracy: {accuracy}")
    
    # Try saving model weights
    try:
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = f"/media/ethan/model_saves/vit/{accuracy}-{epoch}-vit_384"
        torch.save({
            'save_dir': save_path,
            'state_dict': state_dict},
            save_path)
    except:
        continue