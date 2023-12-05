# Evaluate Bayesian DenseNet model with CheXpert + MIMIC dataset
# Varience in confidence scores are used as a metric to quantify uncertainty

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

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

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

################ functions for evaluation ################

def mc_evaluate(net, test_loader, criterion, device):
    net.eval()
    test_loss = 0
    targets, outputs, probs = [], [], []
    paths = []

    with torch.no_grad():
        for batch_id, (image_path, data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            prob = F.softmax(output)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            probs += [prob]
            test_loss += batch_loss

            paths += [image_path]

        test_loss /= (batch_id + 1)
    return torch.cat(probs).unsqueeze(0).cpu().numpy(), F.one_hot(torch.cat(targets), 2).cpu().numpy(), test_loss, paths

def mutual_info(mc_prob):
    eps = 1e-5
    mean_prob = mc_prob.mean(axis=0)
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term

def predictive_entropy(prob):
    eps = 1e-5
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)

################ prepare dataset ################

from torch.utils.data import Dataset, DataLoader
import cv2

def prepare_dataset(dataframe):
    x = dataframe['img_paths'].tolist()
    y = dataframe['labels'].tolist()
    return x, y

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

val_x, val_y = prepare_dataset(pd.read_csv('test.csv'))
test_dataset = cxr_dataset(val_x, val_y, transform_val)
test_loader = DataLoader(test_dataset, batch_size=250,
                        shuffle=False, num_workers=4, drop_last=False)

################ prepare models ################

checkpoint_path = "/media/ethan/model_saves/densenet_512_combined/0.7613527138134301_35_54600.ckpt"

import densenet

net = densenet.densenet121().to(device)
net.classifier = nn.Linear(net.classifier.in_features, out_features=2).to(device)

checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

################ run tests ################

mc_count = 100

mc_probs = []
for mc_iter in range(mc_count):
    print('running mc iteration # {}'.format(mc_iter))
    iter_probs, iter_targets, iter_loss, img_paths = mc_evaluate(net, test_loader, criterion, device)
    mc_probs += [iter_probs]
mc_probs = np.concatenate(mc_probs)  # [mc_iter, N, C]

mean_prob = mc_probs.mean(axis=0)
var_pred_entropy = predictive_entropy(mean_prob)
var_pred_MI = mutual_info(mc_probs)
acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), iter_targets.argmax(axis=1))) / mean_prob.shape[0]
print('accuracy={0:.02%}'.format(acc))

true_positive = []
true_negative = []
false_positive = []
false_negative = []

tp, tn, fp, fn = {}, {}, {}, {}

for i in range(len(var_pred_MI)):
    if(mean_prob.argmax(axis=1)[i] == iter_targets.argmax(axis=1)[i]):
        if(iter_targets.argmax(axis=1)[i] == 1):
            true_positive.append(var_pred_entropy[i])
            tp[var_pred_entropy[i]] = img_paths[i]
        else:
            true_negative.append(var_pred_entropy[i])
            tn[var_pred_entropy[i]] = img_paths[i]
    else:
        if(iter_targets.argmax(axis=1)[i] == 1):
            false_negative.append(var_pred_entropy[i])
            fn[var_pred_entropy[i]] = img_paths[i]
        else:
            false_positive.append(var_pred_entropy[i])
            fp[var_pred_entropy[i]] = img_paths[i]

correct = true_positive + true_negative

################ visualize results ################

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist([true_negative, false_positive, false_negative], bins=10, alpha=1.0, label=['true negative', 'false positive', 'false negative'])
axs.set_xlabel("Densenet121 with MC Dropout")
axs.legend()
plt.savefig("figure_saves/densenet121_512_clahe.png")


for threshhold in [0.2, 0.1, 0.05]:
    print("threshhold:", threshhold)
    tn_count = 0
    for data in true_negative:
        if(data < threshhold):
            tn_count += 1
    print("true negative", tn_count)

    fp_count = 0
    for data in false_positive:
        if(data < threshhold):
            fp_count += 1
    print("false positive", fp_count)

    points = []
    for data in false_negative:
        if(data < threshhold):
            points.append(data)
    print("false negative", len(points))

    tp_count = 0
    for data in true_positive:
        if(data < threshhold):
            tp_count += 1

    print("true positive:", tp_count)

    total = tn_count + fp_count + len(points) + tp_count
    print("total:", total)
    print("FN:", len(points)/total, "%\n")