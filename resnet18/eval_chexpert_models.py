# Evaluate Bayesian ResNet model and Probabilistic model with CheXpert dataset
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

import torchvision.transforms as transforms
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
import numpy as np

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

def return_model_answer_bayesian(model, img_path, mc_num):
    model.eval()
    import cv2
    from PIL import Image

    # Preprocessing
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = cv2.imread(img_path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    img = clahe.apply(img)#.astype('uint8') ##
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ##
    image = Image.fromarray(img)
    image = transform_val(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    # Iterate over number of experiments, dropout with a probability of 0.5    
    for mc in range(mc_num):
        predicted = model(image, dropout=True, p=0.5).softmax(1).unsqueeze(1)
        mc_output = torch.cat((mc_output, predicted), dim=1)
    
    # Calculate mean and varience
    mean = mc_output.mean(dim=1)
    var = mc_output.var(dim=1)
    
    return (mean.argmax().item(), np.mean(var.data.cpu().numpy()))

def return_model_answer_probabilistic(model, img_path, mc_num):
    model.eval()
    import cv2
    from PIL import Image
    
    # Preprocessing
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = cv2.imread(img_path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    img = clahe.apply(img)#.astype('uint8') ##
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ##
    image = Image.fromarray(img)
    image = transform_val(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    predicted, means, log_vars = model(image)

    # Iterate over number of experiments, dropout with a probability of 0.5    
    mc_output = predicted.softmax(1).unsqueeze(1)
    for mc in range(mc_num):
        predicted = model.reparameterize(means, log_vars).softmax(1).unsqueeze(1)
        mc_output = torch.cat((mc_output, predicted), dim=1)

    # Calculate mean and varience
    mean = mc_output.mean(dim=1)
    var = mc_output.var(dim=1)
    
    return (mean.argmax().item(), np.mean(var.data.cpu().numpy()))

################ prepare dataset ################

def prepare_dataset(dataframe):
    dataset_df = dataframe[dataframe['Frontal/Lateral'] == 'Frontal'] #take frontal pics only
    df = dataset_df.sample(frac=1., random_state=1)
    df.fillna(0, inplace=True)
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

val_x, val_y = prepare_dataset(pd.read_csv('/media/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/test.csv'))

################ prepare models ################

import bayesian_resnet_2
import probabilistic_resnet

bayesian_model = bayesian_resnet_2.BayesianResNet2(num_classes=2)
probabilistic_model = probabilistic_resnet.ProbabilisticResNet(num_classes=2)

probabilistic_checkpoint = torch.load("/media/ethan/model_saves/probabilistic_chexpert/_best.pth.tar", map_location=device)
bayesian_checkpoint = torch.load("/media/ethan/model_saves/bayesian2_chexpert/_best.pth.tar", map_location=device)

probabilistic_model.load_state_dict(probabilistic_checkpoint['state_dict'])
bayesian_model.load_state_dict(bayesian_checkpoint['state_dict'])

probabilistic_model = probabilistic_model.to(device)
bayesian_model = bayesian_model.to(device)

################ run tests ################

probabilistic_false_negative = []
probabilistic_false_positive = []
probabilistic_correct = []
bayesian_false_negative = []
bayesian_false_positive = []
bayesian_correct = []

for i in range(len(val_x)):
    x = val_x[i]
    y = val_y[i]
    probabilistic_res, probabilistic_var = return_model_answer_probabilistic(probabilistic_model, x, 100)
    bayesian_res, bayesian_var = return_model_answer_bayesian(bayesian_model, x, 100)

    # Varience in confidence scores are used as a metric to quantify uncertainty
    if(y == probabilistic_res):
        probabilistic_correct.append(probabilistic_var)
    else:
        if(y == 0):
            probabilistic_false_positive.append(probabilistic_var)
        else:
            probabilistic_false_negative.append(probabilistic_var)
    
    if(y == bayesian_res):
        bayesian_correct.append(bayesian_var)
    else:
        if(y == 0):
            bayesian_false_positive.append(bayesian_var)
        else:
            bayesian_false_negative.append(bayesian_var)

################ visualize results ################

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist([bayesian_correct, bayesian_false_positive, bayesian_false_negative], bins=10, density=True, alpha=1.0, label=['correct', 'false positive', 'false negative'])
axs.set_xlabel("Bayesian with MC Dropout")
axs.legend()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist([probabilistic_correct, probabilistic_false_positive, probabilistic_false_negative], bins=10, density=True, alpha=1.0, label=['correct', 'false positive', 'false negative'])
axs.set_xlabel("Probabilistic")
axs.legend()
plt.show()