import torch
import numpy as np
import torchvision
import glob as glob
from tqdm import tqdm

# flatten vector
def flatten(x):
    return x.view(x.size(0), -1)

# calculate accuracy over prediction
def accuracy(input, target):
    _, max_indices = torch.max(input.data, 1)
    acc = (max_indices == target).sum().float() / max_indices.size(0)
    return acc.item()

class BayesianResNet2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # use pretrained resnet18 model, modify last conv layer to output desired number of classes
        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        # forcefully set dropout to override evaluation mode
        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer1(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer2(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer3(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer4(x)

        x = self._resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        y = self._resnet.fc(x)

        return y
    
def train_model(model, optimizer, lr_scheduler, criterion, device, save_path, dataloader_train, dataloader_valid, bayesian_dropout_p, epochs):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    e = 0

    for e in range(epochs):

        model.train()
        print("lr =", optimizer.param_groups[0]['lr'])

        epoch_train_loss = []
        epoch_train_acc = []

        x, y, y_pred = None, None, None
        batches = tqdm(dataloader_train)
        for x, y in batches:

            # typecast to avoid errors
            y = y.type(torch.LongTensor)
            
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x, dropout=True, p=bayesian_dropout_p)
            train_loss = criterion(y_pred, y)            
            
            train_loss.backward()
            optimizer.step()

            # print current loss
            batches.set_description("loss: {:4f}".format(train_loss.item()))

            # sum epoch loss
            epoch_train_loss.append(train_loss.item())

            # calculate batch train accuracy
            batch_acc = accuracy(y_pred, y)
            epoch_train_acc.append(batch_acc)

        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_acc = np.mean(epoch_train_acc)
        lr_scheduler.step(epoch_train_loss)

        # go through validation set
        model.eval()
        with torch.no_grad():

            epoch_valid_loss = []
            epoch_valid_acc = []

            batches = tqdm(dataloader_valid)
            for x, y in batches:

                y = y.type(torch.LongTensor)
                x, y = x.to(device), y.to(device)

                y_pred = model(x, dropout=True, p=bayesian_dropout_p)
                valid_loss = criterion(y_pred, y)

                # print current loss
                batches.set_description("loss: {:4f}".format(valid_loss.item()))

                # sum epoch loss
                epoch_valid_loss.append(valid_loss.item())

                # calculate batch train accuracy
                batch_acc = accuracy(y_pred, y)
                epoch_valid_acc.append(batch_acc)

        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_acc = np.mean(epoch_valid_acc)

        print("Epoch {:d}: loss: {:4f}, acc: {:4f}, val_loss: {:4f}, val_acc: {:4f}"
              .format(e,
                      epoch_train_loss,
                      epoch_train_acc,
                      epoch_valid_loss,
                      epoch_valid_acc,
                      ))

        # save epoch information
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_acc)

        if valid_losses[-1] <= np.min(valid_losses):
            filename = "/media/ethan/model_saves/" + save_path + "/" + valid_losses[-1] + "_best.pth.tar"
            print("Saving best weights so far with validation loss: {:4f}".format(valid_losses[-1]))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accuracies,
                'val_losses': valid_losses,
                'val_accs': valid_accuracies,
            }, filename)
            
    return model