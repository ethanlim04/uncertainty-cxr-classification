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

# KL Divergence loss from https://arxiv.org/abs/1312.6114
def kld_loss(mean, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return kld / mean.size(0)  # norm by batch size

# Identity layer used in model
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# model from https://arxiv.org/pdf/1908.00792v1.pdf
class ProbabilisticResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = Identity()

        self._linear_means = torch.nn.Linear(512, num_classes)
        self._linear_log_vars = torch.nn.Linear(512, num_classes)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self._resnet(x)

        means = self._linear_means(x)
        log_vars = self._linear_log_vars(x)

        # reparameterize model only when model is in training mode
        if self.training:
            y = self.reparameterize(means, log_vars)
        else:
            y = means

        return y, means, log_vars
    
def train_model(model, optimizer, lr_scheduler, criterion, device, save_path, dataloader_train, dataloader_valid, lambda_prop, epochs):
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
        is_best = False

        x, y, y_pred = None, None, None
        batches = tqdm(dataloader_train)
        for x, y in batches:

            # typecast to avoid errors
            y = y.type(torch.LongTensor)

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()


            y_pred, means, log_vars = model(x)
            train_loss = criterion(y_pred, y) + lambda_prop * kld_loss(means, log_vars)

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


                y_pred, means, log_vars = model(x)
                valid_loss = criterion(y_pred, y) + lambda_prop * kld_loss(means, log_vars)


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
            is_best = True

        if is_best:
            filename = "/media/ethan/model_saves/" + save_path + "/" + valid_losses[-1] + "_best.pth.tar"
            print("Saving best weights so far with val_loss: {:4f}".format(valid_losses[-1]))
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