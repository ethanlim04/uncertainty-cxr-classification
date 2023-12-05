import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
import os


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

# Bottleneck forward
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

# DenseNet Layer
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    # Apply dropout to each channel to preserve spatial structure
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)

        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=True)
        return new_features

# DenseNet Block with n layers
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

# Transition Block
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer for final dimension reduction
        self.classifier = nn.Linear(num_features, num_classes)

        self.num_features = num_features
        
        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)

def compute_accuracy(target, output):
    num_samples = target.size(0)
    num_correct = torch.sum(target == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy

def train(net, criterion, optimizer, train_loader, test_loader, epochs, device, disp_freq, val_freq, save_dir):
    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(epochs):
        print('--' * 30)
        print('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss = 0
        targets, outputs = [], []

        for batch_id, (data, target) in enumerate(train_loader):
            global_step += 1

            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            train_loss += batch_loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (batch_id + 1) % disp_freq == 0:
                print('==' * 30)
                train_loss /= disp_freq
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))
                print("epoch: {0}, step: {1}, train_loss: {2:.4f} train_accuracy: {3:.02%}"
                           .format(epoch+1, batch_id+1, train_loss, train_acc))
                train_loss = 0
                targets, outputs = [], []

            if (batch_id + 1) % val_freq == 0:
                print('--' * 30)
                print('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(net, criterion, test_loader, save_dir, epoch,
                                               best_loss=best_loss,
                                               best_acc=best_acc,
                                               global_step=global_step)
                net.train()

    return net


def evaluate(net, criterion, test_loader, save_dir, device, epoch, **kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    net.eval()
    test_loss = 0
    targets, outputs = [], []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
        test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))

        # check for improvement
        loss_str, acc_str = '', ''
        if test_loss <= best_loss:
            loss_str, best_loss = '(improved)', test_loss
        if test_acc >= best_acc:
            acc_str, best_acc = '(improved)', test_acc

        # display
        print("validation_loss: {0:.4f} {1}, validation_accuracy: {2:.02%}{3}"
                   .format(test_loss, loss_str, test_acc, acc_str))

        # save checkpoint model
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(save_dir, '{}.ckpt'.format(str(test_acc.item()) + '_' + str(epoch) + '_' + str(global_step)))
        torch.save({
            'global_step': global_step,
            'loss': test_loss,
            'acc': test_acc,
            'save_dir': save_path,
            'state_dict': state_dict},
            save_path)
        print('Model saved at: {}'.format(save_path))
        return best_loss, best_acc
    

@article{cicalese2020kidney,
  title={Kidney Level Lupus Nephritis Classification using Uncertainty Guided Bayesian Convolutional Neural Networks},
  author={Cicalese, Pietro A and Mobiny, Aryan and Shahmoradi, Zahed and Yi, Xiongfeng and Mohan, Chandra and Van Nguyen, Hien},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2020},
  publisher={IEEE}
}