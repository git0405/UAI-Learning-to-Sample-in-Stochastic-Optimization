# Importing
# Python
# libraries
from __future__ import print_function, division
import argparse
import time
import os
import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_set = "cifar10"

lr_list = []
kl_list =[]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        out = F.log_softmax(out, dim=1)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
'''
A simple way to define a noisy data loader. We pick a batch of training
set size and introduce corruptions via directed or random noise models 
on randomly chosen samples
'''


def noisy_loader(params):

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])        

    if data_set == 'cifar10':
        train_data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batchsize'], shuffle=True)
        
    # for index, (data, target, _) in enumerate(train_loader):
    for index, (data, target) in enumerate(train_loader):

        flag = np.random.binomial(1, params['epsilon'], size=(len(target), 1))
        target_noisy = copy.deepcopy(target.numpy())
        if params['clean_data'] is False:
            for index, val in enumerate(flag):
                if val[0] == 1 and params['noise_type'] == 'directed':
                    target_noisy[index] = (target[index] + params['shift']) % 10
                if val[0] == 1 and params['noise_type'] == 'random':
                    out = np.random.randint(0, 10)
                    while out == target_noisy[index]:
                        out = np.random.randint(0, 10)
                    target_noisy[index] = out
            break
        elif params['clean_data'] is True:
            print('Do nothing')
            
    target_noisy = torch.from_numpy(target_noisy)
    print(len(target_noisy))
    train_noisy = torch.utils.data.TensorDataset(data, target_noisy)
    

    if params['optimizer_type'] == 'sgd':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']), shuffle=True, drop_last=True)
    elif params['optimizer_type'] == 'mkl':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=params['k'], shuffle=True, drop_last=True)
    elif params['optimizer_type'] == 'pac-bayes':
        # print("train data len: ", len(train_noisy))
        w = np.full((len(train_noisy)), 1 / len(train_noisy))
        samp = torch.utils.data.WeightedRandomSampler(torch.Tensor(w).to(device), len(train_noisy))

        train_loader_noisy_uni = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']),shuffle=True, num_workers=4)

        train_loader_noisy_train = torch.utils.data.DataLoader(train_noisy,
                                                               batch_size=int(params['frac'] * params['k']),
                                                               shuffle=False, sampler=samp, num_workers=4)
        train_loader_eval = torch.utils.data.DataLoader(train_noisy, batch_size=100, shuffle=False, num_workers=4)

        train_loader_noisy = [train_loader_noisy_uni, train_loader_noisy_train, train_loader_eval, samp]

    return train_loader_noisy,train_noisy


# Training and Testing
# Phase


def train(train_loader_noisy, epoch, run, epsilon, params):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_noisy):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        if params['optimizer_type'] == 'sgd' or params['optimizer_type'] == 'pac-bayes':
            optimizer.zero_grad()
            output = model(data)

            
            loss = F.nll_loss(output, target)
        elif params['optimizer_type'] == 'mkl':
            output = model(data)
            temp_loss = F.nll_loss(output, target.to(device), reduction='none')
            temp = temp_loss.cpu().detach().numpy()
            # Pick the samples with the lowest loss
            index1 = np.argpartition(temp, int(params['frac'] * params['k']))
            data1 = data[index1[:int(params['frac'] * params['k'])], :, :, :].view(int(params['frac'] * params['k']),3, 32, 32)
            target1 = target[index1[:int(params['frac'] * params['k'])]]
            data1, target1 = Variable(data1.to(device)), Variable(target1.to(device))
            output1 = model(data1)
            optimizer.zero_grad()
            loss = F.nll_loss(output1, target1)

        loss.backward()
        optimizer.step()

        if batch_idx % 250 == 0:
            print('Train Run: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, {}'.format(
                run + 1, epoch, batch_idx * len(data), len(train_loader_noisy.dataset),
                100. * batch_idx / len(train_loader_noisy), loss.item(), params['optimizer_type']))
    for param_group in optimizer.param_groups:
        # print(f"Current learning rate: {param_group['lr']}")
        lr_list.append({param_group['lr']})
    return loss.item()


def test(test_loader, run):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)

    
    print('\nTest set {} Run: {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        params['optimizer_type'], run + 1, test_loss, correct, len(test_loader.dataset),
                                  100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def select_optimizer(optimizer_name, params):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'],weight_decay=5e-4)
    elif optimizer_name == 'sgdmomentum':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=5e-4)
    return optimizer


def one_run(train_loader_noisy, model, optimizer, params, results):
    num_epochs = params['num_epochs']
    count = params['current_run']

    time_start = time.time()
    
    
    if params['optimizer_type'] == 'sgd' or params['optimizer_type'] == 'mkl':
        train_loader_noisy_uni = train_loader_noisy

        if params['decayschedule'] != 0:
            scheduler.step()
        
        for epoch in range(1, num_epochs + 1):
        
            results['train_loss'][epoch - 1, count] = train(train_loader_noisy_uni, epoch, params['current_run'], params['epsilon'], params)

            results['test_loss'][epoch - 1, count], results['test_acc'][epoch - 1, count] = test(test_loader, params['current_run'])
            results['time_spent'][epoch - 1, count] = time.time() - time_start
        return results
    
    elif params['optimizer_type'] == 'pac-bayes':
        train_loader_noisy_uni, train_loader_noisy_train, train_loader_eval, samp = train_loader_noisy
        q_prior = torch.full((len(samp.weights),), 1 / len(samp.weights), device=torch.device(device))

        results['train_loss'][0, count] = train(train_loader_noisy_uni, 1, params['current_run'], params['epsilon'], params)

        results['test_loss'][0, count], results['test_acc'][0, count] = test(test_loader, params['current_run'])  
    
    model.eval()
    with torch.no_grad():

        loss_temp = torch.zeros(len(samp.weights), ).to(device)

        for i, data in enumerate(train_loader_eval):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels, reduction='none')
            idlist = torch.arange(i * len(labels), (i + 1) * len(labels))

            loss_temp[idlist[:]] = loss

        q_pos = torch.exp(-loss_temp)
        q_pos = q_pos / torch.sum(q_pos)
        w = q_pos.detach().clone()

        ##===compute kl divergence
        term1 = torch.log(torch.div(q_pos, q_prior))
        kl_div = (torch.mul(q_pos, term1)).sum()
        print(kl_div)
        kl_list.append(kl_div.cpu().numpy().squeeze())

        samp.weights = w.to(device)


    results['time_spent'][0, count] = time.time() - time_start


    for epoch in range(2, num_epochs + 1):
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        if params['decayschedule'] != 0:
            scheduler.step()

        results['train_loss'][epoch - 1, count] = train(train_loader_noisy_train, epoch, params['current_run'], params['epsilon'], params)
        results['test_loss'][epoch - 1, count], results['test_acc'][epoch - 1, count] = test(test_loader, params['current_run'])

        model.eval()
        with torch.no_grad():

            loss_temp = torch.zeros(len(samp.weights), ).to(device)
            pred_temp = np.zeros(len(samp.weights), ) 
            true_label_temp = np.zeros(len(samp.weights), ) 

            for i, data in enumerate(train_loader_eval):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels, reduction='none')
                idlist = torch.arange(i * len(labels), (i + 1) * len(labels))

                loss_temp[idlist[:]] = loss
                
                pred = outputs.data.max(1, keepdim=True)[1].cpu().numpy()
                
                pred_temp[idlist[:]] = pred.squeeze()
                true_label_temp[idlist[:]] = labels.cpu().numpy().squeeze()

            q_pos = torch.exp(-loss_temp)
            q_pos = q_pos / torch.sum(q_pos)
            w = q_pos.detach().clone()

            ##===compute kl divergence
            term1 = torch.log(torch.div(q_pos, q_prior))
            kl_div = (torch.mul(q_pos, term1)).sum()
            print(kl_div)
            kl_list.append(kl_div.cpu().numpy().squeeze())
            samp.weights = w.to(device)

        results['time_spent'][epoch - 1, count] = time.time() - time_start

    return results
# Define
# parameters


def init_params():
    params = {}
    params['lr'] = 0.1 # 0.05 # 0.02 for mnist # 0.005#0.01  # Learning rate 
    params['momentum'] = 0.9  # Momentum parameter
     
    
    params['batchsize'] = 50000  # Size of dataset, parameter used in noisy_loader
    params['decayschedule'] = 60  # Decay Schedule
    params['noise_type'] = 'random'  # Noise type
    #    To use directed noise model of corruption, set 'directed'
    #    To use random noise model of corruption, set 'random'
    params['learningratedecay'] = 0.2  # Learning Rate Decay
    params['eps'] = 1e-08  # Corruption parameter
    params['num_epochs'] = 100  # Number of epochs
    params['optimizer_type'] = 'sgd'  # Optimizer Type:
    
    
    
    #    To run standard stochastic gradient descent, set 'sgd'
    #    To run standard MKL-SGD, set 'mkl' 'pac-bayes'
    params['runs'] = 3  # Number of runs
    params['epsilon'] = 0.4  # Fraction of corrupted data
    params['frac'] = 0.6  # Fraction of samples with lowest loss chosen
    #    Number of gradient updates = params['frac'] * params['k']
    params['clean_data'] = False  # Flag to only consider clean data
    params['shift'] = 2  # Shift parameter in directed noise model
    if params['optimizer_type'] == 'mkl':
        params['k'] = 214 # Number of loss evaluations per batch 16 or 214
    else:
        params['k'] = 214

    results = {}
    results['train_loss'] = np.zeros((params['num_epochs'], params['runs']))
    results['test_loss'] = np.zeros((params['num_epochs'], params['runs']))
    results['test_acc'] = np.zeros((params['num_epochs'], params['runs']))
    results['time_spent'] = np.zeros((params['num_epochs'], params['runs']))


    if data_set == "cifar10":
        transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        testset = datasets.CIFAR10('../data', train=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return params, test_loader, results


# Main
# function
time_start = time.time()
params, test_loader, results = init_params()
for run in range(params['runs']):

    train_loader_noisy,_ = noisy_loader(params)
    # print(len(train_loader_noisy))

    if data_set == "cifar10":
        model = ResNet18()
        
    model.to(device)
    optimizer = select_optimizer('sgdmomentum', params)
    if params['decayschedule'] != 0:
        # scheduler = ReduceLROnPlateau(optimizer, 'min' )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params['decayschedule'], gamma=params['learningratedecay'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    params['current_run'] = run
    

    results = one_run(train_loader_noisy, model, optimizer, params, results)


# Saving parameters

if params['optimizer_type'] == 'pac-bayes':
    if data_set =="mnist":
        file_path = "./results/MNIST_pacb_latest_%s_%s/lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d" \
                    % (params['optimizer_type'], params['noise_type'], params['lr'],
                       params['momentum'], params['epsilon'], params['decayschedule'],
                       params['num_epochs'], params['runs'], params['k'])
    elif data_set == "cifar10":
        
        file_path = "./results/cifar10_pacb_latest_%s_%s/%s_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_clean_%s" \
                    % (params['optimizer_type'], params['noise_type'],params['optimizer_type'], params['lr'],
                       params['momentum'], params['epsilon'], params['decayschedule'],
                       params['num_epochs'], params['runs'], params['k'],params['clean_data'])
        #  file_path_lr = "./results/cifar10_pacb_latest_%s_%s/lr_list_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d" \
        #             % (params['optimizer_type'], params['noise_type'], params['lr'],
        #                params['momentum'], params['epsilon'], params['decayschedule'],
        #                params['num_epochs'], params['runs'], params['k'])
        # np.savez(file_path_lr,lr_list = lr_list)

if params['optimizer_type'] == 'sgd':
    file_path = "./results/cifar10_SGD_wideresnet_%s_%s/%s_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_clean_%s"\
                % (params['optimizer_type'], params['noise_type'],params['optimizer_type'], params['lr'], 
                   params['momentum'], params['epsilon'], params['decayschedule'], 
                   params['num_epochs'], params['runs'], params['k'],params['clean_data'])
elif params['optimizer_type'] == 'mkl':
    file_path = "./results/cifar10_SGD_latest_%s_%s/%s_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_frac_%.2f_clean_%s"\
                % (params['optimizer_type'], params['noise_type'],params['optimizer_type'], params['lr'], 
                   params['momentum'], params['epsilon'], params['decayschedule'], 
                   params['num_epochs'], params['runs'], params['k'], params['frac'],params['clean_data'])


directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
np.savez(file_path, train_loss=results['train_loss'], test_loss=results['test_loss'], test_acc=results['test_acc'], time_spent=results['time_spent'])