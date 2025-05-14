# Importing Python libraries
from __future__ import print_function, division
import argparse
import time
import os
import copy
import math
import matplotlib

matplotlib.use('TkAgg')
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

from dataset import SiameseMNIST, SiameseNetworkDataset,SiameseNetworkDataset_ben,SiameseCUHK03
from models import EmbeddingNet, SiameseNet, Siamese_ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_set = "cuhk03"


def noisy_loader(params):
    if data_set == "mnist":
        dataset = SiameseMNIST(datasets.MNIST('../data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.1307,),
                                                                                                 (0.3081,))])))

    elif data_set == "cifar10":

        dataset = SiameseCIFAR10(datasets.CIFAR10('../data', train=True, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize(
                                                                                    (0.4914, 0.4822, 0.4465),
                                                                                    (0.247, 0.243, 0.261))])))


    elif data_set == "cuhk03":
        dataset = SiameseCUHK03(train=True)


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batchsize'], shuffle=False)
    for index, (data, target) in enumerate(train_loader):
        flag = np.random.binomial(1, params['epsilon'], size=(len(target), 1))
        target_noisy = copy.deepcopy(target.numpy())
        if params['clean_data'] is False:
            for index, val in enumerate(flag):
                if val[0] == 1 and params['noise_type'] == 'directed':
                    target_noisy[index] = (target[index] + params['shift']) % 10

                    target_noisy[index] = np.abs(target[index] - 1)

                if val[0] == 1 and params['noise_type'] == 'random':
                    out = np.random.randint(0, 2)
                    while out == target_noisy[index]:
                        out = np.random.randint(0, 2)
                    target_noisy[index] = out

            break
        elif params['clean_data'] is True:
            print('Do nothing')

    sorted_indices = np.argsort(target_noisy)[::-1].copy()
    target_noisy = target_noisy[sorted_indices]
    target_noisy = torch.from_numpy(target_noisy)


    if params['optimizer_type'] =="ben_london":
        train_noisy = SiameseNetworkDataset_ben(data, target_noisy)
    else:
        train_noisy = SiameseNetworkDataset(data, target_noisy)

    if params['optimizer_type'] == 'sgd':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']),
                                                         shuffle=True, drop_last=True)
    elif params['optimizer_type'] == 'mkl':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=params['k'], shuffle=True,
                                                         drop_last=True)

    elif params['optimizer_type'] == 'ben_london':
        print("train data len: ", len(target_noisy))
        w = np.full((len(target_noisy)), 1 / len(target_noisy))
        samp = torch.utils.data.WeightedRandomSampler(torch.Tensor(w).to(device), len(target_noisy))


        train_loader_noisy_train = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']),
                                                               shuffle=False, sampler=samp, drop_last=True)

        train_loader_noisy = [ train_loader_noisy_train , samp]

    elif params['optimizer_type'] == 'pac-bayes':
        w = np.full((len(target_noisy)), 1 / len(target_noisy))
        samp = torch.utils.data.WeightedRandomSampler(torch.Tensor(w).to(device), len(train_noisy))

        train_loader_noisy_uni = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']), shuffle=True, drop_last=True)
        train_loader_noisy_train = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']), shuffle=False, sampler=samp, drop_last=True)
        train_loader_eval = torch.utils.data.DataLoader(train_noisy, batch_size=1000, shuffle=False)

        train_loader_noisy = [train_loader_noisy_uni, train_loader_noisy_train, train_loader_eval, samp]

    return train_loader_noisy

##======= PAC-Adaptive sampling
def updateSampleWeights(w, idx, f, tau=0.25, alpha=0.5):
    '''
    Updates sample weights w assuming example at index idx was drawn. tau indicates decay (assumed 0 to 1), alpha aggressiveness
    of the update. F is the utility function
    '''
    # decaying towards uniform distribution
    w = (w - 1 / np.size(w)) * math.exp(-tau) + 1 / np.size(w)

    w[idx[:]] = w[idx[:]] * np.exp(alpha * f)
    # normalizing
    w = w / np.sum(w)
    return w


tau = 0.25
alpha = 0.5

def train(train_loader_noisy, epoch, run, epsilon, params,w,samp):
    model.train()
    for batch_idx, (data1, data2, target,idx) in enumerate(train_loader_noisy):
        data1, data2, target = Variable(data1.to(device), requires_grad=True), Variable(data2.to(device),
                                                                                        requires_grad=True), Variable(
            target.to(device))
        if params['optimizer_type'] == 'sgd' or params['optimizer_type'] == 'pac-bayes':
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = F.nll_loss(output, target)

        elif params['optimizer_type'] == 'ben_london':
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = F.nll_loss(output, target)
            _, predicted = torch.max(output, 1)
            f = np.where((predicted == target).cpu().numpy(),0,1)
            w = updateSampleWeights(w, idx.cpu().numpy(),f, tau, alpha )
            samp.weights = torch.Tensor(w).to(device)


        elif params['optimizer_type'] == 'mkl':
            output = model(data)
            temp_loss = F.nll_loss(output, target.to(device), reduction='none')
            temp = temp_loss.cpu().detach().numpy()
            index1 = np.argpartition(temp, int(params['frac'] * params['k']))
            data1 = data[index1[:int(params['frac'] * params['k'])], :, :, :].view(int(params['frac'] * params['k']), 1,
                                                                                   28, 28)
            target1 = target[index1[:int(params['frac'] * params['k'])]]
            data1, target1 = Variable(data1.to(device)), Variable(target1.to(device))
            output1 = model(data1)
            optimizer.zero_grad()
            loss = F.nll_loss(output1, target1)
        loss.backward()
        optimizer.step()

        if batch_idx % 250 == 249:
            print('Train Run: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, {}'.format(
                run + 1, epoch, batch_idx * len(target), len(train_loader_noisy.dataset),
                100. * batch_idx / len(train_loader_noisy), loss.item(), params['optimizer_type']))
    return loss.item()

def test(test_loader, run):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data1, data2 = data

            data1, data2, target = Variable(data1.to(device)), \
                                   Variable(data2.to(device)), Variable(target.to(device))
            output = model(data1, data2)

            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).float().cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set {} Run: {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        params['optimizer_type'], run + 1, test_loss, correct, len(test_loader.dataset),
                                  100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


def select_optimizer(optimizer_name, params):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    elif optimizer_name == 'sgdmomentum':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
    return optimizer


def one_run(train_loader_noisy, model, optimizer, params, results):
    num_epochs = params['num_epochs']
    count = params['current_run']

    if params['optimizer_type'] == 'ben_london':
        train_loader_noisy, samp = train_loader_noisy
        data_len = len(samp.weights)
        q_prior = torch.full((data_len,), 1 / data_len, device=torch.device(device))
        w = np.full((data_len), 1 /data_len)
        train_loader_noisy = train_loader_noisy

        time_start = time.time()
        for epoch in range(1, num_epochs + 1):
            results['train_loss'][epoch - 1, count] = train(train_loader_noisy, epoch, params['current_run'],
                                                    params['epsilon'], params, w, samp)
            results['test_loss'][epoch - 1, count], results['test_acc'][epoch - 1, count] = test(test_loader, params['current_run'])
            results['time_spent'][epoch - 1, count] = time.time() - time_start

    return results


def init_params(lr = 0.02,k = 27,decay = 15,lr_decay = 0.2):
    params = {}
    params['lr'] = lr  # Learning rate
    params['momentum'] = 0.9  # Momentum parameter
    params['k'] = k  # Number of loss evaluations per batch
    params['batchsize'] = 50000  # Size of dataset, parameter used in noisy_loader
    params['decayschedule'] = decay  # Decay Schedule
    params['noise_type'] = 'directed'  # Noise type
    #    To use directed noise model of corruption, set 'directed'
    #    To use random noise model of corruption, set 'random'
    params['learningratedecay'] = lr_decay  # Learning Rate Decay
    params['eps'] = 1e-08  # Corruption parameter
    params['num_epochs'] = 80  # Number of epochs
    params['optimizer_type'] = 'ben_london'  # Optimizer Type:
    #    To run standard stochastic gradient descent, set 'sgd'
    #    To run standard MKL-SGD, set 'mkl'
    params['runs'] = 5  # Number of runs
    params['epsilon'] = 0.1  # Fraction of corrupted data
    params['frac'] = 0.6  # Fraction of samples with lowest loss chosen
    #    Number of gradient updates = params['frac'] * params['k']
    params['clean_data'] = False  # Flag to only consider clean data
    params['shift'] = 2  # Shift parameter in directed noise model

    results = {}
    results['train_loss'] = np.zeros((params['num_epochs'], params['runs']))
    results['test_loss'] = np.zeros((params['num_epochs'], params['runs']))
    results['test_acc'] = np.zeros((params['num_epochs'], params['runs']))
    results['time_spent'] = np.zeros((params['num_epochs'], params['runs']))

    if data_set == "mnist":
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        testset = SiameseMNIST(datasets.MNIST('../data', train=False, transform=transform_test))

    elif data_set == "cifar10":
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        testset = SiameseCIFAR10(datasets.CIFAR10('../data', train=False, download=True, transform=transform_test))


    elif data_set == "cuhk03":

        testset = SiameseCUHK03(train=False)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
    return params, test_loader, results


lr_list= [ 0.01 ]
k_lsit = [54]
lr_decay_list = [0.2 ]
decay_list = [ 20]


for lr in lr_list:
    for k in k_lsit:
        for lr_decay in lr_decay_list:
            for decay in decay_list:

                time_start = time.time()
                params, test_loader, results = init_params(lr = lr,k = k,decay = decay,lr_decay = lr_decay)
                for run in range(params['runs']):
                    train_loader_noisy = noisy_loader(params)
                    model = Siamese_ResNet()
                    model.to(device)

                    optimizer = select_optimizer('sgd', params)
                    if params['decayschedule'] != 0:
                        scheduler = lr_scheduler.StepLR(optimizer, step_size=params['decayschedule'], gamma=params['learningratedecay'])
                    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                    params['current_run'] = run
                    results = one_run(train_loader_noisy, model, optimizer, params, results)





