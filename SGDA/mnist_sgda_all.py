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

from lossfns import cw_train_unrolled
from model import ResNet18

from adversary import *

import torchvision.models as models

# from models import CNNet9l,Res_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_value = 0
np.random.seed(seed_value)

# PyTorch
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# For full determinism in PyTorch (optional)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False


# Defining
# the
# neural
# network
# architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5)
        self.conv2 = nn.Conv2d(15, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=5)
        self.conv2 = nn.Conv2d(15, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Defining
# noisy
# training
# loader
'''
A simple way to define a noisy data loader. We pick a batch of training
set size and introduce corruptions via directed or random noise models 
on randomly chosen samples
'''


def noisy_loader(params):
    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='../data', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(mnist_dataset,
                                               batch_size=params['batchsize'], shuffle=True)

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
    train_noisy = torch.utils.data.TensorDataset(data, target_noisy)

    params['train_size'] = len(target_noisy)
    print("data len:", len(target_noisy))

    # params['train_size'] = len(mnist_dataset)
    # train_noisy = mnist_dataset
    # print("data len:", len(mnist_dataset))

    if params['optimizer_type'] == 'sgd':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']),
                                                         shuffle=True, drop_last=True)
    elif params['optimizer_type'] == 'mkl':
        train_loader_noisy = torch.utils.data.DataLoader(train_noisy, batch_size=params['k'], shuffle=True,
                                                         drop_last=True)
    elif params['optimizer_type'] == 'pac-bayes':
        
        w = np.full((params['train_size']), 1 / params['train_size'])
        samp = torch.utils.data.WeightedRandomSampler(torch.Tensor(w).to(device), params['train_size'])

        train_loader_noisy_uni = torch.utils.data.DataLoader(train_noisy, batch_size=int(params['frac'] * params['k']),
                                                             shuffle=True, drop_last=True)
        train_loader_noisy_train = torch.utils.data.DataLoader(train_noisy,
                                                               batch_size=int(params['frac'] * params['k']),
                                                               shuffle=False, sampler=samp, drop_last=True)
        train_loader_eval = torch.utils.data.DataLoader(train_noisy, batch_size=64, shuffle=False)

        train_loader_noisy = [train_loader_noisy_uni, train_loader_noisy_train, train_loader_eval, samp]

    return train_loader_noisy


# Training and Testing
# Phase
eps = 0.1

def train(train_loader_noisy, epoch, run, epsilon, params):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_noisy):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        if params['optimizer_type'] == 'sgd' or params['optimizer_type'] == 'pac-bayes':

            torch.cuda.empty_cache()
            loss = cw_train_unrolled(model, data, target, device, eps)
            optimizer.zero_grad()

        elif params['optimizer_type'] == 'mkl':
            output = model(data)
            temp_loss = F.nll_loss(output, target.to(device), reduction='none')
            temp = temp_loss.cpu().detach().numpy()
            # Pick the samples with the lowest loss
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

        if batch_idx % 2500 == 0:
            print('Train Run: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, {}'.format(
                run + 1, epoch, batch_idx * len(data), len(train_loader_noisy.dataset),
                100. * batch_idx / len(train_loader_noisy), loss.item(), params['optimizer_type']))
    return loss.item()

def test(test_loader, run):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data.to(device)), Variable(target.to(device))
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).float().cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set {} Run: {} : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        params['optimizer_type'], run + 1, test_loss, correct, len(test_loader.dataset),
                                  100. * correct / len(test_loader.dataset)))
    torch.cuda.empty_cache()
    
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

    time_start = time.time()
    if params['decayschedule'] != 0:
        scheduler.step()
    if params['optimizer_type'] == 'sgd' or params['optimizer_type'] == 'mkl':
        train_loader_noisy_uni = train_loader_noisy

        for epoch in range(1, num_epochs + 1):

            results['train_loss'][epoch - 1, count] = train(train_loader_noisy_uni, epoch, params['current_run'], params['epsilon'], params)

            results['test_loss'][epoch - 1, count], results['test_acc'][epoch - 1, count] = test(test_loader, params['current_run'])
         ##===== do adversarial test
        fgsmAttackTest(model, test_loader, device)
        return results


    elif params['optimizer_type'] == 'pac-bayes':    

        q_prior = torch.full((params['train_size'],), 1 / params['train_size'])# , device=torch.device(device)
        train_loader_noisy_uni, train_loader_noisy_train, train_loader_eval, samp = train_loader_noisy
    
        results['train_loss'][0, count] = train(train_loader_noisy_uni, 1, params['current_run'], params['epsilon'], params)
        results['test_loss'][0, count], results['test_acc'][0, count] = test(test_loader, params['current_run'])
    
        torch.cuda.empty_cache()
    
        model.eval()
                
        for i in range(1):

                        # loss_temp = torch.zeros(len(dataset), ).to(device)
                        loss_temp = torch.zeros(params['train_size'], )

                        for i, (data , labels) in enumerate(train_loader_eval ):
                            # data ,  labels = data.to(device) , labels.to(device).long()
                            data ,  labels = Variable(data.to(device)), Variable(labels.to(device))# .long()
                            # loss = F.nll_loss(outputs, labels, reduction='none')
                            d,loss = cw_train_unrolled(model, data, labels, device, eps,reduction=False)
                            id_list = torch.arange(i * len(labels), (i + 1) * len(labels))

                            loss_temp[id_list[:]] = d

                        q_pos = torch.exp(-loss_temp)
                        q_pos = q_pos / torch.sum(q_pos)
                        w = q_pos.clone()

                        ##===compute kl divergence
                        term1 = torch.log(torch.div(q_pos, q_prior))
                        kl_div = (torch.mul(q_pos, term1)).sum()
                        print(kl_div)

                        samp.weights = w 
        results['time_spent'][0, count] = time.time() - time_start

    for epoch in range(2, num_epochs + 1):
        model.train()
        if params['decayschedule'] != 0:
            scheduler.step()

        results['train_loss'][epoch - 1, count] = train(train_loader_noisy_train, epoch, params['current_run'],
                                                        params['epsilon'], params)
        results['test_loss'][epoch - 1, count], results['test_acc'][epoch - 1, count] = test(test_loader,
        params['current_run'])
        torch.cuda.empty_cache()

        model.eval()
        for i in range(1):

                        loss_temp = torch.zeros(params['train_size'], )# .to(device)

                        for i, (data , labels) in enumerate(train_loader_eval ):
                            # data ,   labels = data.to(device) , labels.to(device).long()
                            data ,  labels = Variable(data.to(device)), Variable(labels.to(device))# .long()
                            d,loss = cw_train_unrolled(model, data, labels, device, eps,reduction=False)
                            id_list = torch.arange(i * len(labels), (i + 1) * len(labels))

                            loss_temp[id_list[:]] = d

                        q_pos = torch.exp(-loss_temp)
                        q_pos = q_pos / torch.sum(q_pos)
                        w = q_pos.clone()

                        ##===compute kl divergence
                        term1 = torch.log(torch.div(q_pos, q_prior))
                        kl_div = (torch.mul(q_pos, term1)).sum()
                        print(kl_div)
                        samp.weights = w
            
        results['time_spent'][epoch - 1, count] = time.time() - time_start

    ##===== do adversarial test
    fgsmAttackTest(model, test_loader, device)

    return results


# Define
# parameters


def init_params():
    params = {}
    params['lr'] = 0.05  # 0.05  # Learning rate
    params['momentum'] = 0.9  # Momentum parameter
    params['k'] = 54  # Number of loss evaluations per batch
    params['batchsize'] = 10000  # Size of dataset, parameter used in noisy_loader
    params['decayschedule'] = 30  # Decay Schedule
    params['noise_type'] = 'random'  # Noise type
    #    To use directed noise model of corruption, set 'directed'
    #    To use random noise model of corruption, set 'random'

    params['test_size'] = 0

    params['train_size'] = 0
    params['adv_size'] = 0

    params['learningratedecay'] = 0.2  # Learning Rate Decay
    params['eps'] = 1e-08  # Corruption parameter
    params['num_epochs'] = 15  # Number of epochsf
    params['optimizer_type'] = 'sgd'  # Optimizer Type:
    #    To run standard stochastic gradient descent, set 'sgd'
    #    To run standard MKL-SGD, set 'mkl'
    params['runs'] = 1  # Number of runs
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

    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    testset = datasets.MNIST('../data', train=False, transform=transform_test)

    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


    params['test_size'] = len(testset)

    params['adv_size'] = len(testset)

    return params, trainset, test_loader,  results


# Main
# function
time_start = time.time()
params, trainset, test_loader, results = init_params()
for run in range(params['runs']):

    train_loader_noisy = noisy_loader(params)
    model = ResNet18() # Net()  # ConvNet

    # model = Res_net()
    model.to(device)
    optimizer = select_optimizer('sgdmomentum', params, weight_decay=5e-4)
    if params['decayschedule'] != 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params['decayschedule'], gamma=params['learningratedecay'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    params['current_run'] = run

    results = one_run(train_loader_noisy, model, optimizer, params, results)

# Saving parameters

if params['optimizer_type'] == 'pac-bayes':
    file_path = "./results/MNIST_pacb_latest_%s_%s/lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_clean_%s" \
                % (params['optimizer_type'], params['noise_type'], params['lr'],
                   params['momentum'], params['epsilon'], params['decayschedule'],
                   params['num_epochs'], params['runs'], params['k'], params['clean_data'])
elif params['optimizer_type'] == 'sgd':
    file_path = "./results/MNIST_SGD_latest_%s_%s/%s_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_clean_%s" \
                % (params['optimizer_type'], params['noise_type'], params['optimizer_type'], params['lr'],
                   params['momentum'], params['epsilon'], params['decayschedule'],
                   params['num_epochs'], params['runs'], params['k'], params['clean_data'])
elif params['optimizer_type'] == 'mkl':
    file_path = "./results/MNIST_SGD_latest_%s_%s/%s_lr_%.4f_momentum_%.4f_eps_%.4f_ds_%d_n_epochs_%d_runs_%d_minibatch_%d_frac_%.2f_clean_%s" \
                % (params['optimizer_type'], params['noise_type'], params['optimizer_type'], params['lr'],
                   params['momentum'], params['epsilon'], params['decayschedule'],
                   params['num_epochs'], params['runs'], params['k'], params['frac'], params['clean_data'])


directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
np.savez(file_path, train_loss=results['train_loss'], test_loss=results['test_loss'], test_acc=results['test_acc'],
         time_spent=results['time_spent'])
