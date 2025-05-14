from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def cw_train_unrolled(model, X, y, device, eps, reduction=True):
    N = X.shape[0]
    X = X.repeat(1, 10, 1, 1).reshape(N * 10, 1, 28, 28)
    # X = X.repeat(1, 2).reshape(N * 2, 2)
    X_copy = X.clone()
    X.requires_grad = True

    y = y.view(-1, 1).repeat(1, 10).view(-1, 1).long().to(device)
    # y = y.view(-1, 1).repeat(1, 2).view(-1, 1).long().to(device)

    index = torch.tensor([jj for jj in range(10)] * N).view(-1, 1).to(device).long()
    # index = torch.tensor([jj for jj in range(2)] * N).view(-1, 1).to(device).long()

    MaxIter_max = 11
    step_size_max = 0.1

    for i in range(MaxIter_max):
        torch.cuda.empty_cache()
        output = model(X)

        maxLoss = (output.gather(1, index) - output.gather(1, y)).mean()

        X_grad = torch.autograd.grad(maxLoss, X, retain_graph=True)[0]
        X = X + X_grad.sign() * step_size_max

        X.data = X_copy.data + (X.data - X_copy.data).clamp(-eps, eps)
        X.data = X.data.clamp(0, 1)

    del X_grad
    torch.cuda.empty_cache()



    preds = model(X)

    # loss = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).max(dim=1)[0].mean()

    d = (-F.log_softmax(preds)).gather(1, y).view(-1, 10).max(dim=1)[0]
    loss = d.mean()
    d=d.detach().cpu()
    if reduction:
        return loss
    else:
        return d, loss

