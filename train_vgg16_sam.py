import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import os
import argparse

from PIL import Image

import glob

#import relative .py files
import dataset
import model

#json
import json

#sam
from sam import SAM

import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
#train,test arguments
parser.add_argument('--train', action='store_true',help='train mode')
parser.add_argument('--dataset', default=None, type=str,help='dataset choices = ["MNIST","CIFAR10"]')
parser.add_argument('--batch_size', default=128, type=int,help='dataset choices = ["MNIST","CIFAR10"]')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs tp train for')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for learning rate scheduler')
parser.add_argument('--patience', default=20, type=int, help='period of iterations which has no improvement before update lr')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay factor for sgd')
parser.add_argument('--test', action='store_true', help='test mode, model is required')
parser.add_argument('--model_path', type=str,help='path to saved model or .pth file, required when resume training')
parser.add_argument('--seed', default=42, type=int, help='seed for random')
parser.add_argument('--rho', default=0.05, type=float, help='rho for SAM')

args = parser.parse_args()

best_acc = 0

current_acc = 0 #for ReduceLROnPlateau scheduler

count_reduce = 0 #count the number of decreases of lr due to ReduceLROnPlateau scheduler

#function

def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

#train
def train(epoch,scheduler):
    model.to(device)
    model.train()
    train_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        #first forward-backward pass
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        train_loss += loss.item() #compute for statistical purpose

        #second forward-backward pass
        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
    
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    print(epoch) #print epoch
    print(correct/total) #print train acc

    return train_loss/len(trainloader), correct/total


    
def test_in_train(epoch):
    global best_acc #declare this allow you make changes to global variable
    global current_acc
    model.eval()
    test_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(correct/total) #print test acc

    acc = correct/len(testloader)
    current_acc = acc
    if acc > best_acc:
        #print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vgg16_sam_ckpt.pth')
        best_acc = acc

    return test_loss/len(testloader), correct/total

def test():
    model.eval()

    test_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    testloader = dataset.create_testset(args.dataset,128)
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Test accuracy = {correct/total}')
     
#main
initialize(args,seed=args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.train:

    model = model.create_model('VGG16').to(device)

    #sam
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    trainloader = dataset.create_trainset(args.dataset,args.batch_size)
    testloader = dataset.create_testset(args.dataset)

    #scheduler keep track of base_optimizer in SAM object
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=args.patience, threshold=0.0001, 
    #                                                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    num_epoch = args.epoch

    train_loss_log = []
    test_loss_log = []
    train_acc_log = []
    test_acc_log = []

    for epoch in range(num_epoch):

        tracked_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(epoch,scheduler)
        test_loss, test_acc = test_in_train(epoch)
        #scheduler.step(current_acc) #track current acc and do step() here
        scheduler.step()
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)

        if optimizer.param_groups[0]['lr'] < tracked_lr:
            count_reduce += 1

    log_dict = {'train_loss': train_loss_log, 'test_loss': test_loss_log,
                            'train_acc': train_acc_log, 'test_acc': test_acc_log, 'best_test_acc': max(test_acc_log),
                            'batch_size': args.batch_size, 'lr': args.lr, 'momentum': args.momentum, 'wd': args.weight_decay,
                            'seed': args.seed, 'gamma': args.gamma, 'scheduler': 'CosineAnnealing', 'epoch': args.epoch, 'rho': args.rho}
    with open(f'train_vgg16_sam.json', 'w') as outfile:
        json.dump(log_dict, outfile)
    
    state = {
            'model': model.state_dict(),
            'acc': test_acc_log[-1],
            'epoch': epoch,
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/vgg16_sam_final_ckpt.pth')

    print(f'#lr reduction = {count_reduce}')
    


if args.test:
    if args.model_path is None: 
        print('.pth file of pretrained model is required')
        exit(0)

    info_dict = torch.load(args.model_path) #see format of .pth file in train function
    model = model.create_model(args.model)
    model.load_state_dict(info_dict['model'])

    model.to(device)
    
    test()