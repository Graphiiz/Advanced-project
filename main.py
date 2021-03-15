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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
#train,test arguments
parser.add_argument('--train', action='store_true',help='train mode')
parser.add_argument('--model', default=None, type=str,help='model choices = ["LeNet","VGG16","ResNet"]')
parser.add_argument('--dataset', default=None, type=str,help='dataset choices = ["MNIST","CIFAR10"]')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
parser.add_argument('--gamma', default=0.5, type=int, help='gamma for learning rate scheduler')
parser.add_argument('--step-size', default=50, type=int, dest='step_size',help='gamma for learning rate scheduler')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--test', action='store_true', help='test mode, model is required')
#parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--model-path', dest='model_path', type=str,help='path to saved model or .pth file, required when resume training')

args = parser.parse_args()

best_acc = 0

#function

#train
def train(epoch):
    model.to(device)
    model.train()
    train_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'finish epoch #{epoch}')
    print(f'Training accuracy = {correct/total}')
    
def test(epoch):
    global best_acc #declare this allow you make changes to global variable
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
    print(f'Test accuracy = {correct/total}')
    acc = correct/len(testloader)
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def create_scheduler(model_name,optimizer):
    if model_name.lower() == 'lenet':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,8,12,13,14], gamma=0.5)
        return scheduler
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return scheduler

     
#main
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model is not None:
    if (args.test or args.decomp) == False:
        print('test mode or decomp mode is required')
        exit(0)
if args.train:
    if args.model is None:
        print('model type is required.')
        exit(0)
    print('Create model...')
    model = model.create_model(args.model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = create_scheduler(args.model,optimizer)
    print('Start training the model...')
    print('==> Preparing data..')
    trainloader = dataset.create_trainset(args.dataset)
    testloader = dataset.create_testset(args.dataset)
    print('==> Datasets are ready')
    num_epoch = args.epoch
    for epoch in range(num_epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

if args.test:
    if args.model_path is None: 
        print('.pth file of pretrained model is required')
        exit(0)
    print('Load model...')
    info_dict = torch.load(args.model_path) #see format of .pth file in train function
    model = model.create_model(args.model)
    model.load_state_dict(info_dict['model'])
    print('load model successfully')
    model.to(device)
    model.eval()
    print('Start testing the model...')
    test_loss = 0 #to be used later, don't use it yet
    correct = 0
    total = 0
    testloader = dataset.create_testset(args.dataset)
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f'Test accuracy = {correct/len(testloader)}')
    

