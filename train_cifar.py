from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import os
import argparse
import time

from resnet import ResNet18, ResNet50
from cifar_models import ConvNet
from data_prepare import *

parser = argparse.ArgumentParser(description='Train CIFAR model')
parser.add_argument('--data_root', type=str, default='data', help='CIFAR data root')
parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR-10/100/svhn')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='model output directory')
parser.add_argument('--saved_model', type=str, default='', help='load from saved model and test only')
parser.add_argument('--model', type=str, default='resnet18', help='type of model')
parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

print(args)
if args.dataset == 'cifar10':
    transform_train = cifar_transform_train
    transform_test = cifar_transform_test
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 10
    input_size =  32
elif args.dataset == 'cifar100':
    transform_train = cifar_transform_train
    transform_test = cifar_transform_test
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 100
    input_size =  32
else:
    print("unknown dataset")
    exit()
linear_base = input_size * input_size / 2

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model == 'resnet18':
    net = ResNet18(num_classes=num_classes, linear_base=linear_base)
elif args.model == 'resnet50':
    net = ResNet50(num_classes=num_classes, linear_base=linear_base)
elif args.model == 'convnet':
    net = ConvNet(num_classes=num_classes)

best_acc = 0
if args.saved_model != '':
    checkpoint = torch.load(args.saved_model)
    net.load_state_dict(checkpoint['net'], strict=False)
    #best_acc = checkpoint['acc']
    
net = net.to(device)
criterion = nn.CrossEntropyLoss()
torch.manual_seed(args.seed)

def train(epoch, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('==>>> train loss: {:.6f}, accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))

def test(epoch=-1):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)[:, :num_classes]
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('==>>> test loss: {:.6f}, accuracy: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '%s/%s_%s_seed_%d.pth' % (args.save_dir, args.dataset, args.model, args.seed))
        best_acc = acc


if args.saved_model != '':
    test()
    exit()
        
optimizer = optim.Adam(net.parameters(), lr=args.lr)
first_drop, second_drop = False, False
for epoch in range(args.epochs):
    train(epoch, optimizer)
    test(epoch)
    if num_classes == 2:
        fpr_fnr_test()
    if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        first_drop = True
    if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        second_drop = True

print(best_acc)
state = {
    'net': net.state_dict(),
    'epoch': args.epochs,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, '%s/%s_%s_seed_%d.pth' % (args.save_dir, args.dataset, args.model, args.seed))
