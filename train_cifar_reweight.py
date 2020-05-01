from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np

import os
import argparse
import time

from resnet import ResNet18, ResNet50
from cifar_models import ConvNet
from data_prepare import *
from logger import set_logger
from torch.distributions.categorical import Categorical 
parser = argparse.ArgumentParser(description='Train CIFAR model')
parser.add_argument('--data_root', type=str, default='data', help='CIFAR data root')
parser.add_argument('--dataset', type=str, default='cifar10', help='CIFAR-10/100/svhn')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='model output directory')
parser.add_argument('--saved_model', type=str, default='', help='load from saved model and test only')
parser.add_argument('--model', type=str, default='resnet18', help='type of model')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_seeds', type=int, default=3, help='number of seeds for ensemble the features')
parser.add_argument('--cpr', type=float, default=1.0, help='compression ratio of the dataset; should be 1.0 if train_mode is reweight')
parser.add_argument('--alpha', type=float, default=0, help='combinaion coefficient between uniform and loss based probability')
parser.add_argument('--select_way', type=str, default='sample', help='sample / mle')
parser.add_argument('--prob_style', type=str, default='max_loss', help='max_loss / aul / forgeting_events / loss_droping')

args = parser.parse_args()

saving_name = '{}_{}_{}_alpha_{}_seed_{}_num_seeds_{}_cpr_{}_{}'.format(args.dataset, args.prob_style, args.model, args.alpha, args.seed, args.num_seeds, args.cpr, args.select_way)
logger = set_logger("", "{}/log_{}.txt".format(args.save_dir, saving_name))

logger.info(args)
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
    logger.info("unknown dataset")
    exit()
linear_base = input_size * input_size / 2

torch.manual_seed(args.seed)
if args.prob_style == 'max_loss':
    prob = torch.load('{}/prob_{}_{}_seed_{}_cpr_{}_cpr_num_seeds_{}.pth'.format("results", args.dataset, args.model, 0, 0.1, args.num_seeds))["prob"]
else:
    prob = torch.load('{}/prob_{}_{}_{}.pth'.format("results", args.dataset, args.model, args.prob_style))["prob"]
    
if args.select_way == "sample":
    weight = torch.zeros([len(trainset)])
    def sample_class(i):
        prob_i = prob * 1
        prob_i[np.asarray(trainset.targets) != i] = 0
        prob_i = prob_i / prob_i.sum()
        prob_i = prob_i * args.alpha + (1 - args.alpha) / np.sum(np.asarray(trainset.targets) == i)
        m = Categorical(prob_i)
        sampled_idx = m.sample(sample_shape=torch.Size([int(args.cpr * len(trainset)/ num_classes) ]))
        weight[np.asarray(trainset.targets) == i] = num_classes /(len(trainset) * prob_i[np.asarray(trainset.targets) == i])
        return sampled_idx
    sampled_idx = torch.cat([sample_class(i) for i in range(num_classes)])
    trainset.targets = np.stack([np.array(trainset.targets), weight], axis=1)
elif args.select_way == "sample_uni_weight":
    def sample_class(i):
        prob_i = prob * 1
        prob_i[np.asarray(trainset.targets) != i] = 0
        prob_i = prob_i / prob_i.sum()
        prob_i = prob_i * args.alpha + (1 - args.alpha) / np.sum(np.asarray(trainset.targets) == i)
        m = Categorical(prob_i)
        sampled_idx = m.sample(sample_shape=torch.Size([int(args.cpr * len(trainset) / num_classes) ]))
        return sampled_idx
    sampled_idx = torch.cat([sample_class(i) for i in range(num_classes)])
    trainset.targets = np.stack([np.array(trainset.targets), torch.ones([len(trainset)])], axis=1)
elif args.select_way == "mle":
    num_uni = int((1 - args.alpha) * args.cpr * len(trainset))
    num_loss = int(args.alpha * args.cpr * len(trainset))
    def sample_class(i):
        sorted_idx = torch.arange(len(trainset))[np.array(trainset.targets) == i][torch.argsort(-prob[np.array(trainset.targets) == i])]
        sampled_loss_idx = sorted_idx[: int(num_loss / num_classes)]
        uni_idx_candidate = sorted_idx[int(num_loss / num_classes):]
        sampled_uni_idx = uni_idx_candidate[torch.randperm(len(uni_idx_candidate))[: int(num_uni / num_classes)]]
        #logger.info(sampled_loss_idx)
        #logger.info(sampled_uni_idx)
        sampled_idx = torch.cat([sampled_loss_idx, sampled_uni_idx])
        return sampled_idx
    sampled_idx = torch.cat([sample_class(i) for i in range(num_classes)])
    trainset.targets = np.stack([np.array(trainset.targets), torch.ones([len(trainset)])], axis=1)
    
trainset.data = trainset.data[sampled_idx]
trainset.targets = trainset.targets[sampled_idx]

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
criterion = nn.CrossEntropyLoss(reduction="none")

def train(epoch, optimizer):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        weights = targets[:, 1].float()
        targets = targets[:, 0].long()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss = torch.mean(loss * weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    logger.info('==>>> train loss: {:.6f}, accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))

def test(epoch=-1):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets
            outputs = net(inputs)[:, :num_classes]
            loss = torch.mean(criterion(outputs, targets))
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        logger.info('==>>> test loss: {:.6f}, accuracy: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if epoch % 50 == 0:
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, '{}/{}.pth'.format(args.save_dir, saving_name))
    if acc > best_acc:
        best_acc = acc

if args.saved_model != '':
    test()
    exit()
        
optimizer = optim.Adam(net.parameters(), lr=args.lr)
first_drop, second_drop = False, False
for epoch in range(args.epochs):
    train(epoch, optimizer)
    test(epoch)
    if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        first_drop = True
    if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        second_drop = True

logger.info(best_acc)
state = {
    'net': net.state_dict(),
    'epoch': args.epochs,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, '{}/{}.pth'.format(args.save_dir, saving_name))
