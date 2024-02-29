'''CIFAR-10 for the test accuracy of stop condition '''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR-10 for the test accuracy of stop condition')
parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
parser.add_argument('--model',default="Res18", type=str, help='[Res18, Res34, Res50, Wide28-10]')
parser.add_argument('--batchsize', default=256, type=int, help='training batch size')
parser.add_argument('--optimizer',default="sgd", type=str, help='[momentum,sgd,adam,rmsgrad,adamw]')
parser.add_argument('--use_wandb', default=False, type=str, help='Set to True if using wandb.')
args = parser.parse_args()
#Comment out the following code when you don't use decaying 1~3.
#parser.add_argument('--decaynumber', default=1, type=int, help='decaysing learning rate number')
#parser.add_argument('--stepsize', default=int(50000 * args.epoch/args.batchsize), type=int, help='step size')
#args = parser.parse_args()


if args.use_wandb:
    wandb_project_name = "×××××"
    wandb_exp_name = f"×××××"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "×××××")
    wandb.init(settings=wandb.Settings(start_method='fork'))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if args.model == "Res18":
    net = ResNet18_for_10()
elif args.model == "Res34":
    net = ResNet34_for_34()
elif args.model == "Res50":
    net = ResNet50_for_50()
elif args.model == "Wide28-10":
    net = WideResNet28_10()

net = net.to(device)
if device == 'cuda:0':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#definition of decaying 1~3
def func(steps):
  b = steps + 1
  b = b ** (0.25 * args.decaynumber)
  return 1/b

################definition of decay1~3 learning late ###############################                           

class NewStepLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, lr_lambda, step_size, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.step_size = step_size
        self.m_last_epoch = 1

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)

    # Modification                                                                                                    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            #return [group['lr'] for group in self.optimizer.param_groups]                                            
            return [base_lr * lmbda(self.m_last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        self.m_last_epoch += 1
        return [base_lr * lmbda(self.m_last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

####################################################################################

criterion = nn.CrossEntropyLoss()
if args.optimizer == "momentum":
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)                 #Momentum
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))                           #Adam
elif args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)                                   #SGD
elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9)                                  #RMSProp      
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999))                          #AdamW

#Designation of decreasing step size.
#Comment out the following decaying number's code that you don't use.

#decaying1~3
'''optim.lr_scheduler.LambdaLR = NewStepLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func, step_size=args.stepsize)'''

# Training
steps = 0

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global steps
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        train_acc = correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #Update decaying step size.
        #Comment out the following code when using constant stepsize.
        '''last_lr = scheduler.get_last_lr()[0]
        if args.use_wandb:
            wandb.log({'last_lr': last_lr})
        scheduler.step()'''

        steps += 1
        if args.use_wandb:
            wandb.log({'loss':train_loss/(batch_idx+1),
                       'train_acc':train_acc,
                       'steps':steps})

    
def test(epoch):
    net.eval()
    #Designation of the value of stop condition.
    break_test_acc = 0.9
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = correct/total
    if args.use_wandb:
        wandb.log({'test_acc': test_acc})
    
    if test_acc > break_test_acc:
        if args.use_wandb:
            wandb.log({'loss':test_loss/(batch_idx+1),
                        'test_acc':test_acc,
                        'steps':steps})
        sys.exit()

for epoch in range(start_epoch, start_epoch+200):
    print('\nEpoch: %d' % epoch)
    train(epoch)
    test(epoch)