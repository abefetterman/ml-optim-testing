import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

from smooth_adam import SmoothAdam
from wrn import WideResNet

import wandb
wandb.init(project="optim_testing")

num_classes = 10
batch_size_train, batch_size_test = 8, 32
num_epochs = 150
input_size = 224
valid_size = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create training and validation datasets
image_dataset_train = torchvision.datasets.CIFAR10('./data', download=True, transform=train_transforms)
image_dataset_val = torchvision.datasets.CIFAR10('./data', download=True, transform=test_transforms)

num_train = len(image_dataset_train)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Create training and validation dataloaders
dataloader_train = torch.utils.data.DataLoader(
    image_dataset_train, batch_size=batch_size_train, num_workers=4, 
    sampler=train_sampler
    )
dataloader_val = torch.utils.data.DataLoader(
    image_dataset_val, batch_size=batch_size_test, num_workers=4, 
    sampler=valid_sampler
    )

model = WideResNet(n_groups=3, N=3, n_classes=num_classes, k=6)

model = model.to(device)

wandb.watch(model)

params_to_train = [param for param in model.parameters() if param.requires_grad==True]
# optimizer = SmoothAdam(params_to_train, lr=0.001, eta=2.0)
optimizer = optim.SGD(params_to_train, lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    model.train()
    train_loss = 0.0
    train_correct = 0
    for inputs,labels in tqdm(dataloader_train):
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _,preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item() * inputs.size(0)
        train_correct += torch.sum(preds == labels.data)
        # print('loss {} / correct {}'.format(loss.item(),torch.sum(preds == labels.data).double()/inputs.size(0)))
    
    train_loss = train_loss / len(train_idx)
    train_correct = train_correct.double() / len(train_idx)
    
    model.eval()
    val_loss = 0.0
    val_correct = 0.0
    for inputs,labels in dataloader_val:
        inputs,labels = inputs.to(device),labels.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _,preds = torch.max(outputs, 1)
        
            val_loss += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels.data)
            # print('loss {} / correct {}'.format(loss.item(),torch.sum(preds == labels.data).double()/inputs.size(0)))
    
    val_loss = val_loss / len(valid_idx)
    val_correct = val_correct.double() / len(valid_idx)
    
    print('Train loss: {:.4f}, acc: {:.4f};\t Val loss: {:.4f} acc: {:.4f}'.format(
        train_loss, train_correct, val_loss, val_correct
    ))
    wandb.log({
        'Train Loss': train_loss,
        'Train Accuracy': train_correct,
        'Validation Loss': val_loss,
        'Validation Accuracy': val_correct,
    })
    
