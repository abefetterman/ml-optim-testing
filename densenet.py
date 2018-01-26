from methods.tracker import DefaultTracker
from methods.accuracy import topk_accuracy
from methods.standard import train, validate
from models.densenet import DenseNet3
from dataloaders.cifar10 import Cifar10
from hybrid import Hybrid

import torch
import torch.nn as nn

from utils import ConfigAsArgs
from sacred import Experiment
from observer import add_observer
ex = Experiment('DenseNet')
add_observer(ex)

@ex.config
def config():
    epochs = 100 #(int) number of total epochs to run
    start_epoch = 0 #(int) manual epoch number (useful on restarts)
    batch_size = 64 #(int) mini-batch size
    lr = 0.1 #(float) initial learning rate
    rho_adam = 0.01 #(float) fraction of learning rate for pure adam
    etam = 0.0 #(float) fraction of magnitude from adam
    etad = 1.0 #(float) fraction of direction from adam
    momentum = 0.9 #(float) momentum
    weight_decay = 1e-4 #(float) weight decay
    print_freq = 100 #(int) print frequency
    layers = 100 #(int) number of layers
    growth = 12 #(int) new channels per layer
    droprate = 0.0 #(float) dropout probability
    augment = True #(bool) whether to use standard augmentation
    reduction = 0.5 #(float) compression rate in transition stage
    bottleneck = True #(bool) whether to use bottleneck block
    name = 'DenseNet' #(string) name of Experiment
    datadir = './data/' #(string) path to data
    cuda = True

@ex.capture
def create_model(layers, growth, reduction, bottleneck, droprate, cuda):
    model = DenseNet3(layers, 10, growth, reduction=reduction,
                    bottleneck=bottleneck, dropRate=droprate)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if (cuda):
        model = model.cuda()
    return model

@ex.capture
def create_dataloader(datadir, batch_size, augment, cuda):
    return Cifar10(datadir, batch_size, augment, cuda=cuda)

@ex.capture
def create_optimizer(model, lr, rho_adam, etam, etad, weight_decay):
    return Hybrid(model.parameters(), lr,
                                etam=etam, etad=etad, rho_adam=rho_adam,
                                nesterov=True,
                                weight_decay=weight_decay)

@ex.capture
def create_criterion(cuda):
    criterion = nn.CrossEntropyLoss()
    if (cuda):
        criterion=criterion.cuda()
    return criterion

@ex.capture
def create_tracker(print_freq):
    return DefaultTracker(get_accuracy=topk_accuracy, print_freq=print_freq)

@ex.automain
def main(cuda, start_epoch, epochs):
    model = create_model()
    optimizer = create_optimizer(model)
    loader = create_dataloader()
    criterion = create_criterion()
    tracker = create_tracker()
    for epoch in range(start_epoch, epochs):
        print('===== EPOCH {} ====='.format(epoch))
        train(model, criterion, optimizer, loader.train, tracker=tracker, cuda=cuda)
        print('Epoch {0} val. loss: {1:.6f}'.format(epoch, tracker.avg_loss()))
        print('Epoch {0} val. accuracy: {1:.3f}'.format(epoch, tracker.avg_accuracy()))
        validate(model, criterion, loader.test, tracker=tracker, cuda=cuda)
        print('Epoch {0} test loss: {1:.6f}'.format(epoch, tracker.avg_loss()))
        print('Epoch {0} test accuracy: {1:.3f}'.format(epoch, tracker.avg_accuracy()))
        
