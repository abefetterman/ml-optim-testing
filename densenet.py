from methods.standard import Standard
from models.densenet import DenseNet3
from dataloaders.cifar10 import Cifar10
from hybrid import Hybrid

import torch

from utils import ConfigAsArgs
from sacred import Experiment
from observer import add_observer
ex = Experiment('DenseNet')
add_observer(ex)

@ex.config
def config():
    epochs = 300 #(int) number of total epochs to run
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
    args = ConfigAsArgs({
        "epochs": epochs,
        "start_epoch": start_epoch,
        "lr": lr,
        "print_freq": print_freq,
        "name": name,
        "cuda": cuda
    })
    strict = {"args": args}

@ex.capture
def create_model(layers, growth, reduction, bottleneck, droprate):
    return DenseNet3(layers, 10, growth, reduction=reduction,
                    bottleneck=bottleneck, dropRate=droprate)

@ex.capture
def create_dataloader(datadir, batch_size, augment):
    return Cifar10(datadir, batch_size, augment)

@ex.capture
def create_optimizer(model, lr, rho_adam, etam, etad, weight_decay):
    return Hybrid(model.parameters(), lr,
                                etam=etam, etad=etad, rho_adam=rho_adam,
                                nesterov=True,
                                weight_decay=weight_decay)

@ex.automain
def my_main(args):
    model = create_model()
    optimizer = create_optimizer(model)
    loader = create_dataloader()
    method = Standard(model, optimizer, loader, args)
    method.run()
