from methods.cifar10 import Cifar10
from models.densenet import DenseNet3
import torch

from utils import ConfigAsArgs
from sacred import Experiment
ex = Experiment('DenseNet')
# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

class ConfigAsArgs:
    def __init__(self, entries):
        self.__dict__.update(entries)

@ex.config
def config():
    epochs = 300 #(int) number of total epochs to run
    start_epoch = 0 #(int) manual epoch number (useful on restarts)
    batch_size = 64 #(int) mini-batch size
    lr = 0.1 #(float) initial learning rate
    momentum = 0.9 #(float) momentum
    weight_decay = 1e-4 #(float) weight decay
    print_freq = 1 #(int) print frequency
    layers = 100 #(int) number of layers
    growth = 12 #(int) new channels per layer
    droprate = 0.0 #(float) dropout probability
    augment = True #(bool) whether to use standard augmentation
    reduction = 0.5 #(float) compression rate in transition stage
    bottleneck = True #(bool) whether to use bottleneck block
    resume = '' #(string) path to latest checkpoint
    name = 'DenseNet_BC_100_12' #(string) name of Experiment
    datadir = './data/' #(string) path to data
    cuda = True
    args = ConfigAsArgs({
        "epochs": epochs,
        "start_epoch": start_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "print_freq": print_freq,
        "augment": augment,
        "resume": resume,
        "name": name,
        "datadir": datadir,
        "cuda": cuda
    })
    strict = {"args": args}

@ex.capture
def create_model(layers, growth, reduction, bottleneck, droprate):
    return DenseNet3(layers, 10, growth, reduction=reduction,
                    bottleneck=bottleneck, dropRate=droprate)

@ex.capture
def create_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=weight_decay)

@ex.automain
def my_main(args):
    model = create_model()
    optimizer = create_optimizer(model)
    method = Cifar10(model, optimizer, args)
    method.load()
    method.run()
