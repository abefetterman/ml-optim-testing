
import time
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DefaultTracker(object):
    def __init__(self, get_accuracy=None, print_freq=100):
        self.get_accuracy=get_accuracy
        self.print_freq = print_freq
        self.reset()
    def reset(self, batch_size=0):
        self.batch_time = AverageMeter()
        self.losses = AverageMeter()
        self.accuracy = AverageMeter()

        self.batch_size = batch_size
        self.end = time.time()
    def update(self, i, output, target, loss, size):
        self.losses.update(loss, size)
        if (self.get_accuracy):
            new_accuracy = get_accuracy(output.data, target)
            self.accuracy.update(new_accuracy, size)

        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

        if i % self.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      i, self.batch_size, batch_time=self.batch_time,
                      loss=self.losses))


def train(model, criterion, optimizer, train_loader, \
            cuda=True, tracker=DefaultTracker()):
    # switch to train mode
    model.train()
    if (tracker):
        tracker.reset(len(train_loader))

    for i, (input, target) in enumerate(train_loader):
        if (cuda):
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (tracker):
            tracker.update(i, output.data, target, loss.data[0], input.size(0))
