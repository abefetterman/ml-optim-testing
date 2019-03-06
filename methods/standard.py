import torch
from methods.tracker import DefaultTracker

default_tracker = DefaultTracker()

def train(model, criterion, optimizer, train_loader, \
            cuda=True, tracker=default_tracker):
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
            tracker.update(i, output.data, target, loss.item(), input.size(0))

def validate(model, criterion, val_loader, \
            cuda=True, tracker=default_tracker):
    """Perform validation on the validation set"""
    if (not tracker):
        # no use validating if we're not going to track the result!
        return None
    # switch to evaluate mode
    model.eval()
    if (tracker):
        tracker.reset(len(val_loader))

    for i, (input, target) in enumerate(val_loader):
        if (cuda):
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if (tracker):
            tracker.update(i, output.data, target, loss.data[0], input.size(0))

    return tracker
