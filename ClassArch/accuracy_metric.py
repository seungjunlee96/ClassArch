import torch


def accuracy(pred, target):
    pred = pred.float()
    correct = 0
    for i in range(target.size()[0]):
        if (pred[i] == pred[i].max()).nonzero() == target[i]:
            correct += 1
    
    return correct / target.size()[0]