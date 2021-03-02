import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)

        return 1 - dice

#loss factory
def loss_fn(config):
    if config.criterion == 'dice':
        return DiceLoss(**config.criterion_params[config.criterion])
