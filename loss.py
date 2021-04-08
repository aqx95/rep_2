import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1.0):
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, bce_weight=0.2, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.diceloss = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)

        dice_loss = self.diceloss(inputs, targets)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (self.bce_weight*BCE) + (1-self.bce_weight)*dice_loss

        return Dice_BCE


#loss factory
def loss_fn(config):
    if config.criterion == 'dice':
        return DiceLoss(**config.criterion_params[config.criterion])

    if config.criterion == 'dicebce':
        return DiceBCELoss(**config.criterion_params[config.criterion])
