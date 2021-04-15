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

    def forward(self, inputs, targets, cls=None, non_empty=None, smooth=1):
        #flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)

        dice_loss = self.diceloss(inputs, targets)
        if cls != None:
          cls = torch.flatten(cls).float()
          non_empty = torch.flatten(non_empty).float()
          BCE = F.binary_cross_entropy(cls, non_empty, reduction='mean')
        else:
          BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (self.bce_weight*BCE) + (1-self.bce_weight)*dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return focal_loss


#loss factory
def loss_fn(config):
    if config.criterion == 'dice':
        return DiceLoss(**config.criterion_params[config.criterion])

    if config.criterion == 'dicebce':
        return DiceBCELoss(**config.criterion_params[config.criterion])

    if config.criterion == 'jaccard':
        return IoULoss()

    if config.criterion == 'focal':
        return FocalLoss(**config.criterion_params[config.criterion])
