import torch

class LossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.count = 0

    def update(self, batch_loss, batch_size):
        self.loss += batch_loss*batch_size
        self.count += batch_size

    @property
    def avg(self):
        return self.loss/self.count


class DiceMeter:
    def __init__(self, axis=1):
        self.axis = axis
    def reset(self):
        self.inter
        self.union = 0,0
    def accumulate(self, pred, target):
        pred, target = torch.flatten(torch.sigmoid(pred)), torch.flatten(target)
        self.inter += (pred*target).float().sum().item()
        self.union += (pred+target).float().sum().item()

    @property
    def value(self):
        return 2.0 * self.inter/self.union if self.union > 0 else None
