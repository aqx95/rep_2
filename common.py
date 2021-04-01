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
        self.reset()
    def reset(self):
        self.inter = 0.0
        self.union = 0.0
    def accumulate(self, pred, target):
        pred, target = torch.flatten(torch.sigmoid(pred).float()), torch.flatten(target.float())
        self.inter += (pred*target).sum().item()
        self.union += (pred+target).sum().item()

    @property
    def avg(self):
        return 2.0 * self.inter/self.union if self.union > 0 else None
