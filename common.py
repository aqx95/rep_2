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


def get_dice_coeff(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).int()
    return 2.0*(pred*target).sum() / (pred.sum()+target.sum() + 1.0)
