import torch
import os
import logging


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
    def __init__(self, axis=1, smooth=1.0):
        self.axis = axis
        self.reset()
        self.smooth = smooth

    def reset(self):
        self.inter = 0.0
        self.union = 0.0

    def accumulate(self, pred, target):
        pred, target = torch.flatten(pred), torch.flatten(target)
        self.inter += (pred*target).sum().item()
        self.union += (pred+target).sum().item()

    @property
    def avg(self):
        return (2.0 * self.inter + self.smooth)/(self.union + self.smooth)


class DicethrMeter:
    def __init__(self, ths=np.arange(0.1,0.9,0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, pred, target):
        pred, target = torch.flatten(pred), torch.flatten(target)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*target).float().sum().item()
            self.union[i] += (p+target).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0*self.inter/self.union,
                            torch.zeros_like(self.union))
        return dices



def log(config, name):
    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)
    log_file = os.path.join(config.LOG_PATH, 'log.txt')
    open(log_file, 'w+').close()

    console_log_format = "%(levelname)s %(message)s"
    file_log_format = "%(levelname)s: %(asctime)s: %(message)s"

    #Configure logger
    logging.basicConfig(level=logging.INFO, format=console_log_format)
    logger = logging.getLogger(name)

    #File handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(file_log_format)
    handler.setFormatter(formatter)

    #Add handler to logger
    logger.addHandler(handler)

    return logger
