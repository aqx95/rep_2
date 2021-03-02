import torch.nn as nn
from segmentation_models_pytorch.unet import Unet

class HuBMAPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Unet(encoder_name=config.encoder,
                         encoder_weights='imagenet',
                         classes=1,
                         activation=None)

    def forward(self, x):
        img_mask = self.model(x)
        return img_mask
