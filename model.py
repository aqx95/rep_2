import torch.nn as nn
from segmentation_models_pytorch.unet import Unet

class HuBMAPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.use_cls:
          config.aux_params = None
        self.model = Unet(encoder_name=config.encoder,
                         encoder_weights='imagenet',
                         classes=1,
                         activation='sigmoid',
                         aux_params=config.aux_params)

    def forward(self, x):
        img_mask = self.model(x)
        return img_mask

## Model factory
def create_model(config):
    if config.net == 'unet':
        return HuBMAPModel(config)
