
class GlobalConfig:
    IMG_PATH = '/kaggle/input/hubmap-256x256/train'
    MASK_PATH = '/kaggle/input/hubmap-256x256/masks'
    LOG_PATH = 'log'
    num_split = 5
    seed = 2020
    criterion = 'dice'
    criterion_params = {'dice': {'weight':None,'size_average':True}
                       }
    train_step_scheduler = True
    lr = 0.1
    encoder = 'se_resnext50_32x4d'