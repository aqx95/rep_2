import os
import gc
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data.distributed import DistributedSampler
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from config import GlobalConfig
from logger import log
from loss import loss_fn
from model import HuBMAPModel
from engine import train_one_epoch, validate_one_epoch
from data import HuBMAPData, get_train_transform, get_valid_transform


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _mp_fn(rank, flags):
    device = xm.xla_device()
    best_dice=0

    train_sampler = DistributedSampler(dataset = flags['TRAIN_DS'],
                                      num_replicas = xm.xrt_world_size(),
                                      rank = xm.get_ordinal(),
                                      shuffle = True)

    train_dl = DataLoader(dataset = flags['TRAIN_DS'],
                          batch_size = flags['BATCH_SIZE'],
                          sampler = train_sampler,
                          num_workers = 0)

    val_sampler = DistributedSampler(dataset = flags['VAL_DS'],
                                 num_replicas = xm.xrt_world_size(),
                                 rank = xm.get_ordinal(),
                                 shuffle = False)

    val_dl = DataLoader(dataset = flags['VAL_DS'],
                                  batch_size = flags['BATCH_SIZE'],
                                  sampler = val_sampler,
                                  num_workers = 0)


    fold_model = flags['FOLD_MODEL']
    fold_model.to(device)
    logger = flags['LOGGER']

    lr = flags['LR']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
    criterion = loss_fn(config)
    xm.master_print("Engine ready for training!")

    for e_no, epoch in enumerate(range(flags['EPOCHS'])):
        train_paraloader = pl.ParallelLoader(train_dl, [device]).per_device_loader(device)
        train_one_epoch(e_no, train_paraloader, fold_model, optimizer,
                        criterion, device, config, logger, scheduler)
        del train_paraloader
        gc.collect()

        val_paraloader = pl.ParallelLoader(val_dl, [device]).per_device_loader(device)
        loss, dice_coeff = validate_one_epoch(e_no, val_paraloader, fold_model,
                                              criterion, device, config, logger)
        del val_paraloader
        gc.collect()

        if dice_coeff > best_dice:
            xm.master_print("Saving Best Model | Dice Improvement: {} -----> {}".format(best_dice, dice_coeff))
            xm.save(fold_model.state_dict(), f"model_{flags['FOLD_NO']}.pth")
            best_dice = dice_coeff



###MAIN
if __name__ == '__main__':
    config = GlobalConfig
    seed_everything(config.seed)

    filename = np.array(os.listdir(config.IMG_PATH))
    groups = [x.split('_')[0] for x in filename]
    group_fold = GroupKFold(n_splits=config.num_split)

    for fold, (t_idx, v_idx) in enumerate(group_fold.split(filename, groups=groups)):
        print("Fold: {}".format(fold+1))
        print("-"*40)

        train_id = filename[t_idx]
        valid_id = filename[v_idx]

        train_ds = HuBMAPData(img_ids=train_id, config=config, transform=get_train_transform(config))
        valid_ds = HuBMAPData(img_ids=valid_id, config=config, transform=get_valid_transform(config))

        model = HuBMAPModel(config)
        model.float()

        FLAGS = {'FOLD_NO': fold,
                 'TRAIN_DS': train_ds,
                 'VAL_DS': valid_ds,
                 'FOLD_MODEL': model,
                 'LOGGER': logger,
                 'LR': config.lr,
                 'BATCH_SIZE' : 8,
                 'EPOCHS' : 30}

        xmp.spawn(fn = _mp_fn,
                  args = (FLAGS,),
                  nprocs = 8,
                  start_method = 'fork')


        print('\n')

        del train_ds, val_ds, model
