import os
import gc
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

from config import GlobalConfig
from common import log
from loss import loss_fn
from model import create_model
from engine import Fitter
from data import prepare_loader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_image(dataloader, path):
    image, mask = dataloader.__getitem__(6)
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=bsize)
    for j in range(8):
        axes[0, j].imshow(image[j])
        axes[0, j].set_title(j)
        axes[0, j].axis('off')
        axes[1, j].imshow(mask[j])
        axes[1, j].axis('off')
    plt.savefig(os.path.join(path, 'loader_image.png'))


def train_single_fold(df_folds, config, device, fold):
    model = create_model(config).to(device)
    train_id = df_folds[df_folds['fold'] != fold].filename.values
    valid_id = df_folds[df_folds['fold'] == fold].filename.values

    train_loader, valid_loader = prepare_loader(train_id, valid_id, config)
    plot_image(train_loader, config.SAVE_PATH)  #output img
    #Begin fitting single fold
    fitter = Fitter(model, device, config)
    logger.info("Fold {} data preparation DONE...".format(fold))
    best_checkpoint = fitter.fit(train_loader, valid_loader, fold)
    valid_pred = best_checkpoint['oof_pred']
    logger.info("Finish Training Fold {}".format(fold))

    return valid_pred


def train_loop(df_folds: pd.DataFrame, config, device, fold_num:int=None, train_one_fold=False):
    val_pred = []

    if train_one_fold:
        _oof_pred = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold_num)
        val_pred.append(val_pred)
        curr_fold_dice = sklearn.metrics.roc_auc_score(_oof_df['label'], _oof_df['oof_pred'])
        logger.info("Fold {} AUC Score: {}".format(fold_num, curr_fold_dice))

    else:
        for fold in (number+1 for number in range(config.num_folds)):
            _oof_df = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_auc = sklearn.metrics.roc_auc_score(_oof_df['label'], _oof_df['oof_pred'])
            logger.info("Fold {} AUC Score: {}".format(fold, curr_fold_auc))
            logger.info("-------------------------------------------------------------------")

        oof_auc = sklearn.metrics.roc_auc_score(oof_df['label'], oof_df['oof_pred'])
        logger.info("5 Folds OOF AUC Score: {}".format(oof_auc))
        oof_df.to_csv(f"oof_{config.model_name}.csv")


###MAIN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hubmap')
    parser.add_argument('--num-epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--image-size', type=int, default=512, help='image size for training')
    parser.add_argument('--train-one-fold', type=bool, default=False, help='train one/all folds')
    args = parser.parse_args()

    #overwrite settings
    config = GlobalConfig
    config.num_epochs = args.num_epochs
    config.image_size = args.image_size
    config.train_one_fold = args.train_one_fold

    seed_everything(config.seed)

    #initialise logger
    logger = log(config, 'training')
    logger.info(config.__dict__)
    logger.info("-------------------------------------------------------------------")

    #Generate folds
    train = pd.DataFrame()
    filename = np.array(os.listdir(config.IMG_PATH))
    train['filename'] = filename
    groups = [x.split('_')[0] for x in filename]
    group_fold = GroupKFold(n_splits=config.num_folds)

    train['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(group_fold.split(filename, groups=groups)):
        train.loc[valid_idx, 'fold'] = fold+1

    #training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loop(df_folds=train, config=config, device=device, fold_num=1,
               train_one_fold= config.train_one_fold)
