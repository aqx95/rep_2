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


def plot_image(dataloader, path, fold):
    if not os.path.exists(path):
        os.makedirs(path)
    image, mask, non_empty = next(iter(dataloader)) #img(bs, C, H, W), mask(bs, H, W)
    image = image.permute(0,2,3,1)
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8)
    for j in range(8):
        axes[0, j].imshow(image[j])
        axes[0, j].set_title("Non-empty: {}".format(non_empty[j]))
        axes[0, j].axis('off')
        axes[1, j].imshow(mask[j])
        axes[1, j].axis('off')
    del dataloader
    plt.savefig(os.path.join(path, f'loader_image_{fold}.png'))


def plot_threhold(dice):
    dices = dice.value
    noise_ths = dice.ths
    best_dice = dices.max()
    best_thr = noise_ths[dices.argmax()]
    plt.figure(figsize=(8,4))
    plt.plot(noise_ths, dices, color='blue')
    plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max(), colors='black')
    d = dices.max() - dices.min()
    plt.text(noise_ths[-1]-0.1, best_dice-0.1*d, f'DICE = {best_dice:.3f}', fontsize=12);
    plt.text(noise_ths[-1]-0.1, best_dice-0.2*d, f'TH = {best_thr:.3f}', fontsize=12);
    plt.show()


def train_single_fold(df_folds, config, device, fold):
    model = create_model(config).to(device)
    train_id = df_folds[df_folds['fold'] != fold].filename.values
    valid_id = df_folds[df_folds['fold'] == fold].filename.values

    train_loader, valid_loader = prepare_loader(train_id, valid_id, config)
    plot_image(train_loader, config.SAVE_PATH, fold)  #output img
    #Begin fitting single fold
    fitter = Fitter(model, device, config)
    logger.info("Fold {} data preparation DONE...".format(fold))
    best_fold_dice, fold_thres = fitter.fit(train_loader, valid_loader, fold)
    plot_threshold(fold_thres)
    logger.info("Finish Training Fold {}".format(fold))

    return best_fold_dice


def train_loop(df_folds: pd.DataFrame, config, device, fold_num:int=None, train_one_fold=False):
    val_pred = []

    if train_one_fold:
        fold_dice = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold_num)
        logger.info("Fold {} Valid Dice Score: {}".format(fold_num, fold_dice))

    else:
        for fold in (number+1 for number in range(config.num_folds)):
            fold_dice = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold)
            val_pred.append(fold_dice)
            logger.info("Fold {} Valid Dice Score: {}".format(fold, fold_dice))
            logger.info("-------------------------------------------------------------------")

        mean_oof_dice = sum(val_pred) / len(val_pred)
        logger.info("5 Folds OOF Dice Score: {}".format(mean_oof_dice))


###MAIN
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hubmap')
    parser.add_argument('--num-epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--image-size', type=int, default=512, help='image size for training')
    parser.add_argument('--train-one-fold', action='store_true', help='train one/all folds')
    args = parser.parse_args()
    print(args)

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
    kfold = KFold(n_splits=config.num_folds,
                  random_state=config.seed,
                  shuffle=True)

    train['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(filename)):
        train.loc[valid_idx, 'fold'] = fold+1

    #training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loop(df_folds=train, config=config, device=device, fold_num=1,
               train_one_fold= config.train_one_fold)
