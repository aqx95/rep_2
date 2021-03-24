import os
import gc
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

from config import GlobalConfig
from logger import log
from loss import loss_fn
from model import create_model
from engine import Fitter
from data import prepare_loader
from common import get_dice_coeff


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_single_fold(df_folds, config, device, fold):
    model = create_model(config).to(device)
    train_id = df_folds[df_folds['fold'] != fold].filename.values
    valid_id = df_folds[df_folds['fold'] == fold].filename.values

    train_loader, valid_loader = prepare_loader(train_id, valid_id, config)
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
    config.model = args.model
    config.model_name = args.model_name

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
        train.loc[train_idx, 'fold'] = fold+1

    #training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loop(df_folds=train, config=config, device=device, fold_num=1,
               train_one_fold= config.train_one_fold)
