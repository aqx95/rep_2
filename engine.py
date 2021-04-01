import os
import logging
import datetime
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from loss import loss_fn
from common import DiceMeter, LossMeter

class Fitter:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logging.getLogger('training')

        self.best_dice = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.oof = None
        self.monitored_metric = None

        if not os.path.exists(self.config.SAVE_PATH):
            os.makedirs(self.config.SAVE_PATH)
        if not os.path.exists(self.config.LOG_PATH):
            os.makedirs(self.config.LOG_PATH)

        self.loss = loss_fn(config).to(self.device)
        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                **config.optimizer_params[config.optimizer])
        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(optimizer=self.optimizer,
                                **config.scheduler_params[config.scheduler])


    def fit(self, train_loader, valid_loader, fold):
        self.logger.info("Starts Training with {} on Device: {}".format(self.config.encoder, self.device))

        for epoch in range(self.config.num_epochs):
            self.logger.info("LR: {}".format(self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_one_epoch(train_loader)
            self.logger.info("[RESULTS] Train Epoch: {} | Train Loss: {:.3f}".format(self.epoch, train_loss))
            valid_loss, dice_coeff = self.validate_one_epoch(valid_loader)
            self.logger.info("[RESULTS] Validation Epoch: {} | Valid Loss: {:.3f} | Dice: {:.3f}".format(self.epoch, valid_loss, dice_coeff))

            self.monitored_metrics = dice_coeff
            #self.oof = val_pred

            if self.best_loss > valid_loss:
                self.best_loss = valid_loss
            if self.best_dice < dice_coeff:
                self.logger.info(f"Epoch {self.epoch}: Saving model... | Dice improvement {self.best_dice} -> {dice_coeff}")
                self.save(os.path.join(self.config.SAVE_PATH, '{}_fold{}.pt').format(self.config.encoder, fold))
                self.dice = dice_coeff

            #Update Scheduler
            if self.config.val_step_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

            self.epoch += 1

        fold_checkpoint = self.load(os.path.join(self.config.SAVE_PATH, '{}_fold{}.pt').format(self.config.encoder, fold))
        return fold_checkpoint


    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = LossMeter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for step, (img, mask) in pbar:
            img, mask = img.to(self.device), mask.to(self.device)
            batch_size = img.shape[0]

            mask_pred = self.model(img) #(bs, 1, h, w)
            loss = self.loss(mask_pred, mask)
            summary_loss.update(loss.item(), batch_size)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.config.train_step_scheduler:
                self.scheduler.step(self.epoch + step/len(train_loader))

            description = f"Train Steps: {step}/{len(train_loader)} Summary Loss: {summary_loss.avg:.3f}"
            pbar.set_description(description)

        return summary_loss.avg


    def validate_one_epoch(self, valid_loader):
        self.model.eval()
        dice = DiceMeter()
        summary_loss = LossMeter()
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

        with torch.no_grad():
            for step, (img, mask) in pbar:
                img, mask = img.to(self.device), mask.to(self.device)
                batch_size = img.shape[0]

                mask_pred = self.model(img)
                loss = self.loss(mask_pred, mask)
                summary_loss.update(loss.item(), batch_size)
                dice.accumulate(mask_pred, mask)

                description = f"Valid Steps: {step}/{len(valid_loader)} Summary Loss: {summary_loss.avg:.3f} Dice: {dice.avg:.3f}"
                pbar.set_description(description)


        return summary_loss.avg, dice.avg


    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_auc": self.best_dice,
                "epoch": self.epoch,
                #"oof_pred": self.oof
            }, path
        )


    def load(self, path):
        checkpoint = torch.load(path)
        return checkpoint
