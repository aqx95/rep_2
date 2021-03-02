import gc
import logging
import torch
from tqdm import tqdm
import torch_xla
import torch_xla.core.xla_model as xm

from common import get_dice_coeff, reduce

def train_one_epoch(epoch, train_loader, model, optimizer, criterion, device, config, logger, scheduler=None):
    model.train()
    losses = []
    dice_coeffs = []

    for steps, (img, mask) in enumerate(train_loader):
        img, mask = img.to(device), mask.to(device)

        mask_pred = model(img.float())
        loss = criterion(mask_pred, mask.float())
        loss.backward()
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()

        if config.train_step_scheduler:
            scheduler.step(epoch+steps/len(train_loader))

        loss_reduce = xm.mesh_reduce('Train_Loss', loss, reduce)
        losses.append(loss_reduce.item())

        dice_coeff = get_dice_coeff(torch.squeeze(mask_pred), mask.float())
        dice_coeffs.append(xm.mesh_reduce('Train_DiceCoeff', dice_coeff, reduce).item())

        del img, mask, mask_pred

    xm.master_print(f"Epoch: {epoch} | Train Loss: {reduce(losses): .4f} | Train Dice: {reduce(dice_coeffs): .4f}")


def validate_one_epoch(epoch, valid_loader, model, criterion, device, config, logger):
    model.eval()
    losses = []
    dice_coeffs = []

    with torch.no_grad():
        for steps, (img, mask) in enumerate(valid_loader):
            img, mask = img.to(device), mask.to(device)

            mask_pred = model(img.float())
            loss = criterion(mask_pred,mask.float())
            losses.append(xm.mesh_reduce('val_loss_reduce',
                                         loss,
                                         reduce).item())

            dice_coeff = get_dice_coeff(torch.squeeze(mask_pred),
                                        mask.float())
            dice_coeffs.append(xm.mesh_reduce('val_dice_reduce',
                                              dice_coeff,
                                              reduce).item())

    total_dice_coeff = reduce(dice_coeffs)
    total_loss = reduce(losses)

    xm.master_print(f'Val Loss : {total_loss: .4f}, Val Dice : {total_dice_coeff: .4f}')

    return total_loss, total_dice_coeff
