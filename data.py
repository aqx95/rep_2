import cv2
import os
from torch.utils.data import Dataset, DataLoader
from albumentations import *
from albumentations.pytorch import ToTensorV2

class HuBMAPData(Dataset):
    def __init__(self, img_ids, config, transform=None):
        self.img_id = img_ids
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.config.IMG_PATH, self.img_id[idx])
        mask_path = os.path.join(self.config.MASK_PATH, self.img_id[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sample = self.transform(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask


#Transformation
def get_train_transform(config):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
            ], p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0),
        ])


def get_valid_transform(config):
    return Compose([
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ToTensorV2(p=1.0),
            ])