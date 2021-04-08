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
            img, mask = sample['image'].float(), sample['mask'].float()

        return img, mask


#Transformation
def get_train_transform(config):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(p=0.3),
        ShiftScaleRotate(p=0.3),
        RandomBrightnessContrast(p=0.3),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(),
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
              ], p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0),
        ])

def get_valid_transform(config):
    return Compose([
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(p=1.0),
            ])


#Prepare dataloaders
def prepare_loader(train_id, valid_id, config):
    train_ds = HuBMAPData(img_ids=train_id, config=config, transform=get_train_transform(config))
    valid_ds = HuBMAPData(img_ids=valid_id, config=config, transform=get_valid_transform(config))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2)

    valid_loader = DataLoader(
        valid_ds,
        batch_size = config.batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader, valid_loader
