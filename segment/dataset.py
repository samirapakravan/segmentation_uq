import os
import numpy as np

import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
  def __init__(self, df, data_dir, predict_only=False):
    self.df = df
    self.data_dir = data_dir
    if predict_only:
      self.transform = A.Compose([A.Resize(width=320, height=320),
                                ], is_check_shapes = 0)
    else:
      self.transform = A.Compose([A.Resize(width=320, height=320),
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  ], is_check_shapes = 0)
    

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image_path = self.df.iloc[idx].images
    image_path = os.path.join(self.data_dir, image_path)
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    mask_path = self.df.iloc[idx].masks
    mask_path = os.path.join(self.data_dir, mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask, axis=-1)

    transformed = self.transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    # pytorch uses (c,h,w) our image is (h,w,c)
    transformed_image = np.transpose(transformed_image, (2,0,1)).astype(np.float32)
    transformed_mask = np.transpose(transformed_mask, (2,0,1)).astype(np.float32)

    transformed_image = torch.Tensor(transformed_image) / 255.0
    transformed_mask = torch.round(torch.Tensor(transformed_mask) / 255.0)
    #transformed_image = torch.transpose(transformed_image, (2,0,1))  # cannot be used
    #transformed_mask = torch.transpose(transformed_mask, (2,0,1))

    return transformed_image, transformed_mask