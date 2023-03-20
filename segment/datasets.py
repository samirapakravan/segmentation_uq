import os
from typing import List

import numpy as np
import pandas as pd
import urllib.request
import tarfile
from zipfile import ZipFile
import gdown
from glob import glob
from PIL import Image

import cv2
import albumentations as A

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


DataSets: List[str] = ["custom", "pascal", "carvana"]


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
    return transformed_image, transformed_mask


class VocDataset(Dataset):
  def __init__(self, save_path: str):
    if not os.path.exists(save_path):
      self.get_archive(save_path)
      self.extract(save_path)
    self.root = os.path.join(save_path, 'VOCdevkit/VOC2012')
    self.target_dir = os.path.join(self.root, 'SegmentationClass')
    self.images_dir = os.path.join(self.root, 'JPEGImages')
    file_list = os.path.join(self.root, 'ImageSets/Segmentation/trainval.txt')
    self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    image_id = self.files[index]
    image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
    label_path = os.path.join(self.target_dir, f"{image_id}.png")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, label = cv2.threshold(label, 1, 255, cv2.THRESH_BINARY)
    label = cv2.resize(label, (256,256))
    # pytorch uses (c,h,w) our image is (h,w,c)
    transformed_image = np.transpose(image, (2,0,1)).astype(np.float32)
    transformed_mask = label[np.newaxis, ...].astype(np.float32)
    transformed_image = torch.Tensor(transformed_image) / 255.0
    transformed_mask = torch.round(torch.Tensor(transformed_mask) / 255.0)
    return transformed_image, transformed_mask
  
  def get_archive(self, save_path, filename='devkit'):
    os.makedirs(save_path, exist_ok=True)
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    urllib.request.urlretrieve(url, f"{save_path}/{filename}.tar")

  def extract(self, save_path, filename='devkit'):
    tar_file = tarfile.open(f"{save_path}/{filename}.tar")
    tar_file.extractall(save_path)
    tar_file.close()
    os.remove(f"{save_path}/{filename}.tar")


class CarvanaDataset(Dataset):
  def __init__(self, save_path: str):
    if not os.path.exists(save_path):
      self.get_archive(save_path)
      self.extract(save_path)
    self.transforms = T.Compose([T.Resize((256, 256)),
                                 T.ToTensor(),])
    file_path = os.path.join(save_path, 'train/*.*')
    file_mask_path = os.path.join(save_path, 'train_masks/*.*')
    self.images = sorted(glob(file_path))
    self.image_mask = sorted(glob(file_mask_path))

  def __getitem__(self, index: int):
    image = Image.open(self.images[index]).convert('RGB')
    image_mask = Image.open(self.image_mask[index]).convert('L')
    if self.transforms:
        image = self.transforms(image)
        image_mask = self.transforms(image_mask)
    return image, image_mask

  def __len__(self):
      return len(self.images)

  def get_archive(self, path):
    os.makedirs(path, exist_ok=True)
    gdown.download(url='https://drive.google.com/file/d/1seaq6sKfs2N7l65ZmgI9PxGXuruUctNG/view?usp=sharing', output = f"{path}/train_masks.zip", quiet=False, fuzzy=True)
    gdown.download(url='https://drive.google.com/file/d/1vgIhJai8PBajw70M3kSmUViawEPIdPOy/view?usp=share_link', output = f"{path}/train.zip", quiet=False, fuzzy=True)

  def extract(self, path):
    with ZipFile(f"{path}/train_masks.zip", 'r') as zf:
      zf.extractall(path)
    with ZipFile(f"{path}/train.zip", 'r') as zf:
      zf.extractall(path)
    os.remove(f"{path}/train_masks.zip")
    os.remove(f"{path}/train.zip")
    



def get_custom_train_test_datasets(csv_file: str = '/workspace/data/Human-Segmentation-Dataset-master/train.csv',
                                   data_dir: str = '/workspace/data/',
                                   test_size: float = 0.15,
                                   predict_only: bool = False,
                                   ):
    url = "https://github.com/parth1620/Human-Segmentation-Dataset-master/archive/refs/heads/master.zip"
    filename="Human-Segmentation-Dataset-master"
    path = os.path.join("/workspace/data", filename)
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)
      urllib.request.urlretrieve(url, f"{path}/{filename}.zip")
      with ZipFile(f"{path}/{filename}.zip", 'r') as zf:
        zf.extractall(path)
      os.remove(f"{path}/{filename}.zip")

    df = pd.read_csv(csv_file)
    df.info()
    X_train, X_test = train_test_split(df, test_size=test_size)
    X_train = CustomDataset(X_train, data_dir=data_dir, predict_only=predict_only)
    X_test = CustomDataset(X_test, data_dir=data_dir, predict_only=predict_only)
    return X_train, X_test


def get_pascal_train_test_datasets(save_path: str = '/workspace/data/VOC',
                                   test_size: float = 0.20,
                                   ):
    data = VocDataset(save_path=save_path)
    X_train, X_test = train_test_split(data, test_size=test_size)
    return X_train, X_test


def get_carvana_train_test_datasets(save_path: str = '/workspace/data/Carvana',
                                    test_size: float = 0.3,
                                    ):
  data = CarvanaDataset(save_path=save_path)
  X_train, X_test = train_test_split(data, test_size=test_size)
  return X_train, X_test
  


def get_data_set(data_set: str = "custom", test_size: float = 0.15, predict_only: bool = False):
    """Choose from [custom, pascal, carvana] datasets."""
    if data_set == "custom":
        X_train, X_test = get_custom_train_test_datasets(test_size=test_size, predict_only=predict_only)
    elif data_set == "pascal":
        X_train, X_test = get_pascal_train_test_datasets(test_size=test_size)
    elif data_set == "carvana":
       X_train, X_test = get_carvana_train_test_datasets(test_size=test_size)
    return X_train, X_test