import os
import numpy as np
import pandas as pd
import urllib.request
import tarfile
import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

    df = pd.read_csv(csv_file)
    df.info()
    X_train, X_test = train_test_split(df, test_size=test_size)
    X_train = CustomDataset(X_train, data_dir=data_dir, predict_only=predict_only)
    X_test = CustomDataset(X_test, data_dir=data_dir, predict_only=predict_only)
    return X_train, X_test
  

class VocDataset(Dataset):
  def __init__(self, url, path):
    if not os.path.exists(path):
      self.get_archive(path, url)
      self.extract(path)
    self.root = os.path.join(path, 'VOCdevkit/VOC2012')
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
  
  def get_archive(self, path, url, filename='devkit'):
    os.makedirs(path, exist_ok=True)
    urllib.request.urlretrieve(url, f"{path}/{filename}.tar")

  def extract(self, path, filename='devkit'):
    tar_file = tarfile.open(f"{path}/{filename}.tar")
    tar_file.extractall(path)
    tar_file.close()
    os.remove(f"{path}/{filename}.tar")


def get_pascal_train_test_datasets(url: str = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
                                   path: str = '/workspace/data/VOC',
                                   test_size: float = 0.20,
                                   ):
    data = VocDataset(url=url, path=path)
    X_train, X_test = train_test_split(data, test_size=test_size)
    return X_train, X_test