from typing import List
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from segmentation_models_pytorch.losses import DiceLoss # for image segmentation task
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from segment.helper import show_image, plot_loss
from segment.nn_models import UNet
from segment.datasets import (get_custom_train_test_datasets,
                              get_pascal_train_test_datasets,
                              )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----- Parameters
# Data
DataSets: List[str] = ["custom", "pascal"] 
data_set : str = DataSets[1]

# Trainer
epochs: int = 50
batch_size: int = 1

# optimizer parameters
amp: bool = True
gradient_clipping: float = 1.0
optim_type: str = "adam"
learning_rate: float = 1e-4
weight_decay: float = 1e-8
momentum: float = 0.999

# Unet parameters
bilinear: bool = False
unet_base_exp: int = 6
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(unet_base_exp + i))


# ----- Loading data into dataloaders
if data_set == "custom":
    X_train, X_test = get_custom_train_test_datasets(test_size=0.15)
elif data_set == "pascal":
    X_train, X_test = get_pascal_train_test_datasets(test_size=0.30)

print(f"Size of Train Dataset : {len(X_train)}")
print(f"Size of Valid Dataset : {len(X_test)}")

train_loader = DataLoader(dataset=X_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)

test_loader = DataLoader(dataset=X_test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=2)

print(f"num of batches in train: {len(train_loader)}, and test: {len(test_loader)}")


# ----- instantiate model
model = UNet(n_channels=3,
             n_classes=1,
             bilinear=bilinear,
             ddims=unet_dims,
             UQ=True,
             )
model.to(device)
summary(model, X_train[0][0].shape)


# ----- Loss criterion
criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()


# ----- Optimizer + Scheduler + AMP
if optim_type=="adam":
    optimizer =  torch.optim.Adam(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay,)
elif optim_type=="rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum,
                                    foreach=True,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)


# ----- Training Loop
train_losses = []
test_losses = []
best_test_loss= np.Inf

for i in range(epochs):
    # ---- train
    train_epoch_loss = 0.0
    for images, masks in tqdm(train_loader):
        images =  images.to(device)
        masks = masks.to(device)
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            masks_pred = model(images)
            loss = criterion(masks_pred, masks.float())
            loss += DiceLoss(mode='binary')(torch.sigmoid(masks_pred), masks.float())
        train_epoch_loss += loss / len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

    # ---- test
    with torch.no_grad():
        test_epoch_loss = 0.0
        for images, masks in tqdm(test_loader):
            images =  images.to(device)
            masks = masks.to(device)
            masks_pred = model(images)
            t_loss = criterion(masks_pred, masks.float())
            t_loss += DiceLoss(mode='binary')(torch.sigmoid(masks_pred), masks.float())
            test_epoch_loss += t_loss / len(test_loader)
        scheduler.step(test_epoch_loss)
        if test_epoch_loss < best_test_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print('MODEL SAVED')
            best_test_loss = test_epoch_loss

    # ---- logging
    train_losses.append(train_epoch_loss.detach().cpu().numpy())
    test_losses.append(test_epoch_loss.detach().cpu().numpy())
    print(f'Epoch:{i}, train_loss: {train_epoch_loss}, test_loss: {test_epoch_loss}')


# ----- Inference
model.load_state_dict(torch.load('best_model.pt'))

idx = 5
image, mask = X_test[idx]
out = model(image.to(device).unsqueeze(0)) # (C,H,W) --> (1,C,H,W)  to add batch dim
out = torch.sigmoid(out)
out = (out>0.5)*0.1

show_image(image, mask, out.detach().cpu().squeeze(0))
plt.savefig('test.png')
plt.close()

plot_loss(np.arange(epochs), train_losses, test_losses)
plt.savefig('losses.png')
plt.close()
