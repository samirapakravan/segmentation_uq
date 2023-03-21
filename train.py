from typing import List
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from omegaconf import OmegaConf

from segmentation_models_pytorch.losses import DiceLoss # for image segmentation task
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from segment.helper import show_image, plot_loss
from segment.nn_models import UNet
from segment.datasets import get_data_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----- Parameters
cfg = OmegaConf.load("conf/config.yaml")

# Data
data_set : str = cfg.trainer.data_set

# Trainer
epochs: int = cfg.trainer.epochs
batch_size: int = cfg.trainer.batch_size
test_size: float = cfg.trainer.test_size

# optimizer parameters
amp: bool = cfg.optim.amp
gradient_clipping: float = cfg.optim.gradient_clipping
optim_type: str = cfg.optim.optim_type
learning_rate: float = cfg.optim.learning_rate
weight_decay: float = cfg.optim.weight_decay
momentum: float = cfg.optim.momentum

# Unet parameters
bilinear: bool = cfg.model.bilinear
unet_base_exp: int = cfg.model.unet_base_exp
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(unet_base_exp + i))


# ----- Loading data into dataloaders
X_train, X_test = get_data_set(data_set=data_set, test_size=test_size)
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
def compose_loss_fn(n_classes: int = 1,
                    alpha: float = 1.0,
                    beta: float = 1.0,):
    """The overall loss function is a linear combination of BCE/CE and Dice"""
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(mode='binary')
    total = 0.5 * (alpha + beta)
    alpha /= total 
    beta /= total 
    def loss_fn(masks_pred, masks):
        loss = alpha * criterion(masks_pred, masks.float())
        loss += beta * dice_loss(masks_pred, masks.float())
        return loss
    return loss_fn

loss_fn = compose_loss_fn(n_classes=model.n_classes,
                          alpha=cfg.model.loss.bce_frac,
                          beta=cfg.model.loss.dice_frac)


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
elif optim_type=="sgd":    
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=learning_rate, 
                                momentum=momentum)

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
            loss = loss_fn(masks_pred, masks)

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
            t_loss = loss_fn(masks_pred, masks)
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
out = (out > 0.5) * 0.1

show_image(image, mask, out.detach().cpu().squeeze(0))
plt.savefig('test.png')
plt.close()

plot_loss(np.arange(epochs), train_losses, test_losses)
plt.savefig('losses.png')
plt.close()
