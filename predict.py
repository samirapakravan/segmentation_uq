from typing import List
import os
import tqdm
from omegaconf import OmegaConf

import torch

from segment.inference import inference
from segment.analytics import uq_analytics
from segment.helper import save_image_mask_avg_std
from segment.datasets import (get_custom_train_test_datasets,
                              get_pascal_train_test_datasets,)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Parameters
cfg = OmegaConf.load("conf/config.yaml")

num_predictions: int = cfg.predict.num_predictions # number of images to segment from the dataset
num_monte_carlo: int = cfg.predict.num_monte_carlo # number of evaluations for each image for UQ

# dataset
DataSets: List[str] = ["custom", "pascal"]
data_set : str = cfg.predict.data_set
output_path: str = os.path.join("/workspace/output", data_set)

# model
bilinear: bool = cfg.model.bilinear
unet_base_exp: int = cfg.model.unet_base_exp
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(unet_base_exp + i))


# ----- Loading data into dataloaders
if data_set == "custom":
    X_pred, _ = get_custom_train_test_datasets(test_size=0.01, predict_only=True)
elif data_set == "pascal":
    X_pred, _ = get_pascal_train_test_datasets(test_size=0.01)
print(f"Size of Predict Dataset : {len(X_pred)}")


# ----- Load trained model
model = inference(n_channels=3,
                  n_classes=1,
                  bilinear=bilinear,
                  ddims=unet_dims,
                  device=device)


# ------ Quantify Uncertainties
uqx = uq_analytics(cfg=cfg,
                   save_path='/workspace/output/analytics',
                   base_filename='uq')

avg_out = torch.zeros_like(X_pred[0][1]).to(device=device)
std_out = torch.zeros_like(X_pred[0][1]).to(device=device)

for idx in tqdm.trange(min((len(X_pred), num_predictions))):
    # evaluate uncertainty by multiple calls of model on each image
    avg_out.zero_()
    std_out.zero_()
    for _ in range(num_monte_carlo):
        out, image, mask = model.predict(data=X_pred, idx=idx)
        avg_out += out.squeeze() / num_monte_carlo
        std_out += torch.square(out.squeeze()) / num_monte_carlo
    std_out = torch.sqrt(torch.abs(std_out - torch.square(avg_out)))

    # save uncertainty heatmap
    save_image_mask_avg_std(output_path, idx, image, mask, avg_out, std_out)

    # evaluate quantified analytics
    uqx.evaluate_uq(std_out=std_out)

# save metrics of uncertainty on all considered images
uqx.save_metrics()
