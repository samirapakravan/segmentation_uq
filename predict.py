from typing import List
import os
import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from segment.inference import inference
from segment.analytics import uq_analytics
from segment.helper import save_image_mask_avg_std
from segment.datasets import get_data_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Parameters
cfg = OmegaConf.load("conf/config.yaml")

num_predictions: int = cfg.predict.num_predictions # number of images to segment from the dataset
num_monte_carlo: int = cfg.predict.num_monte_carlo # number of evaluations for each image for UQ

# dataset
data_set : str = cfg.predict.data_set
output_path: str = os.path.join("/workspace/output", data_set)

# model
bilinear: bool = cfg.model.bilinear
unet_base_exp: int = cfg.model.unet_base_exp
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(unet_base_exp + i))


# ----- Loading data into dataloaders
X_pred, _ = get_data_set(data_set=data_set, test_size=0.01, predict_only=True)
print(f"\nSize of Predict Dataset : {len(X_pred)}")


# ----- Load trained model
model = inference(n_channels=3,
                  n_classes=1,
                  bilinear=bilinear,
                  ddims=unet_dims,
                  device=device)


# ------ Quantify Uncertainties
uqx = uq_analytics(cfg=cfg,
                   save_path='/workspace/output/analytics',
                   base_filename=cfg.exp_name)


# ------ Evaluation
def eval(output, mask, eps=1e-5):
    with torch.no_grad():
        output = output.squeeze().to(device=device)
        mask = mask.squeeze().to(device=device)

        preds = (torch.sigmoid(output) > 0.5).float()
        num_correct = ((preds == mask).sum())
        num_pixels = torch.numel(preds)
        pixel_acc = num_correct/num_pixels

        dice_score = (2 * (preds * mask).sum()) / ((preds + mask).sum() + eps)

    return pixel_acc, dice_score

pixel_acc_t = dice_score_t = 0
for idx in tqdm.trange(min((len(X_pred), num_predictions))):
    output, image, mask = model.predict(data=X_pred, idx=idx)
    pixel_acc, dice_score = eval(output, mask,eps=1e-5)
    pixel_acc_t += pixel_acc
    dice_score_t += dice_score

n = min((len(X_pred), num_predictions))    
print(f"\n Accuracy:{pixel_acc_t/n :<3.5f},\n Dice_score:{dice_score_t/n :<3.5f}\n")

#-------------

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
uqx.save_metrics_file()
uqx.save_metrics_histogram()