from typing import List
import os
import pdb
import tqdm
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
import torch 

from segment.helper import show_image
from segment.nn_models import UNet
from segment.dataset import CustomDataset
from segment.inference import inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Parameters
CSV_FILE: str = '/workspace/data/Human-Segmentation-Dataset-master/train.csv'
DATA_DIR: str = '/workspace/data/'

unet_base_exp: int = 6
unet_dims: List = []
for i in range(5):
    unet_dims.append(2**(unet_base_exp + i))


# ----- Loading data into dataloaders
df = pd.read_csv(CSV_FILE)
df.info()
X_pred, _ = train_test_split(df, test_size=0.01)
X_pred = CustomDataset(X_pred, data_dir=DATA_DIR, predict_only=True)
print(f"Size of Predict Dataset : {len(X_pred)}")


# ----- Load trained model
model = inference(n_channels=3,
                n_classes=1,
                bilinear=False,
                ddims=unet_dims,
                device=device)


# ------ Iterate and Save
def save_image_mask_out(idx, image, mask, out):
    show_image(image, mask, out.detach().cpu().squeeze(0))
    filename = f'pred_{idx}_.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def save_image_mask_avg_std(idx, image, mask, avg_out, std_out):
    show_image(image, mask, avg_out.detach().cpu(), std_out.detach().cpu())
    filename = f'pred_{idx}.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

output_path = '/workspace/output'
os.makedirs(output_path, exist_ok=True)
num_mc = 100
avg_out = torch.zeros_like(X_pred[0][1]).to(device=device)
std_out = torch.zeros_like(X_pred[0][1]).to(device=device)

for idx in tqdm.trange(len(X_pred)):
    avg_out.zero_()
    std_out.zero_()
    for _ in range(num_mc):
        out, image, mask = model.predict(data=X_pred, idx=idx)
        avg_out += out.squeeze() / num_mc
        std_out += torch.square(out.squeeze()) / num_mc
    std_out = torch.sqrt(torch.abs(std_out - torch.square(avg_out)))

    save_image_mask_avg_std(idx, image, mask, avg_out, std_out)
    # save_image_mask_out(idx, image, mask, out)