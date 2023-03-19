from typing import List
import os
import tqdm
import matplotlib.pyplot as plt 
import torch 

from segment.helper import show_image
from segment.datasets import (get_custom_train_test_datasets,
                              get_pascal_train_test_datasets,
                              )
from segment.inference import inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Parameters
num_predictions: int = 10 # number of images to segment from the dataset
num_monte_carlo: int = 100 # number of evaluations for each image for UQ

# dataset
DataSets: List[str] = ["custom", "pascal"]
data_set : str = DataSets[0]
output_path: str = os.path.join("/workspace/output", data_set)

# model
bilinear: bool = False
unet_base_exp: int = 6
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

os.makedirs(output_path, exist_ok=True)

avg_out = torch.zeros_like(X_pred[0][1]).to(device=device)
std_out = torch.zeros_like(X_pred[0][1]).to(device=device)

for idx in tqdm.trange(min((len(X_pred), num_predictions))):
    avg_out.zero_()
    std_out.zero_()
    for _ in range(num_monte_carlo):
        out, image, mask = model.predict(data=X_pred, idx=idx)
        avg_out += out.squeeze() / num_monte_carlo
        std_out += torch.square(out.squeeze()) / num_monte_carlo
    std_out = torch.sqrt(torch.abs(std_out - torch.square(avg_out)))
    save_image_mask_avg_std(idx, image, mask, avg_out, std_out)
    # save_image_mask_out(idx, image, mask, out)