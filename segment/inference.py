from typing import List
from functools import partial

import torch 
import torch.nn as nn
from segment.nn_models import UNet


class inference(nn.Module):
    def __init__(self, 
                 n_channels: int = 3,
                 n_classes: int = 1,
                 bilinear: bool = False,
                 ddims: List = [64, 128, 256, 512, 1024],
                 device: str = 'cuda'):
        self.device = device
        model = UNet(n_channels=n_channels,
                    n_classes=n_classes,
                    bilinear=bilinear,
                    ddims=ddims,
                    )
        model.to(device)
        model.load_state_dict(torch.load('best_model.pt'))
        self.predict = partial(self.get_segment_fn, model=model)

    def get_segment_fn(self, model, data, idx):
        image, mask = data[idx]
        out = model(image.to(self.device).unsqueeze(0))
        out = torch.sigmoid(out)
        out = (out > 0.5) * 0.1
        return out, image, mask
