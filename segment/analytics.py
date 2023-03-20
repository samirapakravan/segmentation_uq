import os
import pickle
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
import numpy as np


class uq_analytics:
    def __init__(self,
                 cfg: OmegaConf = None,
                 save_path: str = '/workspace/output/analytics',
                 base_filename: str = 'uq'):
        self.metrics_store = []
        self.analytics_path = save_path
        self.save_filename = self.hash_filename(cfg, base_filename)
        self.analytics_filepath = os.path.join(self.analytics_path, self.save_filename)
        os.makedirs(self.analytics_path, exist_ok=True)

    def evaluate_uq(self,
                    std_out: torch.Tensor):
        L_frobenius = torch.linalg.norm(std_out.squeeze(), ord='fro')
        L_max_value = torch.linalg.norm(std_out.squeeze(), ord=1)
        L_largest_singular_value = torch.linalg.norm(std_out.squeeze(), ord=2)
        self.metrics_store.append([L_frobenius.detach().cpu().numpy(),
                                    L_max_value.detach().cpu().numpy(),
                                    L_largest_singular_value.detach().cpu().numpy()])
    
    def hash_filename(self,
                      cfg: OmegaConf,
                      base_filename: str):
        suffix = '-'.join("%s=%r" % (key,val) for (key,val) in cfg.model.items())
        save_filename = base_filename + '-' + suffix + '.pkl'
        return save_filename

    def save_metrics(self):
        self.metrics_store = np.array(self.metrics_store)
        with open(self.analytics_filepath, 'wb') as handle: 
            pickle.dump(self.metrics_store, handle)
        
        plt.figure(figsize=(8,8))
        plt.hist(self.metrics_store[:,0], bins=10, density=True, color='r', label='Frobenius', histtype='step')
        plt.hist(self.metrics_store[:,1], bins=10, density=True, color='g', label='max value', histtype='step')
        plt.hist(self.metrics_store[:,2], bins=10, density=True, color='b', label='max singular value', histtype='step')
        plt.axvline(x=np.median(self.metrics_store[:,0]), color='r', linestyle='--', linewidth=3)
        plt.axvline(x=np.median(self.metrics_store[:,1]), color='g', linestyle='--', linewidth=3)
        plt.axvline(x=np.median(self.metrics_store[:,2]), color='b', linestyle='--', linewidth=3)
        plt.ylabel('density', fontsize=25)
        plt.xlabel('uncertainty', fontsize=25)
        plt.ylim([0, 1])
        plt.legend(fontsize=10)
        plt.savefig(self.analytics_filepath + '.png')
        plt.close()

    def load_metrics(self):
        with open(self.analytics_filepath, 'rb') as handle: 
            self.metrics_store = pickle.load(handle)
        return self.metrics_store