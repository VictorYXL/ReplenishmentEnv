import torch
import numpy as np
import pdb


class RewardScaler(object):
    def __init__(self, demean=True, destd=True):
        super(RewardScaler, self).__init__()
        self.demean = demean
        self.destd = destd
        self.mean = None
        self.std = None
        self.EPS = 1e-6

    def fit(self, ep_batch):
        # Shape: #batch x #episode x #SKU x #lambda
        data = ep_batch['individual_rewards']
        self.mean = torch.mean(data, dim=(0, 1, 2), keepdim=True)
        self.std = torch.std(data, dim=(0, 1, 2), keepdim=True)

    def transform(self, ep_batch):
        data = ep_batch['individual_rewards']
        if self.demean:
            data = data - self.mean 
        if self.destd:
            data = data / self.std
        ep_batch.data.episode_data['individual_rewards'] = data
        return ep_batch