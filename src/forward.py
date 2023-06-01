import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import transforms
import torch.nn.functional as F

class forward():

    def __init__(self, img_size, timesteps=300, start=0.0001, end=0.02):
        self.img_size = img_size

        betas = torch.linspace(start, end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_index_from_list(vals, t, x_shape):

        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device='cpu'):

        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            sqrt_one_minus_alphas_cumprod_t, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


    def load_trandformed_Dataset(self):

        data_transforms = transforms.Compose[
            transforms.Resize(self.img_size, self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t:(t*2) - 1)
        ]

        train = torchvision.datasets.StanfordCars(root="/data", download=True,
                                                  transform=data_transforms,
                                                  split='test')
        test = torchvision.datasets.StanfordCars(root="/data", download=True,
                                         transform=data_transforms, split='test')

        return ConcatDataset([train, test])