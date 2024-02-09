"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.1
"""
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# def compute_mean_std(loader):
#     channels_sum, channels_squared_sum, num_samples = 0, 0, 0

#     for data, _ in loader:
#         # print("Data shape:", data.shape)
#         data = data.view(data.size(0), -1)
#         channels_sum += torch.mean(data, dim=1)
#         channels_squared_sum += torch.mean(data ** 2, dim=1)
#         num_samples += data.size(1)

#         # channels_sum += torch.mean(data, dim=[0, 1, 2])
#         # channels_squared_sum += torch.mean(data ** 2, dim=[0, 1, 2])
#         # num_batches += 1

#     mean = channels_sum / num_samples
#     std = (channels_squared_sum / num_samples - mean ** 2) ** 0.5
#     print(mean, std)
#     return mean, std
def create_loader(dataset,param_train):
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=param_train["batch_size"], shuffle=True,num_workers=param_train["num_workers"])
    return loader
