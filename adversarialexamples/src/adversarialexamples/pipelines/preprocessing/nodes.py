"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.1
"""
from torch.utils.data import DataLoader

def create_loader(dataset,param_train):
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=param_train["batch_size"], shuffle=False,num_workers=param_train["num_workers"])
    return loader
