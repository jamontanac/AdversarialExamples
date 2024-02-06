from kedro.io import AbstractDataset
import torch
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

class PytorchFlexibleDataset(AbstractDataset):
    def __init__(self, dataset_name: str, root: str, 
                 train: bool, mean: Tuple[float, float, float], 
                 std: Tuple[float, float, float], 
                 normalize: bool = True):
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self._filepath = Path(root)
        self.train = train

        self.available_datasets = {
            'CIFAR10': datasets.CIFAR10,
            'PaintingStyle': datasets.ImageFolder  # Assuming the dataset is organized in folders by class
            # You can add custom dataset handlers here if needed
        }
        self._set_transforms()

    def _set_transforms(self):
        if self.normalize:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),  # Adjust crop size if necessary
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def _load(self) -> Any:
        if self.dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        dataset_class = self.available_datasets[self.dataset_name]
        if self.dataset_name == 'PaintingStyle':
            if self.train:
                original_dataset = dataset_class(root=str(self._filepath), 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.mean, self.std)])
                                                )
                augmented_dataset =  dataset_class(root=str(self._filepath), 
                                                transform = self.transform)
                dataset = ConcatDataset([original_dataset,augmented_dataset])

            else:
                dataset =  dataset_class(root=str(self._filepath), 
                                                transform = self.transform)
        elif  self.dataset_name=='CIFAR10':
            if self.train:
                original_dataset = dataset_class(root=str(self._filepath),
                                                train=self.train,
                                                download=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.mean, self.std)])
                                                )
                augmented_dataset = dataset_class(root=str(self._filepath),
                                                train=self.train,
                                                download=False,
                                                transform=self.transform
                                                )
                dataset  = ConcatDataset([original_dataset,augmented_dataset])
            else:
                dataset =  dataset_class(root=str(self._filepath),
                                                train=self.train,
                                                download=False,
                                                transform=self.transform
                                                ) 
        else:
            pass

        return dataset


# class PytorchFlexibleDataset(AbstractDataset):
#     def __init__(self, dataset_name: str,
#                     root: str, train: bool,
#                     mean: Tuple[float,float,float],
#                     std: Tuple[float,float,float], 
#                     normalize: bool = True):
#         self.dataset_name = dataset_name
#         self.mean = mean
#         self.std = std
#         self.normalize = normalize
#         self._filepath = Path(root)
#         self.train = train

#         self.available_datasets = {
#             'CIFAR10': datasets.CIFAR10,
#             # Add more datasets here as needed, e.g., 'Imagenette': datasets.ImageFolder
#         }
#         if normalize:
#             if train:
#                 self.transform = transforms.Compose([
#                     transforms.RandomCrop(32, padding=4),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean, std)
#                 ])
#             else:
#                 self.transform = transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean, std)
#                 ])

#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#             ])

#     def _load(self) -> Any:
#         if self.dataset_name not in self.available_datasets:
#             raise ValueError(f"Dataset {self.dataset_name} not supported.")

#         dataset_class = self.available_datasets[self.dataset_name]
#         # Some datasets might not use the 'train' parameter, handle such cases as needed
        
#         if self.normalize and self.train:
#             dataset_original = dataset_class(
#                 root=self._filepath, 
#                 train=self.train, 
#                 download=False, 
#                 transform=transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize(self.mean, self.std)
#                 ])
#             )
#             dataset_augmented = dataset_class(
#                 root=self._filepath, 
#                 train=self.train, 
#                 download=False, 
#                 transform=self.transform
#             )
#             dataset = torch.utils.data.ConcatDataset([dataset_original, dataset_augmented])
#         elif self.normalize and not self.train:
#             dataset = dataset_class(
#                 root=self._filepath, 
#                 train=self.train, 
#                 download=False, 
#                 transform=self.transform
#             )
#         return dataset

#     def _save(self, data: Any) -> None:
#         raise NotImplementedError("Saving datasets is not supported.")

#     def _describe(self) -> Dict[str, Any]:
#         return dict(
#             dataset_name=self.dataset_name,
#             root=self._filepath,
#             train=self.train,
#             transform = self.transform,
#             normalize = self.normalize,
#             mean = self.mean,
#             std = self.std
#         )



class PytorchDatasetModel(AbstractDataset):
    def __init__(self,filepath:str, model: torch.nn.Module = None):
        self._model = model
        self._filepath = Path(filepath)

    def _load(self) -> Dict:
        if torch.cuda.is_available():
            model = torch.load(self._filepath)
        else:
            model = torch.load(self._filepath, map_location='cpu')
        return model

    def _save(self,model:torch.nn.Module) -> None:
        torch.save(model, self._filepath)

    def _describe(self) -> dict[str, Any]:
        return dict(filepath=self._filepath)
        
    
class PytorchDatasetDict(AbstractDataset):
    def __init__(self,filepath:str):
        self._filepath = Path(filepath)

    def _load(self) -> Dict:
        model = torch.load(self._filepath)
        return model

    def _save(self,data: Dict) -> None:
        torch.save(data, self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)