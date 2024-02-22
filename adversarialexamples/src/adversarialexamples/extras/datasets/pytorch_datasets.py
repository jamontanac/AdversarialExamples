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
                 normalize: bool = True,
                 image_size: Tuple[int, int] = (32, 32),
                 adversarial_generation: bool = False,):
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self._filepath = Path(root)
        self.train = train
        self.image_size = image_size
        self.adversarial_generation = adversarial_generation

        self.available_datasets = {
            'CIFAR10': datasets.CIFAR10,
            'PaintingStyle': datasets.ImageFolder,
            'Architecture': datasets.ImageFolder,
            'Intel': datasets.ImageFolder,
            # Assuming the dataset is organized in folders by class
            # You can add custom dataset handlers here if needed
        }
        self._set_transforms()

    def _set_transforms(self):
        resize_and_crop = transforms.Resize(self.image_size)
        basic_transforms = [ transforms.ToTensor() ]
        if self.normalize:
            basic_transforms.append(transforms.Normalize(self.mean, self.std))

        self.original_transform = transforms.Compose([resize_and_crop,*basic_transforms])
        #photogrametry transforms
        random_color = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.random_photogrametry_transform = transforms.Compose([*basic_transforms,random_color])
        #rotation transform
        self.rotate_transform = transforms.Compose([*basic_transforms,transforms.RandomRotation(degrees=(0,180))])
        

    def _load(self) -> Any:
        if self.dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")

        dataset_class = self.available_datasets[self.dataset_name]

        if self.available_datasets[self.dataset_name]==datasets.ImageFolder:

            Distort_dataset = dataset_class(root=str(self._filepath),
                                                         transform= self.random_photogrametry_transform)
                                
            Original_dataset = dataset_class(root=str(self._filepath),
                                                transform = self.original_transform)
            Rotated_dataset = dataset_class(root=str(self._filepath),
                                                transform = self.rotate_transform)
            
            if self.train:
                dataset = ConcatDataset([Original_dataset,Rotated_dataset,Distort_dataset])

            else:
                if self.adversarial_generation:
                    dataset = ConcatDataset([Original_dataset])
                else:
                    dataset = ConcatDataset([Original_dataset,Distort_dataset])
                
                
        if self.available_datasets[self.dataset_name]==datasets.CIFAR10:
            Distort_dataset = dataset_class(root=str(self._filepath),
                                                            train=self.train,
                                                            download=False,
                                                            transform = self.random_photogrametry_transform
                                                            )
            Original_dataset = dataset_class(root=str(self._filepath),
                                                train=self.train,
                                                download=False,
                                                transform=self.original_transform
                                                )
            Rotated_dataset = dataset_class(root=str(self._filepath),
                                                train=self.train,
                                                download=False,
                                                transform=self.rotate_transform
                                                )
            if self.train:
                dataset  = ConcatDataset([Original_dataset,Rotated_dataset,Distort_dataset])
            else:
                if self.adversarial_generation:
                    dataset = ConcatDataset([Original_dataset])
                else:
                    dataset = ConcatDataset([Original_dataset,Distort_dataset]) 
        else:
            pass

        return dataset

    def _save(self, data: Any) -> None:
        raise NotImplementedError("Saving datasets is not supported.")

    def _describe(self) -> Dict[str, Any]:
        return dict(
            dataset_name=self.dataset_name,
            root=self._filepath,
            train=self.train,
            original_transform = self.original_transform,
            random_photogrametry_transform = self.random_photogrametry_transform,
            rotate_transform = self.rotate_transform,                        
            normalize = self.normalize,
            mean = self.mean,
            std = self.std,
            image_size = self.image_size,
            adversarial_generation = self.adversarial_generation
        )
    def  _exists(self) -> bool:
        return self._filepath.exists()




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
    def  _exists(self) -> bool:
        return self._filepath.exists()
        
    
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
    def  _exists(self) -> bool:
        return self._filepath.exists()
