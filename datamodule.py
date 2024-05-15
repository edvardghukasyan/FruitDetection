import os

import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class FruitsDatamodule(pl.LightningDataModule):
    
    def __init__(self, batch_size: int, num_workers: int, data_dir: str):
        super().__init__()
        
        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__data_path = data_dir
        
        self._train_set = None
        self._val_set = None
        self._test_set = None
    
    def prepare_data(self) -> None:
        self._train_set = ImageFolder(
            os.path.join(self.__data_path, "Training"),
            transform=torchvision.transforms.ToTensor()
        )
        self._val_set = ImageFolder(
            os.path.join(self.__data_path, "Validation"),
            transform=torchvision.transforms.ToTensor()
        )
        self._test_set = ImageFolder(
            os.path.join(self.__data_path, "Test"),
            transform=torchvision.transforms.ToTensor()
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_set, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_set, batch_size=self.__batch_size, num_workers=self.__num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_set, batch_size=self.__batch_size, num_workers=self.__num_workers)
