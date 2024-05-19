import json
import os.path
import sys
from datetime import datetime

import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import ResNet18_Weights

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from data import FruitsDatamodule
from training import FruitDetector


def get_network(num_classes: int):
    resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    for param in resnet18.parameters():
        param.requires_grad = False
    
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    resnet18.add_module("softmax", nn.Softmax())
    
    return resnet18


def train(data_dir: str, mapping_path: str, num_classes: int, batch_size: int, num_workers: int, num_epochs: int):
    print("Creating the network")
    network = get_network(num_classes=num_classes)
    print("Creating the datamodule")
    datamodule = FruitsDatamodule(
        batch_size=batch_size, num_workers=num_workers,
        data_dir=data_dir, mapping_path=mapping_path
    )
    print("Creating the algorithm")
    algorithm = FruitDetector(num_classes=num_classes, network=network)
    print("Creating model checkpoint callback")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=f"../checkpoints/{now}",
        filename="epoch_{epoch:02d}",
        auto_insert_metric_name=False
    )
    print("Creating the logger")
    logger = TensorBoardLogger(f"../logs/{now}")
    print("Creating the progress bar")
    progress_bar = RichProgressBar()
    print("Creating the trainer")
    trainer = Trainer(
        max_epochs=num_epochs,
        callbacks=[model_checkpoint, progress_bar],
        log_every_n_steps=1,
        logger=logger
    )
    print("Starting the training")
    trainer.fit(algorithm, datamodule=datamodule)
    print("Testing")
    trainer.test(dataloaders=datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"), "r") as config:
        train_config = json.load(config)["train_config"]
        train(**train_config)
