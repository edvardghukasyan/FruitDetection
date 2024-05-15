import click
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import Trainer

from algorithm import FruitDetector
from datamodule import FruitsDatamodule


def get_network(num_classes: int):
    resnet50 = models.resnet50(pretrained=True)
    
    for param in resnet50.parameters():
        param.requires_grad = False
    
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    
    return resnet50


@click.command()
@click.option("--data_dir", default="../fruits360_processed")
@click.option("--num_classes", default=7)
@click.option("--batch_size", default=8)
@click.option("--num_workers", default=8)
@click.option("--num_epochs", default=5)
@click.option("--precision", default=16)
def train(data_dir: str, num_classes: int, batch_size: int, num_workers: int, num_epochs: int, precision: int):
    network = get_network(num_classes=num_classes)
    datamodule = FruitsDatamodule(batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    algorithm = FruitDetector(num_classes=num_classes, network=network)
    trainer = Trainer(max_epochs=num_epochs, precision=precision)
    trainer.fit(algorithm, datamodule=datamodule)


if __name__ == "__main__":
    train()
