from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torch.optim import Adam
from torcheval.metrics.functional import multiclass_f1_score


class FruitDetector(pl.LightningModule):
    
    def __init__(
        self,
        num_classes: int,
        network: nn.Module,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._network = network
        self._loss = nn.CrossEntropyLoss()
    
    @property
    def network(self):
        return self._network
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fc(*args, **kwargs)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return Adam(filter(lambda p: p.requires_grad, self.parameters()))
    
    def _step(self, batch):
        input_image, label = batch
        pred = self._network(input_image)
        loss = self._loss(pred, nn.functional.one_hot(label, num_classes=self._num_classes).to(torch.float16))
        f1 = multiclass_f1_score(pred, label, average="macro", num_classes=self._num_classes)
        
        return dict(loss=loss, f1=f1)
    
    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch)
    
    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch)
    
    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch)
