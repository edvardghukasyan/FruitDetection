from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score


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
        self._f1 = F1Score(task="multiclass", num_classes=self._num_classes, average="macro")
        self._accuracy = Accuracy(task="multiclass", num_classes=self._num_classes)
        
        self._training_step_outputs = defaultdict(list)
        self._validation_step_outputs = defaultdict(list)
        self._test_step_outputs = defaultdict(list)
    
    @property
    def network(self):
        return self._network
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fc(*args, **kwargs)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return Adam(filter(lambda p: p.requires_grad, self.parameters()))
    
    def _step(self, batch, split: str):
        input_image, label = batch
        pred = self._network(input_image)
        loss = self._loss(pred, nn.functional.one_hot(label, num_classes=self._num_classes).to(torch.float16))
        f1 = self._f1(pred, label)
        accuracy = self._accuracy(pred, label)
        
        self.log(f"loss", loss, prog_bar=True)
        self.log(f"f1", f1, prog_bar=True)
        self.log(f"accuracy", accuracy, prog_bar=True)
        
        return dict(loss=loss, f1=f1, accuracy=accuracy)
    
    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch, split="train")
    
    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch, split="val")
    
    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self._step(batch, split="test")
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        for metric_name, metric_values in outputs.items():
            self._training_step_outputs[metric_name].append(metric_values.detach().cpu())
    
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        for metric_name, metric_values in outputs.items():
            self._validation_step_outputs[metric_name].append(metric_values.detach().cpu())
    
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        for metric_name, metric_values in outputs.items():
            self._test_step_outputs[metric_name].append(metric_values.detach().cpu())
    
    @staticmethod
    def _calculate_epoch_metrics(outputs: dict[str, list]) -> dict:
        epoch_metrics = {}
        
        for metric_name, metric_values in outputs.items():
            epoch_metrics[metric_name] = sum(metric_values) / len(metric_values)
        
        return epoch_metrics
    
    def _epoch_end(self, outputs, split):
        epoch_metrics = self._calculate_epoch_metrics(outputs)
        epoch_metrics = {f'{split}_{k}': v for k, v in epoch_metrics.items()}
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)
        else:
            print(f"""\n{epoch_metrics}\n""")
    
    def on_train_epoch_end(self) -> None:
        self._epoch_end(self._training_step_outputs, split='train')
        self._training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        self._epoch_end(self._training_step_outputs, split='val')
        self._training_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        self._epoch_end(self._training_step_outputs, split='test')
        self._training_step_outputs.clear()
