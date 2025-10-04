import lightning.pytorch as pl
import torch.nn.functional as F
from torch import optim
import torch
import torch.nn as nn
import json
from datetime import datetime
import pandas as pd
from vit_pytorch.cct_3d import CCT
from vit_pytorch.simple_vit_3d import SimpleViT
from vit_pytorch.vit_3d import ViT
from vit_pytorch.cvt_3d import CvT
from vit_pytorch.efficient_3d import ViT as EfficientViT
from vit_pytorch.mobile_vit_3d import MobileViT
from vit_pytorch.scalable_vit_3d import ScalableViT3D
from vit_pytorch.parallel_vit_3d import ViT as ParallelViT
from swin_transformer_3d import SwinTransformer3D
from nystrom_attention import Nystromformer

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    ConfusionMatrix
)


class LightningTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        self.model = self._get_model()
        self.example_input_array = torch.rand(self.hparams['input_dim'])
        self.num_classes = self.hparams['num_classes']
        self.loss = nn.CrossEntropyLoss() if self.num_classes > 1 else nn.BCEWithLogitsLoss()
        self.task = 'multiclass' if self.num_classes > 1 else 'binary'
        
        # metrics
        self.test_preds = []
        self.test_labels = []
        self.train_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.val_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.test_acc = Accuracy(task=self.task, num_classes=self.num_classes)
        self.precision = Precision(task=self.task, num_classes=self.num_classes)
        self.recall = Recall(task=self.task, num_classes=self.num_classes)
        self.f1_score = F1Score(task=self.task, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.task, num_classes=self.num_classes)
        self.confmat = ConfusionMatrix(task=self.task, num_classes=self.num_classes)
        
    def _get_model(self):
        model_name = self.hparams['model_name']
        
        with open(f'./config/{model_name}_config.json') as json_file:
            config = json.load(json_file)
        
        if model_name == 'swint_3d':
            return SwinTransformer3D(**config)
        elif model_name == 'cct_3d':
            return CCT(**config)
        elif model_name == 'mobile_vit_3d':
            return MobileViT(**config)
        elif model_name == 'vit_3d':
            return ViT(**config)
        elif model_name == 'simple_vit_3d':
            return SimpleViT(**config)
        elif model_name == 'scalable_vit_3d':
            return ScalableViT3D(**config)
        elif model_name == 'parallel_vit_3d':
            return ParallelViT(**config)
        elif model_name == 'cvt_3d':
            return CvT(**config)
#         elif model_name == 'efficient_vit_3d'
#             efficient_transformer = Nystromformer(
#                 dim = 512,
#                 depth = 12,
#                 heads = 8,
#                 num_landmarks = 256
#             )
#             return EfficientViT(
#                 **config,
#                 transformer=efficient_transformer
#             )
        
    def forward(self, x):
        return self.model(x)
        
        return [optimizer], [self.lr_scheduler]
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['min_lr'], weight_decay=self.hparams['weight_decay'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['T_max'])
#         self.lr_scheduler = CosineAnnealingWarmupRestarts(
#             optimizer,
#             first_cycle_steps=100,
#             cycle_mult=1.0,
#             max_lr=self.hparams['max_lr'],
#             min_lr=self.hparams['min_lr'],
#             warmup_steps=50,
#             gamma=self.hparams['gamma']
#         )
        return [optimizer], [self.lr_scheduler]
        
    def _calculate_loss(self, batch):
        seq_features, labels = batch
        preds = self(seq_features)
        loss = self.loss(preds, labels)

        return preds, labels, loss

    def training_step(self, batch, batch_idx):
        preds, labels, loss = self._calculate_loss(batch)
        
        self.train_acc(preds, labels)
        
        metrics = {
            'train_loss': loss,
            'train_acc': self.train_acc
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.lr_scheduler.get_lr()[0] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._calculate_loss(batch)

        self.val_acc(preds, labels)
        
        metrics = {
            'val_loss': loss,
            'val_acc': self.val_acc
        }
        
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) 
        

    def test_step(self, batch, batch_idx):
        seq_features, labels = batch
        preds = self(seq_features)
        
        # compute metrics
        self.test_acc(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1_score(preds, labels)
        self.auroc(preds, labels)
        
        metrics = {
            'accuracy': self.test_acc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auroc': self.auroc
        }
        
        # log metrics
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        
        
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)

        self._save_confusion_matrix(all_preds, all_labels)
        
        
        results = {
            'accuracy': self.test_acc(all_preds, all_labels),
            'precision': self.precision(all_preds, all_labels),
            'recall': self.recall(all_preds, all_labels),
            'f1_score': self.f1_score(all_preds, all_labels),
            'auroc': self.auroc(all_preds, all_labels)
        }
        
        self.log_dict(results, sync_dist=True)

        # Free memory
        self.test_preds.clear()
        self.test_labels.clear()
        
        
    def _save_confusion_matrix(self, preds, labels):
        root_path = self.hparams['root_path']
        model_name = self.hparams['model_name']
        edges = self.hparams['edges']
        confmat = self.confmat(preds, labels).detach().cpu().numpy().astype(int)
        print(f'Confusion matrix:\n{confmat}')

        df_cm = pd.DataFrame(confmat)
        df_filename = f'{root_path}/logs/{model_name}/{edges}/confusion_matrices/confmat-{datetime.now()}.csv'
        df_cm.to_csv(df_filename, index=False)
        

