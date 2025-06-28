## Trainer class for Multi-GPU training
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.pipeline.sync import Pipe
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR
from torchinfo import summary
from models import SwinTransformer3D
from torchmetrics import Accuracy
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from vit_pytorch.cct_3d import CCT
from pynvml import *

import os
import gc
import sys
import json
import random
import math
from glob import glob
import SimpleITK as sitk
import numpy as np
from datetime import datetime

classes_dict = {
    'PD_Only' : 0,
    'apathy' : 1,
    'ICD' : 2,
    'apathy+ICD' : 3,
}

base_data_path = '/home/carlos.jimenez/data'

seed = 1337
model_name = 'CCT'

## HYPERPARAMETERS
HYPERPARAMETERS = {
    'num_classes' : 4,
    'in_channels' : 1,
    'n_output_channels' : 64,
    # input_dim = (197,233,189) # (w, h, d)
    'batch_size' : 8,
    'epochs' : 200,
    'gamma' : 0.7,
    'lr' : 1e-4,
    'weight_decay' : 1e-3,
    'window_size' : 3,
    'save_every' : 1,
}


class MRIDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img)
        img = torch.from_numpy(img)
        img = np.expand_dims(img, axis=0)
        label_name = img_path.split('/')[4]
        label = classes_dict[label_name]

        return img, label



def print_gpu_utilization(gpu_id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"[GPU{gpu_id}] memory occupied: {info.used//1024**2} MB."

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "4445"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    
    
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
if model_name == 'SwinT3D':
    log_file = f"models/{model_name}/logs/batch_{HYPERPARAMETERS['batch_size']}_epochs_{HYPERPARAMETERS['epochs']}_swin_{swintr_3d_params['window_size']}_patch_{swintr_3d_params['patch_size']}_{timestamp}.txt"
elif model_name == 'CCT':
    log_file = f"models/{model_name}/logs/batch_{HYPERPARAMETERS['batch_size']}_epochs_{HYPERPARAMETERS['epochs']}_convlayers_{conv_layers}_layers_{layers}_heads_{heads}_{timestamp}.txt"

def write_to_logfile(text, stdout=True):
    with open(log_file, 'a') as f:
        if stdout:
            print(text)
        f.write(text+'\n')

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.accuracy = Accuracy(task="multiclass", num_classes=HYPERPARAMETERS['num_classes']).to(gpu_id)
        self.acc_batch = 0.
        self.loss_batch = 0.
        self.val_acc_batch = 0.
        self.val_loss_batch = 0.

    def _train_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        acc = self.accuracy(output, targets).detach().cpu().numpy()
        
        self.acc_batch += acc
        self.loss_batch += loss.item()
        
        # memory efficient
        del source, targets, output
        gc.collect()
        torch.cuda.empty_cache()
        
    
    def _train_dataset(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        write_to_logfile(f"[GPU{self.gpu_id}] Epoch {epoch+1} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        
        print('before train loop...')
        print(print_gpu_utilization(self.gpu_id))
        
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
    
            self._train_batch(source, targets)
        
    def _eval_batch(self, source, targets):
        with torch.no_grad():
            output = self.model(source)
            val_loss = F.cross_entropy(output, targets)
            
        val_acc = self.accuracy(output, targets).detach().cpu().numpy()
    
        self.val_acc_batch += val_acc
        self.val_loss_batch += val_loss.item()
        
        # memory efficient
        del source, targets, output
        gc.collect()
        torch.cuda.empty_cache()
    
    
    def _eval_dataset(self, epoch):    
        self.test_data.sampler.set_epoch(epoch)
        
        for source, targets in self.test_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            self._eval_batch(source, targets)
    
    
    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        PATH = f"models/{model_name}/checkpoints/runs/run_{sys.argv[1]}/checkpoint_epoch_{epoch+1}.pt"
        torch.save(state, PATH)
        write_to_logfile(f"Epoch {epoch+1} | Training checkpoint saved at {PATH}")

    def fit(self, max_epochs: int):
        for epoch in range(max_epochs):    
            # training
            torch.cuda.empty_cache()
            initial_lr = self.lr_scheduler.get_lr()[0]
#             if epoch >= 2:
#                 self.lr_scheduler.step()
            self.model.train()
            self._train_dataset(epoch)
            print(print_gpu_utilization(self.gpu_id))
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                final_lr = self.lr_scheduler.get_lr()[0]
                self.acc_batch /= len(self.train_data)
                self.loss_batch /= len(self.train_data)
                write_to_logfile(f'-> [TRAIN] Epoch: {epoch+1} loss: {self.loss_batch:.4f} acc: {self.acc_batch:.4f} lr_i:{initial_lr:.10f} lr_f: {final_lr:.10f}')
            
            
            # evaluation
            self.model.eval()
            self._eval_dataset(epoch)
            print(print_gpu_utilization(self.gpu_id))
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self.val_acc_batch /= len(self.test_data)
                self.val_loss_batch /= len(self.test_data)
                write_to_logfile(f'-> [EVAL] Epoch: {epoch+1} val_loss: {self.val_loss_batch:.4f} val_cc: {self.val_acc_batch:.4f}')
                self._save_checkpoint(epoch)
                
            self.acc_batch = 0.
            self.loss_batch = 0.
            self.val_acc_batch = 0.
            self.val_loss_batch = 0.


# Model instantiation and parameters
def load_train_objs(train_list: list, in_channels: int, num_classes: int):
    train_transforms = transforms.Compose([
        transforms.Resize((224,224))
    ])
    train_set = MRIDataset(train_list, transform=train_transforms)  # load your dataset
    
    model = CCT(
        img_size = image_size,          # image size
        num_frames = frames,
        embedding_dim = dim,
        n_conv_layers = conv_layers,
        frame_kernel_size = kernel_size,
        kernel_size = kernel_size+4,
        stride = 2,
        padding = 0,
        pooling_kernel_size = kernel_size,
        pooling_stride = 2,
        pooling_padding = 0,
        num_layers = layers,
        num_heads = heads,
        mlp_radio = 1.,
        num_classes = num_classes,
        n_input_channels = 1,
        is_compact=True,
        positional_embedding = 'learnable'
    )
#     PATH = 'models/CCT/checkpoints/runs/run_1/checkpoint_epoch_20.pt'
    # PATH = 'models/CCT/checkpoints/runs/run_8/checkpoint_epoch_51.pt'
#     checkpoint = torch.load(PATH)
#     print('Loading weights...')
#     model.load_state_dict(checkpoint)
#     model = SwinTransformer3D(**swintr_3d_params)
    
#     model = MeshNet(n_channels=1, n_classes=HYPERPARAMETERS['num_classes'], large=False, dropout_p=0.1,dim=32//4)
#     model = CNNResnet(1, ResBlock, outputs=HYPERPARAMETERS['num_classes'])
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERPARAMETERS['lr'], weight_decay=HYPERPARAMETERS['weight_decay'])

    # scheduler
    # lr_scheduler = StepLR(optimizer, step_size=1, gamma=HYPERPARAMETERS['gamma'])
#     steps = len(self.train_data)

#     lf = lambda x: x+(1e-4)
#     lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=20)
#     lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, T_0=2)
#     lf = lambda x: (((1 + math.cos(x * math.pi / HYPERPARAMETERS['epochs'])) / 2) ** 1.0) * 0.9 + 0.1  # cosine
#     lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
#     lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=150, cycle_mult=1.0, max_lr=0.1, min_lr=1e-4, warmup_steps=50, gamma=0.5)
    
    return train_set, model, optimizer, lr_scheduler

def load_eval_objs(test_list: list):
    train_transforms = transforms.Compose([
        transforms.Resize((224,224))
    ])
    eval_set = MRIDataset(test_list, transform=train_transforms)  # load your dataset

    return eval_set

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def train_test_path_split():
    icd_data_paths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted*edges.nii.gz'))
    icd_data_paths, _ = train_test_split(icd_data_paths, test_size=0.22,random_state=seed)

    apathy_icd_data_paths = sorted(glob(f'{base_data_path}/apathy+ICD/nifty/*brain_extracted*edges.nii.gz'))
    apathy_data_paths = sorted(glob(f'{base_data_path}/apathy/nifty/*brain_extracted*edges.nii.gz'))

    pd_only_data_paths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*edges.nii.gz'))
    pd_only_data_paths, _ = train_test_split(pd_only_data_paths, test_size=0.80,random_state=seed)


    data_paths = icd_data_paths + apathy_icd_data_paths + apathy_data_paths + pd_only_data_paths

#     data_paths = [path for path in data_paths if not ('edges' in path or 'contrast' in path)]
    data_paths = [path for path in data_paths if not ('contrast' in path)]

#     print(f'ICD: {len(icd_data_paths)}')
#     print(f'Apathy+ICD: {len(apathy_icd_data_paths)}')
#     print(f'Apathy only: {len(apathy_data_paths)}')
#     print(f'PD only: {len(pd_only_data_paths)}')
#     print(f'Total: {len(data_paths)}')
#     data_paths = [path for path in data_paths if not ('edges' in path  or 'contrast' in path or 'denoised_brain_extracted_rotated' in path)]

    labels = [path.split('/')[4] for path in data_paths]
    
    train_list, test_list = train_test_split(data_paths, test_size=0.2, shuffle=True,stratify=labels, random_state=seed)
#     print(f'Length train list: {len(train_list)}')
#     print(f'Length test list: {len(test_list)}')
    return train_list, test_list
    
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_list, test_list = train_test_path_split()
    in_channels = HYPERPARAMETERS['in_channels']
    num_classes = HYPERPARAMETERS['num_classes']

    train_dataset, model, optimizer, lr_scheduler = load_train_objs(train_list, in_channels, num_classes)
    test_dataset = load_eval_objs(test_list)
    
    train_data = prepare_dataloader(train_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)
    
    trainer = Trainer(model, train_data, test_data, optimizer, lr_scheduler, rank, save_every)
    trainer.fit(total_epochs)
    
    destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Run numbers is missing')
        
    seed_everything(seed)
    print('Running...')
    world_size = torch.cuda.device_count()
    print(f'World size: {world_size}')
    epochs = HYPERPARAMETERS['epochs']
    print(f'Total epochs: {epochs}')
    save_every = HYPERPARAMETERS['save_every']
    batch_size = HYPERPARAMETERS['batch_size']
#     print(f'Batch size: {batch_size}')
    print('Memory Usage:')
    
    for i in range(world_size):
        print(print_gpu_utilization(i))
    
    
    
    mp.spawn(main, args=(world_size, save_every, epochs, batch_size), nprocs=world_size)