import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)

from torch.utils import data
import lightning as L
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.tuner import Tuner
from lightning_transformer import LightningTransformer
from glob import glob
from operator import itemgetter
import SimpleITK as sitk
from datetime import datetime
from time import time
import numpy as np
import json
import torch
import argparse
import sys


classes_dict = {
    'PD_Only' : 0,
    'apathy' : 1,
    'ICD' : 2,
    'apathy+ICD' : 3,
}

root_path = '/home/carlos.jimenez/ai-in-health/ICD-Transformers'
base_data_path = f'{root_path}/data-edges'
k = 5
seed = 1337
batch_size = 2
epochs = 120


## HYPERPARAMETERS
HYPERPARAMETERS = {
    'num_classes' : 4,
    'input_dim': (batch_size,1,189,233,197), # (d, h, w)
    'batch_size' : batch_size,
    'gamma' : 0.7,
    'min_lr' : 1e-4,
    'max_lr' : 1e-3,
    'T_max': 25,
    'weight_decay' : 1e-4,
    'window_size' : 3,
    'root_path': root_path
}


class MRIDataset(data.Dataset):
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
        label_name = img_path.split('/')[-3]
        label = classes_dict[label_name]

        return img, label

def train_model(pl_model, train_loader, val_loader):
    print(f'Use edges: {edges}')
    trainer = L.Trainer(
        default_root_dir=f'{root_path}/logs/{HYPERPARAMETERS["model_name"]}/{edges}',
        accelerator="gpu",
        enable_model_summary=False,
        devices=devices,
#         strategy="ddp_find_unused_parameters_true",
        strategy="ddp",
        num_sanity_val_steps=20,
        max_epochs=epochs,
        fast_dev_run=False,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", verbose=True),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    
    tik = time()
    print(f'Starting training timestamp: {tik}')
    trainer.fit(pl_model, train_loader, val_loader)
    tok = time()
    print(f'Finishing training timestamp: {tok}')
    
    training_time = tok-tik
    return trainer, training_time

def train_test_path_split():
    icd_data_paths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted*edges.nii.gz'))
    icd_data_paths, _ = train_test_split(icd_data_paths, test_size=0.25,random_state=seed)

    apathy_icd_data_paths = sorted(glob(f'{base_data_path}/apathy+ICD/nifty/*brain_extracted*edges.nii.gz'))
    apathy_data_paths = sorted(glob(f'{base_data_path}/apathy/nifty/*brain_extracted*edges.nii.gz'))

    pd_only_data_paths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*edges.nii.gz'))
    pd_only_data_paths, _ = train_test_split(pd_only_data_paths, test_size=0.80,random_state=seed)

    data_paths = icd_data_paths + apathy_icd_data_paths + apathy_data_paths + pd_only_data_paths
    data_paths = [path for path in data_paths if 'contrast' not in path]
    
    test_paths = sorted(glob(f'{base_data_path}/*/nifty/*registered_brain_extracted_edges.nii.gz'))
    test_paths = [path for path in test_paths if 'contrast' not in path]

    if not use_edges:
        test_paths = [f"{path[:-(len('_edges.nii.gz'))]}.nii.gz" for path in test_paths]
        data_paths = [f"{path[:-(len('_edges.nii.gz'))]}.nii.gz" for path in data_paths]

    data_paths = list(set(data_paths) - set(test_paths))
    
    print(f'Edges: {use_edges}')
    print(f'ICD: {len(icd_data_paths)}')
    print(f'Apathy+ICD: {len(apathy_icd_data_paths)}')
    print(f'Apathy only: {len(apathy_data_paths)}')
    print(f'PD only: {len(pd_only_data_paths)}')
    print(f'TRAIN/VAL paths: {len(data_paths)}')
    print(f'TEST paths: {len(test_paths)}')
    
    
    train_labels = [classes_dict[path.split('/')[-3]] for path in data_paths]

    return data_paths, train_labels, test_paths

def kfold_training(k, seed):
    """
        Stratified K-fold cross validation
    """
    splits = StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
    for i, (train_idx, val_idx) in enumerate(splits.split(train_paths, train_labels)):
        HYPERPARAMETERS['kfold'] = i+1
        pl_model = LightningTransformer(model, HYPERPARAMETERS)
        
        train_list = itemgetter(*train_idx)(train_paths)
        val_list = itemgetter(*val_idx)(train_paths)
        
        train_loader = data.DataLoader(MRIDataset(train_list), batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
        val_loader = data.DataLoader(MRIDataset(val_list), batch_size=batch_size, shuffle=False, num_workers=2)
        
        trainer = train_model(pl_model, train_loader, val_loader)
        
def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='ViT MRI Classifier',
        description='ICD classification using Vision Transformers',
        epilog='Text at the bottom of help'
    )
    
    parser.add_argument('--devices', default=torch.cuda.device_count())      # option that takes a value
    parser.add_argument('--model')
    parser.add_argument('--use-edges', action='store_true')  # on/off flag
    
    return parser.parse_args()

def torch_setup():
    torch.set_float32_matmul_precision('high')
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    torch_setup()
    args = parse_arguments()
    global use_edges
    global edges
    global devices
    
    use_edges = args.use_edges
    edges = 'edges' if use_edges else 'no_edges'
    
    HYPERPARAMETERS['model_name'] = args.model
    HYPERPARAMETERS['edges'] = edges
    model_name = args.model
    devices = args.devices
    
    train_paths, train_labels, test_paths = train_test_path_split()
    
    # Training / Validation
    train_list, val_list = train_test_split(train_paths, test_size=0.2, shuffle=True, stratify=train_labels, random_state=seed)
    print(len(train_list), len(val_list),len(test_paths))
    train_loader = data.DataLoader(MRIDataset(train_list), batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    val_loader = data.DataLoader(MRIDataset(val_list), batch_size=batch_size, shuffle=False, num_workers=2)
    
    pl_model = LightningTransformer(HYPERPARAMETERS)
    trainer, training_time = train_model(pl_model, train_loader, val_loader)
    
    torch.distributed.destroy_process_group()
    
    # Testing
    if trainer.is_global_zero:
        print(f'Checkpoint path: {trainer.checkpoint_callback.best_model_path}')
        pl_model = LightningTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        test_loader = data.DataLoader(MRIDataset(test_paths), batch_size=1, shuffle=False, num_workers=4)
        test_trainer = L.Trainer(devices=1, accelerator='gpu', enable_model_summary=False) # Only use 1 gpu for testing
        test_result = test_trainer.test(pl_model, dataloaders=test_loader, verbose=False)
        
        with open(f'./config/{model_name}_config.json') as json_file:
            config = json.load(json_file)

        results = {
            'model_name': model_name,
            'hparams': HYPERPARAMETERS,
            'config': config,
            'metrics' : test_result,
            'training_time': training_time
        }

        print(results['metrics'])

        with open(f'{root_path}/logs/{model_name}/{edges}/history/history-{datetime.now()}.json', 'w') as result_file:
            json.dump(results, result_file)

        print("We're done!")


if __name__ == '__main__':
    main()