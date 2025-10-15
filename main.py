import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm

from data import load_dataset
from network import Network, MRIEncoder
from loss import *
from train_test import *
from utils.exp_helper import parse_args_from_config, load_config, set_seed



cfg = load_config('defaults.yaml')
cfg = parse_args_from_config(cfg)

# Load params from config
set_seed(cfg['seed'])
root = cfg['exp_details']['data_root']
dataset = cfg['exp_details']['dataset']
device = cfg['primary_device']
batch_size = cfg['train_params']['batch_size']
optim = cfg['optimizer']['optim']
lr = cfg['optimizer']['lr']
paradigm = cfg['train_params']['loss_fn']

exp_id = f'{dataset}_{paradigm}_{optim}{lr}'
out_dir = os.path.join(cfg['exp_details']['out_dir'], exp_id)
if os.path.exists(out_dir):
    if not cfg['exp_details']['exist_ok']: raise OSError("Experiment exists!")
else: os.mkdir(out_dir)

with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
    yaml.dump(cfg, f)


# Load data
train_surv_ID, valid_surv_ID, test_surv_ID, time_steps = load_dataset(root, dataset)

train_dataloader   = DataLoader(train_surv_ID, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader   = DataLoader(valid_surv_ID, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader    = DataLoader(test_surv_ID, batch_size=batch_size, shuffle=False, drop_last=True)


# Load model
if dataset == 'ADNI': 
    model = MRIEncoder(in_channel=1, feat_dim=1024, out_dim=len(time_steps))
else: 
    model = Network(len(time_steps), in_dim=cfg['model']['in_dim'], hid_dim=cfg['model']['mlp_hid_dim'], backbone=cfg['model']['backbone'])

pretrained_path = cfg['model']['pretrained_path']
if pretrained_path:
    print(f"\nLoading pretrained model at {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path))

model.to(device)


# Load optim
optimizer = cfg['optimizer']['optim']
if optimizer == 'Adam':
    optim = Adam(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
elif optimizer == 'SGD':
    optim = SGD(model.parameters(), lr=cfg['optimizer']['lr'], momentum=cfg['optimizer']['momentum'])
else: raise NotImplementedError(f'Optimizer {optimizer} not implemented')


# Configure loss
paradigm = cfg['train_params']['loss_fn']
if paradigm == 'NLL': loss_fn = NLLLoss(reduction=cfg['train_params']['reduction'], device=device)
elif paradigm == 'DeepHit': loss_fn = DeepHitLoss(weight=cfg['train_params']['lambda'], device=device)


# Train
train_losses = []
test_losses = []
Cs = []
best_C = 0
best_loss = 1e10
for epoch in tqdm(range(cfg['train_params']['epochs'])):

    train_loss = train_step(model, train_dataloader, optim, loss_fn, device)
    valid_loss, C = test_step(model, valid_dataloader, loss_fn, device, time_step=time_steps, method=paradigm)

    if np.isnan(train_loss): 
        with open(os.path.join(out_dir, 'error.txt'), 'w') as f:
            f.write(f'Got nan loss!')
        raise ValueError('Got nan loss!')
    

    if C > best_C:
        best_C = C
        with open(os.path.join(out_dir, 'best_model_C.txt'), 'w') as f:
            f.write(f'Epoch: {epoch}')
        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_C.pt'))
    if valid_loss < best_loss:
        best_loss = valid_loss
        with open(os.path.join(out_dir, 'best_model_loss.txt'), 'w') as f:
            f.write(f'Epoch: {epoch}')
        torch.save(model.state_dict(), os.path.join(out_dir, 'best_model_loss.pt'))

    
    if (epoch + 1) % cfg['train_params']['checkpoint'] == 0:
        p = epoch + 1
        torch.save(model.state_dict(), os.path.join(out_dir, f'e{p}.pt'))

    train_losses.append(train_loss)
    test_losses.append(valid_loss)
    Cs.append(C)

    # Save Figures
    # Loss
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.title('Loss')
    plt.grid(False)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    # Metrics
    plt.figure(figsize=(10, 7))
    plt.plot(Cs, label=f'C-td ({train_surv_ID.name})')
    plt.title('Concordance index over time')
    plt.grid(False)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'C.png'))
    plt.close()

