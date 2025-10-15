import os
import torch
import argparse
import torchvision.transforms as T
import numpy as np
import random
import csv
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pycox.evaluation import EvalSurv
from torch import nn
from torch.utils.data import DataLoader

from utils.exp_helper import set_seed, load_config
from data import load_dataset
from train_test import test_step
from network import load_pretrained_model


if __name__ == '__main__':


    
    exp_path = '/home/WVU-AD/jdt0025/Documents/exp/METABRIC_DeepHit_Adam0.0001'
    cfg = load_config(os.path.join(exp_path, 'config.yaml'))

    set_seed(cfg['seed'])

    batch_size = 32

    




    ########################
    # Load data
    root = cfg['exp_details']['data_root']
    train_surv, valid_surv, test_surv, time_steps = load_dataset(root, cfg['exp_details']['dataset'])


    train_dataloader  = DataLoader(train_surv, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader   = DataLoader(test_surv, batch_size=batch_size, shuffle=False)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = cfg['primary_device']
    print(f'\nUsing {device} device\n')



    if 'train_params' in cfg.keys(): method = cfg['train_params']['loss_fn']
    else: method = cfg['paradigm']

    model = load_pretrained_model(os.path.join(exp_path, 'best_model_C.pt'), 
                                            out_dim=len(time_steps),
                                            cfg=cfg)
    model.eval()
    model.to(device)


    _, C = test_step(model, test_dataloader, loss_fn=None, device=device, time_step=time_steps)
    print(C)


    


