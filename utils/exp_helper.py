import os
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
import random

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    """Reconstructs a nested dictionary from a flattened one."""
    result = {}
    for k, v in d.items():
        parts = k.split(sep)
        d_ref = result
        for part in parts[:-1]:
            d_ref = d_ref.setdefault(part, {})
        d_ref[parts[-1]] = v
    return result

def parse_args_from_config(config):
    flat_config = flatten_dict(config)
    parser = argparse.ArgumentParser()

    for key, val in flat_config.items():
        arg_type = type(val) if val is not None else str

        if isinstance(val, bool):
            parser.add_argument(f'--{key}', type=str2bool, default=val)
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=val)

    args = parser.parse_args()
    flat_agrs = vars(args)
    return unflatten_dict(flat_agrs)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)