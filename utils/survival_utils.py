import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.interpolate import interp1d
from pycox.evaluation import EvalSurv

def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError
    # input = input.sum(1, keepdim=True) - pad_col(input, where='start').cumsum(1)
    input = input.sum(1, keepdim=True) - input.cumsum(1)
    return input


def get_survival_curves(pred, method='DeepHit'):
    '''
    From: https://github.com/havakv/pycox/blob/master/pycox/models/pmf.py
    Get survival curve for batch of predictions
    '''

    if method == 'MTLR':
        pred = cumsum_reverse(pred, dim=1)    

    if method in ['DeepHit', 'PMF', 'MTLR', 'SurvRNC', 'RPS']:
        pmf = nn.functional.softmax(pred, dim=1)


    else: raise ValueError(f'Expected param method to be one of [PMF, MTLR, DeepHit, distance], got {method}')

    # Cumsum and inverse probs
    return 1 - torch.cumsum(pmf, dim=1)


def get_metrics(predictions, time_range, times, events, method='DeepHit'):
    survival_curves = get_survival_curves(predictions, method=method)
    ev = EvalSurv(pd.DataFrame(survival_curves.T, time_range), np.array(times), np.array(events), censor_surv='km')
    return ev.concordance_td('antolini')