import torch
import numpy as np
from utils.survival_utils import get_metrics
import torchvision.transforms as T
import torchio.transforms as tio

def train_step(model, dataloader, optim, loss_fn, device='cuda'):
  model.train()

  running_loss = 0

  for X, t, e, y in dataloader:
    X = X.to(device)
    e = e.to(device)
    t = t.type(torch.int64).to(device)

    optim.zero_grad()

    h, _ = model(X)


    loss = loss_fn(h, t, e)
    running_loss += loss.item()
    loss.backward()
    optim.step()


  return running_loss / len(dataloader)

def test_step(model, dataloader, loss_fn, device, time_step=None, method='DeepHit'):
    model.eval()

    running_loss = 0
    times = []
    events = []
    predictions = []
    for X, t, e, _ in dataloader:
        times += t.tolist()
        events += [bool(x) for x in e]

        X = X.to(device)
        e = e.to(device)
        t = t.type(torch.int64).to(device)

        h, _ = model(X)

        if loss_fn:
            loss = loss_fn(h, t, e)
            running_loss += loss.item()

        predictions += h.cpu().tolist()

    C = get_metrics(torch.tensor(predictions), time_step, times, events, method=method)
    return running_loss / len(dataloader), C


