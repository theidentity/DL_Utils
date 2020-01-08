import numpy as np
import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
from joblib import delayed, Parallel
from sklearn import metrics
import pickle
import contextlib
import time


# ----------------- Utility Fns -----------------
def unique(arr):
    items, counts = np.unique(arr, return_counts=True)
    for item, count in zip(items, counts):
        print('%s - %d' % (item, count))
    print('%d unique in %d items' % (len(items), len(arr)))


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def run_parallel(n_jobs, fn, params):
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*x) for x in tqdm(params))


# ----------------- PyTorch Layers -----------------
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = (-1,) + tuple(shape)

    def forward(self, x):
        return x.view(self.shape)


# ----------------- Named Average Meter -----------------
class NamedAverageMeter():
    def __init__(self, meter_names):
        self.meter_names = meter_names
        self.reset()

    def reset(self):
        self.meters = {}
        total = count = 0
        for name in self.meter_names:
            self.meters[name] = [total, count]

    def update(self, update_dict):
        for name in self.meter_names:
            total, count = update_dict[name]
            if isinstance(total, torch.Tensor):
                total = total.item()
            self.meters[name][0] += total
            self.meters[name][1] += count

    def avg(self):
        avgs = {}
        for name in self.meter_names:
            total, count = self.meters[name]
            if count != 0:
                avg = total / count
            else:
                avg = 0
            avgs[name] = avg
        return avgs

# ----------------- Metric Tracker -----------------


class MetricTracker(object):
    def __init__(self, metrics_to_track, auroc_class=None):
        super(MetricTracker, self).__init__()
        for item in metrics_to_track:
            assert item in ('acc', 'f1', 'cm', 'auroc', 'ap')
            if item in ('auroc', 'ap'):
                assert auroc_class is not None

        self.metrics_to_track = metrics_to_track
        self.auroc_class = auroc_class
        self.reset()

    def reset(self):
        self.y_scores = None
        self.y_true = None
        self.y_pred = None

    def update(self, model_op, ground_truth):
        if self.y_scores is None:
            self.y_scores = model_op
            self.y_true = ground_truth
        else:
            self.y_scores = torch.cat([self.y_scores, model_op])
            self.y_true = torch.cat([self.y_true, ground_truth])

    def get_metrics(self):
        if self.y_scores.is_cuda:
            y_scores = self.y_scores.cpu().numpy()
            y_true = self.y_true.cpu().numpy()
        else:
            y_scores = self.y_scores.numpy()
            y_true = self.y_true.numpy()

        y_pred = np.argmax(y_scores, axis=1)
        results = {}

        for item in self.metrics_to_track:
            if item == 'acc':
                results[item] = metrics.accuracy_score(y_true, y_pred)
            elif item == 'cm':
                results[item] = metrics.confusion_matrix(y_true, y_pred)
            elif item == 'f1':
                results[item] = metrics.f1_score(y_true, y_pred, average='macro')
            elif item == 'auroc':
                results[item] = metrics.roc_auc_score(y_true, y_scores[:, self.auroc_class])
            elif item == 'ap':
                results[item] = metrics.average_precision_score(y_true, y_scores[:, self.auroc_class])

        return results


@contextlib.contextmanager
def disable_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


@contextlib.contextmanager
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    np.random.seed()
    seed = np.random.randint(0, 1e6)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class EMA_Averager():
    def __init__(self, model, ema_decay=.999):
        self.model = model
        self.ema_decay = ema_decay
        self.ema_model = self.get_ema_model()

    def step(self):
        alpha = self.ema_decay
        one_minus_alpha = 1 - alpha
        for src, dst in zip(self.model.parameters(), self.ema_model.parameters()):
            dst.mul_(alpha)
            dst.add_(one_minus_alpha*src.data)

    def update_bn(self):
        tmp_model = pickle.loads(pickle.dumps(self.ema_model))
        self.ema_model.load_state_dict(self.model.state_dict())
        for src, dst in zip(tmp_model.parameters(), self.ema_model.parameters()):
            dst[:] = src[:]

    def get_ema_model(self):
        ema_model = pickle.loads(pickle.dumps(self.model))
        for src in ema_model.parameters():
            src.detach_()
        return ema_model

def make_inf_dl(dl):
    while True:
        try:
            data_iter = iter(dl)
            yield next(data_iter)
        except StopIteration:
            del(data_iter)
            data_iter = iter(dl)

def accuracy(op, y):
    pred = torch.argmax(op, dim=1)
    num_correct = torch.sum(pred == y)
    return num_correct            