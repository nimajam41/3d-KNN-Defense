import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from .augmentation import *


def split_name(f):
    splitted_arr = f.split("_")
    return int(splitted_arr[0]), int(splitted_arr[1]), int(splitted_arr[2])


def evaluate(model, loader, targeted=False):
    correct, total = 0, 0
    model.eval()

    for idx, data in enumerate(loader):
        if targeted:
            pc, vic, _ = data
        else:
            pc, vic = data
        if torch.cuda.is_available():
            pc, vic = pc.cuda(), vic.cuda()

        outp = model(pc.transpose(1, 2).float())
        if isinstance(outp, tuple):
            outp = outp[0]

        __, preds = torch.max(outp.data, 1)
        correct += (preds == vic).sum()
        total += pc.shape[0]

    return correct / total
    
    
def pointnet_loss_function(output, true, rot64, device, alpha=0.001):
    criterion = nn.CrossEntropyLoss()

    batch_size = output.shape[0]

    i64 = torch.eye(64, requires_grad=True, device=device).repeat(batch_size, 1, 1)

    mat64 = torch.bmm(rot64, rot64.transpose(1, 2))

    dif64 = nn.MSELoss(reduction='sum')(mat64, i64) / batch_size

    loss1 = criterion(output, true)
    loss2 = dif64
    loss = loss1 + alpha * loss2

    return loss, loss1, loss2


def convert_to_numpy(path):
    sorted_files = sorted(os.listdir(path))

    pointclouds = np.zeros((2250, 1024, 3))
    reals = np.zeros(2250)
    targets = np.zeros(2250)

    for i in range(len(sorted_files)):
        f = sorted_files[i]
        victim, target, batch_num = split_name(f)
        pointclouds_batch = np.load(path + f)

        pointclouds[5 * i: 5 * (i + 1)] = pointclouds_batch
        reals[5 * i: 5 * (i + 1)] = victim
        targets[5 * i: 5 * (i + 1)] = target

    np_path = path + "numpy/"
    os.mkdir(np_path)

    np.save(np_path + 'samples', pointclouds)
    np.save(np_path + 'victim_labels', reals)
    np.save(np_path + 'target_labels', targets)


def fast_hausdorff_distance(x, xa):
    """one-sided Hausdorff distance from xa to x"""
    adv , orig = xa[:, None], x[:, None]
    dists = ((adv ** 2).reshape(-1, 1) + (orig ** 2).reshape(1, -1) - \
      2 * adv @ orig.T)
    
    dists = np.maximum(dists, 0)
    dists = np.sqrt(dists)
    
    return np.max(np.min(dists, 1)[0])[0]


def fast_chamfer_distance(x, xa):
    """one-sided Chamfer distance from xa to x"""
    adv , orig = xa[:, None], x[:, None]
    dists = ((adv ** 2).reshape(-1, 1) + (orig ** 2).reshape(1, -1) - \
      2 * adv @ orig.T)
    
    dists = np.maximum(dists, 0)
    dists = np.sqrt(dists)

    return np.mean(np.min(dists, 1)[0])


def euclidean_distance(x, xa):
    xa_sq = (xa ** 2).sum(1)[:, None]  
    x_sq = (x ** 2).sum(1)[None, :] 
    cross_term = xa @ x.T 

    dists = (xa_sq - 2 * cross_term + x_sq)
    dists = np.maximum(dists, 0)

    return np.sqrt(dists)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1. - eps) + (1. - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()

    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


data_transforms = {
    "test": {
        "pointnet": transforms.Compose(
            [
                Normalize(),
                ToTensor()
            ]
        ),
        "pointnet2": transforms.Compose(
            [
                Normalize(),
                ToTensor()
            ]
        ),
        "dgcnn": transforms.Compose(
            [
                ToTensor()
            ]
        ),
        "pct": transforms.Compose(
            [
                ToTensor()
            ]
        )
    },

    "train": {
        "pointnet": transforms.Compose(
            [
                Normalize(),
                RandomNoise(),
                RandomRotate(),
                ToTensor()
            ]
        ),
        "pointnet2": transforms.Compose(
            [
                Normalize(),
                RandomRotate(),
                RandomRotatePerturbation(),
                RandomScale(),
                RandomShift(),
                RandomNoise(),
                ToTensor()
            ]
        ),
        "dgcnn": transforms.Compose(
            [
                Translate(),
                Shuffle(),
                ToTensor()
            ]
        ),
        "pct": transforms.Compose(
            [
                RandomDropPoint(),
                Translate(),
                Shuffle(),
                ToTensor()
            ]
        )
    }
}

to_tensor_transform = transforms.Compose(
    [
        ToTensor()
    ]
)