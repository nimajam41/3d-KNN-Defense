# credits: https://github.com/Wuziyi616/IF-Defense/tree/main/baselines/attack/util

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsAdvLoss(nn.Module):
    def __init__(self, kappa=0.):
        super(LogitsAdvLoss, self).__init__()
        self.kappa = kappa


    def forward(self, logits, targets):
        B, K = logits.shape

        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()

        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        
        loss = torch.clamp(other_logits - real_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        return loss
    

class UntargetedLogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0.):
        super(UntargetedLogitsAdvLoss, self).__init__()
        self.kappa = kappa


    def forward(self, logits, targets):
        B, K = logits.shape

        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()

        one_hot_targets = torch.zeros(B, K).cuda().scatter_(
            1, targets, 1).float()  # to one-hot
        
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max((1. - one_hot_targets) * logits -
                                 one_hot_targets * 10000., dim=1)[0]
        
        loss = torch.clamp(real_logits - other_logits + self.kappa, min=0.)
        return loss.mean()
