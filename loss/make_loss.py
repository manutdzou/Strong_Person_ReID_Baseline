# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .imptriplet_loss import ImpTripletLoss
from .retriplet_loss import ReTripletLoss

def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    imptriplet = ImpTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    retriplet = ReTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    elif sampler == 'softmax_imptriplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + imptriplet(feat, target)[0]
    elif sampler == 'softmax_retriplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + retriplet(feat, target)[0]
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_imptriplet, '
              'but got {}'.format(sampler))
    return loss_func
