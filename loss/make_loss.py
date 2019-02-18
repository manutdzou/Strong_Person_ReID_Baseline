# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .new_triplet_loss import NewTripletLoss


def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    newtriplet = NewTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_newtriplet':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target) + newtriplet(feat, target)[0]
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_new-triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
