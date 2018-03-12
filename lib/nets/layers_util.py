# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from lib.roi_pooling.roi_pool import RoIPool

class Conv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False, bias=True):
    super(Conv2d, self).__init__()
    padding = int((kernel_size - 1) / 2) if same_padding else 0
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
    self.relu = nn.ReLU(inplace=True) if relu else None

  def forward(self, x):
    x = self.conv(x)
    if self.bn is not None:
      x = self.bn(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class FPN_ROI_Pooling(nn.Module):
  def __init__(self, pooled_height, pooled_width, feat_strides):
    super(FPN_ROI_Pooling, self).__init__()
    self.roi_pool_p2 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[0])
    self.roi_pool_p3 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[1])
    self.roi_pool_p4 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[2])
    self.roi_pool_p5 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[3])
    self.roi_pool_p6 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[4])

  def forward(self, features, rois):
    feat_list = list()
    if rois[0] is not None:
      feat_p2 = self.roi_pool_p2(features[0], rois[0])
      feat_list.append(feat_p2)
    if rois[1] is not None:
      feat_p3 = self.roi_pool_p3(features[1], rois[1])
      feat_list.append(feat_p3)
    if rois[2] is not None:
      feat_p4 = self.roi_pool_p4(features[2], rois[2])
      feat_list.append(feat_p4)
    if rois[3] is not None:
      feat_p5 = self.roi_pool_p5(features[3], rois[3])
      feat_list.append(feat_p5)
    if rois[4] is not None:
      feat_p6 = self.roi_pool_p6(features[4], rois[4])
      feat_list.append(feat_p6)

    return torch.cat(feat_list, dim=0)


class FC(nn.Module):
  def __init__(self, in_features, out_features, relu=True):
    super(FC, self).__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU(inplace=True) if relu else None

  def forward(self, x):
    x = self.fc(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, requires_grad=False):
  v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
  if is_cuda:
    v = v.cuda()
  return v


def set_trainable(model, requires_grad):
  for param in model.parameters():
    param.requires_grad = requires_grad

def set_BN_eval(model, eval_if=True):
   for mod in model.modules():
     if isinstance(mod, nn.BatchNorm2d):
       if eval_if:
        set_trainable(mod, False)
        mod.eval()

def flip(x, dim):
  dim = x.dim() + dim if dim < 0 else dim
  inds = tuple(slice(None, None) if i != dim
               else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
               for i in range(x.dim()))
  return x[inds]

