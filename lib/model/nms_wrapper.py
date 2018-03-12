# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.config import cfg
from lib.nms.pth_nms import pth_nms


def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return pth_nms(dets, thresh)


from lib.nms.nms_test.cpu_nms import cpu_nms
from lib.nms.nms_test.gpu_nms import gpu_nms


def nms_test(dets, thresh, force_cpu=False):
  """Dispatch to either CPU or GPU NMS implementations."""

  if dets.shape[0] == 0:
    return []
  if cfg.USE_GPU_NMS and not force_cpu:
    return gpu_nms(dets, thresh, device_id=0)
  else:
    return cpu_nms(dets, thresh)