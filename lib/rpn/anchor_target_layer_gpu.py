# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.model.config import cfg
import numpy as np
import numpy.random as npr
import torch
from lib.utils.bbox import bbox_overlaps
from lib.model.bbox_transform import bbox_transform

DEBUG = False

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A = num_anchors
  total_anchors = all_anchors.size()[0]
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # pytorch (bs, c, h, w)
  height, width = rpn_cls_score.size()[2:4]

  # only keep anchors inside the image
  inds_inside = (
    (all_anchors.data[:, 0] >= -_allowed_border) &
    (all_anchors.data[:, 1] >= -_allowed_border) &
    (all_anchors.data[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors.data[:, 3] < im_info[0] + _allowed_border)  # height
  ).nonzero()[:, 0].long()

  if DEBUG:
    print('total_anchors', total_anchors)
    print('inds_inside', inds_inside.size()[0])

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  if DEBUG:
    print('anchors.shape', anchors.size())

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = inds_inside.new(inds_inside.size()[0]).fill_(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt) shape is A x G
  overlaps = bbox_overlaps(
    anchors.data,
    gt_boxes[:, :4].data)
  max_overlaps, argmax_overlaps = torch.max(overlaps, dim=1)
  gt_max_overlaps, gt_argmax_overlaps = torch.max(overlaps, dim=0)
  gt_argmax_overlaps = (overlaps == (gt_max_overlaps.unsqueeze(0).expand_as(overlaps))).nonzero()[:, 0]

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1

  # fg label: above threshold IOU
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # subsample positive labels if we have too many
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  fg_inds = (labels == 1).nonzero()[:, 0]
  if fg_inds.numel() > num_fg:
    inds = fg_inds.new(
      npr.choice(np.arange(0, fg_inds.numel()), size=int((len(fg_inds) - num_fg)), replace=False)).long()
    disable_inds = fg_inds[inds]
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - (labels == 1).sum()
  bg_inds = (labels == 0).nonzero()[:, 0]
  if bg_inds.numel() > num_bg:
    inds = bg_inds.new(
      npr.choice(np.arange(0, bg_inds.numel()), size=int((len(bg_inds) - num_bg)), replace=False)).long()
    disable_inds = bg_inds[inds]
    labels[disable_inds] = -1

  bbox_targets = _compute_targets(anchors.data, gt_boxes[argmax_overlaps][:, :4].data)
  bbox_inside_weights = bbox_targets.new(inds_inside.size()[0], 4).zero_()
  # only the positive ones have regression targets
  inds = (labels == 1).nonzero().view(-1)
  # dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
  # dim2_inds = inds.new((0,1,2,3)).view(-1,4).expand_as(dim1_inds)
  dim_value = bbox_targets.new(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS).view(-1, 4).expand(inds.size(0), 4)
  bbox_inside_weights[inds, :] = dim_value

  bbox_outside_weights = bbox_targets.new(inds_inside.size()[0], 4).zero_()
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = (labels >= 0).sum()
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        (labels == 1).sum())
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        (labels == 0).sum())

  inds = (labels == 1).nonzero().view(-1)
  # dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
  # dim2_inds = inds.new((0,1,2,3)).view(-1,4).expand_as(dim1_inds)
  dim_value = bbox_targets.new(positive_weights).view(-1, 4).expand(inds.size(0), 4)
  bbox_outside_weights[inds, :] = dim_value

  inds = (labels == 0).nonzero().view(-1)
  # dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
  # dim2_inds = inds.new((0,1,2,3)).view(-1,4).expand_as(dim1_inds)
  dim_value = bbox_targets.new(negative_weights).view(-1, 4).expand(inds.size(0), 4)
  bbox_outside_weights[inds, :] = dim_value

  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  # labels
  labels = labels.view((1, height, width, A)).permute(0, 3, 1, 2).contiguous()
  labels = labels.view((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .view((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .view((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .view((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if data.dim() == 1:
    ret = data.new(count).fill_(fill)
    ret[inds] = data
  else:
    ret_dim = (count,) + data.size()[1:]
    ret = data.new(*ret_dim).fill_(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  return bbox_transform(ex_rois, gt_rois)
