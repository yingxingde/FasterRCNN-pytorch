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
from lib.model.bbox_transform_cpu import bbox_transform

DEBUG = False

def anchor_target_layer(rpn_cls_score_list, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A_s = num_anchors
  total_anchors = all_anchors.shape[0]
  # K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # pytorch (bs, c, h, w)
  heights = [rpn_cls_score.shape[2] for rpn_cls_score in rpn_cls_score_list]
  widths = [rpn_cls_score.shape[3] for rpn_cls_score in rpn_cls_score_list]

  # only keep anchors inside the image
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  if DEBUG:
    print('inds_inside', len(inds_inside))
    print('total anchors', total_anchors)

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  argmax_overlaps = overlaps.argmax(axis=1)
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  gt_argmax_overlaps = overlaps.argmax(axis=0)
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

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
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    # positive_weights = np.ones((1, 4))
    # negative_weights = np.zeros((1, 4))
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  begin_cnt = 0
  end_cnt = 0
  begin_cnt_bbox = 0
  end_cnt_bbox = 0
  labels_list = list()
  bbox_targets_list = list()
  bbox_inside_weights_list = list()
  bbox_outside_weights_list = list()
  for height, width, A in zip(heights, widths, A_s):
    begin_cnt = end_cnt
    end_cnt += 1*height*width*A
    labels_part = labels[begin_cnt:end_cnt]

    # labels
    labels_part = labels_part.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels_part = labels_part.reshape((1, 1, A * height, width)).reshape((-1,))
    labels_list.append(labels_part)

    # begin_cnt_bbox = end_cnt_bbox
    # end_cnt_bbox += 1*height*width*A*4
    # bbox_targets_part = bbox_targets[begin_cnt_bbox:end_cnt_bbox]
    # bbox_inside_weights_part = bbox_inside_weights[begin_cnt_bbox:end_cnt_bbox]
    # bbox_outside_weights_part = bbox_outside_weights[begin_cnt_bbox:end_cnt_bbox]
    #
    # # bbox_targets
    # bbox_targets_part = bbox_targets_part.reshape((1, height, width, A * 4))
    # bbox_targets_list.append(bbox_targets_part)
    #
    # # bbox_inside_weights
    # bbox_inside_weights_part = bbox_inside_weights_part.reshape((1, height, width, A * 4))
    # bbox_inside_weights_list.append(bbox_inside_weights_part)
    #
    # # bbox_outside_weights
    # bbox_outside_weights_part = bbox_outside_weights_part.reshape((1, height, width, A * 4))
    # bbox_outside_weights_list.append(bbox_outside_weights_part)

  assert total_anchors == end_cnt
  labels = np.concatenate(labels_list, axis=0)
  # labels
  # labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  # labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  # bbox_targets = bbox_targets \
  #   .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  # bbox_inside_weights = bbox_inside_weights \
  #   .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  # bbox_outside_weights = bbox_outside_weights \
  #   .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
