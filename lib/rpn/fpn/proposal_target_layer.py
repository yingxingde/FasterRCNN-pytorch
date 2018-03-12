# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform
from lib.utils.bbox import bbox_overlaps
import torch
from torch.autograd import Variable


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    # add the ground-truth to rois will cause zero loss! not good for visuallization
    if cfg.TRAIN.GT_OFFSET:
      gt_easyboxes = gt_boxes
      jittered_gt_boxes = _jitter_gt_boxes(gt_easyboxes)
      zeros = rpn_rois.data.new(gt_boxes.size()[0] * 2, 1).zero_()
      zeros = Variable(zeros)
      jit_cat = torch.cat((gt_easyboxes[:, :-1], jittered_gt_boxes[:, :-1]), 0)
      jit_cat = torch.cat((zeros, jit_cat), 1)
      all_rois = torch.cat((all_rois, jit_cat), 0)
    else:
      zeros = rpn_rois.data.new(gt_boxes.shape[0], 1).zero_()
      zeros = Variable(zeros)
      zeros_cat = torch.cat((zeros, gt_boxes[:, :-1]), dim=1)
      all_rois = torch.cat((all_rois, zeros_cat), dim=0)
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)
    del zeros

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
  inds = (clss > 0).nonzero().view(-1)
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)
    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return torch.cat(
    [labels.unsqueeze(1), targets], 1)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)
  labels = gt_boxes[gt_assignment, [4]]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(
      npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(
      npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(
      npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif bg_inds.numel() > 0:
    to_replace = bg_inds.numel() < rois_per_image
    bg_inds = bg_inds[torch.from_numpy(
      npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds].contiguous()
  roi_scores = all_scores[keep_inds].contiguous()

  bbox_target_data = _compute_targets(
    rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)

  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

def _jitter_gt_boxes(gt_boxes, jitter=0.05):
  """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
  gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
  """
  jittered_boxes = gt_boxes.clone()
  ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
  hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0

  width_rand = Variable((jittered_boxes.data.new(jittered_boxes.size()[0]).random_(0, 1) - 0.5) * jitter)
  width_offset = width_rand * ws
  height_rand = Variable((jittered_boxes.data.new(jittered_boxes.size()[0]).random_(0, 1) - 0.5) * jitter)
  height_offset = height_rand * hs
  # width_offset = (torch.rand(jittered_boxes.size()[0]) - 0.5) * jitter * ws
  # height_offset = (torch.rand(jittered_boxes.size()[0]) - 0.5) * jitter * hs
  del width_rand
  del height_rand
  jittered_boxes[:, 0] = jittered_boxes[:, 0] + width_offset
  jittered_boxes[:, 2] = jittered_boxes[:, 2] + width_offset
  jittered_boxes[:, 1] = jittered_boxes[:, 1] + height_offset
  jittered_boxes[:, 3] = jittered_boxes[:, 3] + height_offset

  return jittered_boxes