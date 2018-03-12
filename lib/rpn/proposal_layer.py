# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform_inv, clip_boxes
from lib.model.nms_wrapper import nms
import torch
from torch.autograd import Variable



def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """
    rpn_cls_prob = [n, h, w, c=a*2]
    rpn_bbox_pred = [n, h, w, c=4*a]
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  # 0:num_anchors is the probability of BG
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  bbox_deltas = rpn_bbox_pred.view((-1, 4))
  scores = scores.contiguous().view((-1,1))

  # 1.Convert anchors into proposals via bbox transformations
  proposals = bbox_transform_inv(anchors, bbox_deltas)

  # 2.clip predicted boxes to image
  proposals = clip_boxes(proposals, im_info[:2])

  # # 3.remove predicted boxes with either height or width < threshold
  # # (NOTE: convert min_size to input image scale stored in im_info[2])
  # keep = _filter_boxes(proposals, _feat_stride[0] * im_info[2])
  # proposals = proposals[keep, :]
  # scores = scores[keep]

  # 4.sort all (proposal, score) pairs by score from highest to lowest
  # 5.take top pre_nms_topN (e.g. Train12000 Test6000)
  _, order = scores.view(-1).sort(descending=True)
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order.data, :]
  scores = scores[order.data, :]

  # 6.Non-maximal suppression apply nms (e.g. threshold = 0.7)
  keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

  # 7.Pick th top region proposals after NMS (e.g. Train2000 Test300)
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep, :]

  # 8.Only support single image as input
  batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
  blob = torch.cat((batch_inds, proposals), 1)
  del batch_inds
  return blob, scores

def _filter_boxes(boxes, min_size):
  """Remove all boxes with any side smaller than min_size."""
  ws = boxes[:, 2] - boxes[:, 0] + 1
  hs = boxes[:, 3] - boxes[:, 1] + 1
  keep = ((ws >= min_size) & (hs >= min_size)).nonzero()[:, 0]
  return keep
