# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import cv2
# gc.set_threshold(100, 10, 10)

from lib.model.config import cfg
from lib.rpn.fpn.generate_anchors_global import generate_anchors_global
from lib.rpn.fpn.anchor_target_layer_cpu import anchor_target_layer
from lib.rpn.fpn.proposal_target_layer import proposal_target_layer
from lib.rpn.fpn.proposal_layer import proposal_layer
from lib.nets.layers_util import *
from lib.model.nms_wrapper import nms_test as nms
from lib.roi_pooling.roi_pool import RoIPool, RoIPoolFunction
from lib.model.bbox_transform_cpu import clip_boxes, bbox_transform_inv
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.transforms as torchtrans

Debug = False

class RPN(nn.Module):
  def __init__(self, net):
    super(RPN, self).__init__()

    self._network = net

    self.cross_entropy = None
    self.loss_box = None
    self.cache_dict = {}


  def _init_network(self):
    self._network._init_network()
    self.rpn_conv = nn.Conv2d(self._network._channels['head'], 512, (3, 3), padding=1)
    self.rpn_score = nn.Conv2d(512, self._network._num_anchors * 2, (1, 1))
    self.rpn_bbox = nn.Conv2d(512, self._network._num_anchors * 4, (1, 1))

  def forward(self, im_data, im_info, gt_boxes=None):
    c2 = self._network._layers['c2'](im_data)

    c3 = self._network._layers['c3'](c2)

    c4 = self._network._layers['c4'](c3)

    c5 = self._network._layers['c5'](c4)

    p5 = self._network._layers['p5'](c5)

    p6 = self._network._layers['p6'](p5)

    p4_fusion = F.upsample(p5, size=c4.size()[-2:], mode='bilinear')+\
                 self._network._layers['p5_p4_lateral'](c4)

    p4 = self._network._layers['p4'](p4_fusion)

    p3_fusion = F.upsample(p4, size=c3.size()[-2:], mode='bilinear')+\
                 self._network._layers['p4_p3_lateral'](c3)

    p3 = self._network._layers['p3'](p3_fusion)

    p2_fusion = F.upsample(p3, size=c2.size()[-2:], mode='bilinear') + \
                self._network._layers['p3_p2_lateral'](c2)

    p2 = self._network._layers['p2'](p2_fusion)

    p_list = [p2, p3, p4, p5, p6]

    if Debug:
      c_list = [c2, c3, c4, c5]
      print('p_list:')
      for p in p_list:
        print(p.size())
      print('c_list:')
      for c in c_list:
        print(c.size())
      print(len(p_list))

    rpn_cls_prob_final_list = list()
    rpn_bbox_score_list = list()
    rpn_cls_score_list = list()
    rpn_cls_score_reshape_list = list()
    for feature in p_list:
      rpn_feature = self.rpn_conv(feature)
      # cls
      # n a*2 h w
      rpn_cls_score = self.rpn_score(rpn_feature)
      rpn_cls_score_list.append(rpn_cls_score)
      # n 2 a*h w
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2)
      rpn_cls_score_reshape_list.append(rpn_cls_score_reshape)
      # n 2 a*h w
      rpn_cls_prob = F.softmax(rpn_cls_score_reshape, 1)
      # n a*2 h w  to  n h w a*2
      rpn_cls_prob_final = self._reshape_layer(rpn_cls_prob, self._network._num_anchors * 2).permute(0, 2, 3, 1).contiguous()
      rpn_cls_prob_final_list.append(rpn_cls_prob_final)
      # bbox
      rpn_bbox_score = self.rpn_bbox(rpn_feature)
      rpn_bbox_score = rpn_bbox_score.permute(0, 2, 3, 1).contiguous()
      rpn_bbox_score_list.append(rpn_bbox_score)

    if Debug:
      print('RPN:')
      for i in rpn_cls_prob_final_list:
        print(i.size())

    # generate anchors
    self._generate_anchors(rpn_cls_score_list)
    rois, scores = self._region_proposal(rpn_cls_prob_final_list, rpn_bbox_score_list, im_info)

    if Debug:
      print('rpn rois:', rois.size())

    # generating training labels and build the rpn loss
    if self.training:
      assert gt_boxes is not None
      rpn_data = self._anchor_target_layer(rpn_cls_score_list, gt_boxes, im_info)
      self.cross_entropy, self.loss_box = self._build_loss(rpn_cls_score_reshape_list, rpn_bbox_score_list, rpn_data)

    # list cache
    self.cache_dict['rpn_cls_prob_final_list'] = rpn_cls_prob_final_list
    self.cache_dict['rpn_bbox_score_list'] = rpn_bbox_score_list
    self.cache_dict['rpn_cls_score_list'] = rpn_cls_score_list
    self.cache_dict['rpn_cls_score_reshape_list'] = rpn_cls_score_reshape_list

    return rois, scores, p_list

  @property
  def loss(self):
    return self.cross_entropy + self.loss_box * cfg.TRAIN.LOSS_RATIO

  def _build_loss(self, rpn_cls_score_reshape_list, rpn_bbox_score_list, rpn_data, sigma_rpn=3):
    rpn_cls_score = [rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
                     for rpn_cls_score_reshape in rpn_cls_score_reshape_list]
    rpn_cls_score = torch.cat(rpn_cls_score, dim=0)
    rpn_label = rpn_data[0].view(-1)

    # cls loss
    rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())

    if Debug:
      print('rpn_keep', rpn_keep.size())
      print('fg:', rpn_label.data.eq(1).sum())
      print('fg:', rpn_label.data.eq(0).sum())

    assert rpn_keep.numel() == cfg.TRAIN.RPN_BATCHSIZE

    if cfg.CUDA_IF:
      rpn_keep = rpn_keep.cuda()
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    rpn_bbox_score = [rpn_bbox_score.view((-1, 4)) for rpn_bbox_score in rpn_bbox_score_list]
    rpn_bbox_score = torch.cat(rpn_bbox_score, dim=0)
    rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_score, rpn_bbox_targets, rpn_bbox_inside_weights,
                                        rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[0, 1])
    return rpn_cross_entropy, rpn_loss_box

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

  def _reshape_layer(self, x, d):
    '''
    :param x: n [a1 b1,a2 b2] h w
    :param d: d
    :return: n 2 a*h w
    '''
    input_shape = x.size()
    x = x.view(
      input_shape[0],
      int(d),
      int(float(input_shape[1] * input_shape[2]) / float(d)),
      input_shape[3]
    )
    return x

  def _generate_anchors(self, rpn_cls_score_list):
    # anchors [A*K, 4]
    heights = [rpn_cls_score.size()[-2] for rpn_cls_score in rpn_cls_score_list]
    widths = [rpn_cls_score.size()[-1] for rpn_cls_score in rpn_cls_score_list]
    anchors, num_anchors = generate_anchors_global(
      feat_strides=self._network._feat_stride,
      heights=heights,
      widths=widths,
      anchor_scales=self._network._anchor_scales,
      anchor_ratios=self._network._anchor_ratios
    )
    self._anchors = Variable(torch.from_numpy(anchors)).float()
    self._num_anchors = torch.from_numpy(num_anchors)
    if cfg.CUDA_IF:
      self._anchors = self._anchors.cuda()
      self._num_anchors = self._num_anchors.cuda()
    self.cache_dict['anchors_cache'] = self._anchors
    self.cache_dict['num_anchors_cache'] = self._num_anchors

  def _region_proposal(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
    cfg_key = 'TRAIN' if self.training else 'TEST'
    rois, rpn_scores = proposal_layer(rpn_cls_prob_list=rpn_cls_prob_reshape, rpn_bbox_pred_list=rpn_bbox_pred,
                                      im_info=im_info, cfg_key=cfg_key,
                                      _feat_stride=self._network._feat_stride,
                                      anchors=self._anchors,
                                      num_anchors_list=self._num_anchors)
    if cfg_key == 'TEST':
      leveled_rois = self.rois_split_level(rois)
      self.cache_dict['rpn_leveled_rois'] = leveled_rois
      return leveled_rois, rpn_scores
    else:
      self.cache_dict['rois'] = rois
      self.cache_dict['rpn_scores'] = rpn_scores
      return rois, rpn_scores

  @staticmethod
  def rois_split_level(rois):
    def calc_level(width, height):
      value = width * height
      if value == 0:
        inner = 0
      else:
        inner = 4 + np.log2(np.sqrt(value) / 224)
      return min(6, max(2, int(inner)))

    level = lambda roi: calc_level(roi[3] - roi[1], roi[4] - roi[2])  # roi: [0, x0, y0, x1, y1]

    rois_data = rois.data.cpu().numpy()

    leveled_rois = [None] * 5
    leveled_idxs = [[], [], [], [], []]
    for idx, roi in enumerate(rois_data):
      level_idx = level(roi) - 2
      leveled_idxs[level_idx].append(idx)

    if Debug:
      print('leveled_rois:')
      for i in range(5):
        print(len(leveled_idxs[i]))

    for level_index in range(0, 5):
      if len(leveled_idxs[level_index]) != 0:
        k = torch.from_numpy(np.asarray(leveled_idxs[level_index]))
        if cfg.CUDA_IF:
          k = k.cuda()
        leveled_rois[level_index] = rois[k]
    return leveled_rois

  def _anchor_target_layer(self, rpn_cls_score_list, gt_boxes, im_info):
    rpn_cls_score_l = [rpn_cls_score.data for rpn_cls_score in rpn_cls_score_list]
    gt_boxes = gt_boxes.data.cpu().numpy()
    all_anchors = self._anchors.data.cpu().numpy()
    all_num_anchors = self._num_anchors.cpu().numpy()
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(rpn_cls_score_list=rpn_cls_score_l, gt_boxes=gt_boxes, im_info=im_info,
                          _feat_stride=self._network._feat_stride,
                          all_anchors=all_anchors,
                          num_anchors=all_num_anchors)
    rpn_labels = np_to_variable(rpn_labels, is_cuda=cfg.CUDA_IF, dtype=torch.LongTensor)
    rpn_bbox_targets = np_to_variable(rpn_bbox_targets, is_cuda=cfg.CUDA_IF)
    rpn_bbox_inside_weights = np_to_variable(rpn_bbox_inside_weights, is_cuda=cfg.CUDA_IF)
    rpn_bbox_outside_weights = np_to_variable(rpn_bbox_outside_weights, is_cuda=cfg.CUDA_IF)
    self.cache_dict['rpn_labels'] = rpn_labels
    self.cache_dict['rpn_bbox_targets'] = rpn_bbox_targets
    self.cache_dict['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self.cache_dict['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

class FasterRCNN(nn.Module):
  def __init__(self, net, classes=None):
    super(FasterRCNN, self).__init__()
    assert (classes is not None), 'class can not be none!'
    self._classes = np.array(classes)
    self._num_classes = len(classes)

    self._rpn = RPN(net=net)
    # loss
    self.cross_entropy = None
    self.loss_box = None
    self.cache_dict = {}
    self.metrics_dict = {}

  def init_fasterRCNN(self):
    self._rpn._init_network()
    self.roi_pool = FPN_ROI_Pooling(7, 7, self._rpn._network._feat_stride)
    self.score_fc = nn.Linear(self._rpn._network._channels['tail'], self._num_classes)
    self.bbox_fc = nn.Linear(self._rpn._network._channels['tail'], self._num_classes * 4)

  def _predict(self, im_data, im_info, gt_boxes):
    # benchmark because now the input size are not fixed
    cudnn.benchmark = False

    rois, rpn_scores, features = self._rpn(im_data, im_info, gt_boxes)

    if self.training:
      roi_data = self._proposal_target_layer(rpn_rois=rois, gt_boxes=gt_boxes,
                                             rpn_scores=rpn_scores)
      rois = roi_data[0]
    else:
      roi_data = None

    pooled_features = self.roi_pool(features, rois)
    self.cache_dict['pooled_features'] = pooled_features

    if self.training:
      assert pooled_features.size()[0] == cfg.TRAIN.BATCH_SIZE
      # benchmark because now the input size are fixed
      cudnn.benchmark = True

    x = self._rpn._network._head_to_tail(pooled_features)

    cls_score = self.score_fc(x)
    cls_prob = F.softmax(cls_score, 1)
    bbox_pred = self.bbox_fc(x)

    if self.training:
      self.cross_entropy, self.loss_box = self._build_loss(cls_score, bbox_pred, roi_data)

    if self.training:
      return cls_prob, bbox_pred, roi_data[1]
    else:
      leveled_rois_t = [i for i in rois if i is not None]
      new_rois = torch.cat(leveled_rois_t, 0)
      return cls_prob, bbox_pred, new_rois

  def forward(self, im_data, im_info, gt_boxes=None):
    im_data = np_to_variable(im_data, is_cuda=cfg.CUDA_IF).permute(0, 3, 1, 2)
    self.cache_dict['im_data'] = im_data
    gt_boxes = np_to_variable(gt_boxes, is_cuda=cfg.CUDA_IF) if gt_boxes is not None else None
    self.cache_dict['gt_boxes'] = gt_boxes

    cls_prob, bbox_pred, rois = self._predict(im_data, im_info, gt_boxes)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # in proposal_target_layer target has done regularization
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(
        bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(
        bbox_pred)
      bbox_pred = bbox_pred.mul(Variable(stds)).add(Variable(means))

    if not self.training:
      # clear middle memory
      self._delete_cache()

    return cls_prob, bbox_pred, rois

  @property
  def loss(self):
    return self.cross_entropy + self.loss_box * cfg.TRAIN.LOSS_RATIO

  def _build_loss(self, cls_score, bbox_pred, roi_data, sigma_rpn=1):
    label = roi_data[2].squeeze()
    assert label.dim() == 1
    fg_cnt = torch.sum(label.data.ne(0))
    bg_cnt = label.data.numel() - fg_cnt
    self.metrics_dict['fg'] = fg_cnt
    self.metrics_dict['bg'] = bg_cnt

    _, predict = cls_score.data.max(1)
    label_data = label.data
    tp = torch.sum(predict.eq(label_data)&label_data.ne(0)) if fg_cnt > 0 else 0
    tf = torch.sum(predict.eq(label_data)&label_data.eq(0)) if bg_cnt > 0 else 0
    self.metrics_dict['tp'] = tp
    self.metrics_dict['tf'] = tf

    # cls
    assert cfg.TRAIN.BATCH_SIZE == label.numel()
    cross_entropy = F.cross_entropy(cls_score, label)
    # bbox
    bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[3:]
    loss_box = self._rpn._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
                                    bbox_outside_weights, sigma=sigma_rpn)

    return cross_entropy, loss_box

  def init_special_bbox_fc(self, dev=0.001):
    def _gaussian_init(m, dev):
      m.weight.data.normal_(0.0, dev)
      if hasattr(m.bias, 'data'):
        m.bias.data.zero_()
    model = self.bbox_fc
    for m in model.modules():
      if isinstance(m, nn.Conv2d):
        _gaussian_init(m, dev)
      elif isinstance(m, nn.Linear):
        _gaussian_init(m, dev)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _delete_cache(self):
    for dic in [self._rpn.cache_dict, self.cache_dict]:
      for key in dic.keys():
        del dic[key]
    # gc.collect()

  def _proposal_target_layer(self, rpn_rois, gt_boxes, rpn_scores):
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposal_target_layer(
        rpn_rois, rpn_scores, gt_boxes, self._num_classes)
    leveled_rois, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      self.rois_split_level(rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    labels = labels.long()
    bbox_targets = Variable(bbox_targets)
    bbox_inside_weights = Variable(bbox_inside_weights)
    bbox_outside_weights = Variable(bbox_outside_weights)
    self.cache_dict['labels'] = labels
    self.cache_dict['bbox_targets'] = bbox_targets
    self.cache_dict['bbox_inside_weights'] = bbox_inside_weights
    self.cache_dict['bbox_outside_weights'] = bbox_outside_weights
    return leveled_rois, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

  @staticmethod
  def rois_split_level(rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    def calc_level(width, height):
      value = width * height
      if value == 0:
        inner = 0
      else:
        inner = 4 + np.log2(np.sqrt(value) / 224)
      return min(6, max(2, int(inner)))

    level = lambda roi: calc_level(roi[3] - roi[1], roi[4] - roi[2])  # roi: [0, x0, y0, x1, y1]

    rois_data = rois.data.cpu().numpy()

    leveled_rois = [None] * 5
    leveled_labels = [None] * 5
    leveled_bbox_targets = [None] * 5
    leveled_bbox_inside_weights = [None] * 5
    leveled_bbox_outside_weights = [None] * 5
    leveled_idxs = [[], [], [], [], []]
    for idx, roi in enumerate(rois_data):
      level_idx = level(roi) - 2
      leveled_idxs[level_idx].append(idx)

    for level_index in range(0, 5):
      if len(leveled_idxs[level_index]) != 0:
        k = torch.from_numpy(np.asarray(leveled_idxs[level_index]))
        if cfg.CUDA_IF:
          k = k.cuda()
        leveled_rois[level_index] = rois[k]
        leveled_labels[level_index] = labels[k]
        leveled_bbox_targets[level_index] = bbox_targets[k]
        leveled_bbox_inside_weights[level_index] = bbox_inside_weights[k]
        leveled_bbox_outside_weights[level_index] = bbox_outside_weights[k]

    leveled_rois_t = [i for i in leveled_rois if i is not None]
    leveled_labels = [i for i in leveled_labels if i is not None]
    leveled_bbox_targets = [i for i in leveled_bbox_targets if i is not None]
    leveled_bbox_inside_weights = [i for i in leveled_bbox_inside_weights if i is not None]
    leveled_bbox_outside_weights = [i for i in leveled_bbox_outside_weights if i is not None]
    new_rois = torch.cat(leveled_rois_t, 0)
    new_labels = torch.cat(leveled_labels, 0)
    new_bbox_targets = torch.cat(leveled_bbox_targets, 0)
    new_bbox_inside_weights = torch.cat(leveled_bbox_inside_weights, 0)
    new_bbox_outside_weights = torch.cat(leveled_bbox_outside_weights, 0)

    return leveled_rois, new_rois, new_labels, new_bbox_targets, new_bbox_inside_weights, new_bbox_outside_weights

  def train_operation(self, blobs, optimizer, image_if=False, clip_parameters=None):
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']

    # forward
    result_cls_prob, result_bbox_pred, result_rois = self(im_data, im_info, gt_boxes)
    loss = self.loss + self._rpn.loss

    # backward
    if optimizer is not None:
      optimizer.zero_grad()
      loss.backward()
      if clip_parameters is not None:
        nn.utils.clip_grad_norm(self._parameters, max_norm=10)
      optimizer.step()

    loss = loss.data.cpu()[0]
    rpn_cls_loss = self._rpn.cross_entropy.data.cpu()[0]
    rpn_bbox_loss = self._rpn.loss_box.data.cpu()[0]
    fast_rcnn_cls_loss = self.cross_entropy.data.cpu()[0]
    fast_rcnn_bbox_loss = self.loss_box.data.cpu()[0]

    image = None
    if image_if:
      image = self.visual_image(blobs, result_cls_prob, result_bbox_pred, result_rois)

    # clear middle memory
    self._delete_cache()

    return (loss, rpn_cls_loss, rpn_bbox_loss, fast_rcnn_cls_loss, fast_rcnn_bbox_loss), image

  def visual_image(self, blobs, result_cls_prob, result_bbox_pred, result_rois):
    new_gt_boxes = blobs['gt_boxes'].copy()
    new_gt_boxes[:, :4] = new_gt_boxes[:, :4]
    image = self.back_to_image(blobs['data']).astype(np.uint8)

    im_shape = image.shape
    pred_boxes, scores, classes = self.interpret_faster_rcnn_scale(result_cls_prob, result_bbox_pred, result_rois,
                                                                  im_shape, min_score=0.1)
    image = self.draw_photo(image, pred_boxes, scores, classes, new_gt_boxes)
    image = torchtrans.ToTensor()(image)
    image = vutils.make_grid([image])
    return image

  @staticmethod
  def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
      return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]

  def interpret_faster_rcnn_scale(self, cls_prob, bbox_pred, rois, im_shape, nms=True, clip=True, min_score=0.0):
    # find class
    scores, inds = cls_prob.data.max(1)
    scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

    keep = np.where((inds > 0) & (scores >= min_score))
    scores, inds = scores[keep], inds[keep]

    # Apply bounding-box regression deltas
    keep = keep[0]
    box_deltas = bbox_pred.data.cpu().numpy()[keep]
    box_deltas = np.asarray([
      box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
    ], dtype=np.float)
    boxes = rois.data.cpu().numpy()[keep, 1:5]
    if len(keep) != 0:
      pred_boxes = bbox_transform_inv(boxes, box_deltas)
    else:
      pred_boxes = boxes
    if clip and pred_boxes.shape[0] > 0:
      pred_boxes = clip_boxes(pred_boxes, im_shape)
    # nms
    if nms and pred_boxes.shape[0] > 0:
      pred_boxes, scores, inds = self.nms_detections(pred_boxes, scores, 0.3, inds=inds)

    return pred_boxes, scores, self._classes[inds]

  def draw_photo(self, image, dets, scores, classes, gt_boxes):
      # im2show = np.copy(image)
      im2show = image
      # color_b = (0, 191, 255)
      for i, det in enumerate(dets):
          det = tuple(int(x) for x in det)
          r = min(0+i*10, 255)
          r_i = i / 5
          g = min(150+r_i*10, 255)
          g_i = r_i / 5
          b = min(200+g_i, 255)
          color_b_c = (r,g,b)
          cv2.rectangle(im2show, det[0:2], det[2:4], color_b_c, 2)
          cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                      1.0, (0, 0, 255), thickness=1)
      for i, det in enumerate(gt_boxes):
          det = tuple(int(x) for x in det)
          gt_class = self._classes[det[-1]]
          cv2.rectangle(im2show, det[0:2], det[2:4], (255, 0, 0), 2)
          cv2.putText(im2show, '%s' % (gt_class), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                      1.0, (0, 0, 255), thickness=1)
      return im2show

  def back_to_image(self, img):
    image = img[0] + cfg.PIXEL_MEANS
    image = image[:,:,::-1].copy(order='C')
    return image