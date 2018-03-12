from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.config import cfg
import lib.roi_data_layer.roidb as rdl_roidb
from lib.roi_data_layer.layer import RoIDataLayer
from lib.nets.layers_util import *
from lib.utils.timer import Timer
try:
  import cPickle as pickle
except ImportError:
  import pickle
import numpy as np

import os
import torch
import sys
import pprint
import time
import cv2
import h5py


class SolverWrapper(object):
  def __init__(self, network, imdb, roidb, valroidb, model_dir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.model_dir = model_dir
    self.tbdir = os.path.join(model_dir, 'train_log')
    if not os.path.exists(self.tbdir):
      os.makedirs(self.tbdir)
    self.pretrained_model = pretrained_model

  def set_learn_strategy(self, learn_dict):
    self._disp_interval = learn_dict['disp_interval']
    self._valid_interval = learn_dict['disp_interval']*5
    self._use_tensorboard = learn_dict['use_tensorboard']
    self._use_valid = learn_dict['use_valid']
    self._save_point_interval = learn_dict['save_point_interval']
    self._lr_decay_steps = learn_dict['lr_decay_steps']

  def train_model(self, resume=None, max_iters=100000):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    self.prepare_construct(resume)

    net = self.net
    # training
    train_loss = 0
    rpn_cls_loss = 0
    rpn_bbox_loss = 0
    fast_rcnn_cls_loss = 0
    fast_rcnn_bbox_loss = 0
    tp, tf, fg, bg = 0., 0., 0, 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()
    for step in range(self.start_step, max_iters + 1):
      blobs = self.data_layer.forward()

      if step % self._valid_interval == 0 and self._use_tensorboard:
        loss_r, image_r = net.train_operation(blobs, self._optimizer, image_if=True, clip_parameters=self._parameters)
        self._tensor_writer.add_image('Image', image_r, step)
      else:
        loss_r, image_r = net.train_operation(blobs, self._optimizer, image_if=False, clip_parameters=self._parameters)

      train_loss += loss_r[0]
      rpn_cls_loss += loss_r[1]
      rpn_bbox_loss += loss_r[2]
      fast_rcnn_cls_loss += loss_r[3]
      fast_rcnn_bbox_loss += loss_r[4]
      fg += net.metrics_dict['fg']
      bg += net.metrics_dict['bg']
      tp += net.metrics_dict['tp']
      tf += net.metrics_dict['tf']
      step_cnt += 1

      if step % self._disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
          step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
        tp_text = 'step {}, tp: {}/{}, tf: {}/{}'.format(
          step, int(tp/step_cnt), int(fg/step_cnt), int(tf/step_cnt), int(bg/step_cnt)
        )
        pprint.pprint(log_text)
        pprint.pprint(tp_text)

        if self._use_tensorboard:
          self._tensor_writer.add_text('Train', log_text, global_step=step)
          # Train
          avg_rpn_cls_loss = rpn_cls_loss / step_cnt
          avg_rpn_bbox_loss = rpn_bbox_loss / step_cnt
          avg_fast_rcnn_cls_loss = fast_rcnn_cls_loss / step_cnt
          avg_fast_rcnn_bbox_loss = fast_rcnn_bbox_loss / step_cnt

          self._tensor_writer.add_scalars('TrainSetLoss', {
            'RPN_cls_loss': avg_rpn_cls_loss,
            'RPN_bbox_loss': avg_rpn_bbox_loss,
            'FastRcnn_cls_loss': avg_fast_rcnn_cls_loss,
            'FastRcnn_bbox_loss': avg_fast_rcnn_bbox_loss
          }, global_step=step)
          self._tensor_writer.add_scalar('Learning_rate', self._lr, global_step=step)

        re_cnt = True

      if self._use_valid and step % self._valid_interval == 0:
        total_valid_loss = 0.0
        valid_rpn_cls_loss = 0.0
        valid_rpn_bbox_loss = 0.0
        valid_fast_rcnn_cls_loss = 0.0
        valid_fast_rcnn_bbox_loss = 0.0
        valid_step_cnt = 0
        valid_tp, valid_tf, valid_fg, valid_bg = 0., 0., 0, 0
        start_time = time.time()

        valid_length = self._disp_interval
        for valid_batch in range(valid_length):
          # get one batch
          blobs = self.data_layer_val.forward()

          if self._use_tensorboard and valid_batch % valid_length == 0:
            loss_r, image_r = net.train_operation(blobs, None, image_if=True)
            self._tensor_writer.add_image('Image_Valid', image_r, step)
          else:
            loss_r, image_r = net.train_operation(blobs, None, image_if=False)

          total_valid_loss += loss_r[0]
          valid_rpn_cls_loss += loss_r[1]
          valid_rpn_bbox_loss += loss_r[2]
          valid_fast_rcnn_cls_loss += loss_r[3]
          valid_fast_rcnn_bbox_loss += loss_r[4]
          valid_fg += net.metrics_dict['fg']
          valid_bg += net.metrics_dict['bg']
          valid_tp += net.metrics_dict['tp']
          valid_tf += net.metrics_dict['tf']
          valid_step_cnt += 1

        duration = time.time() - start_time
        fps = valid_step_cnt / duration

        log_text = 'step %d, valid average loss: %.4f, fps: %.2f (%.2fs per batch)' % (
          step, total_valid_loss / valid_step_cnt, fps, 1. / fps)
        pprint.pprint(log_text)

        if self._use_tensorboard:
          self._tensor_writer.add_text('Valid', log_text, global_step=step)
          # Valid
          avg_rpn_cls_loss_valid = valid_rpn_cls_loss / valid_step_cnt
          avg_rpn_bbox_loss_valid = valid_rpn_bbox_loss / valid_step_cnt
          avg_fast_rcnn_cls_loss_valid = valid_fast_rcnn_cls_loss / valid_step_cnt
          avg_fast_rcnn_bbox_loss_valid = valid_fast_rcnn_bbox_loss / valid_step_cnt
          valid_tpr = valid_tp*1.0 / valid_fg
          valid_tfr = valid_tf*1.0 / valid_bg
          real_total_loss_valid = valid_rpn_cls_loss + valid_rpn_bbox_loss + valid_fast_rcnn_cls_loss + valid_fast_rcnn_bbox_loss

          # Train
          avg_rpn_cls_loss = rpn_cls_loss / step_cnt
          avg_rpn_bbox_loss = rpn_bbox_loss / step_cnt
          avg_fast_rcnn_cls_loss = fast_rcnn_cls_loss / step_cnt
          avg_fast_rcnn_bbox_loss = fast_rcnn_bbox_loss / step_cnt
          tpr = tp*1.0 / fg
          tfr = tf*1.0 / bg
          real_total_loss = rpn_cls_loss + rpn_bbox_loss + fast_rcnn_cls_loss + fast_rcnn_bbox_loss

          self._tensor_writer.add_scalars('Total_Loss', {
            'train': train_loss / step_cnt,
            'valid': total_valid_loss / valid_step_cnt
          }, global_step=step)
          self._tensor_writer.add_scalars('Real_loss', {
            'train': real_total_loss / step_cnt,
            'valid': real_total_loss_valid / valid_step_cnt
          }, global_step=step)
          self._tensor_writer.add_scalars('RPN_cls_loss', {
            'train': avg_rpn_cls_loss,
            'valid': avg_rpn_cls_loss_valid
          }, global_step=step)
          self._tensor_writer.add_scalars('RPN_bbox_loss', {
            'train': avg_rpn_bbox_loss,
            'valid': avg_rpn_bbox_loss_valid
          }, global_step=step)
          self._tensor_writer.add_scalars('FastRcnn_cls_loss', {
            'train': avg_fast_rcnn_cls_loss,
            'valid': avg_fast_rcnn_cls_loss_valid
          }, global_step=step)
          self._tensor_writer.add_scalars('FastRcnn_bbox_loss', {
            'train': avg_fast_rcnn_bbox_loss,
            'valid': avg_fast_rcnn_bbox_loss_valid
          }, global_step=step)
          self._tensor_writer.add_scalars('tpr', {
            'train': tpr,
            'valid': valid_tpr
          }, global_step=step)
          self._tensor_writer.add_scalars('tfr', {
            'train': tfr,
            'valid': valid_tfr
          }, global_step=step)

          self._tensor_writer.add_scalars('ValidSetLoss', {
            'RPN_cls_loss': avg_rpn_cls_loss_valid,
            'RPN_bbox_loss': avg_rpn_bbox_loss_valid,
            'FastRcnn_cls_loss': avg_fast_rcnn_cls_loss_valid,
            'FastRcnn_bbox_loss': avg_fast_rcnn_bbox_loss_valid
          }, global_step=step)

      if (step % self._save_point_interval == 0) and step > 0:
        save_name, _ = self.save_check_point(step)
        print('save model: {}'.format(save_name))

      if step in self._lr_decay_steps:
        self._lr *= self._lr_decay
        self._optimizer = self._train_optimizer()

      if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        rpn_cls_loss = 0
        rpn_bbox_loss = 0
        fast_rcnn_cls_loss = 0
        fast_rcnn_bbox_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

    if self._use_tensorboard:
        self._tensor_writer.export_scalars_to_json(os.path.join(self.tbdir, 'all_scalars.json'))

  def save_check_point(self, step):
    net = self.net

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    # store the model snapshot
    filename = os.path.join(self.model_dir, 'fasterRcnn_iter_{}.h5'.format(step))
    h5f = h5py.File(filename, mode='w')
    for k, v in net.state_dict().items():
      h5f.create_dataset(k, data=v.cpu().numpy())

    # store data information
    nfilename = os.path.join(self.model_dir, 'fasterRcnn_iter_{}.pkl'.format(step))
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm
    # current learning rate
    lr = self._lr

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(lr, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(step, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def load_check_point(self, step):
    net = self.net
    filename = os.path.join(self.model_dir, 'fasterRcnn_iter_{}.h5'.format(step))
    nfilename = os.path.join(self.model_dir, 'fasterRcnn_iter_{}.pkl'.format(step))
    print('Restoring model snapshots from {:s}'.format(filename))

    if not os.path.exists(filename):
      print('The checkPoint is not Right')
      sys.exit(1)

    # load model
    h5f = h5py.File(filename, mode='r')
    for k, v in net.state_dict().items():
      param = torch.from_numpy(np.asarray(h5f[k]))
      v.copy_(param)

    # load data information
    with open(nfilename, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      lr = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val
      self._lr = lr

    if last_snapshot_iter == step:
      print('Restore over ')
    else:
      print('The checkPoint is not Right')
      raise ValueError

    return last_snapshot_iter

  def weights_normal_init(self, model, dev=0.01):
    import math
    def _gaussian_init(m, dev):
      m.weight.data.normal_(0.0, dev)
      if hasattr(m.bias, 'data'):
        m.bias.data.zero_()

    def _xaiver_init(m):
      nn.init.xavier_normal(m.weight.data)
      if hasattr(m.bias, 'data'):
        m.bias.data.zero_()

    def _hekaiming_init(m):
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
      if hasattr(m.bias, 'data'):
        m.bias.data.zero_()

    def _resnet_init(model, dev):
      if isinstance(model, list):
        for m in model:
          self.weights_normal_init(m, dev)
      else:
        for m in model.modules():
          if isinstance(m, nn.Conv2d):
            _hekaiming_init(m)
          elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
          elif isinstance(m, nn.Linear):
            _gaussian_init(m, dev)

    def _vgg_init(model, dev):
      if isinstance(model, list):
        for m in model:
          self.weights_normal_init(m, dev)
      else:
        for m in model.modules():
          if isinstance(m, nn.Conv2d):
            _gaussian_init(m, dev)
          elif isinstance(m, nn.Linear):
            _gaussian_init(m, dev)
          elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    if cfg.TRAIN.INIT_WAY == 'resnet':
      _vgg_init(model, dev)
    elif cfg.TRAIN.INIT_WAY == 'vgg':
      _vgg_init(model, dev)
    else:
      raise NotImplementedError


  def prepare_construct(self, resume_iter):
    # init network
    self.net.init_fasterRCNN()

    # Set the random seed
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    # Set learning rate and momentum
    self._lr = cfg.TRAIN.LEARNING_RATE
    self._lr_decay = 0.1
    self._momentum = cfg.TRAIN.MOMENTUM
    self._weight_decay = cfg.TRAIN.WEIGHT_DECAY

    # load model
    if resume_iter:
      self.start_step = resume_iter + 1
      self.load_check_point(resume_iter)
    else:
      self.start_step = 0
      self.weights_normal_init(self.net, dev=0.01)
      # refer to caffe faster RCNN
      self.net.init_special_bbox_fc(dev=0.001)
      if self.pretrained_model != None:
        self.net._rpn._network._load_pre_trained_model(self.pretrained_model)
        print('Load parameters from Path: {}'.format(self.pretrained_model))
      else:
        pass

    # model
    self.net.train()
    if cfg.CUDA_IF:
      self.net.cuda()

    # resnet fixed BN should be eval
    if cfg.TRAIN.INIT_WAY == 'resnet':
      self.net._rpn._network._bn_eval()

    # set optimizer
    self._parameters = [params for params in self.net.parameters() if params.requires_grad==True]
    self._optimizer = self._train_optimizer()

    # tensorboard
    if self._use_tensorboard:
      import tensorboardX as tbx
      self._tensor_writer = tbx.SummaryWriter(log_dir=self.tbdir)

  def _train_optimizer(self):
    parameters = self._train_parameter()
    optimizer = torch.optim.SGD(parameters, momentum=self._momentum)
    return optimizer

  def _train_parameter(self):
    params = []
    for key, value in self.net.named_parameters():
      if value.requires_grad == True:
        if 'bias' in key:
          params += [{'params': [value],
                      'lr': self._lr * (cfg.TRAIN.DOUBLE_BIAS+1),
                      'weight_decay': 0}]
        else:
          params += [{'params': [value],
                      'lr': self._lr ,
                      'weight_decay': self._weight_decay}]
    return params



def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb

def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, valroidb, model_dir,
              learn_dict, resume,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  sw = SolverWrapper(network, imdb, roidb, valroidb, model_dir,
                     pretrained_model=pretrained_model)
  sw.set_learn_strategy(learn_dict)

  sw.train_model(resume, max_iters)



