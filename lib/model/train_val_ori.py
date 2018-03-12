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
import PIL.Image as Image
import os
import torch
import sys
import pprint
import time
import cv2
import h5py
import torchvision.utils as vutils
import torchvision.transforms as torchtrans

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

      im_data = blobs['data']
      im_info = blobs['im_info']
      gt_boxes = blobs['gt_boxes']
      # forward
      result_cls_prob, result_bbox_pred, result_rois = net(im_data, im_info, gt_boxes)

      loss = net.loss + net._rpn.loss

      train_loss += loss.data.cpu()[0]
      rpn_cls_loss += net._rpn.cross_entropy.data.cpu()[0]
      rpn_bbox_loss += net._rpn.loss_box.data.cpu()[0]
      fast_rcnn_cls_loss += net.cross_entropy.data.cpu()[0]
      fast_rcnn_bbox_loss += net.loss_box.data.cpu()[0]
      step_cnt += 1

      # backward
      self._optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm(self._parameters, max_norm=10)
      self._optimizer.step()
      # clear middle memory
      net._delete_cache()

      if step % self._disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
          step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
        pprint.pprint(log_text)

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

      if self._use_tensorboard and step % self._valid_interval == 0:
        new_gt_boxes = gt_boxes.copy()
        new_gt_boxes[:, :4] = new_gt_boxes[:, :4]
        image = self.back_to_image(blobs['data']).astype(np.uint8)

        im_shape = image.shape
        pred_boxes, scores, classes = net.interpret_faster_rcnn_scale(result_cls_prob, result_bbox_pred, result_rois,
                                                                      im_shape, min_score=0.1)
        image = self.draw_photo(image, pred_boxes, scores, classes, new_gt_boxes)
        image = torchtrans.ToTensor()(image)
        image = vutils.make_grid([image])
        self._tensor_writer.add_image('Image', image, step)

      if self._use_valid and step % self._valid_interval == 0:
        total_valid_loss = 0.0
        valid_rpn_cls_loss = 0.0
        valid_rpn_bbox_loss = 0.0
        valid_fast_rcnn_cls_loss = 0.0
        valid_fast_rcnn_bbox_loss = 0.0
        valid_step_cnt = 0
        start_time = time.time()

        valid_length = self._disp_interval
        net.eval()
        for valid_batch in range(valid_length):
          # get one batch
          blobs = self.data_layer_val.forward()

          im_data = blobs['data']
          im_info = blobs['im_info']
          gt_boxes = blobs['gt_boxes']

          # forward
          result_cls_prob, result_bbox_pred, result_rois = net(im_data, im_info, gt_boxes)
          valid_loss = net.loss + net._rpn.loss

          total_valid_loss += valid_loss.data.cpu()[0]
          valid_rpn_cls_loss += net._rpn.cross_entropy.data.cpu()[0]
          valid_rpn_bbox_loss += net._rpn.loss_box.data.cpu()[0]
          valid_fast_rcnn_cls_loss += net.cross_entropy.data.cpu()[0]
          valid_fast_rcnn_bbox_loss += net.loss_box.data.cpu()[0]
          valid_step_cnt += 1
        net.train()
        duration = time.time() - start_time
        fps = valid_step_cnt / duration

        log_text = 'step %d, valid average loss: %.4f, fps: %.2f (%.2fs per batch)' % (
          step, total_valid_loss / valid_step_cnt, fps, 1. / fps)
        pprint.pprint(log_text)

        if self._use_tensorboard:
          self._tensor_writer.add_text('Valid', log_text, global_step=step)
          new_gt_boxes = gt_boxes.copy()
          new_gt_boxes[:, :4] = new_gt_boxes[:, :4]
          image = self.back_to_image(blobs['data']).astype(np.uint8)

          im_shape = image.shape
          pred_boxes, scores, classes = net.interpret_faster_rcnn_scale(result_cls_prob, result_bbox_pred, result_rois,
                                                                        im_shape, min_score=0.1)
          image = self.draw_photo(image, pred_boxes, scores, classes, new_gt_boxes)
          image = torchtrans.ToTensor()(image)
          image = vutils.make_grid([image])
          self._tensor_writer.add_image('Image_Valid', image, step)

        if self._use_tensorboard:
          # Valid
          avg_rpn_cls_loss_valid = valid_rpn_cls_loss / valid_step_cnt
          avg_rpn_bbox_loss_valid = valid_rpn_bbox_loss / valid_step_cnt
          avg_fast_rcnn_cls_loss_valid = valid_fast_rcnn_cls_loss / valid_step_cnt
          avg_fast_rcnn_bbox_loss_valid = valid_fast_rcnn_bbox_loss / valid_step_cnt
          real_total_loss_valid = valid_rpn_cls_loss + valid_rpn_bbox_loss + valid_fast_rcnn_cls_loss + valid_fast_rcnn_bbox_loss

          # Train
          avg_rpn_cls_loss = rpn_cls_loss / step_cnt
          avg_rpn_bbox_loss = rpn_bbox_loss / step_cnt
          avg_fast_rcnn_cls_loss = fast_rcnn_cls_loss / step_cnt
          avg_fast_rcnn_bbox_loss = fast_rcnn_bbox_loss / step_cnt
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

          self._tensor_writer.add_scalars('ValidSetLoss', {
            'RPN_cls_loss': avg_rpn_cls_loss_valid,
            'RPN_bbox_loss': avg_rpn_bbox_loss_valid,
            'FastRcnn_cls_loss': avg_fast_rcnn_cls_loss_valid,
            'FastRcnn_bbox_loss': avg_fast_rcnn_bbox_loss_valid
          }, global_step=step)

          # self._tensor_writer.add_scalars('TrainSetLoss', {
          #   'RPN_cls_loss': avg_rpn_cls_loss,
          #   'RPN_bbox_loss': avg_rpn_bbox_loss,
          #   'FastRcnn_cls_loss': avg_fast_rcnn_cls_loss,
          #   'FastRcnn_bbox_loss': avg_fast_rcnn_bbox_loss
          # }, global_step=step)
          # self._tensor_writer.add_scalar('Learning_rate', self._lr, global_step=step)

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
          gt_class = self.net._classes[det[-1]]
          cv2.rectangle(im2show, det[0:2], det[2:4], (255, 0, 0), 2)
          cv2.putText(im2show, '%s' % (gt_class), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                      1.0, (0, 0, 255), thickness=1)
      return im2show

  def back_to_image(self, img):
    image = img[0] + cfg.PIXEL_MEANS
    image = image[:,:,::-1].copy(order='C')
    return image

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



