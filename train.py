# -*- codingL utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.train_val import get_training_roidb, train_net
from lib.model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_model_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
from lib.nets.vgg16 import VGG16
from lib.nets.resnet import Resnet
from lib.nets.network import FasterRCNN
from lib.nets.network_fpn import FasterRCNN as FPN
from lib.nets.fpn import FPN_Resnet

import argparse
import pprint
import numpy as np
import sys
import os

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='./', type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      default='./',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--resume', dest='resume',
                      help='resume checkpoint',
                      default=None, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    # print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = lib.datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  # args.max_iters = 100000
  # args.tag = 'vgg16_3'
  # args.resume = 80000
  # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # train set
  imdb, roidb = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # # output directory where the models are saved
  # output_dir = get_output_dir(imdb, args.tag)
  # output_dir = os.path.join(output_dir, cfg.TRAIN.SNAPSHOT_PREFIX)
  # print('Output will be saved to `{:s}`'.format(output_dir))

  # model directory where the summaries are saved during training
  model_dir = get_output_model_dir(imdb, args.tag)
  model_dir = os.path.join(model_dir, cfg.TRAIN.SNAPSHOT_PREFIX)
  print('Model will be saved to `{:s}`'.format(model_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = FasterRCNN(VGG16(feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'vgg'
  # elif args.net == 'res18':
  #   net = FasterRCNN(Resnet(resnet_type=18, feat_strdie=(16,),
  #                          anchor_scales=cfg.ANCHOR_SCALES,
  #                          anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
  #   cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'res50':
    net = FasterRCNN(Resnet(resnet_type=50, feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'res101':
    net = FasterRCNN(Resnet(resnet_type=101, feat_strdie=(16,),
                           anchor_scales=cfg.ANCHOR_SCALES,
                           anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'fpn50':
    net = FPN(FPN_Resnet(resnet_type=50, feat_strdie=(4, 8, 16, 32, 64),
                         anchor_scales=cfg.ANCHOR_SCALES,
                         anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  elif args.net == 'fpn101':
    net = FPN(FPN_Resnet(resnet_type=101, feat_strdie=(4, 8, 16, 32, 64),
                         anchor_scales=cfg.ANCHOR_SCALES,
                         anchor_ratios=cfg.ANCHOR_RATIOS), imdb.classes)
    cfg.TRAIN.INIT_WAY = 'resnet'
  else:
    raise NotImplementedError

  learn_dict = {
    'disp_interval': cfg.TRAIN.DISPLAY,
    'use_tensorboard': True,
    'use_valid': True,
    'save_point_interval': cfg.TRAIN.SAVE_POINT_INTERVAL,
    'lr_decay_steps': cfg.TRAIN.STEPSIZE
  }
  resume = args.resume

  train_net(net, imdb, roidb, valroidb, model_dir, learn_dict, resume,
            pretrained_model=args.weight, max_iters=args.max_iters)

