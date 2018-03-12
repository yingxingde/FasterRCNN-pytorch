# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lib.model.config import cfg
from lib.nets.network import Network
from lib.nets.layers_util import *


class VGG(nn.Module):
  def __init__(self, features, num_classes=1000):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      # nn.Linear(4096, num_classes),
    )

def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)

vgg_cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}


class VGG16(Network):
  def __init__(self, feat_strdie=(16, ), anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    Network.__init__(self)
    self._channels['head'] = 512
    self._channels['tail'] = 4096
    self._feat_stride = feat_strdie
    self._anchor_scales = anchor_scales
    self._anchor_ratios = anchor_ratios
    self._num_anchors = len(anchor_scales)*len(anchor_ratios)
    # self._init_network()

  def _init_network(self, bn=False):
    self._vgg = VGG(make_layers(vgg_cfg['D']))

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self._vgg.features[layer].parameters(): p.requires_grad = False

    self._layers['head'] = self._vgg.features

    self._layers['tail'] = self._vgg.classifier

  def _image_to_head(self, input):
    return self._layers['head'](input)


  def _head_to_tail(self, pool5):
    x = pool5.view(pool5.size()[0], -1)
    return self._layers['tail'](x)

  def _load_pre_trained_model(self, pre_trained_model):
    pre_model = torch.load(pre_trained_model)
    state_dict = self._vgg.state_dict()
    pre_model_dict = {k: v for k, v in pre_model.items() if k in state_dict}
    # # imagenet pre-trained modal is RGB order, but this project is BGR order
    for key in state_dict.keys():
      if 'classifier' in key:
        key_split = key.split('.')
        new_key = [key_split[0]]+[str(int(key_split[1])+1)]+[key_split[2]]
        pre_model_dict[key] = pre_model['.'.join(new_key)]
      else:
        pass
    state_dict.update(pre_model_dict)
    self._vgg.load_state_dict(pre_model_dict)



if __name__ == '__main__':
  pre_trained_model = '../../data/pretrained_model/vgg16_caffe.pth'
  pre_model = torch.load(pre_trained_model)
  for key in pre_model.keys():
    print(key)
  #   if 'features.0.weight' in key:
  #     print(key)
  #     print(pre_model[key][0,0:5,0,0])
  #     print(flip(pre_model[key], 1)[0,-5:, 0, 0])
  #     print(pre_model[key][0,0:5,0,0])
  vgg = VGG16()
  vgg._init_network()
  for k in vgg.state_dict().keys():
    print(k)
  # for layer in range(10):
  #   print(vgg._vgg.features[layer])