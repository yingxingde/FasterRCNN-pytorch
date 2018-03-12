# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from lib.nets.layers_util import *
from lib.nets.network import Network
from lib.model.config import cfg


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Resnet_Ori(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(Resnet_Ori, self).__init__()
    self.block = block
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    # self.avgpool = nn.AvgPool2d(7)
    # self.fc = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

class Resnet(Network):
  def __init__(self, resnet_type, feat_strdie=(16, ), anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    Network.__init__(self)
    self._resnet_type = resnet_type
    self._channels['head'] = None
    self._channels['tail'] = None
    self._feat_stride = feat_strdie
    self._anchor_scales = anchor_scales
    self._anchor_ratios = anchor_ratios
    self._num_anchors = len(anchor_scales) * len(anchor_ratios)

  def _init_network(self):
    if self._resnet_type == 18:
      layers = [2, 2, 2, 2]
      self._resnet = Resnet_Ori(BasicBlock, layers)
    elif self._resnet_type == 34:
      layers = [3, 4, 6, 3]
      self._resnet = Resnet_Ori(BasicBlock, layers)
    elif self._resnet_type == 50:
      layers = [3, 4, 6, 3]
      self._resnet = Resnet_Ori(Bottleneck, layers)
    elif self._resnet_type == 101:
      layers = [3, 4, 23, 3]
      self._resnet = Resnet_Ori(Bottleneck, layers)
    else:
      raise NotImplementedError

    set_trainable(self._resnet.conv1, False)
    set_trainable(self._resnet.bn1, False)
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      set_trainable(self._resnet.layer3, False)
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      set_trainable(self._resnet.layer2, False)
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      set_trainable(self._resnet.layer1, False)

    self._layers['head'] = nn.Sequential(self._resnet.conv1,
                                         self._resnet.bn1,
                                         self._resnet.relu,
                                         self._resnet.maxpool,
                                         self._resnet.layer1,
                                         self._resnet.layer2,
                                         self._resnet.layer3)
    self._layers['tail'] = nn.Sequential(self._resnet.layer4)
    self._channels['head'] = self._resnet.block.expansion*256
    self._channels['tail'] = self._resnet.block.expansion*512

  def _bn_eval(self):
    set_BN_eval(self._resnet.bn1, True)
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    set_BN_eval(self._resnet.layer4, True)
    set_BN_eval(self._resnet.layer3, True)
    set_BN_eval(self._resnet.layer2, True)
    set_BN_eval(self._resnet.layer1, True)
    # if cfg.RESNET.FIXED_BLOCKS >= 3:
    #   set_BN_eval(self._resnet.layer3, True)
    # if cfg.RESNET.FIXED_BLOCKS >= 2:
    #   set_BN_eval(self._resnet.layer2, True)
    # if cfg.RESNET.FIXED_BLOCKS >= 1:
    #   set_BN_eval(self._resnet.layer1, True)

  def _image_to_head(self, input):
    return self._layers['head'](input)

  def _head_to_tail(self, pool5):
    x = self._layers['tail'](pool5)
    x = x.mean(3).mean(2)
    x = x.view(x.size()[0], -1)
    return x

  def _load_pre_trained_model(self, pre_trained_model):
    pre_model = torch.load(pre_trained_model)
    state_dict = self._resnet.state_dict()
    pre_model_dict = {k: v for k, v in pre_model.items() if k in state_dict}
    print('Load keys/Model.State_dict keys: {} / {}'.format(len(pre_model_dict.keys()), len(state_dict)))
    # #imagenet pre-trained modal is RGB order, but this project is BGR order
    # for key in pre_model_dict.keys():
    #   if 'conv1.weight' == key:
    #     # weigth = [out_channels, in_channels, kernel, kernel]
    #     pre_model_dict[key] = flip(pre_model_dict[key], 1)
    #     print('CONV1 RGB to BGR Success')
    #   else:
    #     pass
    state_dict.update(pre_model_dict)
    self._resnet.load_state_dict(pre_model_dict)

    # for k, v in state_dict.items():
    #   if isinstance(pre_model[k], torch.FloatTensor):
    #     if 'conv1.weight' == k:
    #       v.copy_(flip(pre_model[k], 1))
    #       print('CONV1 RGB to BGR Success')
    #     else:
    #       v.copy_(pre_model[k])
    #   elif isinstance(pre_model[k], torch.nn.Parameter):
    #     if 'conv1.weight' == k:
    #       v.copy_(flip(pre_model[k].data, 1))
    #       print('CONV1 RGB to BGR Success')
    #     else:
    #       v.copy_(pre_model[k].data)
    #   else:
    #     raise ValueError

if __name__ == '__main__':
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '7'
  pre_model = torch.load('/home/yxd/projects/cervix/FasterRCNN_torch/data/pretrained_model/resnet101_caffe.pth')
  res = Resnet(resnet_type=101)
  res._init_network()
  # for k, v in res._resnet.state_dict().items():
  #   print(k, v.size())
  # print(len(res._resnet.state_dict()))
  res._load_pre_trained_model('/home/yxd/projects/cervix/FasterRCNN_torch/data/pretrained_model/resnet101_caffe.pth')








