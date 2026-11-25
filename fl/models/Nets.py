#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
## Add softmax function ##

import torch
from torch import nn
import torch.nn.functional as F
# from thop import profile
import math
# from .quantization import *


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(1 if args.dataset == 'fmnist' else 3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNGate(nn.Module):
    def __init__(self, args):
        super(CNNGate, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1 if args.dataset == 'fmnist' else 3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.args.num_classes)

        for p in self.parameters():
            p.requires_grad = False

        self.gate = nn.Linear(32 * 32 * (3 if args.dataset == 'cifar' else 1), 1) if args.struct else nn.Linear(
            16 * 5 * 5, 1)
        self.pfc1 = nn.Linear(16 * 5 * 5, 120)
        self.pfc2 = nn.Linear(120, 84)
        self.pfc3 = nn.Linear(84, args.num_classes)

    def forward(self, x1):
        x = F.max_pool2d(F.relu(self.conv1(x1)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)

        z = F.relu(self.pfc1(x))
        z = F.relu(self.pfc2(z))
        z = self.pfc3(z)

        g = torch.sigmoid(self.gate(torch.flatten(x1, 1))) if self.args.struct else torch.sigmoid(self.gate(x))
        # g =
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y * g + z * (1 - g), g, z
        # return z


# '''
# Modified from https://github.com/pytorch/vision.git
# '''
# import math
#
# import torch.nn as nn
# import torch.nn.init as init
#
# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
#
#
# class VGG(nn.Module):
#     '''
#     VGG model
#     '''
#
#     def __init__(self, features, has_gate=False, struct=False):
#         super(VGG, self).__init__()
#         self.has_gate = has_gate
#         self.struct = struct
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10),
#         )
#         if has_gate:
#             self.pclassifier = nn.Sequential(
#                 nn.Dropout(),
#                 nn.Linear(512, 512),
#                 nn.ReLU(True),
#                 nn.Dropout(),
#                 nn.Linear(512, 512),
#                 nn.ReLU(True),
#                 nn.Linear(512, 10),
#             )
#             if self.struct:
#                 self.gate = nn.Linear(3 * 32 * 32, 1)
#             else:
#                 self.gate = nn.Linear(512, 1)
#         # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()
#
#     def forward(self, input):
#         x = self.features(input)
#         x = x.view(x.size(0), -1)
#         if self.has_gate:
#             if self.struct:
#                 g = torch.sigmoid(self.gate(torch.flatten(input, 1)))
#             else:
#                 g = torch.sigmoid(self.gate(x))
#             y = self.classifier(x)
#             z = self.pclassifier(x)
#             return y * g + z * (1-g), g, z
#         else:
#             x = self.classifier(x)
#             return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def test_gradients(self, gradient: torch.Tensor):
#         """
#         Make use of the given gradients to run a test, then revert back to the previous status
#         """
#         cache = self.get_flatten_parameters()
#         test_param = cache + gradient
#         self.load_parameters(test_param)
#         loss, acc = self.test_outcome()
#         self.load_parameters(cache)
#         return loss, acc
#
#     def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):
#         param = self.get_flatten_parameters()
#         loss = 0
#         acc = 0
#         for epoch in range(local_epoch):
#             batch_idx = 0
#             while batch_idx * batch_size < X.size(0):
#                 lower = batch_idx * batch_size
#                 upper = lower + batch_size
#                 X_b = X[lower: upper]
#                 y_b = y[lower: upper]
#                 self.optimizer.zero_grad()
#                 out = self.forward(X_b)
#                 loss_b = self.loss(out, y_b)
#                 loss_b.backward()
#                 self.optimizer.step()
#                 loss += loss_b.item()
#                 pred_y = torch.max(out, dim=1).indices
#                 acc += torch.sum(pred_y == y_b).item()
#                 batch_idx += 1
#         grad = self.get_flatten_parameters() - param
#         loss /= local_epoch
#         acc = acc / (local_epoch * X.size(0))
#         if revert:
#             self.load_parameters(param)
#         return acc, loss, grad
#
#     def get_flatten_parameters(self):
#         """
#         Return the flatten parameter of the current model
#         :return: the flatten parameters as tensor
#         """
#         out = torch.zeros(0, device=next(self.parameters()).device)
#         with torch.no_grad():
#             for parameter in self.parameters():
#                 out = torch.cat([out, parameter.flatten()])
#         return out
#         # out = torch.zeros(0)
#         # with torch.no_grad():
#         #     for parameter in self.parameters():
#         #         out = torch.cat([out, parameter.flatten()])
#         # return out
#
#     def load_parameters(self, parameters: torch.Tensor, mask=None):
#         """
#         Load parameters to the current model using the given flatten parameters
#         :param mask: only the masked value will be loaded
#         :param parameters: The flatten parameter to load
#         :return: None
#         """
#         start_index = 0
#         for param in self.parameters():
#             with torch.no_grad():
#                 length = len(param.flatten())
#                 to_load = parameters[start_index: start_index + length]
#                 to_load = to_load.reshape(param.size())
#                 if mask is not None:
#                     local_mask = mask[start_index: start_index + length]
#                     local_mask = local_mask.reshape(param.size())
#                     param[local_mask] = to_load[local_mask]
#                 else:
#                     param.copy_(to_load)
#                 start_index += length
#
#
#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
#           512, 512, 512, 512, 'M'],
# }
#
#
# def vgg11():
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg['A']))
#
#
# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg['A'], batch_norm=True))
#
#
# def vgg13():
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg['B']))
#
#
# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True))
#
#
# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg['D']))
#
#
# def gate_vgg16(args):
#     return VGG(make_layers(cfg['D']), has_gate=True, struct=args.struct)
#
#
# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg['D'], batch_norm=True))
#
#
# def vgg19():
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg['E']))
#
#
# def vgg19_bn():
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg['E'], batch_norm=True))


def cifar_test():
    # from utils.options import args_parser
    net = vgg16().to("cuda:0")
    img = torch.randn(100, 3, 32, 32).to("cuda:0")
    flops, params = profile(net, inputs=(img,))
    print(flops, params)
    print(img.size())


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = quan_Conv2d(inplanes,
                                  planes,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = quan_Conv2d(planes,
                                  planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, block, depth, num_classes):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(
            depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = quan_Conv2d(3,
                                      16,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = quan_Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion,
                                     stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnet20_quan(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


# test()
# cifar_test()

'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', ]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        """
        cache = self.get_flatten_parameters()
        test_param = cache + gradient
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc

    def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):
        param = self.get_flatten_parameters()
        loss = 0
        acc = 0
        for epoch in range(local_epoch):
            batch_idx = 0
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size
                X_b = X[lower: upper]
                y_b = y[lower: upper]
                self.optimizer.zero_grad()
                out = self.forward(X_b)
                loss_b = self.loss(out, y_b)
                loss_b.backward()
                self.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()
                batch_idx += 1
        grad = self.get_flatten_parameters() - param
        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))
        if revert:
            self.load_parameters(param)
        return acc, loss, grad

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0, device=next(self.parameters()).device)
        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out
        # out = torch.zeros(0)
        # with torch.no_grad():
        #     for parameter in self.parameters():
        #         out = torch.cat([out, parameter.flatten()])
        # return out

    def load_parameters(self, parameters: torch.Tensor, mask=None):
        """
        Load parameters to the current model using the given flatten parameters
        :param mask: only the masked value will be loaded
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                if mask is not None:
                    local_mask = mask[start_index: start_index + length]
                    local_mask = local_mask.reshape(param.size())
                    param[local_mask] = to_load[local_mask]
                else:
                    param.copy_(to_load)
                start_index += length


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


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # Add back_prop method
    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        :param gradient: the gradient to apply
        :return: the loss and accuracy of the test
        """
        cache = self.get_flatten_parameters()
        test_param = cache + gradient
        self.load_parameters(test_param)
        loss, acc, _ = self.back_prop()
        self.load_parameters(cache)
        return loss, acc

    def back_prop(self, X, y, batch_size, local_epoch, revert=False):
        # [Function content is the same as you provided]
        param = self.get_flatten_parameters()
        loss = 0
        acc = 0
        for epoch in range(local_epoch):
            batch_idx = 0
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size
                X_b = X[lower: upper]
                y_b = y[lower: upper]
                self.optimizer.zero_grad()
                out = self.forward(X_b)
                loss_b = self.loss(out, y_b)
                loss_b.backward()
                self.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()
                batch_idx += 1
        grad = self.get_flatten_parameters() - param
        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))
        if revert:
            self.load_parameters(param)
        return acc, loss, grad

    # Add get_flatten_parameters method
    def get_flatten_parameters(self):
        # [Function content is the same as you provided]
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0, device=next(self.parameters()).device)
        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out

    # Add load_parameters method
    def load_parameters(self, parameters, mask=None):
        # [Function content is the same as you provided]
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                if mask is not None:
                    local_mask = mask[start_index: start_index + length]
                    local_mask = local_mask.reshape(param.size())
                    param[local_mask] = to_load[local_mask]
                else:
                    param.copy_(to_load)
                start_index += length


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
