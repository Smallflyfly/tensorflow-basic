#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：fangpf
@Date    ：2021/6/7 16:20 
'''

import tensorflow as tf


def conv3x3(out_pales, stride=1, groups=1):
    return tf.keras.layers.Conv2D(out_pales, kernel_size=(3, 3), stride=stride, use_bias=False)


def conv1x1(out_planes, stride=1):
    return tf.keras.layers.Conv2D(out_planes, kernel_size=(3, 3), stride=stride, use_bias=False)


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, normal_layer=None):
        super(BasicBlock, self).__init__()
        if normal_layer is None:
            normal_layer = tf.keras.layers.BatchNormalization()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only support group=1 and base_width=64')
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = normal_layer
        self.relu = tf.nn.relu()
        self.conv2 = conv3x3(planes)
        self.bn2 = normal_layer
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=None, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = tf.keras.layers.add([out, identity])
        out = self.relu(out)

        return out


class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self._normal_layer = tf.keras.layers.BatchNormalization()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn1 = self._normal_layer
        self.relu = tf.keras.layers.ReLU()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer1 = self.make_layer_(block, 64, layers[0])
        self.layer2 = self.make_layer_(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer_(block, 256, layers[2],stride=2)
        self.layer4 = self.make_layer_(block, 512, layers[3], stride=2)
        self.avgpool = tf.nn.avg_pool_v2()

    def make_layer_(self, block, planes, blocks, stride=1):
        normal_layer = self._normal_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(planes * block.expansion, stride),
                normal_layer
            ])
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                        normal_layer=normal_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                normal_layer=normal_layer))
        return tf.keras.Sequential(layers)
