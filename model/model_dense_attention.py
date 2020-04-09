# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:58:59 2020

@author: 吕
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models


class pix_attention(nn.Module):
    def __init__(self, channel):
        super(pix_attention, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class channel_attention(nn.Module):
    def __init__(self, channel):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,end=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.channel_attention=channel_attention(growth_rate)
        self.end=end

    def forward(self, x):
        out = super(_DenseLayer, self).forward(x)
        out = self.channel_attention(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        if(self.end):
            return out
        else:
            return torch.cat([x, out], 1)



class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers-1):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        layer = _DenseLayer(num_layers * growth_rate, growth_rate, bn_size, drop_rate,True)
        self.add_module('denselayer%d' % (i + 2), layer)
            

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        
class _Detransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Detransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1 ,bias=False))
        self.add_module('transpose', nn.Upsample(scale_factor= 2 ))  
        
class _conv(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_conv,self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_input_features,
                                          kernel_size=3, stride=1, bias=False,padding=1))
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class net(nn.Module):
    def __init__(self,num_layers=4,bn_size=4,growth_rate=16,drop_rate=0.5):
        super(net,self).__init__()
        self.conv=_conv(3,growth_rate)
        self.transition1 = _Transition(growth_rate,growth_rate)
        self.Detransition1 = _Detransition(growth_rate,growth_rate)
        self.transition0 = _Transition(growth_rate,growth_rate)
        self.Detransition0 = _Detransition(growth_rate,growth_rate)
        self.dense0 = _DenseBlock(num_layers,growth_rate,bn_size,growth_rate,drop_rate)
        self.dense1 = _DenseBlock(num_layers,growth_rate,bn_size,growth_rate,drop_rate)
        self.dense2 = _DenseBlock(num_layers,growth_rate,bn_size,growth_rate,drop_rate)
        self.conv1 = _conv(16,3)
        self.pix_attention0=pix_attention(growth_rate)
        self.pix_attention1=pix_attention(growth_rate)
        self.num_layers=num_layers
        self.growth_rate=growth_rate

    def forward(self,x):
        out=self.conv(x)
        sub1=self.transition1(out)
        sub0=self.transition0(sub1)
        
        sub0=self.dense0(sub0)
        sub1=self.pix_attention0(self.Detransition0(sub0))+sub1
        sub1=self.dense1(sub1)
        sub2=self.pix_attention1(self.Detransition1(sub1))+out
        sub2=self.dense2(sub2)
        
        out=self.conv1(sub2)
        
        return out
        
        
        
        
        
        


