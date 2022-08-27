'''adversarial example generators'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import weights_init


class BasicAG(nn.Module):
    '''generate perturbation given clean inputs'''

    def __init__(self, ngf=16):
        super(BasicAG, self).__init__()
        self.ngf = ngf
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(ngf),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(ngf, ngf * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )  # output shape=[batch_size, ngf*4, 4, 4]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=2,
                               padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, return_feat=False):
        o = self.encoder(x)
        if return_feat:
            return o, self.decoder(o)
        else:
            return self.decoder(o)


class ConvAG(nn.Module):
    ''' adversarial generator with full conv layers
        adapted from https://github.com/TransEmbedBA/TREMBA/blob/master/FCN.py
    '''
    def __init__(self, ngf=16):
        super(ConvAG, self).__init__()
        self.ngf = ngf
        self.encoder = nn.Sequential(
            # conv1_1
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf),
            # conv1_2
            nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 2),
            nn.MaxPool2d(kernel_size=2),
            # conv2_1
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # conv2_2
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.MaxPool2d(kernel_size=2),
            # conv3_1
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # conv3_2
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 8),
            nn.MaxPool2d(kernel_size=2),
            # conv4_1
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # conv4_2
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf // 2),
            nn.MaxPool2d(kernel_size=2),
            )
        self.decoder = nn.Sequential(
            # deconv1_1
            nn.Conv2d(in_channels=ngf // 2, out_channels=ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # deconv1_2
            nn.ConvTranspose2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # deconv2_1
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # deconv2_2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # deconv3_1
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 8),
            # deconv3_2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # deconv4_1
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # deconv4_2
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ngf),
            # fcn
            nn.Conv2d(ngf, 3, kernel_size=1)
            )

    def forward(self, x, return_feat=False):
        o = self.encode(x)
        if return_feat:
            return o, self.decode(o)
        else:
            return self.decode(o)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x)) # range [-1,1]


# Define a resnet block
# modified from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResAdvGenerator(nn.Module):
    '''Adversarial generator with residual blocks'''

    def __init__(self, nc=3):
        super(ResAdvGenerator, self).__init__()

        encoder_lis = [
            # MNIST:3*32*32
            nn.Conv2d(nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*30*30
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*14*14
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*6*6
        ]

        bottle_neck_lis = [ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32), ]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                               stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3,
                               stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, nc, kernel_size=6,
                               stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        self.apply(weights_init)

    def forward(self, x, return_feat=False):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        o = self.decoder(x)
        if return_feat:
            return x, o
        else:
            return o


# generator with residual blocks from cross-domain-perturbation project
# from https://github.com/Muzammal-Naseer/Cross-domain-perturbations/blob/master/generators.py

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

class ResG(nn.Module):
    def __init__(self, ngf=64, inception=False, data_dim='low'):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(ResG, self).__init__()
        self.ngf = ngf
        self.inception = inception
        self.data_dim = data_dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = ngf, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = ngf, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        # Input size = 2*ngf, n/4, n/4
        self.block4 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 4*ngf, n/8, n/8
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 4*ngf, n/8, n/8
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 2*ngf, n/4, n/4
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = ngf, n/2, n/2
        self.upsampl3 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, x, return_feat=False):
        o = self.encode(x)
        x = self.decode(o)
        if return_feat:
            return o, x
        else:
            return x

    def encode(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        return x

    def decode(self, x):
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.upsampl3(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        x = (torch.tanh(x) + 1) / 2 # Output range [0 1]
        return x
    