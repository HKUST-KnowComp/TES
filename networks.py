'''base classifiers'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import weights_init


# For SVHN dataset
class DTN(nn.Module):
    '''
    named_params: 
    ['conv_params.0.weight',
     'conv_params.0.bias',
     'conv_params.1.weight',
     'conv_params.1.bias',
     'conv_params.4.weight',
     'conv_params.4.bias',
     'conv_params.5.weight',
     'conv_params.5.bias',
     'conv_params.8.weight',
     'conv_params.8.bias',
     'conv_params.9.weight',
     'conv_params.9.bias',
     'fc_params.0.weight',
     'fc_params.0.bias',
     'fc_params.1.weight',
     'fc_params.1.bias',
     'classifier.weight',
     'classifier.bias']
    '''

    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def get_output_num(self):
        return self.__in_features

    def forward(self, x, return_feat=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        logits = self.classifier(x)
        if return_feat:
            return x, logits
        else:
            return logits

    def extract_features(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x

    def get_parameters(self, init_lr):
        parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list

    def restore_from_ckpt(self, ckpt_state_dict, exclude_vars):
        '''restore entire model from another model
        Args:
            exclude_vars: a list of string, prefixes of variables to be excluded
        '''
        model_dict = self.state_dict()

        # Fiter out unneccessary keys
        print('restore variables: ')
        filtered_dict = {}
        for n, v in ckpt_state_dict.items():
            if len(exclude_vars) == 0:
                prefix_match = [0]
            else:
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
            if sum(prefix_match) == 0 and v.size() == model_dict[n].size():
                print(n)
                filtered_dict[n] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        # frozen restored params
        print('freeze variables: ')
        if len(exclude_vars) > 0:
            for n, v in self.named_parameters():
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
                # if n is not found in exclude_vars, freeze it
                if sum(prefix_match) == 0:
                    v.requires_grad = False


# blocks for wide-res network
class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor,
                     32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[
                                   1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[
                                   2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[
                                   3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_output_num(self):
        return self.nChannels

    def forward(self, x, return_feat=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        logits = self.fc(out)
        if return_feat:
            return out, logits
        else:
            return logits

    def get_parameters(self, init_lr):
        parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list

    def restore_from_ckpt(self, ckpt_state_dict, exclude_vars):
        '''restore entire model from another model'''
        model_dict = self.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {}
        for n, v in ckpt_state_dict.items():
            if len(exclude_vars) == 0:
                prefix_match = [0]
            else:
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
            if sum(prefix_match) == 0 and v.size() == model_dict[n].size():
                print(n)
                filtered_dict[n] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        # frozen restored params
        print('freeze variables: ')
        if len(exclude_vars) > 0:
            for n, v in self.named_parameters():
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
                # if n is not found in exclude_vars and n is not a var in bn
                # layer, freeze it
                if sum(prefix_match) == 0 and 'bn' not in n:
                    print(n)
                    v.requires_grad = False


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34,
               "ResNet50": models.resnet50, "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNetFc(nn.Module):

    def __init__(self, resnet_name, use_bottleneck=False, bottleneck_dim=256, 
                 new_cls=False, class_num=1000, imagenet_pretrain=True, normalize_op=None):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=imagenet_pretrain)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
        self.normalize_op = normalize_op
    
    def forward(self, x, return_feat=False):
        if self.normalize_op is not None:
            x = self.normalize_op(x)
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        if return_feat:
            return x, y
        else:
            return y

    def get_output_num(self):
        return self.__in_features

    def get_parameters(self, init_lr):
        if self.new_cls:
            parameter_list = [{"params": self.fc.parameters(), "lr": init_lr * 1.}] + \
                             [{"params": self.feature_layers.parameters(), "lr": init_lr}]
            if self.use_bottleneck:
                parameter_list += [{"params": self.bottleneck.parameters(), "lr": init_lr * 1.}]
        else:
            parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn, "VGG19BN": models.vgg19_bn}


class VGGFc(nn.Module):

    def __init__(self, vgg_name, use_bottleneck=False, bottleneck_dim=256, 
                 new_cls=False, class_num=1000, imagenet_pretrain=True, normalize_op=None):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=imagenet_pretrain)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier" + str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(4096, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(4096, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = 4096
        else:
            self.fc = model_vgg.classifier[6]
            self.__in_features = 4096
        self.normalize_op = normalize_op

    def forward(self, x, return_feat=False):
        if self.normalize_op is not None:
            x = self.normalize_op(x)
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        if return_feat:
            return x, y
        else:
            return y

    def get_output_num(self):
        return self.__in_features

    def get_parameters(self, init_lr):
        if self.new_cls:
            parameter_list = [{"params": self.fc.parameters(), "lr": init_lr * 1.}] + \
                             [{"params": self.feature_layers.parameters(), "lr": init_lr}]
            if self.use_bottleneck:
                parameter_list += [{"params": self.bottleneck.parameters(), "lr": init_lr * 1.}]
        else:
            parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list

    def restore_from_ckpt(self, ckpt_state_dict, exclude_vars):
        '''restore entire model from another model
        Args:
            exclude_vars: a list of string, prefixes of variables to be excluded
        '''
        model_dict = self.state_dict()

        # Fiter out unneccessary keys
        print('restore variables: ')
        filtered_dict = {}
        for n, v in ckpt_state_dict.items():
            if len(exclude_vars) == 0:
                prefix_match = [0]
            else:
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
            if sum(prefix_match) == 0 and v.size() == model_dict[n].size():
                print(n)
                filtered_dict[n] = v
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

        # frozen restored params
        print('freeze variables: ')
        if len(exclude_vars) > 0:
            for n, v in self.named_parameters():
                prefix_match = [1 if n.startswith(
                    vn) else 0 for vn in exclude_vars]
                # if n is not found in exclude_vars, freeze it
                if sum(prefix_match) == 0:
                    v.requires_grad = False


class MobileNetFc(nn.Module):

    def __init__(self, use_bottleneck=False, bottleneck_dim=256, 
                 new_cls=False, class_num=1000, imagenet_pretrain=True, normalize_op=None):
        super(MobileNetFc, self).__init__()
        model_mobilenet = models.mobilenet_v2(imagenet_pretrain)
        self.feature_layers = model_mobilenet.features
        self.dropout = nn.Dropout(0.2)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_mobilenet.classifier[1].in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_mobilenet.classifier[1].in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = model_mobilenet.classifier[1].in_features
        else:
            self.fc = model_mobilenet.classifier[1]
            self.__in_features = model_mobilenet.classifier[1].in_features
        self.normalize_op = normalize_op
    
    def forward(self, x, return_feat=False):
        if self.normalize_op is not None:
            x = self.normalize_op(x)
        x = self.feature_layers(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.dropout(x)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        if return_feat:
            return x, y
        else:
            return y

    def get_output_num(self):
        return self.__in_features

    def get_parameters(self, init_lr):
        if self.new_cls:
            parameter_list = [{"params": self.fc.parameters(), "lr": init_lr * 1.}] + \
                             [{"params": self.feature_layers.parameters(), "lr": init_lr}]
            if self.use_bottleneck:
                parameter_list += [{"params": self.bottleneck.parameters(), "lr": init_lr * 1.}]
        else:
            parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list


densenet_dict = {"densenet121": models.densenet121, "densenet169": models.densenet169,
                 "densenet161": models.densenet161, "densenet201": models.densenet201}


class DenseNetFc(nn.Module):

    def __init__(self, densenet_name, use_bottleneck=False, bottleneck_dim=256, 
                 new_cls=False, class_num=1000, imagenet_pretrain=True, normalize_op=None):
        super(DenseNetFc, self).__init__()
        model_densenet = densenet_dict[densenet_name](pretrained=imagenet_pretrain)
        self.feature_layers = model_densenet.features

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_densenet.classifier.in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_densenet.classifier.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = model_densenet.classifier.in_features
        else:
            self.fc = model_densenet.classifier
            self.__in_features = model_densenet.classifier.in_features
        self.normalize_op = normalize_op
    
    def forward(self, x, return_feat=False):
        if self.normalize_op is not None:
            x = self.normalize_op(x)
        x = self.feature_layers(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        if return_feat:
            return x, y
        else:
            return y

    def get_output_num(self):
        return self.__in_features

    def get_parameters(self, init_lr):
        if self.new_cls:
            parameter_list = [{"params": self.fc.parameters(), "lr": init_lr * 10.}] + \
                             [{"params": self.feature_layers.parameters(), "lr": init_lr}]
            if self.use_bottleneck:
                parameter_list += [{"params": self.bottleneck.parameters(), "lr": init_lr * 10.}]
        else:
            parameter_list = [{"params": self.parameters(), "lr": init_lr}]
        return parameter_list


class Discriminator(nn.Module):
    '''discriminator of AdvGAN'''

    def __init__(self, image_nc=3):
        super(Discriminator, self).__init__()
        # MNIST: 1*32*32
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4,
                      stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*15*15
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 2),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RandomLayer(nn.Module):

    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(
            input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i])
                       for i in range(self.input_num)]
        return_tensor = return_list[
            0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class AdversarialNetwork(nn.Module):

    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high,
                           self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
