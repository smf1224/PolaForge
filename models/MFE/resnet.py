import math
import torch
import torch.nn as nn
from ..RGBC import RGBConv

__all__ = ['ResNet', 'resnet50', 'BasicBlock', 'Bottleneck']


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
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


class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        # self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv1 = RGBConv(in_channels=inplanes, out_channels=midplanes, mid_channels=midplanes // 4,
                              kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        # self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv2 = RGBConv(in_channels=inplanes, out_channels=midplanes, mid_channels=midplanes // 4,
                              kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        # x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        # x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None, spm_on=False):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = RGBConv(in_channels=inplanes, out_channels=planes, mid_channels=planes // 4, kernel_size=1,
                              bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        # self.conv2 = RGBConv(in_channels=planes, out_channels=planes, mid_channels=planes//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = RGBConv(in_channels=planes, out_channels=planes * 4, mid_channels=planes // 4, kernel_size=1,
                              bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.spm = None
        if spm_on:
            self.spm = SPBlock(planes, planes, norm_layer=norm_layer)

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.spm is not None:
            out = out * self.spm(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dilated=False, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d, spm_on=False, att_on=False):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        self.spm_on = spm_on
        if deep_base:
            self.conv1 = nn.Sequential(
                # nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                RGBConv(in_channels=3, out_channels=64, mid_channels=16, kernel_size=3, stride=2, padding=1,
                         bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                RGBConv(in_channels=64, out_channels=64, mid_channels=16, kernel_size=3, stride=1, padding=1,
                         bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                RGBConv(in_channels=64, out_channels=128, mid_channels=16, kernel_size=3, stride=1, padding=1,
                         bias=False),
            )

        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 256, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True, att_on=True)
            else:
                self.layer4 = self._make_layer(block, 256, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer, att_on=True)
        else:
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                           norm_layer=norm_layer, att_on=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False, att_on=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                RGBConv(in_channels=self.inplanes, out_channels=planes * block.expansion, mid_channels=planes,
                         kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        spm_on = False
        if planes == 512:
            spm_on = self.spm_on
        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                spm_on=spm_on))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=4,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer,
                                spm_on=spm_on))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i >= blocks - 1 or planes == 512:
                spm_on = self.spm_on
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer, spm_on=spm_on))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer, spm_on=spm_on))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(x)
        return c1, c2, c3, c4


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


