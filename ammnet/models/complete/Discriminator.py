import torch
from .unet3d import Bottleneck3D
from torch import nn
from ..build import MODELS


@MODELS.register_module()
class Discriminator(nn.Module):
    def __init__(self, num_classes, plane=64, norm_layer=nn.BatchNorm3d, bn_momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(num_classes, plane, kernel_size=1, bias=False)
        self.layer1 = Bottleneck3D(plane, plane // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(plane, plane,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(plane, momentum=bn_momentum),
            ), norm_layer=norm_layer)

        self.layer2 = Bottleneck3D(plane, plane // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(plane, plane * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(plane * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer)

        self.layer3 = Bottleneck3D(plane * 2, plane // 2, bn_momentum=bn_momentum, stride=3, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=3, stride=3),
                nn.Conv3d(plane * 2, plane * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(plane * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer)

        self.layer4 = Bottleneck3D(plane * 2, plane // 2, bn_momentum=bn_momentum, expansion=4,
                                   norm_layer=norm_layer, dilation=[1, 1, 1])
        self.linear = nn.Linear(plane * 2 * 5 * 3 * 5, plane * 2)
        self.logit = nn.Linear(plane * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(1)
        x = self.relu(self.linear(x))
        x = self.logit(x)
        return x