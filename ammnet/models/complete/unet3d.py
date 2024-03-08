import torch
from torch import nn
import torch.nn.functional as F
from ..build import MODELS
import numpy as np
import torchvision.transforms.functional as TF


class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


@MODELS.register_module()
class Unet3d(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm3d, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained=None, freeze=False):
        super(Unet3d, self).__init__()

        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)

        self.pre_layer_R11 = nn.Conv3d(feature, feature, 1)
        self.pre_layer_R12 = nn.Conv3d(feature, feature, 1)

        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )

        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )

        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, num_classes, kernel_size=1, bias=True)
            )]
        )

        self.DownScale = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.UpScale = nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            )

        self.pre_layer_R21 = nn.Conv3d(feature, feature, 1)
        self.pre_layer_R22 = nn.Conv3d(feature, feature, 1)

        self.pre_layer_R31 = nn.Conv3d(feature, feature, 1)
        self.pre_layer_R32 = nn.Conv3d(feature, feature, 1)

    def projection(self, x, mapping2d, shape):
        if x.shape[-2:] != mapping2d.shape[-2:]:
            x = F.upsample(x, size=mapping2d.shape[-2:], mode='bilinear')
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)

        segres = x.new_zeros(b, c, *shape).flatten(2).permute(0, 2, 1).contiguous()
        for i, mapping in enumerate(mapping2d):
            segres[i][mapping[mapping != -1]] = x[i][mapping != -1]
            # [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)]
        segres = segres.permute(0, 2, 1).view(b, c, *shape).contiguous()  # B, (channel), 60, 36, 60
        return segres

    def aug3d(self, seg_fea, tsdf_feat):
        if not self.training:
            return seg_fea, tsdf_feat, None

        flip_x = np.random.rand() > 0.5
        if self.training and flip_x:
            seg_fea = torch.flip(seg_fea, dims=[2, ])
            tsdf_feat = torch.flip(tsdf_feat, dims=[2, ])

        flip_z = np.random.rand() > 0.5
        if self.training and flip_z:
            seg_fea = torch.flip(seg_fea, dims=[4, ])
            tsdf_feat = torch.flip(tsdf_feat, dims=[4, ])

        perm_xz = np.random.rand() > 0.5
        if self.training and perm_xz:
            seg_fea = torch.permute(seg_fea, (0, 1, 4, 3, 2))
            tsdf_feat = torch.permute(tsdf_feat, (0, 1, 4, 3, 2))

        aug_info = {'flip_x': flip_x, 'flip_z': flip_z, 'perm_xz': perm_xz}
        return seg_fea, tsdf_feat, aug_info

    def forward(self, feature2d, mapping2d, tsdf_feat=None, shape3d=(60, 36, 60)):
        '''
        project 2D feature to 3D space
        '''
        segres = self.projection(feature2d, mapping2d, shape=shape3d)

        upsample = False
        if tsdf_feat.shape[-3:] != shape3d:
            upsample = True
            segres = F.avg_pool3d(segres, kernel_size=(2, 1, 2), stride=(2, 1, 2))
        '''
        init the 3D feature
        '''

        if self.ThreeDinit:
            pool = self.pooling(segres)

            zero = (segres == 0).float()
            pool = pool * zero
            segres = segres + pool

        '''
        extract 3D feature
        '''

        segres, tsdf_feat, aug_info = self.aug3d(segres, tsdf_feat)

        weightR1 = self.pre_layer_R11(tsdf_feat).sigmoid()
        biasR1 = self.pre_layer_R12(tsdf_feat)
        segres = segres * (1 + weightR1) + biasR1

        semantic1 = self.semantic_layer1(segres)
        semantic2 = self.semantic_layer2(semantic1)
        tsdf_feat2 = self.DownScale(tsdf_feat)
        up_sem1 = self.classify_semantic[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        weightR2 = self.pre_layer_R21(tsdf_feat2).sigmoid()
        biasR2 = self.pre_layer_R22(tsdf_feat2)
        up_sem1 = up_sem1 * (1 + weightR2) + biasR2
        up_sem2 = self.classify_semantic[1](up_sem1)

        tsdf_feat3 = self.UpScale(tsdf_feat2)
        weightR3 = self.pre_layer_R31(tsdf_feat3).sigmoid()
        biasR3 = self.pre_layer_R32(tsdf_feat3)
        up_sem2 = up_sem2 * (1 + weightR3) + biasR3
        pred_semantic = self.classify_semantic[2](up_sem2)
        if upsample:
            pred_semantic = F.upsample(pred_semantic, size=shape3d, mode='trilinear')
        return pred_semantic, aug_info