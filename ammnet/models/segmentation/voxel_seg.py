import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS, build_model_from_cfg
import random


@MODELS.register_module()
class VoxelSSC(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 head_args=None,
                 tsdf_args=None,
                 complete_args=None,
                 with_disc=None,
                 **kwargs):
        super().__init__()
        pretrained = encoder_args.pop('pretrained')
        freeze_encoder = encoder_args.pop('freeze')
        head_pretrianed = head_args.pop('pretrained')

        self.encoder = build_model_from_cfg(encoder_args)
        self.head = build_model_from_cfg(head_args)

        self.with_tsdf = False
        if tsdf_args is not None:
            self.with_tsdf = True
            self.TSDFNet = build_model_from_cfg(tsdf_args)

        if pretrained:
            if not head_pretrianed:
                self.encoder.init_weights(pretrained)     # pretrained for encoder only
            else:
                self.load_model(pretrained)
        if freeze_encoder:
            assert pretrained, 'You freeze the encoder while not loading any pretrained weights'
            self.freeze_encoder()

        self.complete = build_model_from_cfg(complete_args)

        self.with_disc = False
        if with_disc is not None:
            self.with_disc = True
            self.disc = build_model_from_cfg(with_disc)

    def load_model(self, pretrained):
        print('load pretrained weights for both encoder and head')
        if isinstance(pretrained, str):
            print('Load Model: ' + pretrained)
            state_dict = torch.load(pretrained)['state_dict']
        else:
            state_dict = pretrained

        from collections import OrderedDict
        new_state_dict_backbone = OrderedDict()
        new_state_dict_head = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone' in k:
                name = k.replace('backbone.', '')
                new_state_dict_backbone[name] = v
            elif 'head' in k:
                name = k.replace('decode_head.', '')
                if 'linear_fuse' in name:
                    if '.bn.' in name:
                        name = name.replace('linear_fuse.bn', 'linear_fuse.1')
                    else:
                        name = name.replace('linear_fuse.conv', 'linear_fuse.0')
                new_state_dict_head[name] = v
        self.encoder.load_state_dict(new_state_dict_backbone, strict=True)
        self.head.load_state_dict(new_state_dict_head, strict=False)

    def freeze_encoder(self):
        for key, param in self.encoder.named_parameters():
            param.requires_grad = False

    def freeze_head(self):
        for key, param in self.head.named_parameters():
            param.requires_grad = False

    def forward(self, img, mapping2d, tsdf=None, label_weight=None, label3d=None, train_disc=False, **kwargs):
        if not train_disc:
            for param in self.disc.parameters():
                param.requires_grad = False
            shape3d = tsdf.shape[-3:]
            feature2d = self.encoder(img)
            feature2d, semantic2d = self.head(feature2d, img_size=img.shape[-2:])

            tsdf_feat = self.TSDFNet(tsdf)
            pred3d, aug_info = self.complete(feature2d, mapping2d, tsdf_feat, shape3d=shape3d)
            fake_loss = None
            if self.training:
                fake_loss = self.get_disc_loss(pred3d, label_weight=label_weight)
        else:
            for param in self.disc.parameters():
                param.requires_grad = True
            with torch.no_grad():
                shape3d = tsdf.shape[-3:]
                feature2d = self.encoder(img)
                feature2d, semantic2d = self.head(feature2d, img_size=img.shape[-2:])

                tsdf_feat = self.TSDFNet(tsdf)
                pred3d, aug_info = self.complete(feature2d, mapping2d, tsdf_feat, shape3d=shape3d)
            true_loss = self.get_disc_loss(pred3d, label_weight=label_weight, label3d=label3d, train_disc=True)
        return pred3d, semantic2d, aug_info, fake_loss if not train_disc else true_loss

    @staticmethod
    def shuffle(labels, num_classes=12):
        labels = labels.argmax(1)
        b, h, w, d = labels.size()
        noise_labels = labels.clone()
        for i in range(b):
            exist_class = labels[i].unique()[1:]
            noise_num_class = random.randint(1, len(exist_class))
            picked_noise_class = exist_class[torch.randperm(len(exist_class))][:noise_num_class]
            for c in picked_noise_class:
                indices = torch.nonzero(labels[i] == c)
                prob = random.random() * 0.8 + 0.1
                num_noise_samples = min(int(len(indices) * prob), max(10, len(indices)))
                sample_ind = torch.randint(0, len(indices), (num_noise_samples,))
                sample_indices = indices[sample_ind]
                noise_c = random.randint(1, num_classes - 1)
                if c == noise_c:
                    noise_c = c - 1 if c > 1 else random.randint(2, num_classes - 1)
                noise_labels[i, sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]] = noise_c  # 赋随机值
        noise_labels = F.one_hot(noise_labels, num_classes=num_classes).permute(0, -1, 1, 2, 3).float()
        return noise_labels

    @staticmethod
    def random_erase(voxel, label):
        valid_pos = (label != 0) & (label != 255)

        for i in range(voxel.shape[0]):
            pos = torch.where(valid_pos[i])
            num_pos = len(pos[0])
            prob = (random.random() * 0.8) + 0.1      # [0.1, 0.9]
            num_erase = int(num_pos * prob)
            idx = random.sample(range(num_pos), num_erase)
            pos_erase = (pos[0][idx], pos[1][idx], pos[2][idx])
            voxel[i, :, pos_erase[0], pos_erase[1], pos_erase[2]] = 0
        return voxel

    def get_disc_loss(self, pred3d, label_weight, label3d=None, train_disc=False):
        b, c, h, w, d = pred3d.shape
        ground_label = torch.ones(b, 1).to(pred3d.device)
        pred3d = pred3d * label_weight.float().view(b, 1, h, w, d)
        bce_loss = nn.BCEWithLogitsLoss()
        if not train_disc:
            pr_data = self.disc(pred3d)
            fake_loss = bce_loss(pr_data, ground_label)
            return fake_loss

        label3d_, label_weight_ = label3d.clone(), label_weight.clone()
        label3d_[(label3d_ == 255) | (label_weight == 0)] = 0
        label3d_ = label3d_.view(b, h, w, d)
        label3d_ = F.one_hot(label3d_, num_classes=12).permute(0, -1, 1, 2, 3).float()

        predict_label = torch.zeros(b, 1).to(pred3d.device)
        gt_data = self.disc(label3d_)
        label3d_shuffle = label3d_.clone()
        gt_data_shuffle = self.disc(self.shuffle(label3d_shuffle))
        gt_data_erase = self.disc(self.random_erase(label3d_.clone(), label3d.view(b, h, w, d)))
        pr_data = self.disc(pred3d.detach())
        true_loss = bce_loss(pr_data, predict_label) + bce_loss(gt_data, ground_label)\
                    + bce_loss(gt_data_shuffle, predict_label) + bce_loss(gt_data_erase, predict_label)
        return true_loss
