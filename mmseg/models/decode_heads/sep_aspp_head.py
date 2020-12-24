import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from mmseg.ops import DepthwiseSeparableConvModule, resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule
from mmcv.runner import force_fp32
from ..losses import accuracy


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

    def ata_loss(self, target, num_class, batch_size):
        target_b = target.view(batch_size, -1)
        lb_shown = [torch.unique(target_b[bi], sorted=True).long() for bi in range(0, batch_size)]
        for i in range(0, batch_size):
            if lb_shown[i].max()==255:
                lb_shown[i] = lb_shown[i][0:lb_shown[i].shape[0]-1]
        # label_onehot = torch.sign(label_bin)
        ata_mask = 1. - torch.eye(num_class).to('cuda')
        filters = self.conv_seg.weight.repeat([batch_size, 1, 1, 1])
        filters = F.normalize(filters, p=2, dim=1)
        dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs(), bmlc[bi].bool()) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                
                                filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]][:,lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask)[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(bmlc[bi]) for bi in range(0, batch_size)]

        cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix
        return torch.exp(10*cosdist[~torch.isnan(cosdist)].sum()), dist_matrix

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        b, c, h, w = seg_logit.shape
        ataloss, dist_matrix = self.ata_loss(seg_label, c, b)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['ata_loss'] = ataloss
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        return self.forward(inputs)
