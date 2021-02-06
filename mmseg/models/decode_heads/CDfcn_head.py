import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from ..losses import accuracy

@HEADS.register_module()
class CDFCNHead(BaseDecodeHead):
    """Conditional Fully Convolution Classifier

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        # if hasattr(kwargs,'freeze_all'):
        #     self.freeze_all = kwargs.pop('freeze_all')
        super(CDFCNHead, self).__init__(**kwargs)
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        # self.freeze()

    def ata_loss(self, target, num_class, batch_size):
        target_b = target.view(batch_size, -1)
        lb_shown = [torch.unique(target_b[bi], sorted=True).long() for bi in range(0, batch_size)]
        for i in range(0, batch_size):
            if lb_shown[i].max()==255:
                lb_shown[i] = lb_shown[i][0:lb_shown[i].shape[0]-1]
        # label_onehot = torch.sign(label_bin)
        ata_mask = 1. - torch.eye(num_class).to('cuda')
        filters = self.conv_seg.weight.repeat([batch_size, 1, 1, 1])
        # filters = F.normalize(filters, p=2, dim=1)
        dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]

        cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix
        return 10*cosdist[~torch.isnan(cosdist)].sum(), dist_matrix
        # return torch.exp(10*cosdist[~torch.isnan(cosdist)].sum()), dist_matrix
        # return torch.exp(cosdist[~torch.isnan(cosdist)].sum()), dist_matrix
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        b, c, h, w = seg_logit.shape
        # ataloss, dist_matrix = self.ata_loss(seg_label, c, b)
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
        # loss['aux_ata_loss'] = ataloss
        return loss

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def freeze(self):
        if hasattr(self, 'freeze_all'):
            if self.freeze_all:
                self.requires_grad_(False)
