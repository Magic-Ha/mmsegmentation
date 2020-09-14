import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from ..losses import accuracy
import numpy as np
from mmcv.runner import force_fp32

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class PSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class DDPSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        self.weight_mode = kwargs.pop('weight_mode')
        self.sum_weight = kwargs.pop('sum_weight')
        (super(DDPSPHead, self).__init__)(**kwargs)
        self.freeze_parameters()
        assert isinstance(pool_scales, (list, tuple))
        assert self.weight_mode in ('sum', 'new'), "Can not recognize config parameter: train_cfg['weight_mode']"
        #这里是用分组conv把初步的prediction中所含有的类别信息利用上
        #来产生一个delta，用来纠正conv_seg,对每张图进行特化
        #generator的输入应该是初步的prediction结果(B,class_num,H,W)，输出应该是针对conv_seg的weight
        #的修正delta，因此输出形状应该是self.ds_num
        #########################################
        #问题：是应该先把输入整成1*1还是把输出搞个gloabal pooling呢？s
        #这地方先整个gloabal pooling吧
        #TODO:以后可以在整整attention什么的整一个不用gloabal pooling的版本 搞成动态卷积哪样的
        self.ds_num = np.array(self.conv_seg.weight.size()).prod()
        self.dsb_num = np.array(self.conv_seg.bias.size()).prod()

        self.f_compact = nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1)
        self.generator_w = nn.Conv2d((2 * self.num_classes), self.ds_num, kernel_size=1, groups=self.num_classes)
        self.generator_b = nn.Conv2d((2 * self.num_classes), self.dsb_num, kernel_size=1, groups=self.num_classes)
        # self.generator = nn.Conv2d(self.num_classes,int(self.ds_num/self.num_classes),kernel_size=1)
        # self.generator = nn.Conv2d(self.num_classes,self.ds_num,kernel_size=1)

        #每类的mask单独用一个卷积分开，这样给出来的delta也是分别根据每类给出来的

    def freeze_parameters(self):
        self.conv_seg.requires_grad_(False)
        self.psp_modules.requires_grad_(False)
        self.bottleneck.requires_grad_(False)

    def forward(self, inputs, cfg=None):
        """Forward function."""
        if cfg is not None:
            mode = cfg['mode']
        else:
            mode = 'test'
        batch_size = inputs[0].size()[0]
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        temp = self.bottleneck(psp_outs)
        coarse_output = self.cls_seg(temp)
        #现在应该将初步分类结果中的类别信息进行处理，产生delta
        #TODO:以后可以搞一个利用中间层信息的 
        if mode == 'stage1':
            temp.retain_grad()
            coarse_output.retain_grad()
            return coarse_output
        if mode in ('stage2', 'test'):
            temp = temp.detach()
            coarse_output = coarse_output.detach()
            f = self.f_compact(x.detach())
            if mode == 'stage2':
                f.retain_grad()
            delta_p = self.generator_w(torch.cat([coarse_output, f], dim=1))
            delta_b = self.generator_b(torch.cat([coarse_output, f], dim=1))

            delta_p = nn.functional.adaptive_avg_pool2d(delta_p, (1, 1))
            delta_b = nn.functional.adaptive_avg_pool2d(delta_b, (1, 1))

            #conv_seg的weight形状应该是[class_num, channel, 1, 1]
            #现在delta_p 还是[B,ds_num,1,1],所以要想办法把它每张图分别搞成一个weight
            #然后每张图分别去生成自己的final_ds
            #所以可以把Batch搞到channel维度上去
            #比如说本来应该每张图都是150类，现在把最后这个final_ds的输入改成[1,B*channel,H,W]
            #所以相应的卷积用分组卷积分成B个groups 用分组卷积 
            #如果不用分组卷积，那Conv的weight本来应该是 [out_channels,in_channels,kernel,kernel]
            #现在用了分组卷积，分成B组，那weight就应该变成了 [out_channels,in_channels/B,kernel,kernel]
            #而现在的输入是[1,B*channel,H,W], 输出应该为[1,B*class_num,H,W],然后再变成[B,class_num,H,W]
            #所以final_ds输入通道应该是B*channel，输出通道应该是B*class_num
            #所以分组卷积的final_ds的weight形状应该是 [B*class_num, channel, kernel, kernel]
            delta_p = delta_p.view(delta_p.size()[0], self.num_classes, -1, 1, 1)#1*1 kernel size
            delta_b = delta_b.view(delta_b.size()[0], self.num_classes)
            # bias.size()就是out_channels 一个一维度的tensor

            if self.weight_mode == 'sum':
                if self.sum_weight is not None:
                    new_weight = delta_p * self.sum_weight[0] + self.conv_seg.weight.detach() * self.sum_weight[1]
                    new_bias = delta_b * self.sum_weight[0] + self.conv_seg.bias.detach() * self.sum_weight[1]

                else:
                    new_weight = delta_p + self.conv_seg.weight.detach()
                    new_bias = delta_b
            else:
                new_weight = delta_p
            new_weight = new_weight.view([batch_size * self.num_classes, -1, 1, 1])
            new_bias = new_bias.view([batch_size * self.num_classes])
            # bias = self.conv_seg.bias.repeat(batch_size).detach()

            #TODO: 目前这个还没写动态生成bias，以后可以搞

            output = nn.functional.conv2d((temp.view(1, -1, temp.size()[(-2)], temp.size()[(-1)])), new_weight,
              new_bias, groups=batch_size)
            output = output.view(coarse_output.size())
            if mode == 'stage2':
                return (
                 output, coarse_output)
            return output

    @force_fp32(apply_to=('seg_logit', 'coarse_logit'))
    def losses(self, seg_logit, coarse_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        size = seg_label.shape[2:]
        seg_logit = resize(input=seg_logit,
          size=(seg_label.shape[2:]),
          mode='bilinear',
          align_corners=(self.align_corners))
        coarse_logit = resize(input=coarse_logit,
          size=(seg_label.shape[2:]),
          mode='bilinear',
          align_corners=(self.align_corners))
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['fix_loss_seg'] = self.loss_decode(seg_logit,
          seg_label,
          weight=seg_weight,
          ignore_index=(self.ignore_index))
        loss['coarse_loss_seg'] = self.loss_decode(coarse_logit,
          seg_label,
          weight=coarse_seg_weight,
          ignore_index=(self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['acc_seg_coarse'] = accuracy(coarse_logit, seg_label)
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, coarse_logits = self.forward(inputs, train_cfg)
        losses = self.losses(seg_logits, coarse_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)


@HEADS.register_module()
class DD2PSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        (super(DDPSPHead, self).__init__)(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.ds_num = np.array(self.conv_seg.weight.size()).prod()
        print(self.ds_num)
        self.generator = nn.Conv2d((self.num_classes), (self.ds_num), kernel_size=1, groups=(self.num_classes))

    def forward(self, inputs, train_cfg):
        """Forward function."""
        batch_size = inputs[0].size()[0]
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        temp = self.bottleneck(psp_outs)
        coarse_output = self.cls_seg(temp)
        if train_cfg['mode'] == 'stage2':
            temp = temp.detach()
            coarse_output = coarse_output.detach()
        delta_p = self.generator(coarse_output)
        delta_p = nn.functional.adaptive_avg_pool2d(delta_p, (1, 1))
        delta_p = delta_p.view(delta_p.size()[0], self.num_classes, -1, 1, 1)
        new_weight = delta_p + self.conv_seg.weight
        new_weight = new_weight.view([batch_size * self.num_classes, -1, 1, 1])
        bias = self.conv_seg.bias.repeat(batch_size)
        output = nn.functional.conv2d((temp.view(1, -1, temp.size()[(-2)], temp.size()[(-1)])), new_weight,
          bias, groups=batch_size)
        output = output.view(coarse_output.size())
        if mode != 'train':
            return output
        return (output, coarse_output)

    @force_fp32(apply_to=('seg_logit', 'coarse_logit'))
    def losses(self, seg_logit, coarse_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        size = seg_label.shape[2:]
        seg_logit = resize(input=seg_logit,
          size=(seg_label.shape[2:]),
          mode='bilinear',
          align_corners=(self.align_corners))
        coarse_logit = resize(input=coarse_logit,
          size=(seg_label.shape[2:]),
          mode='bilinear',
          align_corners=(self.align_corners))
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = 0.7 * self.loss_decode(seg_logit,
          seg_label,
          weight=seg_weight,
          ignore_index=(self.ignore_index))
        loss['loss_seg'] += 0.3 * self.loss_decode(coarse_logit,
          seg_label,
          weight=coarse_seg_weight,
          ignore_index=(self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['acc_seg_coarse'] = accuracy(coarse_logit, seg_label)
        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, coarse_logits = self.forward(inputs, train_cfg)
        losses = self.losses(seg_logits, coarse_logits, gt_semantic_seg)
        return losses