import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.norm import build_norm_layer
from mmseg.ops import resize
from ..builder import HEADS
# from ..losses import accuracy, accuracy_error_loss
from ..losses import accuracy, accuracy_error_loss, accuracy_ce_error_loss, error_gt
from ..builder import build_loss
from .decode_head import BaseDecodeHead
from mmseg.models.backbones.resnet import ResNetV1c

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
        # self.freeze()

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output

    def freeze(self):
        self.requires_grad_(False)


@HEADS.register_module()
class DDPSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        self.weight_mode = kwargs.pop('weight_mode')
        self.sum_weight = kwargs.pop('sum_weight')
        self.with_bias = kwargs.pop('with_bias')
        self.with_error_ds = kwargs.pop("with_error_ds")
        (super(DDPSPHead, self).__init__)(**kwargs)
        self.freeze_parameters()
        assert isinstance(pool_scales, (list, tuple))
        assert self.weight_mode in ('sum', 'new'), \
            "Can not recognize config parameter: train_cfg['weight_mode']"
        # 这里是用分组conv把初步的prediction中所含有的类别信息利用上
        # 来产生一个delta，用来纠正conv_seg,对每张图进行特化
        # generator的输入应该是初步的prediction结果(B,class_num,H,W)，输出应该是针对conv_seg的weight
        # 的修正delta，因此输出形状应该是self.ds_num
        #########################################
        # 问题：是应该先把输入整成1*1还是把输出搞个gloabal pooling呢？s
        # 这地方先整个gloabal pooling吧
        # TODO:以后可以在整整attention什么的整一个不用gloabal pooling的版本 搞成动态卷积哪样的
        self.ds_num = np.array(self.conv_seg.weight.size()).prod()
        self.dsb_num = np.array(self.conv_seg.bias.size()).prod()

        self.f_compact = nn.Conv2d(2048, self.num_classes,
                                   kernel_size=3, padding=1)
        self.generator_w = nn.Conv2d(
            (2 * self.num_classes),
            self.ds_num,
            kernel_size=1,
            groups=self.num_classes)
        if self.with_bias:
            self.fb_compact = nn.Conv2d(2048, self.num_classes,
                                        kernel_size=3, padding=1)
            self.generator_b = nn.Conv2d(
                (2 * self.num_classes),
                self.dsb_num,
                kernel_size=1,
                groups=self.num_classes)
        # if self.with_error_ds:
        #     self.error_ds = nn.Conv2d(self.num_classes*2, 1,
        #                               kernel_size=3, padding=1)
            # error_ds的输入是coarse_prediction与backbone来的feature(还要考虑位置)
        # self.generator = nn.Conv2d(self.num_classes,
        #                            int(self.ds_num/self.num_classes),
        #                            kernel_size=1)
        # self.generator = nn.Conv2d(self.num_classes,
        #                            self.ds_num,
        #                            kernel_size=1)

        # 每类的mask单独用一个卷积分开，这样给出来的delta也是分别根据每类给出来的

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
        # 现在应该将初步分类结果中的类别信息进行处理，产生delta
        # TODO:以后可以搞一个利用中间层信息的
        if mode == 'stage1':
            temp.retain_grad()
            coarse_output.retain_grad()
            return coarse_output
        if mode in ('stage2', 'test'):
            temp = temp.detach()
            coarse_output = coarse_output.detach()
            f = self.f_compact(x.detach())
            # if mode == 'stage2':
            #     f.retain_grad()
            delta_p = self.generator_w(torch.cat([coarse_output, f], dim=1))
            # delta_p.retain_grad()
            delta_p = nn.functional.adaptive_avg_pool2d(delta_p, (1, 1))
            # conv_seg的weight形状应该是[class_num, channel, 1, 1]
            # 现在delta_p 还是[B,ds_num,1,1],所以要想办法把它每张图分别搞成一个weight
            # 然后每张图分别去生成自己的final_ds
            # 所以可以把Batch搞到channel维度上去
            # 比如说本来应该每张图都是150类，现在把最后这个final_ds的输入改成[1,B*channel,H,W]
            # 所以相应的卷积用分组卷积分成B个groups 用分组卷积
            # 如果不用分组卷积，那Conv的weight本来应该是:
            #                   [out_channels,in_channels,kernel,kernel]
            # 现在用了分组卷积，分成B组，那weight就应该变成了:
            #                   [out_channels,in_channels/B,kernel,kernel]
            # 而现在的输入是[1,B*channel,H,W], 输出应该为[1,B*class_num,H,W],
            # 然后再变成[B,class_num,H,W]
            # 所以final_ds输入通道应该是B*channel，输出通道应该是B*class_num
            # 所以分组卷积的final_ds的weight形状应该是:
            #                   [B*class_num, channel, kernel, kernel]
            delta_p = delta_p.view(batch_size, self.num_classes, -1, 1, 1)
            # 1*1 kernel size

            # bias.size()就是out_channels 一个一维度的tensor

            if self.weight_mode == 'sum':
                if self.sum_weight is not None:
                    new_weight = delta_p * self.sum_weight[0] + \
                        self.conv_seg.weight.detach() * self.sum_weight[1]
                else:
                    new_weight = delta_p + self.conv_seg.weight.detach()
            else:
                new_weight = delta_p
            # bias
            if self.with_bias:
                fb = self.fb_compact(x.detach())
                # if mode == 'stage2':
                #     fb.retain_grad()
                delta_b = self.generator_b(torch.cat([coarse_output, fb], dim=1))
                delta_b = nn.functional.adaptive_avg_pool2d(delta_b, (1, 1))
                delta_b = delta_b.view(batch_size, self.num_classes)
                if self.weight_mode == 'sum':
                    if self.sum_weight is not None:
                        new_bias = delta_b * self.sum_weight[0] + \
                            self.conv_seg.bias.detach() * self.sum_weight[1]

                    else:
                        new_bias = delta_b + self.conv_seg.bias.detach()
                else:
                    new_bias = delta_b
            else:
                new_bias = self.conv_seg.bias.repeat(batch_size).detach()

            # if self.with_error_ds:
            #     error_map = self.error_ds(torch.cat[coarse_output])

            new_weight = new_weight.view([batch_size * self.num_classes,
                                         -1, 1, 1])
            new_bias = new_bias.view([batch_size * self.num_classes])
            # bias = self.conv_seg.bias.repeat(batch_size).detach()
            output = nn.functional.conv2d((temp.view(1, -1, temp.size()[(-2)],
                                           temp.size()[(-1)])),
                                          new_weight,
                                          new_bias, groups=batch_size)
            output = output.view(coarse_output.size())
            if mode == 'stage2':
                return (output, coarse_output)
            return output

    @force_fp32(apply_to=('seg_logit', 'coarse_logit'))
    def losses(self, seg_logit, coarse_logit, seg_label):
        print(seg_label.min())
        print(torch.where(seg_label==255,torch.full_like(seg_label, -1),seg_label).view(-1).max())
        """Compute segmentation loss."""
        loss = dict()
        # size = seg_label.shape[2:]
        seg_logit = resize(
            input=seg_logit,
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        coarse_logit = resize(
            input=coarse_logit,
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
        loss['fix_loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
        loss['coarse_loss_seg'] = self.loss_decode(
            coarse_logit,
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
class EDPSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        self.weight_mode = kwargs.pop('weight_mode')
        self.sum_weight = kwargs.pop('sum_weight')
        self.with_bias = kwargs.pop('with_bias')
        self.with_error_ds = kwargs.pop("with_error_ds")
        (super(EDPSPHead, self).__init__)(**kwargs)
        self.freeze_parameters()
        assert isinstance(pool_scales, (list, tuple))
        assert self.weight_mode in ('sum', 'new'), \
            "Can not recognize config parameter: train_cfg['weight_mode']"

        self.ds_num = np.array(self.conv_seg.weight.size()).prod()
        self.dsb_num = np.array(self.conv_seg.bias.size()).prod()

        self.f_compact = nn.Sequential(
            nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True))

        self.generator_w = nn.Conv2d(
            (2 * self.num_classes),
            self.ds_num,
            kernel_size=1,
            groups=self.num_classes)
        if self.with_bias:
            self.fb_compact = nn.Sequential(
                nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True))
            self.generator_b = nn.Conv2d(
                (2 * self.num_classes),
                self.dsb_num,
                kernel_size=1,
                groups=self.num_classes)
        if self.with_error_ds:
            self.fe_compact = nn.Sequential(
                nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True))

            self.error_ds = nn.Conv2d(self.num_classes*2, 1, kernel_size=1, padding=0)
            # error_ds的输入是coarse_prediction与backbone来的feature(还要考虑位置)

    def freeze_parameters(self):
        # TODO: 后面得把这个搞一个对应的参数放在config里面
        self.conv_seg.requires_grad_(False)
        self.psp_modules.requires_grad_(False)
        self.bottleneck.requires_grad_(False)

    def forward(self, inputs, cfg=None, gt=None):
        if cfg is not None:
            mode = cfg['mode']
        else:
            mode = 'test'
        # test 的时候可以不要gt
        if mode == 'stage3':
            assert gt is not None
        batch_size = inputs[0].size()[0]
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        temp = self.bottleneck(psp_outs)
        coarse_output = self.cls_seg(temp)
        # 现在应该将初步分类结果中的类别信息进行处理，产生delta
        # TODO:以后可以搞一个利用中间层信息的
        if mode == 'stage1':
            return coarse_output
        if mode in ('stage2', 'test', 'stage3'):
            temp = temp.detach()
            coarse_output = coarse_output.detach()
            f = self.f_compact(x.detach())
            # if mode == 'stage2':
            #     f.retain_grad()
            delta_p = self.generator_w(torch.cat([coarse_output, f], dim=1))
            # delta_p.retain_grad()
            delta_p = nn.functional.adaptive_avg_pool2d(delta_p, (1, 1))
            # conv_seg的weight形状应该是[class_num, channel, 1, 1]
            # 现在delta_p 还是[B,ds_num,1,1],所以要想办法把它每张图分别搞成一个weight
            # 然后每张图分别去生成自己的final_ds
            # 所以可以把Batch搞到channel维度上去
            # 比如说本来应该每张图都是150类，现在把最后这个final_ds的输入改成[1,B*channel,H,W]
            # 所以相应的卷积用分组卷积分成B个groups 用分组卷积
            # 如果不用分组卷积，那Conv的weight本来应该是:
            #                   [out_channels,in_channels,kernel,kernel]
            # 现在用了分组卷积，分成B组，那weight就应该变成了:
            #                   [out_channels,in_channels/B,kernel,kernel]
            # 而现在的输入是[1,B*channel,H,W], 输出应该为[1,B*class_num,H,W],
            # 然后再变成[B,class_num,H,W]
            # 所以final_ds输入通道应该是B*channel，输出通道应该是B*class_num
            # 所以分组卷积的final_ds的weight形状应该是:
            #                   [B*class_num, channel, kernel, kernel]
            delta_p = delta_p.view(batch_size, self.num_classes, -1, 1, 1)
            # 1*1 kernel size
            # bias.size()就是out_channels 一个一维度的tensor
            if self.weight_mode == 'sum':
                if self.sum_weight is not None:
                    new_weight = delta_p * self.sum_weight[0] + \
                        self.conv_seg.weight.detach() * self.sum_weight[1]
                else:
                    new_weight = delta_p + self.conv_seg.weight.detach()
            else:
                new_weight = delta_p
            # bias
            if self.with_bias:
                fb = self.fb_compact(x.detach())
                # if mode == 'stage2':
                #     fb.retain_grad()
                delta_b = self.generator_b(torch.cat([coarse_output, fb], dim=1))
                delta_b = nn.functional.adaptive_avg_pool2d(delta_b, (1, 1))
                delta_b = delta_b.view(batch_size, self.num_classes)
                if self.weight_mode == 'sum':
                    if self.sum_weight is not None:
                        new_bias = delta_b * self.sum_weight[0] + \
                            self.conv_seg.bias.detach() * self.sum_weight[1]

                    else:
                        new_bias = delta_b + self.conv_seg.bias.detach()
                else:
                    new_bias = delta_b
            else:
                new_bias = self.conv_seg.bias.repeat(batch_size).detach()

            new_weight = new_weight.view([batch_size * self.num_classes,
                                         -1, 1, 1])
            new_bias = new_bias.view([batch_size * self.num_classes])
            # bias = self.conv_seg.bias.repeat(batch_size).detach()
            output = nn.functional.conv2d((temp.view(1, -1, temp.size()[(-2)],
                                           temp.size()[(-1)])),
                                          new_weight,
                                          new_bias, groups=batch_size)
            output = output.view(coarse_output.size())
            if self.with_error_ds:
                fe = self.fe_compact(x.detach())
                error_map = self.error_ds(torch.cat([coarse_output, fe],
                                                    dim=1))
                if mode != 'test':
                    error_map.retain_grad()
                # FIXME:不一定是这两个函数的问题，只要有retain_grad()他都能传
                error_logit = torch.sigmoid(error_map)
                # error_logit = nn.functional.softmax(error_map, dim=1)
                if mode != 'test':
                    error_logit.retain_grad()
                # output = torch.mul(error_map[:, 0, :, :].unsqueeze(1),
                #                    output) + \加接着下面
                # torch.mul(error_map[:, 1, :, :].unsqueeze(1), coarse_output)
                if mode == 'test' or mode == 'stage2':
                # if mode == 'stage2':
		    		# output = torch.mul(error_logit, output) + \
                    #          torch.mul(1-error_logit, coarse_output)
                    # FIXME: 09/19晚上跑的42.24的结果的是用的直接加coarse_output的
                    # FIXME: 09/19晚上跑的42.24是有错的 根本就不能反传loss 太奇怪了
                    '''
                    output = torch.mul(error_logit, output) + torch.mul(1-error_logit, coarse_output)
                    '''
                    # error_mask = torch.where(error_logit>0.5, error_logit.clone(), torch.full_like(error_logit, 0.))
                    # output = torch.mul(error_mask, output) + coarse_output
                    output = torch.mul(error_logit, output) + coarse_output

                # elif mode == 'stage3':
                else:
                    '''
                    large_coarse_output = resize(
                        input=coarse_output,
                        size=(gt.shape[2:]),
                        mode='bilinear',
                        align_corners=(self.align_corners))

                    correction = self.get_error_target(large_coarse_output, gt)
                    correction = nn.functional.adaptive_avg_pool2d(correction, output.shape[2:])
                    correction = correction.detach()
                    # FIXME: 调试的时候用的东西 以后没用的时候要删掉
                    # if hasattr(self, 'last_fe_weight'):
                    #     assert id(self.last_fe_weight) != id(self.fe_compact[0].weight)
                    #     print(torch.max(self.last_fe_weight - self.fe_compact[0].weight))
                    # self.last_fe_weight = self.fe_compact[0].weight.clone()
                    # self.last_fb_weight = self.fb_compact[0].weight.clone()

                    # bce = nn.BCELoss(reduction='mean')
                    # e_loss = bce(error_logit, correction)

                    error_yes = torch.where(torch.abs(correction-error_logit)<0.2, torch.full_like(correction,1.), torch.full_like(correction,0.))
                    error_yes_num = error_yes.sum()
                    total_num = torch.ones_like(correction).sum()
                    #print(error_yes_num, total_num)
                    self.eds_correct_rate = error_yes_num/total_num
                    '''
                    ##########这是我想弄成互补搞得，可以注释掉，以后好起来了可以改改试试###########
                    #print("error prediction correct rate:",self.eds_correct_rate)
                    # output = torch.mul(error_logit.clone().detach(), output) + torch.mul(1-error_logit.clone().detach(), coarse_output)
                    # output = torch.mul(error_logit.clone().detach(), output) + coarse_output

                    # error_mask = torch.where(error_logit>0.5, error_logit.clone(), torch.full_like(error_logit, 0.))
                    # output = torch.mul(error_mask, output) + coarse_output
                    output = torch.mul(error_logit, output) + coarse_output

                    # if self.eds_correct_rate>0.6:
                    #     output = torch.mul(error_logit.clone().detach(), output) + torch.mul(1-error_logit.clone().detach(), coarse_output)
                    #     # output = torch.mul(error_logit, output) + torch.mul(1-error_logit, coarse_output)
                    # else:
                    #     output = torch.mul(correction, output) + torch.mul(1-correction, coarse_output)
            if mode == 'stage2' or mode == 'stage3':
                if self.with_error_ds:
                    
                    # if mode == 'stage2':
                    #     return (output, coarse_output, error_logit)
                    # else:
                    #     return (output, large_coarse_output, error_logit)
                    
                    return (output, coarse_output, error_logit)
                    # return (output, coarse_output)
                else:
                    return (output, coarse_output)
            return output

    def get_error_target(self, pred, label):
        pred_value, pred_label = pred.max(dim=1)
        correct = pred_label.unsqueeze(1).eq(label)
        target_error = torch.zeros(correct.size(), device=correct.device)
        target_error = torch.where(correct, torch.full_like(target_error, 0.),
                                   torch.full_like(target_error, 1.))
        return target_error

    @force_fp32(apply_to=('seg_logit', 'coarse_logit'))
    def losses(self, seg_logit, coarse_logit, seg_label, error_map=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        if coarse_logit.shape[2:] != seg_label.shape[2:]:
            coarse_logit = resize(
                input=coarse_logit,
                size=(seg_label.shape[2:]),
                mode='bilinear',
                align_corners=(self.align_corners))
        ############################################
        if self.with_error_ds:
            error_logit = resize(
                input=error_map,
                size=(seg_label.shape[2:]),
                mode='bilinear',
                align_corners=(self.align_corners))
        ##############################################
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['fix_loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
        loss['coarse_loss_seg'] = self.loss_decode(
            coarse_logit,
            seg_label,
            weight=coarse_seg_weight,
            ignore_index=(self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['acc_seg_coarse'] = accuracy(coarse_logit, seg_label)
        if self.with_error_ds:
            loss['error_loss_seg'], loss['acc_error_pred'] = accuracy_error_loss(coarse_logit,
                                                            seg_label,
                                                            error_logit)
            #print(s==loss['acc_seg_coarse'])
            # loss['error_loss_seg'] *= 10
            # loss['acc_error_pred'] = self.eds_correct_rate

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
        if train_cfg['mode'] == 'stage3':
            gt = gt_semantic_seg
        else:
            gt = None
        if self.with_error_ds:
            seg_logits, coarse_logits, error_map = self.forward(inputs, train_cfg, gt=gt)
            losses = self.losses(seg_logits,
                                 coarse_logits,
                                 gt_semantic_seg,
                                 error_map)
            # seg_logits, coarse_logits = self.forward(inputs, train_cfg)
            # losses = self.losses(seg_logits, coarse_logits, gt_semantic_seg)

        else:
            seg_logits, coarse_logits = self.forward(inputs, train_cfg, gt=gt)
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
class ED_CE_PSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        self.weight_mode = kwargs.pop('weight_mode')
        
        # self.finput_fusion_mode = kwargs.pop('finput_fusion_mode')
        self.sum_weight = kwargs.pop('sum_weight')
        self.with_bias = kwargs.pop('with_bias')
        self.with_error_ds = kwargs.pop("with_error_ds")
        fe_input_cast = {'res_stage1': [0, 256, 128],
                         'res_stage2': [1, 512, 64],
                         'res_stage3': [2, 1024, 64],
                         'res_stage4': [3, 2048, 64]}
        self.fe_input_config = fe_input_cast[kwargs.pop('fe_input')]
        (super(ED_CE_PSPHead, self).__init__)(**kwargs)
        self.freeze_parameters()
        assert isinstance(pool_scales, (list, tuple))
        assert self.weight_mode in ('sum', 'new'), \
            "Can not recognize config parameter: train_cfg['weight_mode']"

        self.ds_num = np.array(self.conv_seg.weight.size()).prod()
        self.dsb_num = np.array(self.conv_seg.bias.size()).prod()

        self.f_compact = nn.Sequential(
            nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True))

        self.generator_w = nn.Conv2d(
            (2 * self.num_classes),
            self.ds_num,
            kernel_size=1,
            groups=self.num_classes)
        if self.with_bias:
            self.fb_compact = nn.Sequential(
                nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True))

            self.generator_b = nn.Conv2d(
                (2 * self.num_classes),
                self.dsb_num,
                kernel_size=1,
                groups=self.num_classes)
        if self.with_error_ds:
            # self.fe_compact = nn.Sequential(
            #     nn.Conv2d(2048, self.num_classes, kernel_size=3, padding=1),
            #     nn.LeakyReLU(inplace=True))
            self.fe_compact = nn.Sequential(
                nn.Conv2d(self.fe_input_config[1], self.num_classes, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True))
            self.error_ds = nn.Conv2d(self.num_classes*2, 2, kernel_size=1, padding=0)
            self.softmax = torch.nn.LogSoftmax(dim=1)
            # error_ds的输入是coarse_prediction与backbone来的feature(还要考虑位置)

    def freeze_parameters(self):
        # TODO: 后面得把这个搞一个对应的参数放在config里面
        self.conv_seg.requires_grad_(False)
        self.psp_modules.requires_grad_(False)
        self.bottleneck.requires_grad_(False)

    def forward(self, inputs, cfg=None, gt=None):
        if cfg is not None:
            mode = cfg['mode']
        else:
            mode = 'test'
        # test 的时候可以不要gt
        if mode == 'stage3':
            assert gt is not None
        batch_size = inputs[0].size()[0]
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        temp = self.bottleneck(psp_outs)
        coarse_output = self.cls_seg(temp)
        # 现在应该将初步分类结果中的类别信息进行处理，产生delta
        if mode == 'stage1':
            return coarse_output
        if mode in ('stage2', 'test', 'stage3'):
            temp = temp.clone()
            coarse_output = coarse_output.clone()
            f = self.f_compact(x.clone())
            # if mode == 'stage2':
            #     f.retain_grad()
            # TODO: 这里以后要弄一个加和拼接的对比操作
            # if self.finput_fusion_mode == 'sum':
            #     m = coarse_output + 

            # m = torch.cat([coarse_output[:,0,:,:].unsqueeze(1), f[:,0,:,:].unsqueeze(1)], dim=1)
            # for i in range(1,self.num_classes):
            #     m = torch.cat([m, torch.cat([coarse_output[:,i,:,:].unsqueeze(1),f[:,i,:,:].unsqueeze(1)],dim=1)],dim=1)
            
            # FIXME: 这地方是应该用m的直接这样拼起来然后做分组卷积其实就不是对应的了 是一部分用的cp 一部分用的f
            # 这明显不合理,这地方应该要重新做实验了
            delta_p = self.generator_w(torch.cat([coarse_output, f], dim=1))
            # delta_p = self.generator_w(m)
            # delta_p.retain_grad()
            delta_p = nn.functional.adaptive_avg_pool2d(delta_p, (1, 1))
            delta_p = delta_p.view(batch_size, self.num_classes, -1, 1, 1)
            # 1*1 kernel size
            # bias.size()就是out_channels 一个一维度的tensor
            if self.weight_mode == 'sum':
                if self.sum_weight is not None:
                    new_weight = delta_p * self.sum_weight[0] + \
                        self.conv_seg.weight.detach() * self.sum_weight[1]
                else:
                    new_weight = delta_p + self.conv_seg.weight.detach()
            else:
                new_weight = delta_p
            # bias
            if self.with_bias:
                fb = self.fb_compact(x.detach())
                # if mode == 'stage2':
                #     fb.retain_grad()
                delta_b = self.generator_b(torch.cat([coarse_output, fb], dim=1))
                delta_b = nn.functional.adaptive_avg_pool2d(delta_b, (1, 1))
                delta_b = delta_b.view(batch_size, self.num_classes)
                if self.weight_mode == 'sum':
                    if self.sum_weight is not None:
                        new_bias = delta_b * self.sum_weight[0] + \
                            self.conv_seg.bias.detach() * self.sum_weight[1]

                    else:
                        new_bias = delta_b + self.conv_seg.bias.detach()
                else:
                    new_bias = delta_b
            else:
                new_bias = self.conv_seg.bias.repeat(batch_size).detach()

            new_weight = new_weight.view([batch_size * self.num_classes,
                                         -1, 1, 1])
            new_bias = new_bias.view([batch_size * self.num_classes])
            # bias = self.conv_seg.bias.repeat(batch_size).detach()
            output = nn.functional.conv2d((temp.view(1, -1, temp.size()[(-2)],
                                           temp.size()[(-1)])),
                                          new_weight,
                                          new_bias, groups=batch_size)
            output = output.view(coarse_output.size())
            if self.with_error_ds:
                fe = self.fe_compact(inputs[self.fe_input_config[0]])
                if coarse_output.shape[2:] != fe.shape[2:]:
                    coarse_output = resize(
                        input=coarse_output,
                        size=(fe.shape[2:]),
                        mode='bilinear',
                        align_corners=(self.align_corners))
                error_map = self.error_ds(torch.cat([coarse_output, fe],
                                                    dim=1))
                # if mode != 'test':
                #     error_map.retain_grad()
                # error_logit = torch.sigmoid(error_map)
                # error_logit = torch.sigmoid(error_map.clone().detach())

                if output.shape[2:] != error_map.shape[2:]:
                    output = resize(
                        input=output,
                        size=(error_map.shape[2:]),
                        mode='bilinear',
                        align_corners=(self.align_corners))
                # FIXME:这里detach掉了 记得改回来啊
                error_logit = self.softmax(error_map.clone().detach())
                # error_logit = self.softmax(error_map)
                # if mode != 'test':
                #     error_logit.retain_grad()
                # output = torch.mul(error_map[:, 0, :, :].unsqueeze(1),
                #                    output) + \加接着下面
                # torch.mul(error_map[:, 1, :, :].unsqueeze(1), coarse_output)
                # output = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), output) + torch.mul(error_logit[:, 0, :, :].unsqueeze(1), coarse_output)
                output = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), output) + coarse_output


            if mode == 'stage2' or mode == 'stage3':
                if self.with_error_ds:
                    # return (output, coarse_output, error_logit)
                    return (output, coarse_output, error_map)
                    # return (output, coarse_output)
                else:
                    return (output, coarse_output)
            return output

    def get_error_target(self, pred, label):
        pred_value, pred_label = pred.max(dim=1)
        correct = pred_label.unsqueeze(1).eq(label)
        target_error = torch.zeros(correct.size(), device=correct.device)
        target_error = torch.where(correct, torch.full_like(target_error, 0.),
                                   torch.full_like(target_error, 1.))
        return target_error

    @force_fp32(apply_to=('seg_logit', 'coarse_logit'))
    def losses(self, seg_logit, coarse_logit, seg_label, error_map=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        if coarse_logit.shape[2:] != seg_label.shape[2:]:
            coarse_logit = resize(
                input=coarse_logit,
                size=(seg_label.shape[2:]),
                mode='bilinear',
                align_corners=(self.align_corners))
        ############################################
        if self.with_error_ds:
            error_logit = resize(
                input=error_map,
                size=(seg_label.shape[2:]),
                mode='bilinear',
                align_corners=(self.align_corners))
        ##############################################
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['fix_loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
        loss['coarse_loss_seg'] = self.loss_decode(
            coarse_logit,
            seg_label,
            weight=coarse_seg_weight,
            ignore_index=(self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['acc_seg_coarse'] = accuracy(coarse_logit, seg_label)
        if self.with_error_ds:
            # loss['error_loss_seg'], loss['acc_error_pred'] = accuracy_ce_error_loss(coarse_logit,
            #                                                 seg_label,
            #                                                 error_logit)
            loss['error_loss'], loss['eds_acc'], loss['pred_error_rate'], loss['mask_correct_rate'] = accuracy_ce_error_loss(coarse_logit, seg_label, error_logit)
        #     #print(s==loss['acc_seg_coarse'])
        #     # loss['error_loss_seg'] *= 10
        #     # loss['acc_error_pred'] = self.eds_correct_rate

        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        if train_cfg['mode'] == 'stage3':
            gt = gt_semantic_seg
        else:
            gt = None
        if self.with_error_ds:
            seg_logits, coarse_logits, error_map = self.forward(inputs, train_cfg, gt=gt)
            losses = self.losses(seg_logits,
                                 coarse_logits,
                                 gt_semantic_seg,
                                 error_map)
            # seg_logits, coarse_logits = self.forward(inputs, train_cfg)
            # losses = self.losses(seg_logits, coarse_logits, gt_semantic_seg)

        else:
            seg_logits, coarse_logits = self.forward(inputs, train_cfg, gt=gt)
            losses = self.losses(seg_logits, coarse_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, img):
        return self.forward(inputs)


@HEADS.register_module()
class ConditionalFilterLayer(nn.Module):
    def __init__(self, ichn, ochn, same_loss=False):
        super(ConditionalFilterLayer, self).__init__()
        self.ichn = ichn
        self.ochn = ochn

        self.mask_conv = nn.Conv2d(ichn, ochn, kernel_size=1)
        # self.mask_conv.requires_grad_(False)
        self.filter_conv = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)
        self.error_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2.], device='cuda').float(), reduction='mean')#, ignore_index=255)
        # self.filter_conv.requires_grad_(False)
        # self.filter_conv = nn.Sequential(
        #     ConvBnRelu(ochn * ichn, ochn * ichn, 1, 1, 0,
        #                has_bn=False,
        #                has_relu=True,
        #                has_bias=False,
        #                groups=ochn,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer),
        #     nn.Dropout2d(0.1, inplace=False)
        # )
        # self.feat_conv = ConvBnRelu(ichn, ochn*ichn, 3, 1, 1,
        #                             has_bn=True,
        #                             has_relu=True,
        #                             has_bias=True,
        #                             groups=ichn,
        #                             norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                             norm_layer=build_norm_layer)
        # self.generator = nn.Sequential(
        #     # ConvBnRelu(ochn * 2, ochn, 1, 1, 0,
        #     #            has_bn=False,
        #     #            has_relu=True,
        #     #            has_bias=True,
        #     #            norm_cfg=dict(type='SyncBN', requires_grad=True),
        #     #            norm_layer=build_norm_layer),           
        #     ConvBnRelu(ochn, ichn * ochn, 1, 1, 0,
        #                has_bn=False,
        #                has_relu=True,
        #                has_bias=True,
        #                groups=ochn,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer)
        # )
        # self.bias_conv = nn.Conv2d(ochn * ichn, ochn, kernel_size=1,
        #                            groups=ochn)
        self.same_loss = same_loss
        # self.context_g = nn.AdaptiveAvgPool2d((1,1))
        # self.image_conv = nn.Conv2d(ochn, ochn, kernel_size=1)
        # self.image_loss = nn.BCELoss()
        # self.feat_trans_conv = nn.Sequential(
        #     ConvBnRelu(ichn, ochn, 3, 1, 1,
        #                has_bn=True,
        #                has_relu=True,
        #                has_bias=False,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer),
        #     nn.Dropout2d(0.1, inplace=False)
        # )
        # self.fe_compact = nn.Sequential(
        #     ConvBnRelu(ichn, ochn, 3, 1, 1,
        #                has_bn=True,
        #                has_relu=True,
        #                has_bias=True,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer),
        #     ConvBnRelu(ochn, ochn, 3, 1, 1,
        #                has_bn=True,
        #                has_relu=True,
        #                has_bias=True,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer),
        #     ConvBnRelu(ochn, ochn, 3, 1, 1,
        #                has_bn=True,
        #                has_relu=True,
        #                has_bias=False,
        #                norm_cfg=dict(type='SyncBN', requires_grad=True),
        #                norm_layer=build_norm_layer),
        #     nn.Dropout2d(0.1, inplace=False)
        # )

        # self.error_ds = nn.Conv2d(ochn*2+3, 2, kernel_size=1, padding=0)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.freeze()
        self.error_ds = nn.Conv2d(ochn+512, 2, kernel_size=3, padding=1)
        # self.error_ds.requires_grad_(False)
        self.fe_compact = ResNetV1c(depth=18,
        # self.fe_compact2 = ResNetV1c(depth=34,
                                    # num_stages=4,
                                    out_indices=(0, 1, 2, 3),
                                    dilations=(1, 1, 2, 4),
                                    strides=(1, 2, 1, 1),
                                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                                    norm_eval=False,
                                    style='pytorch',
                                    contract_dilation=True)
        # self.fe_compact.requires_grad_(False)
    def freeze(self):
        self.requires_grad_(False)

    def multi_class_dice_loss(self, mask, target, num_class):
        target = torch.where(target==255,torch.full_like(target, 150), target)
        target = F.one_hot(target.long(), num_class + 1)[:, :, :, :150]
        # print(target.size())
        target = target.permute(0, 3, 1, 2)
        mask = mask.contiguous().view(mask.size()[0], num_class, -1)
        target = target.contiguous().view(target.size()[0], num_class,
                                          -1).float()

        a = torch.sum(mask * target, 2)
        b = torch.sum(mask * mask, 2) + 0.001
        c = torch.sum(target * target, 2) + 0.001
        d = (2 * a) / (b + c)
        return (1 - d).mean()

    def image_classification_loss(self, vector, target, num_class, b):
        target = torch.where(target==255,torch.full_like(target, 150), target)
        label = torch.bincount(target[0].long(), minlength=150)[:150].unsqueeze(0)
        # print(label.shape)
        # input("label")
        label = torch.where(label>0., torch.full_like(label, 1), torch.full_like(label, 0))
        # print(label.shape)
        # input("label")

        for i in range(1,b):
            # label = torch.cat((label, torch.bincount(target[i])),dim=0)
            label_i = torch.bincount(target[i].long(), minlength=150)[:150].unsqueeze(0)
            # print(label_i.shape)
            # input("label_i")
            label_i = torch.where(label_i>0., torch.full_like(label_i, 1), torch.full_like(label_i, 0))
            # print(label_i.shape)
            # input("label_i")
            # print(label.shape)
            # input("before cat label")
            label = torch.cat((label, label_i),dim=0)
            # print(label.shape)
            # input("after cat label")
        image_loss = 0.4*self.image_loss(vector, label.float())
        return image_loss

    def forward(self, x, pre_mask, gt=None, num_class=None, delta_mode=False, img=None, conv_weight=None, conv_bias=None):
        feat = x
        pre_mask = self.mask_conv(x)
        mask = torch.sigmoid(pre_mask)
        # mask = self.softmax(pre_mask)

        #.clone().detach()
        # efeat = self.fe_compact(x.clone().detach())
        efeat = self.fe_compact(img)
        # error_map = self.error_ds(torch.cat([efeat, pre_mask.clone().detach()], dim=1))
        # error_map = self.error_ds(torch.cat([efeat[-1], pre_mask.clone().detach()], dim=1))
        # mask_v, mask_label = pre_mask.topk(1, dim=1)
        # mask_onehot = torch.nn.functional.one_hot(mask_label.permute(0,2,3,1), mask.shape[1])
        # mask_onehot = mask_onehot.squeeze(3).permute(0,3,1,2)
        error_map = self.error_ds(torch.cat([efeat[-1], pre_mask], dim=1))
        e_v, e_label = error_map.topk(1, dim=1)
        e_onehot = torch.nn.functional.one_hot(e_label.permute(0,2,3,1), error_map.shape[1])
        e_onehot = e_onehot.squeeze(3).permute(0,3,1,2)
        # error_map = self.error_ds(torch.cat([img, pre_mask.clone().detach()], dim=1))
        # error_logit = torch.sigmoid(error_map.clone().detach())
        # error_logit = self.softmax(error_map.clone().detach())
        error_logit = self.softmax(error_map)
        # error_logit = self.softmax(error_map.clone().detach())

        # error_logit = torch.where(error_logit[:, 0, :, :].unsqueeze(1)-error_logit[:, 1, :, :].unsqueeze(1)>0.4,
        #                           error_logit[:, 0, :, :].unsqueeze(1),
        #                           torch.full_like(error_logit[:, 0, :, :].unsqueeze(1),0.))
        # mask = torch.sigmoid(self.mask_conv(x))
        # if not self.same_loss:
        #     if gt is not None:
        #         dice_loss = self.multi_class_dice_loss(mask, gt, num_class)
        b, k, h, w = mask.size()
        # class_context = self.context_g(pre_mask.clone().detach())
        # class_context = self.context_g(pre_mask)
        # # class_context = class_context.clone().detach()
        # image_vector = self.image_conv(class_context)
        # image_vector = torch.sigmoid(image_vector)
        if not self.same_loss:
            if gt is not None:
                dice_loss = self.multi_class_dice_loss(mask, gt, num_class)
                # error_gt_o = error_gt(pre_mask.clone().detach(), gt).float()
                # e_gt = 1 - error_gt_o
                # feat = torch.mul(e_onehot[:, 0, :, :].unsqueeze(1), feat)
                feat = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), feat)
                # feat = torch.mul(e_gt, feat)
                error_loss, eds_acc, pred_error_rate, mask_correct_rate, correct_correct_rate, error_correct_rate = accuracy_ce_error_loss(pre_mask, gt, error_map, self.error_criterion)
                # error_loss, eds_acc, pred_error_rate, mask_correct_rate, correct_correct_rate, error_correct_rate = accuracy_ce_error_loss(pre_mask, gt, error_logit, self.error_criterion)
                # error_loss, eds_acc, pred_error_rate, mask_correct_rate= accuracy_ce_error_loss(pre_mask, gt, error_map)
                # feat = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), feat)
                # feat = torch.mul(e_gt, feat)
            else:
                # feat = torch.mul(e_onehot[:, 0, :, :].unsqueeze(1), feat)
                feat = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), feat)
                # feat = torch.mul(e_gt, feat)

                # image_loss = self.image_classification_loss(image_vector.view(b, k), gt.view(b,-1),k,b)
        # error_logit = self.softmax(error_map)

        # b k h w
        #10/21 zhushi
        # mask_onehot = mask_onehot.float().view(b, k, -1)
        mask = mask.view(b, k, -1)
        
        # error map 机制
        # feat = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), feat.clone().detach())
        # feat = torch.mul(e_gt, feat)
        # feat = torch.mul(torch.where(error_logit[:, 0, :, :].unsqueeze(1)-error_logit[:, 1, :, :].unsqueeze(1)>0.,
        #                              torch.full_like(error_logit[:, 0, :, :].unsqueeze(1), 1.),
        #                              torch.full_like(error_logit[:, 0, :, :].unsqueeze(1), 0.)), feat)
        # feat_f = nn.functional.adaptive_avg_pool2d(feat_f, (1, 1))
        # filters = self.feat_conv(feat_f)
        # filters = nn.functional.adaptive_avg_pool2d(feat_f, (1, 1))
        # filters = self.generator()
        # filters = nn.functional.adaptive_avg_pool2d(filters, (1, 1))
        #10/21 zhushi
        
        feat = feat.view(b, self.ichn, -1)
        feat = feat.permute(0, 2, 1)
        # if delta_mode :
            # class_feat = torch.bmm(mask, feat) / (h * w)
        # else:
            # class_feat = torch.bmm(mask, feat) / (h * w)
        
        class_feat = torch.bmm(mask, feat) / (h * w)
        # class_feat = torch.bmm(mask_onehot, feat) / (h * w)
        class_feat = class_feat.view(b, k * self.ichn, 1, 1)
        filters = self.filter_conv(class_feat)
        filters = filters.view(b * k, self.ichn, 1, 1)

        x = x.view(-1, h, w).unsqueeze(0)
        if delta_mode:
            # pred = F.conv2d(x, filters.add(self.mask_conv.weight.repeat([b,1,1,1])),
            #                 # bias.add(self.mask_conv.bias.repeat(b)),
            #                 groups=b).view(b, k, h, w)
            # pred = F.conv2d(x, filters.add(conv_weight.repeat([b,1,1,1])),
            #                 # conv_bias.repeat(b),
            #                 # bias.add(self.mask_conv.bias.clone().detach().repeat(b)),
            #                 groups=b).view(b, k, h, w)
            fix_pred = F.conv2d(x, filters,
                                # bias,
                                groups=b).view(b, k, h, w)
            if gt is not None:
                # fix_pred = torch.mul(e_onehot[:, 1, :, :].unsqueeze(1), fix_pred) 
                fix_pred = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), fix_pred)
                # fix_pred = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), fix_pred)
                # fix_pred = torch.mul(error_gt_o, fix_pred) 
            else:
                # fix_pred = torch.mul(e_onehot[:, 1, :, :].unsqueeze(1), fix_pred) 
                fix_pred = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), fix_pred)
                # fix_pred = torch.mul(error_logit[:, 0, :, :].unsqueeze(1), fix_pred)

            pred = fix_pred + pre_mask#.clone().detach()
            # pred = pred + pre_mask.clone().detach()
            # print(delta_mode)
        else:
            pred = F.conv2d(x, filters, groups=b).view(b, k, h, w)

        # 1009跟果子狸讨论之前的做法-跟他想的基本一样的
        # class_relation = torch.bmm(mask.clone().detach().view(b,k,-1), mask.clone().detach().view(b,k,-1).permute(0,2,1)) / (h * w)
        # print(class_relation.shape)
        # pred = torch.bmm(class_relation, pred.view(b,k,-1)).view(b, k, h, w)
        # pred = torch.mul(image_vector.view(b,k,1), pred.view(b,k,-1)).view(b, k, h, w)

        if not self.same_loss:
            if gt is not None:
                # return pred, {'loss_CFlayer': dice_loss, 'loss_imageclass': image_loss, 'image_vector_max':image_vector.max()}
                # return pred, {'loss_CFlayer': dice_loss, 'error_loss': error_loss, 'error_acc': eds_acc, 'pred_error_rate': pred_error_rate, 'mask_correct_rate': mask_correct_rate}
                return pred, {'loss_CFlayer': dice_loss, 'error_loss': error_loss, 'error_acc': eds_acc, 'pred_error_rate': pred_error_rate, 'mask_correct_rate': mask_correct_rate, 'correct_correct_rate': correct_correct_rate, 'error_correct_rate': error_correct_rate}, pre_mask
                # return pred, { 'error_loss': error_loss, 'error_acc': eds_acc, 'pred_error_rate': pred_error_rate, 'mask_correct_rate': mask_correct_rate, 'correct_correct_rate': correct_correct_rate, 'error_correct_rate': error_correct_rate}, pre_mask
                # return pred, {'loss_CFlayer': dice_loss}

            else:
                return pred
        else:
            if gt is not None:
                return pred, pre_mask
            else:
                return pred
            # return pred, mask
        # return pred

@HEADS.register_module()
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=build_norm_layer, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False, norm_cfg=None):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            # self.bn = norm_layer(out_planes, eps=bn_eps)
            self.bn_name, bn = norm_layer(norm_cfg, out_planes)
            self.add_module(self.bn_name, bn)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

@HEADS.register_module()
class CFPSPHead(PSPHead):

    def __init__(self, **kwargs):
        self.delta_mode = kwargs.pop('delta_mode')
        self.same_cfloss = kwargs.pop('same_loss')
        super(CFPSPHead, self).__init__(**kwargs)
        self.head_layer = nn.Sequential(
            ConvBnRelu(2048, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True,
                       has_bias=False,
                       norm_cfg=self.norm_cfg,
                       norm_layer=build_norm_layer),
            nn.Dropout2d(0.1, inplace=False)
        )
        # self.head_layer.requires_grad_(False)
        # if self.same_cfloss:
        #     self.cf_layer = CEConditionalFilterLayer(
        #         512,
        #         self.num_classes,
        #         self.loss_decode,
        #         self.align_corners,
        #         self.sampler,
        #         self.ignore_index
        #         )
        # else:
        self.cf_layer = ConditionalFilterLayer(
            512,
            self.num_classes,
            same_loss=self.same_cfloss)

    def freeze(self):
        self.requires_grad_(False)
    def forward(self, inputs, label=None, img=None):
        """Forward function."""
        x = self._transform_inputs(inputs)
        # psp_outs = [x]
        # psp_outs.extend(self.psp_modules(x))
        # psp_outs = torch.cat(psp_outs, dim=1)
        # psp_outs = self.bottleneck(psp_outs)
        # pspoutput = self.cls_seg(psp_outs)
        output = self.head_layer(inputs[-1])
        # output = self.cls_seg(output)
        if label is not None:
            b, useless, h, w = label.size()
            aux_label = F.interpolate((label.view(b, 1, h, w)).float(),
                                      size=output.shape[2:],
                                    #   scale_factor=0.125,
                                      mode="nearest",
                                    #   align_corners=(self.align_corners))
            )
            # aux_label = aux_label.view(b, h // 8, w // 8)
            aux_label = aux_label.squeeze(1)
        else:
            aux_label = None
        # if img is not None:
        #     ib, iu, ih, iw = img.size()
        #     re_img = F.interpolate(img.float(),
        #                         #    scale_factor=0.125,
        #                            size=output.shape[2:],
        #                            mode="bilinear",
        #                            align_corners=(self.align_corners))
        # final_output = self.cf_layer(output, pspoutput, aux_label, self.num_classes, self.delta_mode, img=img, conv_weight=self.conv_seg.weight, conv_bias=self.conv_seg.bias)
        final_output = self.cf_layer(output, None, aux_label, self.num_classes, self.delta_mode, img=img, conv_weight=self.conv_seg.weight, conv_bias=self.conv_seg.bias)
        # final_output2 = self.cf_layer(final_output, None, aux_label, self.num_classes, self.delta_mode, img=img, conv_weight=self.conv_seg.weight, conv_bias=self.conv_seg.bias)
        if label is not None:
            fm = final_output[0]
            dice_loss = final_output[1]
            pre_mask = final_output[2]
            return fm, dice_loss, pre_mask
            # return fm, dice_loss
        else:
            fm = final_output
            return fm

        # if self.delta_mode:
        #     fm = fm + output
        # return fm

    @force_fp32(apply_to=('seg_logit', 'dice_loss'))
    def losses(self, seg_logit, dice_loss, seg_label, error_map=None, coarse_logit=None):
        """Compute segmentation loss."""
        loss = dict()
        shape = seg_label.shape[2:]
        seg_logit = resize(
            input=seg_logit,
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        if coarse_logit is not None:
            coarse_logit = resize(
                input=coarse_logit,
                size=(seg_label.shape[2:]),
                mode='bilinear',
                align_corners=(self.align_corners))
        ###########################################

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            if coarse_logit is not None:
                coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
        # loss['coarse_loss_seg'] = 0.4*self.loss_decode(
        #     coarse_logit,
        #     seg_label,
        #     weight=coarse_seg_weight,
        #     ignore_index=(self.ignore_index))
        loss['acc'] = accuracy(seg_logit, seg_label)
        if coarse_logit is not None:
            loss['coarse_loss_seg'] = 0.4*self.loss_decode(
                coarse_logit,
                seg_label,
                weight=coarse_seg_weight,
                ignore_index=(self.ignore_index))
            loss['acc_psp'] = accuracy(coarse_logit, seg_label)

        loss.update(dice_loss)

        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        # print(img_metas)
        seg_logits, dice_loss, pre_mask = self.forward(inputs,  label=gt_semantic_seg, img=img)
        losses = self.losses(seg_logits, dice_loss, gt_semantic_seg, coarse_logit=pre_mask)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        # fm, dice_loss, pre_mask = self.forward(inputs, label=kwargs['gt_semantic_seg'][0], img = kwargs['img'])
        fm = self.forward(inputs, label=None, img = kwargs['img'])
        return fm

# @HEADS.register_module()
# class PSPHead(BaseDecodeHead):
#     """Pyramid Scene Parsing Network.

#     This head is the implementation of
#     `PSPNet <https://arxiv.org/abs/1612.01105>`_.

#     Args:
#         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
#             Module. Default: (1, 2, 3, 6).
#     """

#     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
#         super(PSPHead, self).__init__(**kwargs)
#         assert isinstance(pool_scales, (list, tuple))
#         self.pool_scales = pool_scales
#         self.psp_modules = PPM(
#             self.pool_scales,
#             self.in_channels,
#             self.channels,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg,
#             align_corners=self.align_corners)
#         self.bottleneck = ConvModule(
#             self.in_channels + len(pool_scales) * self.channels,
#             self.channels,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         # self.freeze()

#     def forward(self, inputs):
#         """Forward function."""
#         x = self._transform_inputs(inputs)
#         psp_outs = [x]
#         psp_outs.extend(self.psp_modules(x))
#         psp_outs = torch.cat(psp_outs, dim=1)
#         output = self.bottleneck(psp_outs)
#         output = self.cls_seg(output)
#         return output

#     def freeze(self):
#         self.requires_grad_(False)
