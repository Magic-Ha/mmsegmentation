import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmseg.ops import resize
from ..builder import HEADS
# from ..losses import accuracy, accuracy_error_loss
from ..losses import accuracy, accuracy_error_loss, accuracy_ce_error_loss

from .decode_head import BaseDecodeHead


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
            temp = temp.detach()
            coarse_output = coarse_output.detach()
            f = self.f_compact(x.detach())
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
                if mode != 'test':
                    error_map.retain_grad()
                # error_logit = torch.sigmoid(error_map)
                # error_logit = torch.sigmoid(error_map.clone().detach())

                if output.shape[2:] != error_map.shape[2:]:
                    output = resize(
                        input=output,
                        size=(error_map.shape[2:]),
                        mode='bilinear',
                        align_corners=(self.align_corners))
                # FIXME:这里detach掉了 记得改回来啊
                # error_logit = self.softmax(error_map.clone().detach())
                error_logit = self.softmax(error_map)
                # if mode != 'test':
                #     error_logit.retain_grad()
                # output = torch.mul(error_map[:, 0, :, :].unsqueeze(1),
                #                    output) + \加接着下面
                # torch.mul(error_map[:, 1, :, :].unsqueeze(1), coarse_output)
                # output = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), output) + torch.mul(error_logit[:, 0, :, :].unsqueeze(1), coarse_output)
                output = torch.mul(error_logit[:, 1, :, :].unsqueeze(1), output) + coarse_output


            if mode == 'stage2' or mode == 'stage3':
                if self.with_error_ds:
                    return (output, coarse_output, error_logit)
                    # return (output, coarse_output, error_map)
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
            loss['error_loss_seg'], loss['acc_error_pred'] = accuracy_ce_error_loss(coarse_logit,
                                                            seg_label,
                                                            error_logit)
        #     #print(s==loss['acc_seg_coarse'])
        #     # loss['error_loss_seg'] *= 10
        #     # loss['acc_error_pred'] = self.eds_correct_rate

        return loss

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
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
        return self.forward(inputs)


class ConditionalFilterLayer(nn.Module):
    def __init__(self, ichn, ochn):
        super(ConditionalFilterLayer, self).__init__()
        self.ichn = ichn
        self.ochn = ochn
        self.mask_conv = nn.Conv2d(ichn, ochn, kernel_size=1)
        self.filter_conv = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)

    def multi_class_dice_loss(self, mask, target, num_class):
        target = torch.where(target==255,torch.full_like(target, 0), target)
        target = F.one_hot(target.long(), num_class + 1)[:, :, :, 1:]
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

    def ata_loss(self, filters, target, num_class, batch_size):
        target_b = target.view(batch_size, -1)
        lb_shown = [torch.unique(target_b[bi], sorted=True).long() for bi in range(0, batch_size)]
        for i in range(0, batch_size):
            if lb_shown[i].max()==255:
                lb_shown[i] = lb_shown[i][0:lb_shown[i].shape[0]-1]
        # label_onehot = torch.sign(label_bin)
        ata_mask = 1. - torch.eye(num_class).to('cuda')
        batch_mask = torch.zeros_like(ata_mask).to('cuda')
        bml = [batch_mask.index_fill(0, lb_shown[bi].squeeze(), 1) for bi in range(0, batch_size)]
        # bmlc = [ata_mask.mul(bml[bi]).mul(bml[bi].T) for bi in range(0, batch_size)]
        bmlc = [ata_mask.mul(bml[bi]) for bi in range(0, batch_size)]
        # [batch_mask.index_fill(0, label_bin[bi], 1), for bi in range(0, batch_size)]
        # dist_matrix = [torch.masked_select(torch.mm(filters.view(batch_size, num_class, -1)[bi],
        # dist_matrix = [torch.masked_select(torch.mm(filters.view(batch_size, num_class, -1)[bi],

        # dist_matrix = [(torch.mm(filters.view(batch_size, num_class, -1)[bi],
        #                         # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
        #                         # filters.view(batch_size, num_class, -1)[bi].T).abs(), bmlc[bi].bool()) for bi in range(0, batch_size)]
        #                         # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
        #                         filters.view(batch_size, num_class, -1)[bi].T)/2.).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
        #                         # filters.view(batch_size, num_class, -1)[bi].T).mul(bmlc[bi]) for bi in range(0, batch_size)]
        # # for i in range(0,batch_size):
        # #     assert not bool((torch.isnan(dist_matrix[i]).sum()>0).cpu())
        # dist_matrix_mean = [dist_matrix[bi].mean(dim=0) for bi in range(0, batch_size)]

        # filters = F.normalize(filters, p=2, dim=1)
        dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs(), bmlc[bi].bool()) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                #之前一直用这个filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]][:,lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask)[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(bmlc[bi]) for bi in range(0, batch_size)]
        
        # print(dist_matrix[0])
        # print(dist_matrix[0].shape)
        # print(dist_matrix[1].shape)
        # print(dist_matrix[2].shape)
        # print(dist_matrix[3].shape)
        # print(dist_matrix)
        # for i in range(0,batch_size):
        #     assert not bool((torch.isnan(dist_matrix_mean[i]).sum()>0).cpu())
        cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # cosdist = torch.cat([dist_matrix[bi].unsqueeze(0) for bi in range(0, batch_size)], dim=0).sum(0)
        # print(cosdist)
        # print(cosdist.shape)
        # if cosdist>=2.0:
        #     print("error")
        # cosdist = [dist_matrix[bi].mean() for bi in range(0, batch_size)]
        # print(cosdist)
        # return cosdist.mean(), dist_matrix
        # if cosdist.mean()>0.5:
            # print("dayu 0.5 ")
        # assert not bool((torch.isnan(cosdist).sum()>0).cpu())
        # assert not bool(torch.isnan(cosdist.mean()).cpu())
        # assert not bool(torch.isnan(cosdist[~torch.isnan(cosdist)]).sum().cpu()>0)
        
        return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix

    def forward(self, x, gt=None, num_class=None, delta_mode=False):
        flag = True
        feat = x
        pre_mask = self.mask_conv(x)
        mask = torch.sigmoid(pre_mask)
        # mask = torch.sigmoid(self.mask_conv(x))

        if gt is not None:
            dice_loss = self.multi_class_dice_loss(mask, gt, num_class)

        # b k h w
        b, k, h, w = mask.size()
        mask = mask.view(b, k, -1)

        feat = feat.view(b, self.ichn, -1)
        feat = feat.permute(0, 2, 1)
        # b, k, ichn
        class_feat = torch.bmm(mask, feat) / (h * w)
        class_feat = class_feat.view(b, k * self.ichn, 1, 1)

        # b, k*ichn, 1, 1
        filters = self.filter_conv(class_feat)
        filters = filters.view(b * k, self.ichn, 1, 1)

        x = x.view(-1, h, w).unsqueeze(0)
        if delta_mode:
            pred = F.conv2d(x, filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), groups=b).view(b, k, h, w)
            # pred = pred + pre_mask.clone().detach()
            # print(delta_mode)
        else:
            pred = F.conv2d(x, filters, groups=b).view(b, k, h, w)

        if gt is not None:
            # filters
            # cosdist, dist_matrix = self.ata_loss(filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), gt, num_class, b)
            cosdist, dist_matrix = self.ata_loss(filters, gt, num_class, b)
            # cpcosdist, cp_dist_matrix = self.ata_loss(self.mask_conv.weight.repeat([b,1,1,1]), gt, num_class, b)
            # print(cosdist)
            # return pred, {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist,
            return pred, {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist, 'pre_mask': pre_mask}
            # return pred, {'loss_CFlayer': dice_loss}
            # return pred, mask
        return pred

@HEADS.register_module()
class CFPSPHead(PSPHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        self.delta_mode = kwargs.pop('delta_mode')
        self.same_cfloss = kwargs.pop('same_loss')
        super(CFPSPHead, self).__init__(**kwargs)
        self.cf_layer = ConditionalFilterLayer(512, self.num_classes)

    def forward(self, inputs, label=None):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        # output = self.cls_seg(output)
        if label is not None:
            b, useless, h, w = label.size()
            aux_label = F.interpolate((label.view(b, 1, h, w)).float(),
                                      scale_factor=0.125,
                                      mode="nearest")
            aux_label = aux_label.view(b, h // 8, w // 8)
        else:
            aux_label = None
        final_output = self.cf_layer(output, aux_label, self.num_classes, self.delta_mode)
        if label is not None:
            fm = final_output[0]
            dice_loss = final_output[1]
            return fm, dice_loss
        else:
            fm = final_output
            return fm

        # if self.delta_mode:
        #     fm = fm + output
        # return fm

    @force_fp32(apply_to=('seg_logit', 'extra_loss'))
    def losses(self, seg_logit, extra_loss, seg_label, error_map=None):
        """Compute segmentation loss."""
        loss = dict()
        pre_seg_logit = resize(
            input=extra_loss.pop('pre_mask'),
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        seg_logit = resize(
            input=seg_logit,
            size=(seg_label.shape[2:]),
            mode='bilinear',
            align_corners=(self.align_corners))
        # if coarse_logit.shape[2:] != seg_label.shape[2:]:
        #     coarse_logit = resize(
        #         input=coarse_logit,
        #         size=(seg_label.shape[2:]),
        #         mode='bilinear',
        #         align_corners=(self.align_corners))
        ############################################
        # if self.with_error_ds:
        #     error_logit = resize(
        #         input=error_map,
        #         size=(seg_label.shape[2:]),
        #         mode='bilinear',
        #         align_corners=(self.align_corners))
        ##############################################
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            # coarse_seg_weight = self.sampler.sample(coarse_logit, seg_label)
        else:
            seg_weight = None
            # coarse_seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
        # loss['scores_pre_seg'] = 0.4*self.loss_decode(
        #     pre_seg_logit,
        #     seg_label,
        #     weight=seg_weight,
        #     ignore_index=(self.ignore_index))
        # loss['coarse_loss_seg'] = self.loss_decode(
        #     coarse_logit,
        #     seg_label,
        #     weight=coarse_seg_weight,
        #     ignore_index=(self.ignore_index))
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['acc_pre_seg'] = accuracy(pre_seg_logit, seg_label)

        loss.update(extra_loss)

        # loss['loss_dice'] = 0.4 * extra_loss[0]
        # loss['loss_cos_dist'] = 0.5 * extra_loss[1]
        return loss
        # loss['acc_seg_coarse'] = accuracy(coarse_logit, seg_label)
        # if self.with_error_ds:
        #     loss['error_loss_seg'], loss['acc_error_pred'] = accuracy_ce_error_loss(coarse_logit,
        #                                                     seg_label,
        #                                                     error_logit)
        # #     #print(s==loss['acc_seg_coarse'])
        # #     # loss['error_loss_seg'] *= 10
        # #     # loss['acc_error_pred'] = self.eds_correct_rate

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        seg_logits, dice_loss = self.forward(inputs,  label=gt_semantic_seg)
        losses = self.losses(seg_logits, dice_loss, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        return self.forward(inputs)
