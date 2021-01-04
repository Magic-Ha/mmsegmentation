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
        # filters = F.normalize(filters, p=2, dim=1)
        dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs(), bmlc[bi].bool()) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T)/2.).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                filters.view(batch_size, num_class, -1)[bi].T)[lb_shown[bi]] for bi in range(0, batch_size)]
                                
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]][:,lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask)[lb_shown[bi]] for bi in range(0, batch_size)]
                                # filters.view(batch_size, num_class, -1)[bi].T).mul(bmlc[bi]) for bi in range(0, batch_size)]

        cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix
        return torch.exp(cosdist[~torch.isnan(cosdist)].mean()), dist_matrix
        # return torch.exp(10*cosdist[~torch.isnan(cosdist)].sum()), dist_matrix
        # return torch.exp(cosdist[~torch.isnan(cosdist)].sum()), dist_matrix

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


class ConditionalFilterLayer(nn.Module):
    def __init__(self, ichn, ochn):
        super(ConditionalFilterLayer, self).__init__()
        self.ichn = ichn
        self.ochn = ochn
        self.mask_conv = nn.Conv2d(ichn, 256, kernel_size=1)
        self.mask_conv2 = nn.Conv2d(256, ochn, kernel_size=1)
        self.filter_conv = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)
        self.filter_convloop = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)
    def multi_class_dice_loss(self, mask, target, num_class):
        target = torch.where(target==255,torch.full_like(target, 151), target)
        self.max_labelv, self.max_labeli = target.max(dim=1)
        # print("zuida:",int(self.max_labelv.view(-1).min()), "- \t")
        # print("zuixiao:", int(self.max_labelv.view(-1).max()), "! \t")
        # if int(self.max_labelv.view(-1).max())==150:
        #     input("150 found")
        target = F.one_hot(target.long(), num_class+2)[:,:,:,:-2]
        # target = F.one_hot(target.long(), num_class)
        # target = F.one_hot(target.long(), num_class)[:, :, :, 1:]
        # print(target.size())
        target = target.permute(0, 3, 1, 2)
        mask = mask.contiguous().view(mask.size()[0], num_class, -1)
        target = target.contiguous().view(target.size()[0], num_class,
                                          -1).float()

        a = torch.sum(mask * target, 2)
        b = torch.sum(mask * mask, 2) + 0.00001
        # b = torch.sum(mask * mask, 2) + 0.001
        c = torch.sum(target * target, 2) + 0.00001
        # c = torch.sum(target * target, 2) + 0.001
        d = (2 * a + 1) / (b + c + 1)
        # return (1 - d).mean()
        return (1 - d).mean()

    def ata_loss(self, filters, target, num_class, batch_size):
        target_b = target.view(batch_size, -1)
        lb_shown = [torch.unique(target_b[bi], sorted=True).long() for bi in range(0, batch_size)]
        for i in range(0, batch_size):
            if lb_shown[i].max()==255:
                lb_shown[i] = lb_shown[i][0:lb_shown[i].shape[0]-1]
        # label_onehot = torch.sign(label_bin)
        ata_mask = 1. - torch.eye(num_class).to('cuda')
        # ZHE GE BU YAO SHAN
        # batch_mask = torch.zeros_like(ata_mask).to('cuda')
        # bml = [batch_mask.index_fill(0, lb_shown[bi].squeeze(), 1) for bi in range(0, batch_size)]
        # # bmlc = [ata_mask.mul(bml[bi]).mul(bml[bi].T) for bi in range(0, batch_size)]
        # bmlc = [ata_mask.mul(bml[bi]) for bi in range(0, batch_size)]

        # filters = F.normalize(filters, p=2, dim=1)
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
        # cosdist = torch.cat([dist_matrix[bi].sum().unsqueeze(0) for bi in range(0, batch_size)], dim=0)

        # return cosdist[~torch.isnan(cosdist)].mean(), dist_matrix
        return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix

    def cfloop(self, filter_conv, feat, mask, x, b, k, h, w, topk, delta_mode=False):
        
        class_feat = torch.bmm(mask, feat) / (h * w)
        class_feat = class_feat.view(b, k * self.ichn, 1, 1)
        filters = filter_conv(class_feat)
        filters = filters.view(b * k, self.ichn, 1, 1)
        if delta_mode:
            pred = F.conv2d(x, filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), groups=b).view(b, k, h, w)
            # pred = pred + pre_mask.clone().detach()
            # print(delta_mode)
        else:
            pred = F.conv2d(x, filters, groups=b).view(b, k, h, w)
        return filters, pred

    def forward(self, x, gt=None, num_class=None, delta_mode=False, dpm=None):
        feat = x
        # pre_mask = x
        mask = self.mask_conv(x)
        mask = torch.relu(mask)
        mask = self.mask_conv2(mask)
        # mask = torch.sigmoid(mask)
        mask = torch.softmax(mask,dim=1)
        pre_mask = mask
        # mask = torch.sigmoid(self.mask_conv(x))
        # if gt is not None:
        #     dice_loss = self.multi_class_dice_loss(mask, gt, num_class)
            # pass
        # b k h w
        b, k, h, w = mask.size()
        mask = mask.view(b, k, -1)#B*C*N

        feat = feat.view(b, self.ichn, -1)#B*C*N
        feat = feat.permute(0, 2, 1)#B*N*C
        x = x.view(-1, h, w).unsqueeze(0)
        # if dpm is not None:
        #     feat = dpm(feat)
        filters, pred = self.cfloop(self.filter_conv, feat, mask, x, b, k, h, w, delta_mode)
        if gt is not None:
            # filters
            # cosdist, dist_matrix = self.ata_loss(filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), gt, num_class, b)
            cosdist, dist_matrix = self.ata_loss(filters, gt, num_class, b)
            # return pred, {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist,
            return pred, {'loss_cosdist': cosdist,
            # return pred, {'loss_cosdist': cosdist,
                        #    'loss_cpcosdist': cpcosdist, 'pre_mask': pre_mask}
                           'pre_mask': pre_mask}
        return pred

@HEADS.register_module()
class CFDSASPPHead(DepthwiseSeparableASPPHead):
    def __init__(self, **kwargs):
        super(CFDSASPPHead, self).__init__(**kwargs)
        self.cf_layer = ConditionalFilterLayer(512, self.num_classes)
    def forward(self, inputs, label=None):
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
        # output = self.cls_seg(output)
        # output = self.cls_seg(output)
        if label is not None:
            b, useless, h, w = label.size()
            # b, useless, h, w = output.size()
            aux_label = F.interpolate((label.view(b, 1, h, w)).float(),
                                      size=output.size()[2:],
                                      mode="nearest")
            aux_label = aux_label.squeeze(dim=1)
            # aux_label = aux_label.view(b, h // 4, w // 4)
        else:
            aux_label = None
        if self.dropout is not None:
            dpm = self.dropout
        final_output = self.cf_layer(output, aux_label, self.num_classes, False)
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
        # loss['max_segl'] = max(seg_label)
        loss['loss_dice'] = self.cf_layer.multi_class_dice_loss(
                                                pre_seg_logit,
                                                seg_label.squeeze(dim=1),
                                                self.num_classes)
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
        loss['losg_pre_seg'] = 0.4*self.loss_decode(
            pre_seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=(self.ignore_index))
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

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        seg_logits, output = self.forward(inputs,  label=gt_semantic_seg)
        losses = self.losses(seg_logits, output, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        return self.forward(inputs)
