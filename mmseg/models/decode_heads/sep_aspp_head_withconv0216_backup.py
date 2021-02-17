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
        # self.mask_conv = nn.Conv2d(ichn, 256, kernel_size=1)
        # self.mask_conv2 = nn.Conv2d(256, ochn, kernel_size=1)
        self.filter_conv = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)
        self.filter_convloop = nn.Conv2d(ochn * ichn, ochn * ichn, kernel_size=1,
                                     groups=ochn)
        self.intra_s = 16
        self.bottom_num = 4

    def multi_class_dice_loss(self, mask, target, num_class):
        # target = torch.where(target==255,torch.full_like(target, 0), target)
        target = F.one_hot(target.long(), 256)[:, :, :, :150]
        # target = F.one_hot(target.long(), num_class + 1)[:, :, :, 1:]
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
        d = (2 * a) / (b + c)
        # return (1 - d).mean()
        return (1 - d).mean()

    def ata_loss(self, filters, target, num_class, batch_size, filters_2=None):
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
        if filters_2 is None:
            dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                    # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T).abs(), bmlc[bi].bool()) for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T).abs().mul(bmlc[bi]) for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T)/2.).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T)/2.).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]

                                    # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T).softmax(dim=1).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                    filters.view(batch_size, num_class, -1)[bi].T).softmax(dim=1) for bi in range(0, batch_size)]
                                    
                                    # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask).abs()[lb_shown[bi]][:,lb_shown[bi]] for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T).mul(ata_mask)[lb_shown[bi]] for bi in range(0, batch_size)]
                                    # filters.view(batch_size, num_class, -1)[bi].T).mul(bmlc[bi]) for bi in range(0, batch_size)]
        else:
            dist_matrix = [torch.mm(filters.view(batch_size, num_class, -1)[bi],
                                    # filters_2.view(num_class, -1).T).mul(ata_mask).abs()[lb_shown[bi]] for bi in range(0, batch_size)]
                                    filters.view(batch_size, num_class, -1)[bi].T).softmax(dim=1) for bi in range(0, batch_size)]

        cosdist = torch.cat([dist_matrix[bi].mul(ata_mask)[lb_shown[bi]].sum(dim=1).mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # cosdist = torch.cat([dist_matrix[bi].sum(dim=1).mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # cosdist = torch.cat([dist_matrix[bi].mean().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        # cosdist = torch.cat([dist_matrix[bi].sum().unsqueeze(0) for bi in range(0, batch_size)], dim=0)
        eyedist = torch.cat([-torch.log(dist_matrix[bi].mul(1.-ata_mask)[lb_shown[bi]].sum(dim=1).mean().unsqueeze(0)) for bi in range(0, batch_size)], dim=0)

        return cosdist[~torch.isnan(cosdist)].mean(), eyedist[~torch.isnan(eyedist)].mean()
        # return cosdist[~torch.isnan(cosdist)].sum(), dist_matrix
        # return 10*cosdist[~torch.isnan(cosdist)].mean(), dist_matrix

    def cfloop(self, filter_conv, feat, mask, x, b, k, h, w, delta_mode=False):
        class_feat = torch.bmm(mask, feat) / (h * w)
        class_feat = class_feat.view(b, k * self.ichn, 1, 1)
        filters = filter_conv(class_feat)
        filters = filters.view(b * k, self.ichn, 1, 1)
        # if delta_mode:
        #     pred = F.conv2d(x, filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), groups=b).view(b, k, h, w)
        #     # pred = pred + pre_mask.clone().detach()
        #     # print(delta_mode)
        # else:
        #     pred = F.conv2d(x, filters, groups=b).view(b, k, h, w)
        return filters

    def forward(self, x, output_branch, gt=None, num_class=None, delta_mode=False, dpm=None, softmax_mask=False, topk_filter=None, intra_weight=1.0, extra_filter=None):
        feat = x
        pre_mask = x
        d = x.shape[1]
        # mask = self.mask_conv(x)
        # mask = torch.relu(mask)
        # mask = self.mask_conv2(mask)
        if softmax_mask:
            # mask = torch.softmax(mask, dim=1)
            mask = torch.softmax(output_branch, dim=1)
        else:
            # mask = torch.sigmoid(mask)
            mask = torch.sigmoid(output_branch)
        # mask = torch.sigmoid(self.mask_conv(x))
        # mask = torch.softmax(mask, dim=1)

        if gt is not None:
            dice_loss = self.multi_class_dice_loss(mask, gt, num_class)
        # b k h w
        b, k, h, w = mask.size()
        
        # if topk_filter is not None:
        #     # cat_value, cat_result = mask.topk(topk_filter, dim=1)
        #     # max_mask = F.one_hot(cat_result.long(), num_class)
        #     # class_selected_mask = max_mask.sum(dim=1).permute(0,3,1,2).float()
        #     # mask = torch.mul(class_selected_mask, mask)
        #     pass
        # # mask = mask.view(b, k, -1)

        feat = feat.view(b, self.ichn, -1)
        feat = feat.permute(0, 2, 1)
        # x = x.view(-1, h, w).unsqueeze(0)
        if dpm is not None:
            feat = dpm(feat)
        # filters, pred = self.cfloop(self.filter_conv, feat, mask.view(b, k, -1), x.view(-1, h, w).unsqueeze(0), b, k, h, w, delta_mode)
        filters = self.cfloop(self.filter_conv, feat, mask.view(b, k, -1), x.view(-1, h, w).unsqueeze(0), b, k, h, w, delta_mode)
        # gtt = F.one_hot(gt.long(), 256)[:, :, :, :150]
        # gtt = gtt.permute(0, 3, 1, 2).contiguous()
        # filters = self.cfloop(self.filter_conv, feat, gtt.view(b, k, -1).float(), x.view(-1, h, w).unsqueeze(0), b, k, h, w, delta_mode)
        # if gt is not None and topk_filter is None:
        #     # cat_value, cat_result = mask.topk(3, dim=1)
        #     # max_mask = F.one_hot(cat_result.long(), num_class)
        #     # class_selected_mask = max_mask.sum(dim=1).permute(0,3,1,2).float()
        #     # mask = torch.mul(class_selected_mask, mask)
            
        #     target_mask = torch.where(gt==255,torch.full_like(gt, 0), gt)
        #     target_mask = F.one_hot(target_mask.long(), num_class + 1)[:, :, :, 1:]

        #     # cat_value, cat_result = mask.max(dim=1)
        #     # max_mask = F.one_hot(cat_result.long(), num_class)
 
        #     # class_selected_mask = max_mask.permute(0,3,1,2) #4*150*h*w
        #     class_selected_mask = target_mask.permute(0,3,1,2) #4*150*h*w
        #     class_num = class_selected_mask.view(b,k,h*w).sum(dim=-1)
        #     order_num, order_indice = torch.sort(class_num, dim=-1, descending=True)
            
        #     batch_index = torch.arange(0,b,1,device=order_indice.device)
        #     i = 0
        #     for i in range(self.intra_s):
        #         # batch_index = torch.randint(0, b, b, decive=order_indice.device)
        #         # iter_mask = class_selected_mask[batch_index, order_indice[:, i], :, :].unsqueeze(1).repeat(1, d, 1, 1)
        #         # iter_num_flag = (class_num[:, i]>128).float().cuda()
        #         iter_num = order_num[:, i]
        #         iter_batch_num_mask = (iter_num >= self.bottom_num)
        #         if bool(iter_batch_num_mask.sum()):
        #             iter_batch_index = batch_index[iter_batch_num_mask]
        #             iter_num = iter_num[iter_batch_num_mask]
        #             ib = iter_num.shape[0]

        #             # order_indice[:, i][iter_batch_num_mask]
        #             # iter_mask = class_selected_mask[batch_index, order_indice[:, i][iter_num>=256], :, :]
        #             iter_mask = class_selected_mask[iter_batch_index, order_indice[:, i][iter_batch_num_mask], :, :]
        #             #################################
        #             # 这里是减少空间与计算量版本的双循环loss计算                    
        #             iter_selected_feat = x[iter_batch_num_mask].permute(0,2,3,1)[iter_mask.bool()] #B*H*W*D->BN*D
        #             batj_weight_feat = filters.detach().view(b,k,d)[iter_batch_index, order_indice[:, i][iter_batch_num_mask], :]
        #             # batj_weight_feat = filters.view(b,k,d)[iter_batch_index, order_indice[:, i][iter_batch_num_mask], :]
        #             # print("test")
        #             j = 0
        #             p = 0
        #             iter_feat = []
        #             iter_weight_feat = []
        #             for j in range(ib):
        #                 selected_temp = iter_selected_feat[p:p+int(iter_num[j]), :]
        #                 iter_feat.append(selected_temp)
        #                 # batj_mean_feat = selected_temp.mean(dim=0).repeat(int(iter_num[j]), 1)
        #                 # iter_mean_feat.append(batj_mean_feat)
        #                 iter_weight_feat.append(batj_weight_feat[j].repeat(int(iter_num[j]), 1))
        #                 p += int(iter_num[j])
        #             # batch_dist = [( torch.mul(iter_feat[q], iter_feat[q])
        #             #               - 2*torch.mul(iter_feat[q], iter_mean_feat[q])
        #             #               + torch.mul(iter_mean_feat[q], iter_mean_feat[q])).mean().unsqueeze(0) 
        #             #               for q in range(ib)]
        #             # batch_dist = [( torch.mul(iter_feat[q], iter_feat[q])
        #             #               - 2*torch.mul(iter_feat[q], iter_weight_feat[q])
        #             #               + torch.mul(iter_weight_feat[q], iter_weight_feat[q])).mean().unsqueeze(0) 
        #             #               for q in range(ib)]
        #             # batch_dist = [1 - 2*torch.mul(iter_feat[q], iter_weight_feat[q]).mean().unsqueeze(0) 
        #             batch_dist = [(1 - (2*torch.sum(iter_feat[q] * iter_weight_feat[q])/(torch.sum(iter_feat[q] * iter_feat[q])+torch.sum(iter_weight_feat[q] * iter_weight_feat[q]) + 0.00001))).mean().unsqueeze(0)
        #             # batch_dist = [1-torch.mul(F.normalize(iter_feat[q], p=2, dim=1), F.normalize(iter_weight_feat[q], p=2, dim=1)).sum(dim=1)
        #             # batch_dist = [1-torch.mul(F.normalize(iter_feat[q], p=2, dim=1), F.normalize(iter_weight_feat[q], p=2, dim=1)).abs().sum(dim=1)
        #                           for q in range(ib)]
        #             # batch_dist = torch.cat(batch_dist)
        #             batch_dist = torch.cat(batch_dist)

        #             # batch_dist = batch_dist[~torch.isnan(batch_dist)].mean().unsqueeze(0)
                    
        #             ##################################
        #             # 这里是空间有富裕的时候的并行计算
        #             # iter_mask = iter_mask.unsqueeze(1).repeat(1, d, 1, 1)
        #             # iter_feat = torch.mul(x[iter_batch_num_mask], iter_mask)
        #             # iter_mean_feat = iter_feat.view(ib, d, -1).sum(-1)/iter_num.view(ib, 1)
        #             # x2 = torch.mul(iter_feat, iter_feat)
        #             # y2 = torch.mul(iter_mean_feat, iter_mean_feat)
        #             # y2 = y2.view(ib, d, 1, 1).repeat(1, 1, h, w)
        #             # xy = torch.mul(iter_feat, iter_mean_feat.view(ib, d, 1, 1).repeat(1, 1, h, w))
        #             # batch_dist = x2 + y2 - 2*xy
        #             # batch_dist = batch_dist[~torch.isnan(batch_dist)]
        #             # batch_dist = batch_dist[~torch.isinf(batch_dist)].mean()
        #             if i == 0:
        #                 intra_dist = batch_dist
        #             else:
        #                 # intra_dist += batch_dist
        #                 intra_dist = torch.cat((intra_dist, batch_dist), dim=0)
        #         else:
        #             break
        #     intra_dist = intra_dist[~torch.isnan(intra_dist)].mean()*intra_weight
        #     # intra_dist = intra_dist.mean()*intra_weight

        if gt is not None:
            # filters
            # cosdist, dist_matrix = self.ata_loss(filters+self.mask_conv.weight.repeat([b,1,1,1]).clone().detach(), gt, num_class, b)
            
            # 0215 change
            # cosdist, dist_matrix = self.ata_loss(filters, gt, num_class, b)
            cosdist, eyedist = self.ata_loss(filters, gt, num_class, b)
            
            # cosdist_inter, dist_matrix_inter = self.ata_loss(filters, gt, num_class, b, filters_2=extra_filter)
            # mask_conv2_cosdist, mask_conv2_distmatrix = self.ata_loss(self.mask_conv2.weight,  gt, num_class, 1)
            # result_dic = {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist, 'pre_mask': pre_mask}
            # result_dic = {'loss_dice': 0.2*dice_loss, 'loss_cosdist': cosdist, 'pre_mask': pre_mask}
            # result_dic = {'loss_dice': dice_loss, 'pre_mask': pre_mask}
            # result_dic = { 'loss_cosdist': cosdist, 'pre_mask': pre_mask}
            result_dic = { 'loss_cosdist': cosdist, 'loss_eyedist': eyedist, 'pre_mask': pre_mask}
            # result_dic = {'loss_cosdist': cosdist, 'pre_mask': pre_mask, 'loss_cosdist_inter': cosdist_inter}

            # result_dic = {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist, 'loss_mcosd': mask_conv2_cosdist, 'pre_mask': pre_mask}
            # if topk_filter is None:
            #     result_dic.update({'loss_intradist': intra_dist})
            
            # return pred, result_dic
            return result_dic
            
            # return pred, {'loss_CFlayer': dice_loss, 'loss_cosdist': cosdist, 'loss_intradist': intra_dist,
            # # return pred, {'loss_cosdist': cosdist,
            #             #    'loss_cpcosdist': cpcosdist, 'pre_mask': pre_mask}
            #                'pre_mask': pre_mask}
        return pred

@HEADS.register_module()
class CFDSASPPHead(DepthwiseSeparableASPPHead):
    def __init__(self, **kwargs):
        self.intra_weight=kwargs.pop("intra_weight")

        self.topk_filter = kwargs.pop("topk_filter")
        self.softmax_mask = kwargs.pop("softmax_mask")
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
        
        output_branch = self.cls_seg(output)
        # output = self.cls_seg(output)
        if label is not None:
            b, useless, h, w = label.size()
            aux_label = F.interpolate((label.view(b, 1, h, w)).float(),
                                      scale_factor=0.25,
                                      mode="nearest")
            aux_label = aux_label.view(b, h // 4, w // 4)
        else:
            aux_label = None
        if self.dropout is not None:
            dpm = self.dropout
        #原本在这里算cflayer
        # if label is not None:
        #     final_output = self.cf_layer(output, aux_label, self.num_classes, False, softmax_mask=self.softmax_mask, topk_filter=self.topk_filter, intra_weight=self.intra_weight)
        # conv_seg_cosdist, conv_seg_dist_matrix = self.cf_layer.ata_loss(self.conv_seg.weight, aux_label, self.num_classes, 1)
        if label is not None:
            # fm = final_output[0]
            # dice_loss = final_output[1]
            # dice_loss['o_b'] = output_branch
            # # dice_loss['loss_ob_ata'] = 0.4*conv_seg_cosdist
            # # return 0.5*(fm+output_branch), dice_loss
            # return fm, dice_loss

            final_loss = self.cf_layer(output, output_branch, aux_label, self.num_classes, False, softmax_mask=self.softmax_mask, topk_filter=self.topk_filter, intra_weight=self.intra_weight, extra_filter=self.conv_seg.weight)
            return output_branch, final_loss
        else:
            # fm = final_output
            # return fm
            return output_branch

        # if self.delta_mode:
        #     fm = fm + output
        # return fm

    @force_fp32(apply_to=('seg_logit', 'extra_loss'))
    def losses(self, seg_logit, extra_loss, seg_label, error_map=None):
        """Compute segmentation loss."""
        loss = dict()
        extra_loss.pop('pre_mask')
        # pre_seg_logit = resize(
        #     # input=extra_loss.pop('pre_mask'),
        #     input=extra_loss.pop('o_b'),
        #     size=(seg_label.shape[2:]),
        #     mode='bilinear',
        #     align_corners=(self.align_corners))
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
        # loss['loss_ob_seg'] = 0.4*self.loss_decode(
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
        # loss['acc_ob_seg'] = accuracy(pre_seg_logit, seg_label)

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
