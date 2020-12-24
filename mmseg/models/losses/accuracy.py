import torch.nn as nn
import torch

def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res


def accuracy_error_loss(pred, target, error_logit, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    # error_logit = error_logit[:,:,0:2,0:2]
    # pred = pred[:,:,0:2,0:2]
    # target = target[:,0:2,0:2]
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False
    critierion = nn.BCELoss(reduction='mean')
    # critierion = nn.CrossEntropyLoss(reduction='mean')
    # critierion = nn.MSELoss(reduction='mean')
    return_single = True
    maxk = max(topk)
    # maxk = 1
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...) 人家的注释
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    target_error = torch.zeros(correct.size(), device=correct.device)
    target_error = torch.where(correct, torch.full_like(target_error, 0.), torch.full_like(target_error, 1.))
    ##################################
    # error_logit_c = 1-error_logit
    # error_logit = torch.cat([error_logit,error_logit_c],dim=1)
    ##################################
    #print(error_logit.shape)
    #print(target_error.shape)
    #input()
    # loss = critierion(error_logit, target_error.squeeze(0))
    loss = 0.2 * critierion(error_logit.transpose(0, 1).squeeze(0), target_error.squeeze(0).detach())
    error_prediction = error_logit.transpose(0, 1).squeeze(0)
    error_GT = target_error.squeeze(0).detach()
    error_yes = torch.where(torch.abs(error_prediction-error_GT)<0.2, torch.full_like(error_GT,1.), torch.full_like(error_GT,0.))
    # error_yes_num = [error_yes[i].sum() for i in range(error_yes.shape[0])]
    # total_num = [torch.ones_like(error_GT[i]).sum() for i in range(error_yes.shape[0])]
    # eds_correct_rate = [error_yes_num[i]/total_num[i] for i in range(error_yes.shape[0])]
    error_yes_num = error_yes.sum()
    total_num = torch.ones_like(error_GT).sum()
    GT_error_rate = error_GT.sum()/total_num
    # GT_error_num = error_GT.sum()
    # activated_error_num = torch.where(error_logit>0.5, error_logit, torch.full_like(error_logit, 0.)).sum()
    # activated_true_num = torch.where(error_yes==1. and )
    eds_correct_rate = error_yes_num/total_num
    # return (loss, res[0]) if return_single else (loss, res)
    return (loss, eds_correct_rate,GT_error_rate)


def accuracy_ce_error_loss(pred, target, error_logit, critierion, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False
    # critierion = nn.BCELoss(reduction='mean')
    # critierion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.], device='cuda').float(), reduction='mean')#, ignore_index=255)
    # critierion = nn.MSELoss(reduction='mean')
    return_single = True
    maxk = max(topk)
    # maxk = 1
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...) 人家的注释
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #     res.append(correct_k.mul_(100.0 / target.numel()))
    target_error = torch.zeros(correct.size(), device=correct.device)
    # 正确的是0类 不正确的是1类 所以乘output的应该是1类对吧?
    # 因为想让output去预测原本cp分错的像素
    target_error = torch.where(correct, torch.full_like(target_error, 0), torch.full_like(target_error, 1)).long()

    # total_num = target_error.numel()
    # error_value, error_label = error_logit.topk(1, dim=1)
    # pre_error_rate = error_label.sum().float().mul(100./total_num)
    # error_correct = error_label.squeeze(1).eq(target_error.squeeze(0).detach())
    # mask_correct_rate = correct.sum().float().mul(100./total_num)
    # error_yes_num = error_correct.sum().float()
    # eds_correct_rate = error_yes_num.mul(100./total_num)
    # # weight = torch.tensor([2-mask_correct_rate/100 , mask_correct_rate/100], device='cuda')
    # critierion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.], device='cuda').float(), reduction='mean')#, ignore_index=255)
    # # critierion = nn.CrossEntropyLoss(weight=weight.float(), reduction='mean')#, ignore_index=255)
    # loss = critierion(error_logit, target_error.squeeze(0).detach())
    # return (loss, eds_correct_rate, pre_error_rate, mask_correct_rate)
    total_num = target_error.numel()
    error_value, error_label = error_logit.topk(1, dim=1)
    pre_error_rate = error_label.sum().float().mul(100./total_num)
    error_correct = error_label.squeeze(1).eq(target_error.squeeze(0).detach())
    mask_correct_rate = correct.sum().float().mul(100./total_num)
    correct_correct = error_correct.mul(correct)
    error_correct_temp = error_correct.mul(~correct)
    correct_correct_rate = correct_correct.sum().float().mul(100./correct.sum().float())
    error_correct_rate = error_correct_temp.sum().float().mul(100./(~correct).sum().float())
    error_yes_num = error_correct.sum().float()
    eds_correct_rate = error_yes_num.mul(100./total_num)
    # critierion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2.], device='cuda').float(), reduction='mean')#, ignore_index=255)
    # critierion = nn.CrossEntropyLoss(weight=weight.float(), reduction='mean')#, ignore_index=255)
    loss = critierion(error_logit, target_error.squeeze(0))
    # loss = critierion(error_logit[:, 1, :, :].view(error_logit.shape[0], -1), target_error.squeeze(0).float().view(error_logit.shape[0], -1))
    # loss = critierion(error_logit[:, 1, :, :].unsqueeze(1), target_error.squeeze(0).float())
    
    # FIXME:这个float 在CEloss的时候想着要去掉
    return (loss, eds_correct_rate, pre_error_rate, mask_correct_rate, correct_correct_rate, error_correct_rate)
# def error_gt(pred, target, topk=1, thresh=None):
#     assert isinstance(topk, (int, tuple))
#     if isinstance(topk, int):
#         topk = (topk, )
#         return_single = True
#     else:
#         return_single = False
#     # critierion = nn.BCELoss(reduction='mean')
#     critierion = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.], device='cuda').float(), reduction='mean')
#     # critierion = nn.MSELoss(reduction='mean')
#     return_single = True
#     maxk = max(topk)
#     # maxk = 1
#     if pred.size(0) == 0:
#         accu = [pred.new_tensor(0.) for i in range(len(topk))]
#         return accu[0] if return_single else accu
#     assert pred.ndim == target.ndim + 1
#     assert pred.size(0) == target.size(0)
#     assert maxk <= pred.size(1), \
#         f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
#     pred_value, pred_label = pred.topk(maxk, dim=1)
#     # transpose to shape (maxk, N, ...) 人家的注释
#     pred_label = pred_label.transpose(0, 1)
#     correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
#     if thresh is not None:
#         # Only prediction values larger than thresh are counted as correct
#         correct = correct & (pred_value > thresh).t()
#     # res = []
#     # for k in topk:
#     #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#     #     res.append(correct_k.mul_(100.0 / target.numel()))
#     target_error = torch.zeros(correct.size(), device=correct.device)
#     # 正确的是0类 不正确的是1类 所以乘output的应该是1类对吧?
#     # 因为想让output去预测原本cp分错的像素
#     target_error = torch.where(correct, torch.full_like(target_error, 0), torch.full_like(target_error, 1)).long()
#     # loss = critierion(error_logit, target_error.squeeze(0).detach())

#     # error_value, error_label = error_logit.topk(1, dim=1)
#     # error_correct = error_label.squeeze(1).eq(target_error.squeeze(0).detach())

#     # error_yes_num = error_correct.sum().float()
#     # total_num = target_error.numel()
#     # eds_correct_rate = error_yes_num.mul_(100./total_num)

#     return target_error

def error_gt(pred, target, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False
    # critierion = nn.BCELoss(reduction='mean')
    # critierion = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.], device='cuda').float(), reduction='mean')
    # critierion = nn.MSELoss(reduction='mean')
    return_single = True
    maxk = max(topk)
    # maxk = 1
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...) 人家的注释
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #     res.append(correct_k.mul_(100.0 / target.numel()))
    target_error = torch.zeros(correct.size(), device=correct.device)
    # 正确的是0类 不正确的是1类 所以乘output的应该是1类对吧?
    # 因为想让output去预测原本cp分错的像素
    target_error = torch.where(correct, torch.full_like(target_error, 0), torch.full_like(target_error, 1)).long()
    # loss = critierion(error_logit, target_error.squeeze(0).detach())

    # error_value, error_label = error_logit.topk(1, dim=1)
    # error_correct = error_label.squeeze(1).eq(target_error.squeeze(0).detach())

    # error_yes_num = error_correct.sum().float()
    # total_num = target_error.numel()
    # eds_correct_rate = error_yes_num.mul_(100./total_num)
    # pred_acc = 
    return target_error.permute(1,0,2,3)


def cosdist_loss(pred, target, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False
    # critierion = nn.BCELoss(reduction='mean')
    # critierion = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.], device='cuda').float(), reduction='mean')
    # critierion = nn.MSELoss(reduction='mean')
    return_single = True
    maxk = max(topk)
    # maxk = 1
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...) 人家的注释
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #     res.append(correct_k.mul_(100.0 / target.numel()))
    target_error = torch.zeros(correct.size(), device=correct.device)
    # 正确的是0类 不正确的是1类 所以乘output的应该是1类对吧?
    # 因为想让output去预测原本cp分错的像素
    target_error = torch.where(correct, torch.full_like(target_error, 0), torch.full_like(target_error, 1)).long()
    # loss = critierion(error_logit, target_error.squeeze(0).detach())

    # error_value, error_label = error_logit.topk(1, dim=1)
    # error_correct = error_label.squeeze(1).eq(target_error.squeeze(0).detach())

    # error_yes_num = error_correct.sum().float()
    # total_num = target_error.numel()
    # eds_correct_rate = error_yes_num.mul_(100./total_num)
    # pred_acc = 
    return target_error.permute(1,0,2,3)


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
