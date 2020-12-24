import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module()
class ClassChannelLoss(nn.Module):
    def __init__(self,
                 reduction='mean',  # class之间求均值
                 class_weight=None
                 ):
        super(ClassChannelLoss, self).__init__()
        