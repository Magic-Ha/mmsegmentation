from .ann_head import ANNHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .psa_head import PSAHead
from .psp_head import DDPSPHead, PSPHead, EDPSPHead, ED_CE_PSPHead
# # from .psp_head import 
# from .edpsp_head import EDPSPHead

from .sep_aspp_head import DepthwiseSeparableASPPHead, CFDSASPPHead
# from .sep_aspp_head_laplas_modified_gaihuiqude import MAXCFDSASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .uper_head import UPerHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'CFDSASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'DDPSPHead', 'EDPSPHead', 'ED_CE_PSPHead'
]
