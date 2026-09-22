"""Hàm mất mát CenterNet — bản sao train_crop/train_centernet.py (Focal + 0,1·L1 size + 1,0·L1 offset).

Không đổi so với v1 (nhãn yếu, không thêm ràng buộc mới — docs/HUONG_DAN_HUAN_LUYEN_I5 §3)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss cho heatmap (CornerNet/CenterNet). pred, gt: (B,1,H,W)."""
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg
    n_pos = pos.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    return -(neg_loss if n_pos == 0 else (pos_loss + neg_loss) / n_pos)


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H * W).permute(0, 2, 1).contiguous()
    ind = ind.unsqueeze(2).expand(B, ind.size(1), C)
    return feat.gather(1, ind)


def reg_l1(pred: torch.Tensor, ind: torch.Tensor, target: torch.Tensor,
           mask: torch.Tensor) -> torch.Tensor:
    p = _gather(pred, ind)
    m = mask.unsqueeze(2).expand_as(p)
    return F.l1_loss(p * m, target * m, reduction="sum") / (mask.sum() + 1e-4)


class CenterNetLoss(nn.Module):
    def __init__(self, w_hm=1.0, w_size=0.1, w_off=1.0):
        super().__init__()
        self.w_hm, self.w_size, self.w_off = w_hm, w_size, w_off

    def forward(self, outputs, targets):
        hm, wh, off = outputs
        l_hm = focal_loss(hm.float(), targets["hm"])
        l_wh = reg_l1(wh.float(), targets["ind"], targets["wh"], targets["mask"])
        l_off = reg_l1(off.float(), targets["ind"], targets["off"], targets["mask"])
        total = self.w_hm * l_hm + self.w_size * l_wh + self.w_off * l_off
        return total, {"hm": float(l_hm.detach()), "wh": float(l_wh.detach()),
                       "off": float(l_off.detach())}
