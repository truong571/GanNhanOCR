"""PROMPT 2 — Kiến trúc CenterNet (ResNet34 + FPN, tuỳ chọn DCNv2) cho chữ cổ.

Mạng coi mỗi ký tự là MỘT ĐIỂM (tâm chữ) nên không dùng anchor/NMS; nhờ đó hai
chữ dính nét vẫn cho hai cực trị tách biệt trên heatmap.

Luồng:
    Ảnh (B,3,H,W)
      │  ResNet34 (ImageNet) -> C2(/4,64)  C3(/8,128)  C4(/16,256)  C5(/32,512)
      │  FPN top-down (lateral 1×1 + cộng dồn + upsample) -> P2 ở /4, ``fpn_ch`` kênh
      │  (tuỳ chọn) DCNv2 ở conv làm mượt P2 để bám nét cong dính nhau
      ├─ Head Heatmap (1ch, sigmoid, bias=-2.19 ổn định Focal Loss)
      ├─ Head Size    (2ch, [w,h], không kích hoạt)
      └─ Head Offset  (2ch, [dx,dy], không kích hoạt)

Đầu ra ở output stride = 4 (H/4 × W/4). Yêu cầu H, W chia hết 32.

Chạy thử:
    .venv/bin/python test/model_centernet.py --selftest
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDE = 4

# DCNv2 (deformable conv) — tuỳ chọn; nếu không có thì tự rơi về Conv thường.
try:
    from torchvision.ops import DeformConv2d
    _HAS_DCN = True
except Exception:
    _HAS_DCN = False


class _DCNBlock(nn.Module):
    """Deformable Conv v2 (modulated): 1 conv sinh offset+mask, rồi DeformConv2d.

    Dùng để conv làm mượt nét cong/dính của chữ ván khắc. Tự fallback Conv2d khi
    torchvision không có DeformConv2d (vd. một số build CPU/MPS)."""

    def __init__(self, cin, cout, k=3, pad=1):
        super().__init__()
        self.k = k
        if _HAS_DCN:
            # 2*k*k kênh offset + k*k kênh mask (modulated v2)
            self.offmask = nn.Conv2d(cin, 3 * k * k, k, padding=pad)
            nn.init.constant_(self.offmask.weight, 0.0)
            nn.init.constant_(self.offmask.bias, 0.0)
            self.dcn = DeformConv2d(cin, cout, k, padding=pad)
        else:
            self.conv = nn.Conv2d(cin, cout, k, padding=pad)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if _HAS_DCN:
            o = self.offmask(x)
            o1, o2, mask = torch.chunk(o, 3, dim=1)
            offset = torch.cat([o1, o2], dim=1)
            mask = torch.sigmoid(mask)
            x = self.dcn(x, offset, mask)
        else:
            x = self.conv(x)
        return self.act(self.bn(x))


def _conv_block(cin, cout, k=3, pad=1, use_dcn=False):
    if use_dcn:
        return _DCNBlock(cin, cout, k, pad)
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=pad),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CenterNetResNet34FPN(nn.Module):
    """ResNet34 + FPN -> 3 head CenterNet ở output stride 4."""

    def __init__(self, pretrained: bool = True, fpn_ch: int = 64,
                 head_ch: int = 64, use_dcn: bool = False):
        super().__init__()
        if use_dcn and not _HAS_DCN:
            print("[model] DCNv2 không khả dụng -> dùng Conv thường.")
            use_dcn = False
        self.use_dcn = use_dcn and _HAS_DCN

        bb = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # tách các stage để lấy đặc trưng đa quy mô
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)  # -> /4, 64ch
        self.layer1 = bb.layer1      # C2: /4,  64
        self.layer2 = bb.layer2      # C3: /8,  128
        self.layer3 = bb.layer3      # C4: /16, 256
        self.layer4 = bb.layer4      # C5: /32, 512

        # FPN lateral 1×1
        self.lat5 = nn.Conv2d(512, fpn_ch, 1)
        self.lat4 = nn.Conv2d(256, fpn_ch, 1)
        self.lat3 = nn.Conv2d(128, fpn_ch, 1)
        self.lat2 = nn.Conv2d(64, fpn_ch, 1)
        # conv làm mượt sau khi cộng dồn (P2 dùng DCN nếu bật)
        self.smooth2 = _conv_block(fpn_ch, fpn_ch, use_dcn=self.use_dcn)
        self.smooth3 = _conv_block(fpn_ch, fpn_ch)
        self.smooth4 = _conv_block(fpn_ch, fpn_ch)

        def head(out_ch, bias=0.0):
            m = nn.Sequential(
                nn.Conv2d(fpn_ch, head_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(head_ch, out_ch, 1))
            nn.init.normal_(m[0].weight, std=0.001)
            nn.init.constant_(m[-1].weight, 0.0)
            nn.init.constant_(m[-1].bias, bias)
            return m

        # bias heatmap = -2.19 -> sigmoid≈0.1 lúc đầu, ổn định Focal Loss
        self.hm = head(1, bias=-2.19)
        self.wh = head(2, bias=0.0)
        self.off = head(2, bias=0.0)

    # ----- FPN -----
    def _fpn(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)            # /4, fpn_ch
        return p2

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p2 = self._fpn(c2, c3, c4, c5)
        hm = torch.sigmoid(self.hm(p2)).clamp(1e-4, 1 - 1e-4)
        return hm, self.wh(p2), self.off(p2)


def build_model(arch: str = "resnet34_fpn", pretrained: bool = True,
                use_dcn: bool = False, **kw) -> nn.Module:
    """Factory để train/infer dùng chung. Hiện hỗ trợ 'resnet34_fpn'."""
    if arch == "resnet34_fpn":
        return CenterNetResNet34FPN(pretrained=pretrained, use_dcn=use_dcn, **kw)
    raise ValueError(f"arch chưa hỗ trợ: {arch}")


def _selftest():
    net = build_model(pretrained=False, use_dcn=False)
    for img in (256, 512):
        x = torch.randn(2, 3, img, img)
        hm, wh, off = net(x)
        o = img // STRIDE
        assert hm.shape == (2, 1, o, o), hm.shape
        assert wh.shape == (2, 2, o, o), wh.shape
        assert off.shape == (2, 2, o, o), off.shape
        assert float(hm.min()) >= 0 and float(hm.max()) <= 1
    # backward chạy được
    hm, wh, off = net(torch.randn(1, 3, 256, 256))
    hm.mean().backward()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"model_centernet self-test OK | out stride {STRIDE} | DCN avail={_HAS_DCN} "
          f"| params {n_params:.1f}M | hm{tuple(hm.shape)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dcn", action="store_true", help="thử bật DCNv2")
    a = ap.parse_args()
    if a.dcn:
        m = build_model(pretrained=False, use_dcn=True)
        y = m(torch.randn(1, 3, 256, 256))
        print("DCN forward OK:", tuple(y[0].shape))
    elif a.selftest:
        _selftest()
    else:
        print("dùng --selftest")
