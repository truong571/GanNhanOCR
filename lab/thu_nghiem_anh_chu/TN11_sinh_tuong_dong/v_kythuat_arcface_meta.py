"""v_kythuat (03/10): đọc SIÊU DỮ LIỆU các checkpoint encoder (v1 nom-embed, v2 ArcFace, các .pt khác) — CPU, chỉ đọc khoá/hình dạng.
Không chạy mô hình, không sửa tệp nào. Ra: measure_out/_tn11/verify/kythuat/arcface_meta.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_arcface_meta.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)

CKS = ["nom-embed/best.pt", "ArcFace/checkpoints/best.pt", "ArcFace/checkpoints/train_best.pt", "ArcFace/checkpoints/last.pt",
       "ArcFace/nom-embed-arcface/best.pt"]
extra = sorted(str(p.relative_to(REPO)) for p in (REPO / "ArcFace").glob("**/*.pt"))
extra += sorted(str(p.relative_to(REPO)) for p in (REPO / "nom-embed").glob("**/*.pt"))
res = {}
for rel in dict.fromkeys(CKS + extra):
    p = REPO / rel
    if not p.exists():
        res[rel] = "KHÔNG TỒN TẠI"
        continue
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001  (vd. con trỏ git-LFS 133 byte)
        res[rel] = {"bytes": p.stat().st_size, "loi_doc": f"{type(e).__name__}: {str(e)[:120]}"}
        continue
    d = {"bytes": p.stat().st_size, "keys": sorted(map(str, ck.keys())) if isinstance(ck, dict) else str(type(ck))}
    if isinstance(ck, dict):
        for k in ("arch", "embed_dim", "img", "epoch", "k", "s", "m", "args", "val_top1", "head_top1", "note", "tiers", "split", "holdout"):
            if k in ck:
                v = ck[k]
                d[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:300]
        if "head" in ck and isinstance(ck["head"], dict):
            d["head_keys"] = {k: list(v.shape) for k, v in ck["head"].items() if hasattr(v, "shape")}
        if "backbone" in ck and isinstance(ck["backbone"], dict):
            bb = ck["backbone"]
            d["n_backbone_tensors"] = len(bb)
            d["has_layer4_1"] = any(k.startswith("backbone.layer4.1.") for k in bb)
            d["has_layer4_2"] = any(k.startswith("backbone.layer4.2.") for k in bb)   # ResNet34 có layer4.2, ResNet18 không
            d["has_layer3_5"] = any(k.startswith("backbone.layer3.5.") for k in bb)   # ResNet34 có layer3.5
            d["proj_shape"] = [list(bb[k].shape) for k in bb if k.startswith("proj.")]
            d["fc_in_feats"] = [list(bb[k].shape) for k in bb if k.endswith("layer4.1.bn2.weight")]
        if "classes" in ck:
            c = ck["classes"]
            d["n_classes"] = len(c)
    res[rel] = d
(OUT / "arcface_meta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
