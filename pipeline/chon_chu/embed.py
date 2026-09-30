"""embed.py — nhúng crop mọi ô: MultiEnc [nom-embed v1, ArcFace v2] (norm, 512-d) + CNN kiểm ảnh↔chữ 64×64 của
models/gold_exact (vft_T/L, hand_T/L_<Kinh|DungLy>) — nạp qua pipeline.gold_exact.common.Assets (kiểm sha256).
Cache theo md5 byte PNG của crop + sha mô hình (gold_exact.common.EmbCache) ở <cache>/<Sách>/emb/. 0 API.
"""
from __future__ import annotations

import time

import numpy as np

from pipeline.gold_exact import signals_img as SI
from pipeline.gold_exact.common import EmbCache

NET_FILES = {"vft_T": "vft_model_T.pt", "vft_L": "vft_model_L.pt",
             "hand_T_Kinh": "hand_model_T_Kinh.pt", "hand_L_Kinh": "hand_model_L_Kinh.pt",
             "hand_T_DungLy": "hand_model_T_DungLy.pt", "hand_L_DungLy": "hand_model_L_DungLy.pt"}


class Embedder:
    def __init__(self, assets, device: str, cache_dir, log=print):
        import torch
        self.A, self.log = assets, log
        self.dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
        self.cache_dir = cache_dir
        self._enc = None
        self._nets = {}

    def enc(self):
        if self._enc is None:
            self._enc = SI.MultiEnc([self.A.ext(SI.ENC_FILES["v1"]), self.A.ext(SI.ENC_FILES["v2"])], self.dev.type, True)
        return self._enc

    def enc_sha(self):
        import hashlib
        return hashlib.sha256((self.A.sha(SI.ENC_FILES["v1"]) + self.A.sha(SI.ENC_FILES["v2"])).encode()).hexdigest()

    def net(self, name):
        if name not in self._nets:
            self._nets[name] = SI.load_net(self.A.load(NET_FILES[name]), self.dev)
        return self._nets[name]

    def embed(self, keys: list[str], grays: dict, nets: list[str]) -> dict:
        """keys = md5 crop (thứ tự hàng); grays = {md5: ảnh xám} (chỉ cần cho khoá chưa có trong cache).
        Trả {'enc': (N, 512) float32, <net>: (N, 256) float32}."""
        caches = {"enc": EmbCache(self.cache_dir, "enc_v1v2_norm", self.enc_sha())}
        for n in nets:
            caches[n] = EmbCache(self.cache_dir, n, self.A.sha(NET_FILES[n]))
        uniq = list(dict.fromkeys(keys))
        t0 = time.time()
        for name, C in caches.items():
            miss = C.missing(uniq)
            if not miss:
                continue
            self.log(f"[chon_chu] nhúng {name}: {len(miss):,} crop mới (cache {len(uniq) - len(miss):,})")
            for s in range(0, len(miss), 4096):
                part = miss[s:s + 4096]
                gs = [grays[k] for k in part]
                if name == "enc":
                    C.put(part, self.enc().embed(gs))
                else:
                    X64 = np.stack([SI.prep64(g) for g in gs])
                    C.put(part, SI.net_embed(self.net(name), X64, self.dev))
            C.save()
        out = {name: C.get(keys) for name, C in caches.items()}
        self.log(f"[chon_chu] nhúng xong {len(keys):,} ô ({time.time() - t0:.0f}s)")
        return out
