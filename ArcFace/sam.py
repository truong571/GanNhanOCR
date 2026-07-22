"""SAM — Sharpness-Aware Minimization (Foret et al., ICLR 2021).

Flat-minima training is the roadmap-P2 lever for FAILURE PREDICTION: Zhu et al.
(ECCV 2022, "Rethinking Confidence Calibration for Failure Prediction") show that
ordinary training/calibration HURTS the ability to tell correct from wrong
predictions, while flat minima (SAM/SWA) widen the correct-vs-wrong confidence
gap — i.e. they raise the error-detection AUC itself, which is exactly the metric
the current encoder fails (0.57). SAM wraps any base optimizer; use with the
two-step closure in train.py. SWA is applied separately in train.py.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kw):
        assert rho >= 0
        defaults = dict(rho=rho, adaptive=adaptive, **kw)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                         # climb to the local worst point
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]     # go back to the original weights
        self.base_optimizer.step()                  # do the actual update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                norms.append(g.norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):
        raise RuntimeError("SAM requires the two-step form: first_step()/second_step() in train.py")
