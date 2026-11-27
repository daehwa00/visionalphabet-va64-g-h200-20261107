# ruff: noqa: ANN001, ANN201, ANN204
"""Eager, full-precision model shadow for optimizer-step EMA."""

import copy

import torch


class ModelEMA:
    def __init__(self, model, decay, *, warmup=True):
        self.model = copy.deepcopy(model).requires_grad_(requires_grad=False).eval()
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.src_params = tuple(p.detach() for p in model.parameters())
        self.ema_params = tuple(self.model.parameters())
        for param in self.ema_params:
            dtype = torch.complex64 if param.is_complex() else torch.float32
            param.data = param.data.to(dtype=dtype)
        self.float_params = [
            (dst, src)
            for dst, src in zip(self.ema_params, self.src_params, strict=True)
            if not dst.is_complex()
        ]
        self.complex_params = [
            (torch.view_as_real(dst), torch.view_as_real(src))
            for dst, src in zip(self.ema_params, self.src_params, strict=True)
            if dst.is_complex()
        ]
        self.groups = [
            tuple(zip(*pairs, strict=True))
            for pairs in (self.float_params, self.complex_params)
            if pairs
        ]

    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        decay = self.decay
        if self.warmup:
            decay = min(decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for dst, src in self.groups:
            torch._foreach_lerp_(dst, [p.float() for p in src], 1.0 - decay)

    def attach(self, optimizer):
        return optimizer.register_step_post_hook(lambda *_: self.update())

    @torch.no_grad()
    def sync_buffers(self, model):
        for shadow, source in zip(self.model.buffers(), model.buffers(), strict=True):
            shadow.copy_(source)

    def checkpoint(self):
        return {"ema_state_dict": self.model.state_dict(), "ema_num_updates": self.num_updates}

    def restore(self, payload):
        self.model.load_state_dict(payload["ema_state_dict"])
        self.num_updates = payload["ema_num_updates"]
