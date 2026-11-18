import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F


def check(kind, model, runtime, optimizer, mixup, batches):
    root = Path(os.environ["RUN_ROOT"])
    fixture = root / "checkpoint.pt"
    reference = root / "reference.pt"
    if kind == "seed":
        chunks = [next(batches), next(batches)]
        x = torch.cat([v.cuda(memory_format=torch.channels_last) for v, _ in chunks])
        y = torch.cat([t.cuda() for _, t in chunks])
        x, y = mixup(x, y)
        assert optimizer.state
        torch.save((model.state_dict(), optimizer.state_dict(), x.cpu(), y.cpu()), fixture)
        del x, y, chunks
    state, moments, x, y = torch.load(fixture, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(moments)
    x = x.cuda().contiguous(memory_format=torch.channels_last)
    y = y.cuda()
    torch.manual_seed(501)
    torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = runtime(x)
        loss = F.cross_entropy(out[0], y)
    loss.backward()
    params = dict(model.named_parameters())
    missing = [n for n,p in params.items() if p.requires_grad and p.grad is None]
    assert not missing, missing
    current = {"output": {str(i): v.detach().float().cpu() for i,v in enumerate(out)},
               "gradient": {n:p.grad.detach().float().cpu() for n,p in params.items() if p.requires_grad}}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    current["update"] = {n:p.detach().float().cpu()-state[n].float() for n,p in params.items() if p.requires_grad}
    if kind == "seed":
        torch.save(current, reference)
    baseline = torch.load(reference, weights_only=True)
    summary = {}
    for group, tensors in current.items():
        assert tensors.keys() == baseline[group].keys()
        row = {"count":len(tensors), "max_abs":0, "max_rel_l2":0, "failed":[]}
        for n,v in tensors.items():
            ref = baseline[group][n]
            assert ref.shape == v.shape and v.isfinite().all() and ref.isfinite().all()
            delta = v-ref
            absolute = delta.abs().max().item()
            relative = (delta.norm()/ref.norm().clamp_min(1e-12)).item()
            limit = 0.03*ref.abs().max().item()+1e-6 if group=="gradient" else 3e-5 if group=="update" else 0.05
            if absolute>limit or relative>(0.005 if group=="output" else 0.02):
                row["failed"].append(n)
            row["max_abs"]=max(row["max_abs"],absolute)
            row["max_rel_l2"]=max(row["max_rel_l2"],relative)
        summary[group]=row
    print("LAYOUT_GATE", os.environ["ARM"], json.dumps(summary), flush=True)
    if any(v["failed"] for v in summary.values()):
        raise SystemExit(43)
