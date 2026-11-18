"""Untimed five-output, gradient and checkpoint-LR AdamW check using the archived benchmark."""
import json
import os
import sys
from pathlib import Path

import torch

source = Path("benchmark_matched.py")
text = source.read_text()
start = "    restore_optimizer(validation=True)"
end = "    # Restore the identical checkpoint before warmup/timing."
assert text.count(start) == text.count(end) == 1
# The archived main() has already checked and saved its gradients and update at this point.
text = text.split(end)[0].replace(start, "    restore_optimizer()") + '''
    result.update(status="passed", outputs=output_checks, update_validation_scope="checkpoint LR, validation only")
    save()
'''
namespace = {"__name__": "checkpoint_benchmark", "__file__": str(source)}
exec(compile(text, str(source), "exec"), namespace)

output = Path(sys.argv[sys.argv.index("--output")+1])
control_name = os.environ.get("VA_FORWARD_CONTROL", "control-forward")
control = output.with_name(control_name + ".pt")
checks = {}
namespace["output_checks"] = checks
heads = namespace["runner"].heads
objective = heads._training_objective

def capture(model, outputs, targets, other, mixing):
    assert isinstance(outputs, tuple) and len(outputs) == 5
    cpu = tuple(v.detach().float().cpu() for v in outputs)
    if output.stem == control_name:
        torch.save(cpu, control)
    expected = torch.load(control, map_location="cpu", weights_only=True)
    for name, value, ref in zip(("joint","affine","fusion","lrq","descriptor"),cpu,expected,strict=True):
        checks[name] = namespace["error_metrics"](value,ref)
    assert all(v["passed"] for v in checks.values())
    return objective(model, outputs, targets, other, mixing)

heads._training_objective = capture
try:
    namespace["main"]()
finally:
    # Preserve the forward-only outcome even when the subsequent gradient gate rejects a candidate.
    if output.exists() and checks:
        record = json.loads(output.read_text())
        record["outputs"] = checks
        output.write_text(json.dumps(record, indent=2)+"\n")
