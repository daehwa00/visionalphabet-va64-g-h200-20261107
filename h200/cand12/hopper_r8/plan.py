"""Rekey the fixed r4 and qualified r5 launch geometries to this workload."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def workload(micro, arm):
    from lnet.effective_output_vjp import VJP_NAME, vjp_revision
    from lnet.effective_structural_scan import STRUCT_NAME, structural_revision
    from lnet.kernel_launch_config import LaunchScope
    from lnet.output_dot_dispatch import DOT_NAME, dot_revision
    from lnet.path_backward_layout import LAYOUT_NAME, _driver_api_version, _layout_revision
    from lnet.retained_path_dispatch import PROFILE_NAME, step_revision
    from lnet.segmented_path_scan import SEGMENT_NAME, segment_revision

    records = json.loads((HERE / "r4_records.json").read_text())
    if arm != "r4":
        forward = next(k for k in records[PROFILE_NAME] if json.loads(k)["shape"]["operation"] == 0)
        backward = next(k for k in records[STRUCT_NAME] if json.loads(k)["shape"]["coarse"] == 0)
        marker = dict(blocks={"VARIANT": 2}, num_warps=4, num_stages=1)
        records[STRUCT_NAME][forward] = marker
        records[VJP_NAME] = {backward: marker}
        records[PROFILE_NAME][forward]["num_warps"] = 2

    result = {}
    for family, group in records.items():
        result[family] = {}
        for key, geometry in group.items():
            fields = json.loads(key)
            shape = {**fields["shape"], "batch": micro}
            if family == LAYOUT_NAME:
                revision = _layout_revision()
            else:
                parent = step_revision(shape["operation"])
                revision = {
                    PROFILE_NAME: lambda: parent,
                    SEGMENT_NAME: lambda: segment_revision(parent),
                    DOT_NAME: lambda: dot_revision(parent),
                    STRUCT_NAME: lambda: structural_revision(parent),
                    VJP_NAME: lambda: vjp_revision(parent),
                }[family]()
            scope = LaunchScope.build(
                shape=shape, dtype=fields["dtype"], execution_regime=fields["execution_regime"],
                compile_profile=fields["compile_profile"], kernel_revision=revision,
                cuda_driver_version=_driver_api_version(),
            )
            result[family][scope.key] = geometry
    return result
