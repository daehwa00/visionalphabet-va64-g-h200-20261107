"""Exact-scope choice between the verified four-row output VJP and the separate reductions."""

from dataclasses import replace
from functools import cache
from pathlib import Path
import hashlib

from .kernel_launch_config import (
    geometry_store_signature,
    register_kernel_specializer,
    validated_geometry,
)
from .output_dot_kernel import _path_backward_kernel
from .path_backward_layout import _LayoutGraphPass
from .retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD
from .segmented_path_scan import segment_revision, select_segment

DOT_NAME = "retained_output_weight_bias_dot"
BUNDLE = Path(__file__).with_name("validated_output_dots.json")


@cache
def dot_revision(parent_revision):
    h = hashlib.sha256(segment_revision(parent_revision).encode())
    h.update(_path_backward_kernel.cache_key.encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def select_output_dot(scope, base):
    four_row = select_segment(scope, base)
    shape = dict(scope.shape_items)
    if (
        scope.execution_regime != "compiled"
        or shape.get("operation") != VERTICAL_BACKWARD
        or shape.get("height") != 56
    ):
        return four_row
    exact = replace(scope, kernel_revision=dot_revision(scope.kernel_revision))
    selected = validated_geometry(DOT_NAME, exact, BUNDLE)
    if four_row is not None and selected is not None and selected == base:
        return _path_backward_kernel
    return four_row


def output_dot_graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME, select_output_dot)
    signature = geometry_store_signature(DOT_NAME, BUNDLE, omit_empty=True)
    return _LayoutGraphPass(dot_revision(signature), previous) if signature else previous
