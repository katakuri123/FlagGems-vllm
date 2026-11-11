# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performance test for the shifted mHC post->pre operator.

The baseline is vLLM's **production** kernel (the fused TileLang / DeepGEMM
path).  Shapes, inputs and the native resolver come from the shared workload in
``baseline/mhc_shifted_post_pre.py``, so this benchmark and the CUDA baseline
collector cannot drift apart.  vLLM's torch reference is the CPU/fallback
implementation used for numeric checks: it upcasts every stage to fp32 and fuses
nothing, so timing against it produces a large and meaningless SpeedUp.  It is
therefore not used as a performance baseline here (it remains the reference in
``tests/``); when the production kernel is not importable this benchmark reports
latency only.

Run eagerly: ``--mode kernel`` (the default).  The operator dispatches into
several sub-operators with host-side autotune caches, so a cold CUDA-graph
capture invalidates the stream and a warm one only measures the frozen
steady-state.
"""

import importlib.util
import warnings
from pathlib import Path

import pytest

import flaggems_vllm
from flaggems_vllm.ops.mhc.mhc_shifted_post_pre import mhc_shifted_post_pre

from . import base

VENDOR = flaggems_vllm.vendor_name

_SPEC = importlib.util.spec_from_file_location(
    "mhc_shifted_post_pre_workload",
    Path(__file__).resolve().parents[1] / "baseline/mhc_shifted_post_pre.py",
)
workload = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workload)


def _get_baseline():
    """vLLM's production kernel, or ``None`` when it is not importable.

    Only the production implementation gives a meaningful SpeedUp.  The
    resolver lives in the baseline workload so the collector and the benchmark
    probe the same module path; a missing kernel means latency-only, never a
    torch fallback.
    """
    if VENDOR != "nvidia":
        warnings.warn(
            f"mhc_shifted_post_pre benchmark: no production kernel registered "
            f"for vendor {VENDOR!r}; reporting latency without a SpeedUp.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    op = workload.native()
    if op is None:
        warnings.warn(
            "mhc_shifted_post_pre benchmark: vLLM production kernel is not "
            f"importable ({workload.NATIVE_UNAVAILABLE_REASON}); reporting "
            "latency without a SpeedUp.",
            RuntimeWarning,
            stacklevel=2,
        )
    return op


class MHCShiftedPostPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    # One source of truth for shapes and dtypes: baseline/mhc_shifted_post_pre.py.
    # core_shapes.yaml may override shapes, but it must stay identical.
    DEFAULT_SHAPES = workload.SHAPES
    DEFAULT_DTYPES = workload.DTYPES

    def set_more_metrics(self):
        # Not in DEFAULT_METRICS, so only the comprehensive level reports it.
        return ["gbps"]

    def get_input_iter(self, dtype):
        for n, h in self.shapes:
            args, kwargs = workload.build_inputs({"n": n, "h": h}, dtype, self.device)
            yield (*args, kwargs)

    def get_gbps(self, args, latency):
        """Logical HBM traffic of the five stages, in bytes.

        post (x + residual in, residual out) + projection read + collapse read
        (residual out) + layer-input write + norm read/write + aux write.
        """
        residual = args[1]
        n, hc, h = residual.shape
        plane = n * h * 2  # one (N, H) bfloat16 plane
        bytes_moved = (
            plane * 2 * hc  # post: residual in + out
            + plane  # post: x in
            + plane * hc  # projection: residual read
            + plane * hc  # collapse: residual read
            + plane  # layer input write
            + plane * 2  # norm: read + write
            + plane  # aux write
        )
        return bytes_moved / (latency / 1e3) / 1e9


@pytest.mark.mhc_shifted_post_pre
def test_mhc_shifted_post_pre():
    bench = MHCShiftedPostPreBenchmark(
        op_name="mhc_shifted_post_pre",
        torch_op=_get_baseline(),
        gems_op=mhc_shifted_post_pre,
        is_inplace=workload.IS_INPLACE,
    )
    bench.run()
