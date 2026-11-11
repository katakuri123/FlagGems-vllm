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

"""NVIDIA-native baseline for ``mhc_shifted_post_pre``.

The workload is vLLM's DSV4.1 production kernel
(``vllm.models.deepseek_v41.nvidia.ops.mhc.mhc_shifted_post_pre``, TileLang /
DeepGEMM).  The shifted boundary only exists in a vLLM tree that ships DSV4.1,
so ``native()`` probes that module and returns ``None`` with a reason when the
installed vLLM does not provide it.  Do not substitute the FlagGems kernel
here: a self-baseline makes every SpeedUp 1.0 and hides real regressions.

Timing is eager (no CUDA graph): the operator dispatches into several sub-ops
with host-side autotune caches, and a cold capture invalidates the stream.
"""

import importlib

import torch

OP_NAME = "mhc_shifted_post_pre"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

_HC = 4
_MIX = _HC * (_HC + 2)
_SEED = 0

# DSV4/DSV4.1 boundary shapes.  Keep in sync with benchmark/core_shapes.yaml;
# benchmark/test_mhc_shifted_post_pre.py imports SHAPES from this module.
SHAPES = [
    (1, 1280),
    (16, 1280),
    (512, 1280),
    (512, 7168),
    (4096, 7168),
    (128, 5120),
    (2048, 5120),
]

# Release-specific locations of the production kernel.  Add a new entry when
# vLLM moves the module; never fall back to a candidate implementation.
_NATIVE_MODULES = ("vllm.models.deepseek_v41.nvidia.ops.mhc",)


def native():
    """Resolve the vLLM production kernel, or return None with a reason."""
    global NATIVE_UNAVAILABLE_REASON

    reasons = []
    for module_name in _NATIVE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - resolution must not abort collection
            reasons.append(f"{module_name}: {exc!r}")
            continue
        op = getattr(module, "mhc_shifted_post_pre", None)
        if callable(op):
            NATIVE_UNAVAILABLE_REASON = None
            return op
        reasons.append(f"{module_name}.mhc_shifted_post_pre absent")

    NATIVE_UNAVAILABLE_REASON = (
        "; ".join(reasons) if reasons else "no candidate native module"
    )
    return None


def grid():
    return [{"n": n, "h": h} for n, h in SHAPES]


def build_inputs(binding, dtype, device):
    n = int(binding["n"])
    h = int(binding["h"])
    if dtype is not torch.bfloat16:
        raise NotImplementedError("mhc_shifted_post_pre only supports bfloat16")

    # Local generator: build_inputs must be repeatable without perturbing the
    # process-wide RNG (the collector calls it in the main and NCU processes).
    gen = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(n, h, dtype=dtype, device=device, generator=gen)
    residual = torch.randn(n, _HC, h, dtype=dtype, device=device, generator=gen)
    post_layer_mix = (
        torch.rand(n, _HC, 1, dtype=torch.float32, device=device, generator=gen) * 0.3
    )
    comb_res_mix = (
        torch.randn(n, _HC, _HC, dtype=torch.float32, device=device, generator=gen)
        * 0.2
    )
    fn = (
        torch.randn(_MIX, _HC * h, dtype=torch.float32, device=device, generator=gen)
        * 0.02
    )
    hc_scale = torch.tensor([0.1, 0.3, 0.2], dtype=torch.float32, device=device)
    hc_base = torch.randn(_MIX, dtype=torch.float32, device=device, generator=gen) * 0.1
    pre_mix = (
        torch.rand(n, _HC, dtype=torch.float32, device=device, generator=gen) + 0.5
    )
    norm_weight = torch.randn(h, dtype=dtype, device=device, generator=gen) * 0.3 + 1
    args = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,  # rms_eps
        1e-6,  # hc_pre_eps
        1e-6,  # hc_sinkhorn_eps
        1.0,  # hc_post_mult_value
        5,  # sinkhorn_repeat
    )
    kwargs = {
        "pre_mix": pre_mix,
        "norm_weight": norm_weight,
        "norm_eps": 1e-6,
        "capture_aux": True,
    }
    return args, kwargs


def key_shape(binding):
    """Primary key = the first tensor's shape (x is (N, H))."""
    return [int(binding["n"]), int(binding["h"])]


def config(binding, dtype):
    n = int(binding["n"])
    h = int(binding["h"])
    dt = str(dtype)
    f32 = "torch.float32"
    return {
        "inputs": {
            "x": {"shape": [n, h], "dtype": dt},
            "residual": {"shape": [n, _HC, h], "dtype": dt},
            "post_layer_mix": {"shape": [n, _HC, 1], "dtype": f32},
            "comb_res_mix": {"shape": [n, _HC, _HC], "dtype": f32},
            "fn": {"shape": [_MIX, _HC * h], "dtype": f32},
            "hc_scale": {"shape": [3], "dtype": f32},
            "hc_base": {"shape": [_MIX], "dtype": f32},
            "pre_mix": {"shape": [n, _HC], "dtype": f32},
            "norm_weight": {"shape": [h], "dtype": dt},
            "rms_eps": {"scalar": 1e-6},
            "hc_pre_eps": {"scalar": 1e-6},
            "hc_sinkhorn_eps": {"scalar": 1e-6},
            "hc_post_mult_value": {"scalar": 1.0},
            "sinkhorn_repeat": {"scalar": 5},
            "norm_eps": {"scalar": 1e-6},
            "capture_aux": {"scalar": True},
        },
        "outputs": {
            "residual_cur": {"shape": [n, _HC, h], "dtype": dt},
            "post_mix": {"shape": [n, _HC, 1], "dtype": f32},
            "comb_mix": {"shape": [n, _HC, _HC], "dtype": f32},
            "layer_input": {"shape": [n, h], "dtype": dt},
            "next_pre": {"shape": [n, _HC], "dtype": f32},
            "aux": {"shape": [n, h], "dtype": dt},
        },
        "dims": {"N": n, "H": h, "hc_mult": _HC},
        "input_seed": _SEED,
        "graph": False,
    }
