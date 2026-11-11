# Baseline contribution guide

`baseline/` holds one **NVIDIA-native workload definition per operator**. The
collector [`tools/collect_baseline_nvidia.py`](../tools/collect_baseline_nvidia.py)
discovers every `baseline/*.py`, calls `native()` once, then records latency and
NCU counters for each `grid()` binding and `DTYPES` entry. Benchmarks then match
those records through `--base-data`.

This directory is **not** a test suite and **not** an accuracy reference. A
baseline answers one question: "what does the vendor-native kernel cost on the
reference NVIDIA GPU for this exact workload?" Correctness stays in `tests/`.

The machine-readable contract lives in [`__init__.py`](__init__.py); this file is
the practical checklist. See also the acceptance document, section 6.2.

## Non-negotiable rules

1. **NVIDIA-native only.** `native()` must resolve the real vLLM/CUDA callable
   (for example `vllm._custom_ops.fused_add_rms_norm`). Vendor-specific
   implementations for other chips belong in `benchmark/test_<op>.py`, never
   here.
2. **Never use the FlagGems implementation as `native()`.** A self-baseline
   makes every SpeedUp 1.0, hides regressions, and defeats the purpose of
   `--base-data`. If the target vLLM has no native kernel, return `None` and
   record `NATIVE_UNAVAILABLE_REASON`; the operator simply has no CUDA baseline
   yet.
3. **No torch compute.** Only input construction and the native call are
   allowed: `randn`/`rand`/`arange`/`zeros`/`empty`/`tensor`, `to`, `contiguous`,
   `clone`, views. Do not implement the operator in torch "for a baseline".
4. **Lazy-import vLLM inside `native()`.** The collector imports every module in
   this directory during discovery; a top-level `import vllm` that fails would
   get the whole module skipped. Catch the error and return `None` with a reason.
5. **One operator per file, flat directory.** The filename should match
   `OP_NAME`; the collector scans `baseline/*.py` only (not subdirectories) and
   keys the JSON by `OP_NAME`.
6. **The grid and dtypes must cover the benchmark.** For every shape and dtype
   the benchmark can emit, a record must exist. Missing rows produce
   `[vs Base][未匹配]`, and when the vendor has no native op they make `pytest`
   fail (see "How records are matched"). Extra rows are only safe when the
   native kernel supports them; aim for the benchmark's effective set, not a
   superset.
7. **A compound `key_shape` requires `config`.** If the key is not literally the
   first tensor's shape, the benchmark can only find the record through the
   tensor-signature fallback built from `config.inputs`.
8. **Declare in-place mutation.** Set `IS_INPLACE = True` and list
   `MUTATED_INPUTS` when the native call writes into an argument. The collector
   restores those tensors outside the timed region; otherwise warmup
   accumulation pollutes the measurement.
9. **Eager timing only.** Do not add CUDA graphs or your own timing loop. The
   collector uses `do_bench` or CUDA events with graph disabled, per the test
   protocol.
10. **License header, formatting, pre-commit.** Copy the Apache header used by
    the surrounding files and run `pre-commit run --all-files`.

## Contract

Each module must export the following. Required fields are checked by
`_discover_ops`; modules missing any of them are skipped with a message.

| Export | Required | Purpose / notes |
| --- | --- | --- |
| `OP_NAME` | yes | JSON top-level key and `--ops` filter. Must equal the benchmark's `op_name`. |
| `DTYPES` | yes | Every grid binding is collected once per dtype. Must cover every dtype the benchmark runs and be supported by the native kernel. |
| `IS_INPLACE` | yes | Whether the native call writes back into an argument. |
| `MUTATED_INPUTS` | optional | Positional indices or keyword names of mutated tensors. Omitted means "snapshot every tensor argument". |
| `NATIVE_UNAVAILABLE_REASON` | optional, strongly recommended | Human-readable reason retained when `native()` returns `None`. |
| `native()` | yes | Resolve the exact vendor callable; return `None` when unavailable. No candidate/mathematical fallback. |
| `grid()` | yes | List of collection-point dicts. Keep it in sync with the benchmark shapes. |
| `build_inputs(binding, dtype, device)` | yes | Return `(args, kwargs)` for one binding and dtype. Must mirror the benchmark's input construction. |
| `key_shape(binding)` | yes | JSON primary key. Prefer the first tensor shape as a `list`; use a compound list or semantic string otherwise. |
| `config(binding, dtype)` | optional, required for compound keys | Real input/output shapes, dtypes, strides, scalars, aliases and mutation notes. Used by the signature fallback. |
| `unsupported_reason(binding, dtype)` | optional | Return a string for known-unsafe native inputs; the collector writes an explicit `status: "unsupported"` row instead of launching or dropping the point. |

## Skeleton

```python
# Copyright 2026 FlagOS Contributors
# ... Apache-2.0 header ...

"""Native-only workload definition; never imported by production kernels."""

import importlib

import torch

OP_NAME = "moe_sum"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

SHAPES = [(512, 6, 4096), (2048, 6, 4096)]


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        module = importlib.import_module("vllm._custom_ops")
    except Exception as exc:  # resolution must not abort collection
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    op = getattr(module, "moe_sum", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "vllm._custom_ops.moe_sum absent"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, topk, hidden = binding["shape"]
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(num_tokens, topk, hidden, dtype=dtype, device=device, generator=gen)
    out = torch.empty(num_tokens, hidden, dtype=dtype, device=device)
    return (x, out), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    num_tokens, topk, hidden = binding["shape"]
    dt = str(dtype)
    return {
        "inputs": {"input": {"shape": [num_tokens, topk, hidden], "dtype": dt}},
        "outputs": {"output": {"shape": [num_tokens, hidden], "dtype": dt}},
        "dims": {"num_tokens": num_tokens, "topk": topk, "hidden_size": hidden},
    }
```

## Step by step

### 1. Resolve the real native callable

- Inspect the target vLLM image, not a guess: list the module and the callable
  signature (`inspect.signature`), and check how the model layer calls it.
- The entry the model layer uses wins over a "more primitive" entry. For
  example, if the model passes raw gates and pre-activates beta outside the
  kernel, the baseline must call the fused-gate entry and keep beta
  pre-activation outside the timed region.
- vLLM moves modules between releases. Probe a small ordered list of paths,
  keep the reason for the last failure, and never fall back to a different
  implementation. See `chunk_kda` and `mhc_shifted_post_pre` for examples.
- Version-dependent keyword arguments should be detected with
  `inspect.signature` instead of being passed unconditionally.

### 2. Match the benchmark workload

The grid must describe exactly the workloads the benchmark will time:

- Prefer model-derived shapes over synthetic square matrices.
- Either import `SHAPES` from the baseline in the benchmark, or keep the two
  lists (and `core_shapes.yaml`, when present) identical. Drift is the most
  common cause of `[vs Base][未匹配]`.
- The benchmark's effective shape set is the one after `core_shapes.yaml` and
  `--level`/`--shape_file` overrides. If the benchmark can emit a shape, the
  grid must contain it.
- [`mhc_shifted_post_pre.py`](mhc_shifted_post_pre.py) and its benchmark show
  the recommended single-source pattern: the benchmark loads this module by path
  and reuses `SHAPES` and `build_inputs` instead of copying them.

### 3. Build inputs exactly like the benchmark

- Reproduce shape, dtype, stride/layout, scalar parameters, optional `None`
  arguments and aliasing. If the operator receives an output buffer, allocate it
  here too.
- Move host-side preprocessing that the model does outside the kernel (beta
  sigmoid, gate activation, L2 normalization when not fused) into
  `build_inputs`, so the timed region contains only the native kernel.
- Use a local `torch.Generator(...).manual_seed(...)` rather than mutating the
  global RNG. `build_inputs` is called in the collector process and again in
  the NCU subprocess and must be repeatable and side-effect free.
- Do not retain large temporary tensors in module globals; the NCU subprocess
  allocates its own copy and large shapes can otherwise OOM.

### 4. Choose `key_shape` and `config` together

- If the first tensor already identifies the workload, return its shape as a
  `list`. The benchmark then finds the record with a direct lookup.
- If not (varlen, several equally-sized tensor sets, multiple scalar modes),
  build a compound key: append the distinguishing dimension (for example the
  number of sequences) or return a semantic string.
- A compound key **must** come with `config()`. `config["inputs"]` must contain
  the same tensor-shape multiset that the benchmark passes in args and kwargs:
  all tensors, including auxiliary tensors in kwargs and output buffers passed
  as arguments. Scalars go in as `{"scalar": value}`. `dims`, strides, layout,
  alias and mutation notes are for humans and for future audits.

### 5. Declare in-place and unsupported inputs

- If the native call overwrites `args[2]`, set `IS_INPLACE = True` and
  `MUTATED_INPUTS = (2,)`. Prefer explicit locations over snapshotting every
  tensor.
- For known-unsafe native contracts (K != V, unsupported dtype/layout, indexing
  assumptions), implement `unsupported_reason(binding, dtype)` and let the
  collector record an explicit row. Do not let a bad point abort the whole
  operator.

### 6. Validate, collect, refresh

```shell
# Probe the resolver and the first binding/dtype only.
python3 tools/collect_baseline_nvidia.py --check --ops <op_name> --device 0

# Collect one or more operators. NCU must be on PATH for bottleneck counters.
python3 tools/collect_baseline_nvidia.py --device 7 --ops <op_name> \
    --reference-chip H800 --report-dir ncu_reports \
    --output op_vllm_perf_baseline.json

# Verify matching end to end.
pytest -q benchmark/test_<op>.py --level core --iter 1 --warmup 1 \
    --base-data=tools/op_vllm_perf_baseline.json
```

`--check` only exercises the first binding and dtype; it does not prove that
every grid point is launchable. Collection pre-checks the same first point and
skips the whole operator if it fails, so validate large/varlen points
separately.

Adding a module is not enough: the collected JSON must be refreshed and
published according to the acceptance process (the optest `data/` directory),
and the repository copy used by `--base-data` must be updated. A new module with
stale data produces `[vs Base][未匹配]` everywhere.

## Legacy benchmarks

Not every existing `benchmark/test_*.py` follows this standard yet: many older
files still pass a torch function as `torch_op`. Treat those as grandfathered,
not as examples.

- Do not add a new torch-reference baseline. New or modified operators must use
  a vendor-native callable (or `torch_op=None` with `--base-data`).
- When you touch a legacy benchmark for another reason, migrate it in the same
  change: move the native workload into `baseline/`, make `torch_op` the native
  callable or `None`, drop `latency_base`/`speedup` from the metrics when it is
  `None`, and refresh the data.
- Do not delete a test or a benchmark just to "comply". Accuracy tests stay in
  `tests/`; a benchmark covers timing and is not a substitute for them. Deleting
  coverage needs an explicit reason in the PR.

## How records are matched

`benchmark/consts.py` resolves a record in this order:

1. operator name (`op_name` must equal `OP_NAME`);
2. direct shape key: `str(first tensor shape)` against `str(key_shape)`;
3. tensor-signature fallback: the sorted multiset of every tensor shape in
   `args` and `kwargs` against the shapes in `config["inputs"]`;
4. scalar subset disambiguation when several records share a tensor signature.

If the vendor has a native op, a miss only leaves `compared_speedup` empty. If
`torch_op` is `None` (no vendor native) **and** `--base-data` is passed, a miss
raises `RuntimeError` and fails the case. That is why grid and dtype coverage
are part of the contract, not a nice-to-have.

Timing is always eager (no CUDA graph). The collector uses `do_bench` for
non-in-place operators and five warmups plus 25 event-timed samples for
in-place ones; the median is recorded. NCU runs in a subprocess that rebuilds
the inputs and pins the sampled call with an NVTX range.

## Reference chip

The collector writes `_scaling_factors` relative to `--reference-chip`
(default `H800`). The reference chip must be the GPU that actually produced the
latencies, and one JSON must not mix chips. Data collected on H20 (or any other
card) is valid for local analysis, but it is not the official H800/H100
baseline unless the team explicitly changes the reference policy and refreshes
the whole file. Use a separate JSON for such experiments.

## Review checklist

Copy this into the PR description or a review comment:

- [ ] `native()` returns the vLLM/CUDA callable, not a FlagGems or torch
      implementation.
- [ ] `native()` is lazy, returns `None` on failure, and sets
      `NATIVE_UNAVAILABLE_REASON`.
- [ ] `OP_NAME` and the filename match the benchmark's `op_name`.
- [ ] `grid()` matches the benchmark's effective shapes (including
      `core_shapes.yaml` and `--level` overrides).
- [ ] `DTYPES` covers every dtype the benchmark runs.
- [ ] `build_inputs()` mirrors the benchmark's inputs, strides, scalars and
      call convention; host-side preprocessing stays outside the timed region.
- [ ] `key_shape()` is unique; a compound key comes with a complete `config`.
- [ ] `config["inputs"]` tensor-shape multiset equals the benchmark's args and
      kwargs tensors.
- [ ] In-place native calls set `IS_INPLACE` and `MUTATED_INPUTS`.
- [ ] Known-unsafe points return `unsupported_reason`.
- [ ] `--check` passes; collection covers every `(shape, dtype)` pair.
- [ ] `pytest ... --base-data=...` reports no `[vs Base][未匹配]`.
- [ ] JSON data refreshed/published; `_environment` matches the target vLLM
      image.
- [ ] License header and `pre-commit run --all-files` are clean.

## Common rejection reasons

- **Self-baseline**: `native()` points at `flaggems_vllm.*`; every SpeedUp is
  1.0.
- **Torch reference**: a hand-written torch implementation is used as the
  baseline; it is usually much slower than the fused kernel and yields
  meaningless SpeedUp numbers.
- **Shape drift**: the module has a grid, but the benchmark runs different
  shapes from `core_shapes.yaml` or a fixed `set_shapes()`.
- **Dtype drift**: the benchmark runs fp32/fp16/bf16, the module collects only a
  subset.
- **Key without config**: compound key, no `config`, so the signature fallback
  cannot find the record.
- **Config mismatch**: `config["inputs"]` omits tensors passed in kwargs, or
  includes tensors the benchmark does not pass.
- **Unmarked in-place mutation**: warmup keeps mutating the same input and the
  recorded latency is not representative.
- **Top-level vLLM import**: discovery fails and the module silently disappears.
- **No data refresh**: the module is merged but `--base-data` cannot match it.
