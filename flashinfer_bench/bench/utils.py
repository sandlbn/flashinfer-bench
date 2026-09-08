"""Utility functions for benchmark execution."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch as st
import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
from flashinfer_bench.data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import dtype_str_to_torch_dtype, env_snapshot


def _workload_seed(definition: Definition, workload: Workload, trial: int) -> int:
    """Derive a deterministic RNG seed from the workload identity and trial index.

    Seeding from the workload rather than a global counter makes a benchmark run
    reproducible from the dataset alone, and makes two backends generate bit-identical
    inputs for the same workload -- which is what allows a reference implementation to be
    cross-validated against another device.

    Distinct trials still get distinct data, so multiple trials remain a real test of the
    solution rather than the same tensors measured repeatedly.
    """
    h = hashlib.sha256()
    h.update(definition.name.encode())
    h.update(workload.uuid.encode())
    h.update(repr(sorted(workload.axes.items())).encode())
    h.update(str(trial).encode())
    return int.from_bytes(h.digest()[:8], "big") % ((1 << 63) - 1)


def _rand_tensor(
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate a random tensor on the host, then move it to ``device``.

    Generation is always done on the CPU even when the target is an accelerator. Device
    RNG streams differ between backends, so generating on-device would make the same
    workload produce different inputs on CUDA and XPU, and no cross-backend comparison
    of a reference implementation would be meaningful. The cost is one host-to-device
    copy per tensor per trial, amortised against warmup and timed iterations.
    """
    cpu = torch.device("cpu")

    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=cpu, generator=generator).to(device=device)

    # low-precision floats: generate and clamp in fp32, narrow on the target device
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        t = torch.randn(shape, dtype=torch.float32, device=cpu, generator=generator).clamp_(
            -2.0, 2.0
        )
        return t.to(device=device).to(dtype)

    # booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=cpu, generator=generator).to(
            device=device
        )

    # integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=cpu, dtype=dtype, generator=generator).to(
            device=device
        )

    raise ValueError(f"Unsupported random dtype: {dtype}")


def normalize_outputs(
    out: Any,
    *,
    device: torch.device,
    output_names: List[str],
    output_dtypes: Dict[str, torch.dtype],
) -> Dict[str, torch.Tensor]:
    def to_tensor(name: str, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != device else v
        dtype = output_dtypes[name]
        # Python scalar -> 0-D tensor for comparison
        return torch.tensor(v, dtype=dtype, device=device)

    if isinstance(out, dict):
        return {k: to_tensor(k, v) for k, v in out.items() if k in output_dtypes}

    if isinstance(out, torch.Tensor):
        if len(output_names) != 1:
            raise RuntimeError("Single Tensor returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (int, float, bool)):
        if len(output_names) != 1:
            raise RuntimeError("Scalar returned but multiple outputs are defined")
        name = output_names[0]
        return {name: to_tensor(name, out)}

    if isinstance(out, (tuple, list)):
        if len(out) != len(output_names):
            raise RuntimeError(
                f"Tuple/list has {len(out)} elements but {len(output_names)} outputs expected"
            )
        return {name: to_tensor(name, val) for name, val in zip(output_names, out)}

    raise RuntimeError(
        "Unexpected return type; must be Tensor, scalar, or dict[name -> Tensor/scalar]"
    )


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, cfg: ResolvedEvalConfig
) -> Tuple[float, float, bool, float]:
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)

    total_elements = abs_error.numel()
    if total_elements == 0:
        return 0.0, 0.0, False, 1.0

    required_matched_ratio = (
        cfg.required_matched_ratio if cfg.required_matched_ratio is not None else 1.0
    )
    exceeds_tol_mask = (abs_error > cfg.atol) & (rel_error > cfg.rtol)
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < required_matched_ratio

    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())

    return max_abs, max_rel, exceeds_tol, matched_ratio


def is_sampling_operation(definition: Definition) -> bool:
    return getattr(definition, "op_type", None) == "sampling"


_PACKED_SUFFIXES = ("_packed", "_zeros", "_zp", "_qweight", "_qzeros")
_PACKED_NAMES = frozenset({"qweight", "qzeros", "packed_weight"})


def is_packed_quantized(definition: Definition, name: str) -> bool:
    """Whether integer input ``name`` holds packed sub-byte values rather than a number.

    A 4-bit weight tensor is int32 storage carrying eight independent nibbles, so every
    bit is payload. The generic integer generator draws from [-1024, 1024), which leaves
    the top five nibbles of each word at 0 (or 0xF for negatives): the resulting weight
    matrix is mostly zeros, the K-sum is dominated by the constant `-zero * scale` term,
    and the cancellation that produces makes a correct kernel look wrong. Measured on
    `gemm_int4_w4a16_g128_n4096_k2560`, the vLLM XPU kernel matched 94.7% of elements
    against a 95% gate with such data, and passes comfortably with full-width bits.

    Packed tensors want uniform random *bits*. Recognised by name, and only inside a
    definition that declares a quantization tag, so an int32 index tensor -- `m_indptr`,
    `masked_m` in the grouped-GEMM definitions -- keeps its small, meaningful range.
    """
    if not any(str(t).startswith("quantization:") for t in getattr(definition, "tags", ())):
        return False
    lowered = name.lower()
    return lowered in _PACKED_NAMES or lowered.endswith(_PACKED_SUFFIXES)


def is_quantization_scale(definition: Definition, name: str) -> bool:
    """Whether input ``name`` is a per-block/per-tensor quantization scale.

    A scale is a magnitude: a dequantised weight is ``packed * scale``, and every producer
    of one -- FP8 block scales, MXFP4 exponents, AWQ scales -- emits non-negative values.
    Drawing it from a standard normal instead is not merely unrepresentative, it makes the
    definition untestable. Signed scales flip the sign of whole weight blocks, so a K=2560
    accumulation cancels almost completely and the result sits near zero; pointwise
    relative error against it then reaches the hundreds for *any* implementation.

    Measured on Arc B580 with signed scales at M=512, N=4096, K=2560, against the fp32
    reference: pre-scaling in bf16 gave max_rel 2066, dequantising in fp32 then casting
    gave 927, and per-K-block accumulation in fp32 -- the most accurate formulation there
    is -- still gave 754. The spread says the comparison is measuring cancellation, not
    the kernel. With non-negative scales the same kernels agree to ~4e-3, a bf16 ULP.

    Magnitude alone is not enough: the spread has to be realistic too. A per-group scale
    in a checkpoint is ``max|w_group| / (2**(bits-1) - 1)``, so scales differ from one
    another by a few times. Half-normal magnitudes instead span orders of magnitude and
    put substantial mass near zero -- over a [20, 4096] scale tensor the smallest came out
    at 9.7e-06 against a largest of 3.9. Groups whose scale is ~0 then contribute nothing
    while a few dominate, the K-sum cancels, and relative error against those near-zero
    results explodes for any implementation. Measured on
    ``gemm_int4_w4a16_g128_n4096_k2560``: Intel's W4A16 kernel matched 94.3% of elements
    with half-normal scales and 99.4% with a realistic spread, against a 95% gate.

    So these are drawn uniformly from ``[0.5, 1.5]`` -- positive, and a three-fold spread
    rather than a millionfold one.

    Recognised by name because that is what the dataset's definitions already encode
    (``a_scale``, ``b_scale``, ``B_scale_inv``, ``weight_scale_inv``), and narrowed to
    definitions that declare a quantization tag so an unrelated input called ``scale`` --
    an attention softmax scale, say, which is signed and arbitrary -- is untouched.
    """
    if not any(str(t).startswith("quantization:") for t in getattr(definition, "tags", ())):
        return False
    lowered = name.lower()
    return lowered.endswith("_scale") or lowered.endswith("_scale_inv") or lowered == "scale"


def compute_frequency_distribution(
    runnable: Any,
    inputs: List[Dict[str, Any]],
    device: str,
    definition: Definition,
    num_trials: int = 10000,
) -> torch.Tensor:
    inp = inputs[0]

    workload_batch_size = inp["probs"].shape[0] if inp["probs"].dim() > 1 else 1
    vocab_size = inp["probs"].shape[-1]
    counter = torch.zeros(vocab_size, dtype=torch.int64, device=torch.device(device))

    trials_needed = (num_trials + workload_batch_size - 1) // workload_batch_size
    total_samples_collected = 0

    for trial in range(trials_needed):
        with torch.no_grad():
            out = runnable(**inp)

        output_names = list(definition.outputs.keys())
        output_dtypes = {
            k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
        }

        out_normalized = normalize_outputs(
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )

        samples = out_normalized["samples"]

        if samples.dim() == 0:
            sample_idx = samples.item()
            counter[sample_idx] += 1
            total_samples_collected += 1
        else:  # Batch of samples
            for i in range(samples.numel()):
                sample_idx = samples.flatten()[i].item()
                counter[sample_idx] += 1
                total_samples_collected += 1

    frequency = counter.float() / total_samples_collected
    return frequency


_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _ensure_lfs_downloaded(
    file_path: Path,
    repo_root: Optional[Path],
    tensor_name: Optional[str] = None,
    workload_id: Optional[str] = None,
) -> None:
    """If *file_path* is a Git LFS pointer, pull the real content via git lfs."""
    if repo_root is None:
        return
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(len(_LFS_MAGIC))
    except OSError:
        return
    if header != _LFS_MAGIC:
        return
    try:
        rel = file_path.resolve().relative_to(repo_root.resolve())
    except ValueError as e:
        raise ValueError(
            f"Input safetensors path '{file_path}' is outside trace repo root '{repo_root}'"
        ) from e
    include_path = str(rel).replace("\\", "/")
    tensor_label = f" tensor '{tensor_name}'" if tensor_name else ""
    workload_label = f" for workload '{workload_id}'" if workload_id else ""
    print(f"[lfs] Downloading{tensor_label}{workload_label}: {include_path} …")
    git_exe = shutil.which("git")
    if git_exe is None:
        raise RuntimeError("`git` is required for on-demand Git LFS downloads")
    try:
        subprocess.run(
            [git_exe, "lfs", "pull", "--include", include_path],
            cwd=str(repo_root),
            check=True,
            timeout=500,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timed out downloading LFS object: {include_path}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`git lfs pull` failed for: {include_path}") from e


def load_safetensors(
    definition: Definition, workload: Workload, trace_set_root: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    shapes_list = definition.get_input_shapes(workload.axes)
    input_names = list(definition.inputs.keys())
    expected = dict(zip(input_names, shapes_list))

    safe_tensors: Dict[str, torch.Tensor] = {}
    for name, input_spec in workload.inputs.items():
        if input_spec.type != "safetensors":
            continue

        path = input_spec.path
        if trace_set_root is not None and not Path(path).is_absolute():
            path = str(trace_set_root / path)

        _ensure_lfs_downloaded(
            Path(path), trace_set_root, tensor_name=name, workload_id=workload.uuid
        )
        tensors = st.load_file(path)
        if input_spec.tensor_key not in tensors:
            raise ValueError(f"Missing key '{input_spec.tensor_key}' in '{path}'")
        t = tensors[input_spec.tensor_key]
        # shape check
        if list(t.shape) != expected[name]:
            raise ValueError(f"'{name}' expected {expected[name]}, got {list(t.shape)}")
        # dtype check
        expect_dtype = dtype_str_to_torch_dtype(definition.inputs[name].dtype)
        if t.dtype != expect_dtype:
            raise ValueError(f"'{name}' expected {expect_dtype}, got {t.dtype}")

        try:
            t = t.contiguous().pin_memory()
        except Exception:
            t = t.contiguous()
        safe_tensors[name] = t
    return safe_tensors


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[Dict[str, torch.Tensor]] = None,
    trial: int = 0,
) -> List[Any]:
    """Generate input tensors in definition order.

    Returns a list of input values (tensors or scalars) in the same order
    as definition.inputs.

    Random inputs are deterministic in ``(definition, workload, trial)``: the same trial
    of the same workload yields the same tensors on every run and on every backend. See
    :func:`_workload_seed`.

    Parameters
    ----------
    trial : int
        Index of the benchmark trial. Different trials produce different data.
    """
    shapes = definition.get_input_shapes(workload.axes)
    dev = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_workload_seed(definition, workload, trial))
    out: List[Any] = []

    for idx, (name, spec) in enumerate(definition.inputs.items()):
        dtype = dtype_str_to_torch_dtype(spec.dtype)

        if name in workload.inputs and workload.inputs[name].type == "safetensors":
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = safe_tensors[name]
            out.append(t_cpu.to(device=dev, non_blocking=True))
        elif name in workload.inputs and workload.inputs[name].type == "scalar":
            out.append(workload.inputs[name].value)
        else:  # random
            shape = shapes[idx]

            if shape is None:
                value = _rand_tensor((), dtype, dev, generator).item()
            else:
                value = _rand_tensor(shape, dtype, dev, generator)

                if is_sampling_operation(definition) and name == "probs":
                    value = torch.softmax(value, dim=-1)  # convert logits to probs for sampling
                elif is_quantization_scale(definition, name):
                    # A positive magnitude with a realistic spread. Neither a sign nor a
                    # near-zero scale occurs in a checkpoint, and both make a correct
                    # kernel look wrong. See is_quantization_scale.
                    value = torch.rand(
                        value.shape, dtype=torch.float32, device="cpu", generator=generator
                    ).add_(0.5).to(device=value.device, dtype=value.dtype)
                elif is_packed_quantized(definition, name) and not value.dtype.is_floating_point:
                    # Every bit is payload. See is_packed_quantized.
                    value = torch.randint(
                        torch.iinfo(value.dtype).min,
                        torch.iinfo(value.dtype).max,
                        value.shape,
                        dtype=value.dtype,
                        device="cpu",
                        generator=generator,
                    ).to(value.device)

            out.append(value)
    return out


_MAX_EMBEDDED_LOG_BYTES = 5 * 1024 * 1024


def _read_and_cleanup_log(
    log_path: Optional[str], *, limit: int = _MAX_EMBEDDED_LOG_BYTES
) -> Optional[str]:
    """Read log file content and delete it. Returns None if path is None or file missing."""
    if not log_path:
        return None

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    try:
        with open(log_path, "rb") as fh:
            data = fh.read(limit + 1)
    except (FileNotFoundError, OSError):
        return None
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass

    truncated = len(data) > limit
    if truncated:
        data = data[:limit]

    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n[log truncated]\n"
    return text


def make_eval(
    status: EvaluationStatus,
    device: str,
    log_path: Optional[str] = None,
    correctness: Optional[Correctness] = None,
    performance: Optional[Performance] = None,
    extra_msg: Optional[str] = None,
) -> Evaluation:
    log_text = _read_and_cleanup_log(log_path) or ""
    if extra_msg:
        log_text = log_text + "\n" + extra_msg if log_text else extra_msg
    return Evaluation(
        status=status,
        log=log_text,
        environment=env_snapshot(device),
        timestamp=datetime.now().isoformat(),
        correctness=correctness,
        performance=performance,
    )
