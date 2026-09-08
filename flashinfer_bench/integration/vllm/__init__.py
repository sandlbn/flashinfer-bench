"""vLLM integration: put benchmarked kernels into a running vLLM server.

vLLM builds every device-specific layer as a ``CustomOp``, and ``dispatch_forward``
selects ``forward_xpu`` on Intel. Patching that method is enough to substitute a kernel --
no change to how models are constructed, and no fork.

Three properties this deliberately keeps:

**It fails open.** A missing vLLM, a definition with no solution, an unpatched target: all
of them leave the server running vLLM's own kernel. ``PatchManager.patch`` returns False
rather than raising when a target is absent, and every wrapper passes a ``fallback``.

**It is opt-in.** Silently swapping kernels under a serving stack is not debuggable, so
installation is gated on ``FIB_VLLM_INTEGRATION`` and logs what it patched.

**It is reversible.** ``PatchManager`` records the original and
``uninstall_vllm_integrations`` restores it, which is what makes an A/B measurement in one
process possible.

The same call path also feeds tracing, since ``apply(...)`` is the shared entry point for
dispatch and workload collection -- installing these adapters in a real server is how
serving-shaped workloads get collected.

**Patching is how you prove a win, not how you ship one.** FlashInfer reaches vLLM a
different way: vLLM upstream registers ``FlashInferBackend`` in its attention-backend
enum and calls into it through a maintained interface. Nothing is patched, the choice is
configuration, and refactors on either side are caught by that contract. This package
patches instead because a benchmarking tool has to work against stacks it does not
control -- which is the right trade for measuring, and a weak one for deploying.

That distinction ranks the adapters here. ``RMSNorm`` and ``SiluAndMul`` hook
``CustomOp.forward_xpu``, which exists precisely so backends can differ: a real seam, and
about as stable as patching gets. ``GatedMLPAdapter`` hooks ``Qwen2MLP.forward`` -- a model
implementation detail, needing one entry per model class and breaking silently on a
refactor. Treat it as a measurement vehicle. The durable form of that optimization is a
fused-MLP path vLLM selects deliberately, the way it selects an attention backend.

**Install it in the process that owns the weights.** vLLM's V1 engine runs the model in an
``EngineCore`` subprocess, so calling :func:`install_vllm_integrations` from the launching
script patches a copy of the class that never executes a forward pass -- the server runs
unchanged, and nothing reports an error. Either keep the engine in-process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, which is what the benchmark harness does), or have
the worker install it: ``FIB_VLLM_INTEGRATION`` is inherited across the process boundary
precisely so a worker-side hook can act on it without extra plumbing.
"""

from __future__ import annotations

import logging
import os
from typing import List

from flashinfer_bench.integration.patch_manager import get_manager

from .adapters.activation import SiluAndMulAdapter
from .adapters.mlp import GatedMLPAdapter
from .adapters.rmsnorm import RMSNormAdapter

logger = logging.getLogger(__name__)

ENV_VAR = "FIB_VLLM_INTEGRATION"
"""Set truthy to install. Off by default: a serving stack should not change silently."""


def _enabled() -> bool:
    return os.environ.get(ENV_VAR, "").lower() in ("1", "true", "yes", "on")


def install_vllm_integrations(force: bool = False) -> List[str]:
    """Patch vLLM's Intel layers to route through recorded solutions.

    Returns the names of the targets actually patched, so a caller can log or assert on
    them. Idempotent, and a no-op when vLLM is absent.
    """
    if not force and not _enabled():
        logger.debug("%s is not set; leaving vLLM unpatched.", ENV_VAR)
        return []

    manager = get_manager()
    patched: List[str] = []
    for adapter in (RMSNormAdapter(), SiluAndMulAdapter(), GatedMLPAdapter()):
        try:
            targets = adapter.targets()
        except Exception:  # pragma: no cover - an adapter that cannot describe itself
            continue
        for spec in targets:
            if manager.patch(spec, adapter.make_wrapper):
                patched.append(spec.name)

    if patched:
        logger.info("flashinfer-bench patched vLLM: %s", ", ".join(patched))
    else:
        logger.info(
            "flashinfer-bench found no vLLM targets to patch; the server keeps its own " "kernels."
        )
    return patched


def uninstall_vllm_integrations() -> None:
    """Restore vLLM's own implementations."""
    get_manager().unpatch_all()


__all__ = ["install_vllm_integrations", "uninstall_vllm_integrations", "ENV_VAR"]
