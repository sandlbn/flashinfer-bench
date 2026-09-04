"""
PyTorch example: Load and run the distributed .so kernel using tvm-ffi.
"""

from pathlib import Path

import torch
import tvm_ffi

from flashinfer_bench.device import list_devices


def main():
    # The distributed .so is framework-agnostic, but it is built for one backend.
    # Run it on whichever accelerator this machine actually has.
    devices = list_devices()
    device = devices[0] if devices else "cpu"
    print(f"Running on {device}")

    dist_dir = Path("distributed")
    so_path = dist_dir / "kernel.so"

    entry_symbol = None
    for line in (dist_dir / "kernel_metadata.txt").read_text().split("\n"):
        if line.startswith("Entry Symbol:"):
            entry_symbol = line.split(":", 1)[1].strip()
            break

    if entry_symbol is None:
        raise ValueError("Entry symbol not found in metadata")

    mod = tvm_ffi.load_module(str(so_path))
    kernel_fn = getattr(mod, entry_symbol)

    print(f"Loaded kernel: {entry_symbol}")

    # Prepare inputs: C = A @ B.T
    M, N, K = 1024, 4096, 4096

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16, device=device)

    torch.manual_seed(1)
    B = torch.randn(N, K, dtype=torch.float16, device=device)

    C = torch.empty(M, N, dtype=torch.float16, device=device)

    # Run kernel: C = A @ B.T
    kernel_fn(A, B, C)

    # Verify against PyTorch reference
    reference = torch.matmul(A, B.T)
    max_diff = torch.abs(C - reference).max().item()

    print(f"Max diff vs reference: {max_diff:.6f}")
    print(f"Correct: {max_diff < 1e-2}")


if __name__ == "__main__":
    main()
