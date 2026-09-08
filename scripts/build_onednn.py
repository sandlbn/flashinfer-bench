"""Build a specific oneDNN version from source and point flashinfer-bench at it.

Three oneDNN builds are typically in play on an Intel box and they are rarely the same:

  * the one oneAPI installs, which `FIB_ONEDNN_DIR` finds by default;
  * the one torch bundles, which is what `torch.matmul` -- and therefore a GEMM
    definition's `reference` -- actually executes;
  * whatever upstream has released since.

That matters twice over. A solution linking a different oneDNN than the reference it is
measured against produces a cross-library ratio, not a kernel result. And newer releases add
capability: v3.13 ships an optimized `gated_mlp` implementation (`micro_horz`) where v3.12
had only a reference one, and the Graph API's `gated_mlp` fusion pattern is the public route
to a fused MLP.

This builds a chosen tag into its own prefix so several versions can coexist, and prints the
export that selects it. It never touches the oneAPI install.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/uxlfoundation/oneDNN.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", help="Tag to build, e.g. v3.13.2. Required unless --list.")
    ap.add_argument("--src", type=Path, default=Path("tmp/oneDNN"))
    ap.add_argument(
        "--prefix",
        type=Path,
        default=None,
        help="Install prefix (default: ~/.cache/flashinfer_bench/onednn/<version>).",
    )
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--list", action="store_true", help="List available tags and exit.")
    a = ap.parse_args()

    if not a.version and not a.list:
        sys.exit("--version is required (or pass --list to see the tags).")
    if not (a.src / ".git").exists():
        run(["git", "clone", "--filter=blob:none", REPO, str(a.src)])

    if a.list:
        out = subprocess.run(
            ["git", "-C", str(a.src), "ls-remote", "--tags", "--refs", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        import re

        def key(t):  # version-aware, so v3.13 sorts above v3.9
            return [int(x) for x in re.findall(r"\d+", t)]

        tags = sorted(
            {ln.rsplit("/", 1)[-1] for ln in out.splitlines() if "/v" in ln},
            key=key,
        )
        stable = [t for t in tags if re.fullmatch(r"v[\d.]+", t)]
        print("\n".join(f"  {t}" for t in stable[-10:]))
        return

    # Toolchain is only needed for an actual build, not for --list.
    if not shutil.which("cmake"):
        sys.exit("cmake is required.")
    compiler = os.environ.get("FIB_SYCL_COMPILER") or shutil.which("icpx")
    if not compiler:
        sys.exit("icpx not found. source /opt/intel/oneapi/setvars.sh, or set FIB_SYCL_COMPILER.")

    prefix = a.prefix or Path.home() / ".cache/flashinfer_bench/onednn" / a.version
    if (prefix / "lib" / "libdnnl.so").exists():
        print(f"  already built: {prefix}")
    else:
        run(["git", "-C", str(a.src), "fetch", "--depth", "1", "origin", "tag", a.version])
        run(["git", "-C", str(a.src), "checkout", "--detach", a.version])
        build = a.src / f"build-{a.version}"
        build.mkdir(parents=True, exist_ok=True)
        run(
            [
                "cmake",
                "..",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                "-DCMAKE_BUILD_TYPE=Release",
                # SYCL GPU runtime, which is what an Intel GPU build needs. The CPU runtime
                # stays on the default; we are not measuring CPU here.
                "-DDNNL_CPU_RUNTIME=SEQ",
                "-DDNNL_GPU_RUNTIME=SYCL",
                f"-DCMAKE_C_COMPILER={Path(compiler).with_name('icx')}",
                f"-DCMAKE_CXX_COMPILER={compiler}",
                # The Graph API carries the gated_mlp fusion pattern; without it a newer
                # oneDNN buys much less.
                "-DONEDNN_BUILD_GRAPH=ON",
                "-DDNNL_BUILD_TESTS=OFF",
                "-DDNNL_BUILD_EXAMPLES=OFF",
            ],
            cwd=build,
        )
        run(["cmake", "--build", ".", "-j", str(a.jobs), "--target", "install"], cwd=build)

    print("\n  Built. Select it with:\n")
    print(f"      export FIB_ONEDNN_DIR={prefix}\n")
    print("  Then confirm both sides agree:\n")
    print('      python -c "from flashinfer_bench.integration.providers import \\')
    print("          onednn_link_version, onednn_runtime_version; \\")
    print('          print(onednn_link_version()); print(onednn_runtime_version())"\n')
    print("  A mismatch is recorded in every trace and logged as a warning -- a solution and")
    print("  the reference it is measured against should use the same oneDNN.")


if __name__ == "__main__":
    main()
