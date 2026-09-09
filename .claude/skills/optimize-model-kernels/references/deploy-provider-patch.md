# Deploying a provider patch: edit the vendor's source, rebuild, select it for one process

`mechanisms.md` says which axes a provider's op admits. This file says how a change to that
provider's source reaches a running stack, and how to prove it did.

The mechanism's whole appeal is that it puts nothing in the call path. The serving stack
already calls the provider's op through the dispatcher; if the object the dispatcher
resolves to is one you built, the stack calls your kernel with no interception, no wrapper
and no per-call cost. There is nothing to wire. That is also what makes the proof
load-bearing: a rebuild that is not selected produces a clean null result which reads
exactly like an honest negative.

`scripts/bound_candidates.py` decides whether this is worth doing at all. It admits
`provider_patch` on the fact that the class is a provider kernel **and the source was
bundled**, then prices it like every other mechanism. A resolution run without `--bundle`
rejects every candidate at `source_present` while the source sits in the clone the whole
time, so re-resolve before reading anything into that gate.

## Before anything is built: the resolution has to see the provider

`scripts/pull_kernel_source.py` classes an op from `torch._C._dispatch_dump`, and an op is
in a process only once whatever registers it has been imported. Run it with
`--from-harnesses`, which carries the `PROVIDERS` the discovery recorded, and check the
`providers :` line it prints names the extension module. With nothing imported, every
provider op resolves as a Python-registered custom op with no source -- indistinguishable
from the provider not being installed.

## Detect the installation shape

```bash
python -c "import <provider pkg>, pathlib; print(pathlib.Path(<provider pkg>.__file__).parent)"
python -c "import <provider pkg>.<ext> as m; print(m.__file__)"
```

A path inside the interpreter's `site-packages` is a wheel: nothing edited in a clone
reaches a process until it has been rebuilt and selected. A path inside a checkout is an
editable install, and the project's own in-place build is enough. Both cases still need the
identity proofs below; neither is confirmed by the version string, which is stamped from
the clone's tags and says nothing about which object is loaded.

## Build only the extension that carries the op

The vendor's `setup.py` builds every extension the project has. Read the project's own
`CMakeLists.txt` header for whether a single target and component can be built and installed
alone, and for their names; where it can, turning the other kernel groups off is what makes
the rebuild cheap enough to iterate on.

```bash
( source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" >/dev/null 2>&1
  source "${CMPLR_ROOT:-${ONEAPI_ROOT:-/opt/intel/oneapi}/compiler/latest}/env/vars.sh" >/dev/null 2>&1
  command -v icpx                      # nothing below is the vendor's toolchain without this
  cmake -G Ninja -S <clone> -B tmp/provider-builds/<id>/build \
    -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
    -DCMAKE_INSTALL_PREFIX=tmp/provider-builds/<id>/install \
    -D<the op's kernel group>=ON  -D<every other kernel group>=OFF
  cmake --build tmp/provider-builds/<id>/build --target <ext> -j "${MAX_JOBS:-4}"
  cmake --install tmp/provider-builds/<id>/build --component <ext> )
```

Traps this invocation exists to avoid:

- **Sourcing the top-level `setvars.sh` may leave the compiler off `PATH`.** Source the
  compiler component's own `env/vars.sh` as well, and check `command -v icpx` before
  configuring. Without it CMake silently selects the system C++ and configures a build that
  is not the vendor's toolchain -- it may even succeed.
- **Cap the parallelism.** These are large SYCL translation units; `-j` at the core count
  will thrash a shared host. Watch memory during the first build and set `MAX_JOBS` from it.
- **Restrict the ahead-of-time device list** to the part's target from the capability
  record (the project reads it from its own environment variables). The vendor's wheel
  compiles for several parts and that time buys nothing here.
- **A dependency the project fetches at configure time** is cloned from the network unless
  it is pointed at a local checkout through the project's own source-dir variable, even when
  the kernel group that uses it is off.

## Select the rebuilt object for one process

Do not install over the vendor's package: the comparison has to be reversible and the stock
object has to stay untouched. Build an overlay directory that is the provider package with
one file replaced, and put it first on `PYTHONPATH` for the arm under test.

```bash
SEL=tmp/provider-builds/<id>/select/<provider pkg>
mkdir -p "$SEL"
STOCK=$(python -c "import <provider pkg>, pathlib; print(pathlib.Path(<provider pkg>.__file__).parent)")
for f in "$STOCK"/*; do ln -s "$f" "$SEL/$(basename "$f")"; done   # everything you did not build
ln -sf "$PWD/tmp/provider-builds/<id>/install/<provider pkg>/<ext>" "$SEL/<ext>"
PYTHONPATH=tmp/provider-builds/<id>/select <command>
```

Link the rest of the package rather than shadowing the directory with only what you built:
a provider ships several extensions and its Python modules, and an overlay that omits them
hides them.

## Prove the rebuilt object is the one running

Run every applicable proof inside the arm being measured, under the same environment.

Source: the dispatcher's own registration record, the loader's module path, and the
kernel-name column of a device profile.

| Proof | Command | What separates the two builds |
| --- | --- | --- |
| loaded object | `python -c "import <pkg>.<ext> as m; print(m.__file__)"` | the overlay path against the vendor package's |
| mapped object | `grep <ext> /proc/self/maps` | resolves the symlink, so it names the file the loader opened |
| registration | `python -c "import torch, <pkg>.<ext>; print(torch._C._dispatch_dump('<ns>::<op>'))"` | the vendor's line cites its build container's source root, a directory that does not exist here; yours cites the clone |
| op set | the project's schemas before and after, compared | a build with kernel groups off can register fewer ops; check rather than assume |
| kernel identity | a device profile's kernel-name column | give the patched functor a suffix and the name in that column is the proof |
| inside a serving worker | the worker's `/proc/<pid>/maps`, or keep the engine in-process for the proof run | the model runs in a subprocess, so a proof taken in the launcher proves nothing |

## The rebuilt-unpatched control

Rebuild the *unmodified* source first and carry it as its own arm. It separates two things
that a single A/B confounds: what the edit did, and what rebuilding with a different
compiler, a different device list and different feature flags did on its own.

Check the control against the vendor's build the way any arm is checked -- the ops it
registers, and a bitwise digest of each hot op's output at the shapes the model presents.
An unmodified source that does not reproduce the vendor's output bit for bit means the
build differs from the vendor's in a way that will be charged to the edit later.

Then measure it: `scripts/measure_serving_win.py --plain-arm`, arms differing only by the
`PYTHONPATH` that selects the build. With the source unmodified that run is an A/A, and its
verdict is the noise floor any later provider-patch claim has to clear.

## Revert

Nothing was installed, so revert is dropping `PYTHONPATH`. Keep the edit as a diff of the
clone (`git -C <clone> diff`) beside the build, because the build directory is the only
other record of what was compiled.

## Sources

- `scripts/pull_kernel_source.py` -- the class per op, the bundle, and the `PROVIDERS` it imports
- `scripts/bound_candidates.py` -- the gates `provider_patch` passes and the one it is priced at
- `scripts/measure_serving_win.py` -- the plain-arm A/B, its token-digest gate and its verdicts
- `mechanisms.md` -- the axes a provider's op admits, indexed by regime
- `gates.md` -- what makes a number count
