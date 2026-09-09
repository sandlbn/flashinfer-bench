"""Derive a trial harness from baseline.py that calls the patched entry point.

The patched provider registers the replacement kernel under `_C::fused_add_rms_norm_sg`
*in addition to* the vendor's `_C::fused_add_rms_norm`, so one process can hold both and
`kernel_trials benchmark` can pair them. It is the same op with the same schema; the extra
name exists only for the measurement and is not part of the delivered diff, where the
replacement takes over `fused_add_rms_norm` itself.

  python mkvariant.py <out.py> [<op suffix, default _sg>]
"""
import pathlib, sys

out = pathlib.Path(sys.argv[1])
suffix = sys.argv[2] if len(sys.argv) > 2 else "_sg"
base = (pathlib.Path(__file__).parent / "baseline.py").read_text()
old = """class Model(nn.Module):
    def forward(self, t0, t1, t2):
        return _op()(t0, t1, t2, 1e-06) or t0
"""
assert base.count(old) == 1
new = '''VARIANT = "fused_add_rms_norm%s"


class Model(nn.Module):
    def forward(self, t0, t1, t2):
        _op()  # imports whatever registers the provider's ops
        return getattr(torch.ops._C, VARIANT)(t0, t1, t2, 1e-06) or t0
''' % (suffix,)
out.write_text(base.replace(old, new))
print(out)
