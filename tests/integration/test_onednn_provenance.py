"""A trace must say which oneDNN produced it, on both sides of the comparison.

Recording the directory is not enough. `/opt/intel/oneapi/dnnl/latest` is a symlink that
moves, and the only way to change Intel's GEMM kernel catalog is to rebuild oneDNN and put
it at the same path -- so a patched library and a released one record identically.

Worse, the two sides can differ without anyone noticing: a GEMM definition's `reference` is
`torch.matmul`, which runs torch's bundled oneDNN, while a SYCL solution links whatever
FIB_ONEDNN_DIR points at. Observed on this machine: solutions linked 3.11.4 while torch ran
3.12.3, so every "our oneDNN kernel vs torch.matmul" number spanned two libraries.
"""

import pytest

from flashinfer_bench.integration import providers


class TestOneDnnIsIdentifiedByVersion:
    def test_link_version_is_not_a_bare_path(self):
        v = providers.onednn_link_version()
        if v is None:
            pytest.skip("oneDNN not discoverable here")
        head = v.split(" ")[0]
        assert "+" in head, f"expected version+commit, got {head!r}"
        major = head.split(".")[0]
        assert major.isdigit(), f"expected a numeric major version, got {head!r}"

    def test_link_version_records_the_resolved_path_not_the_symlink(self):
        v = providers.onednn_link_version()
        if v is None:
            pytest.skip("oneDNN not discoverable here")
        assert "latest" not in v, (
            "recorded the moving 'latest' symlink; a later toolchain update silently "
            "changes what this trace claims to have measured"
        )

    def test_provenance_exposes_both_sides(self):
        prov = providers.provider_provenance()
        if "onednn" not in prov:
            pytest.skip("oneDNN not installed")
        assert "+" in prov["onednn"], "oneDNN recorded without a version"

    def test_a_mismatch_is_reported_rather_than_silently_averaged(self):
        prov = providers.provider_provenance()
        link, runtime = prov.get("onednn"), prov.get("env:onednn_runtime")
        if not (link and runtime):
            pytest.skip("could not determine both oneDNN versions")
        differ = link.split(" ")[0].split("+")[0] != runtime.split("+")[0]
        assert differ == ("env:onednn_version_mismatch" in prov), (
            "the mismatch flag must track whether the versions actually differ"
        )
