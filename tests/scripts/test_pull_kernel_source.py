"""The command PROVENANCE.md tells the reader to run next must be one the CLI accepts."""

import importlib.util
import json
import pathlib
import re
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_next_step_parses_against_the_real_cli():
    pks = _load("pull_kernel_source")
    kt = _load("kernel_trials")
    text = pks._PROVENANCE.format(
        op="_C::rms_norm",
        series="_C_rms_norm",
        device="xpu:0",
        keys="XPU",
        registered="x.so",
        project="p",
        schema="s",
        files="f",
    )
    commands = re.findall(r"`python scripts/kernel_trials\.py ([^`]*)`", text)
    assert commands, text
    for command in commands:
        args = kt.build_parser().parse_args(command.split())
        assert args.cmd == "init"
        assert args.name == "_C_rms_norm"
        assert args.baseline.endswith("harness.py")


# ---------------------------------------------------------------------------------------
# oneDNN: parsing the library's verbose log, mapping it to source, and bundling it.
# Everything below is synthetic and GPU-free; the line shapes follow oneDNN's own v1
# ``template:`` header, and the source tree is a stand-in laid out the way the real one is.
# ---------------------------------------------------------------------------------------

_TEMPLATE = (
    "onednn_verbose,v1,primitive,info,template:operation,engine,primitive,implementation,"
    "prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time"
)
_BANNER = "onednn_verbose,v1,info,oneDNN v9.8.7 (commit 0123456789abcdef0123456789abcdef01234567)"
_ENGINE = (
    "onednn_verbose,v1,info,gpu,engine,0,backend:Level Zero,name:Vendor(R) Card,"
    "driver_version:1.2.3,binary_kernels:enabled"
)
_DISPATCH = (
    "onednn_verbose,v1,primitive,create:dispatch,gemm,gpu:0,gemm,jit:xe_hp:gemm:any,undef,"
    "src_a:bf16::blocked:ab::f0 src_b:bf16::blocked:ba::f0 dst:bf16::blocked:ab::f0,,,"
    "4x1024:1024x4096,unsupported format tag,src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:79"
)
_NESTED = (
    "onednn_verbose,v1,primitive,create_nested:kernel_cache_hit,gpu:0,gemm,jit:gemm:any,undef,"
    "src_a:bf16::blocked:ab::f0 src_b:bf16::blocked:ba::f0 dst:bf16::blocked:ab::f0,,,"
    "4x1024:1024x4096,0.01"
)
_EXEC = (
    "onednn_verbose,v1,primitive,exec,gpu:0,matmul,jit:gemm:any,undef,"
    "src:bf16::blocked:ab::f0 wei:bf16::blocked:ba::f0 dst:bf16::blocked:ab::f0,"
    "attr-scratchpad:user,,4x1024:1024x4096,0.5"
)
_CONSIDER = (
    "onednn_verbose,v1,info,gpu,gemm,consider:G gemm HHS TNN 16 16 aB wg 4x2 sys,"
    "score:4337337.487821"
)
_LINES = [_BANNER, _ENGINE, _TEMPLATE, _DISPATCH, _CONSIDER, _CONSIDER, _NESTED, _EXEC, _EXEC]


def _run(pks, lines=None):
    return pks.parse_onednn_verbose(_LINES + ["PROBE_KERNEL x"] if lines is None else lines)


def test_parse_reads_version_environment_selection_rejection_and_strategies():
    pks = _load("pull_kernel_source")
    run = _run(pks)
    assert run.version == "9.8.7"
    assert run.commit.startswith("0123456789ab")
    assert run.environment == [_ENGINE.split(",", 3)[3]]
    # Two identical exec lines (warm-up, then profiled) are one fact.
    assert len(run.executed) == 1
    sel = run.executed[0]
    assert (sel["kind"], sel["implementation"], sel["problem_desc"]) == (
        "matmul",
        "jit:gemm:any",
        "4x1024:1024x4096",
    )
    assert sel["attributes"] == "attr-scratchpad:user"
    assert sel["memory_descriptors"].startswith("src:bf16")
    # The nested gemm is a reported selection for its own kind.
    assert [(c["kind"], c["implementation"]) for c in run.created] == [("gemm", "jit:gemm:any")]
    assert run.selected("gemm") == {"jit:gemm:any"} and run.selected("matmul") == {"jit:gemm:any"}
    assert run.rejected == [
        {
            "component": "gemm",
            "engine": "gpu:0",
            "kind": "gemm",
            "implementation": "jit:xe_hp:gemm:any",
            "prop_kind": "undef",
            "memory_descriptors": (
                "src_a:bf16::blocked:ab::f0 src_b:bf16::blocked:ba::f0 dst:bf16::blocked:ab::f0"
            ),
            "attributes": "",
            "problem_desc": "4x1024:1024x4096",
            "reason": "unsupported format tag",
            "path": "src/gpu/intel/gemm/jit_xe_hp_systolic.cpp",
            "line": "79",
        }
    ]
    assert run.considered == [("G gemm HHS TNN 16 16 aB wg 4x2 sys", 4337337.487821)]
    assert run.kinds() == ["gemm", "matmul"]
    assert run.engines() == ["gpu"]
    assert all(line.startswith("onednn_verbose,") for line in run.lines)
    assert len(run.lines) == len(_LINES)
    # The nested gemm is a different (kind, implementation) pair from what executed.
    assert run.summary()["nested"] == [
        {"kind": "gemm", "implementation": "jit:gemm:any", "problem_desc": "4x1024:1024x4096"}
    ]
    assert run.summary()["strategies_scored"] == 1


def test_parse_follows_the_logs_own_column_order():
    """A version that adds or reorders a column must not shift the fields we read."""
    pks = _load("pull_kernel_source")
    template = (
        "onednn_verbose,v1,primitive,info,template:operation,engine,primitive,"
        "implementation,prop_kind,extra_column,memory_descriptors,attributes,auxiliary,"
        "problem_desc,exec_time"
    )
    line = (
        "onednn_verbose,v1,primitive,exec,gpu:0,matmul,some:impl,undef,EXTRA,"
        "src:f32::blocked:ab::f0,attr-x,,8x8:8x8,0.1"
    )
    run = pks.parse_onednn_verbose([template, line])
    assert run.executed[0]["implementation"] == "some:impl"
    assert run.executed[0]["problem_desc"] == "8x8:8x8"
    assert run.executed[0]["attributes"] == "attr-x"


def test_parse_ignores_lines_it_does_not_understand():
    pks = _load("pull_kernel_source")
    run = pks.parse_onednn_verbose(
        [
            "onednn_verbose,v1,info,gpu,runtime:SYCL",
            "onednn_verbose,v1,primitive,create:dispatch,short",
            "onednn_verbose,v1,primitive,exec:check,primitive,unused argument,src/x.cpp:1",
            "garbage",
        ]
    )
    assert run.executed == [] and run.rejected == [] and run.considered == []
    assert run.environment == ["gpu,runtime:SYCL"]


def test_levers_come_from_the_librarys_own_vocabulary():
    pks = _load("pull_kernel_source")
    assert pks.onednn_lever("unsupported format tag")[0] == "operand layout"
    assert pks.onednn_lever("unsupported datatype combination")[0] == "data type"
    assert pks.onednn_lever("unsupported post-ops")[0] == "attributes / post-ops"
    assert pks.onednn_lever("runtime dimension is not supported")[0].startswith("problem shape")
    assert pks.onednn_lever("unsupported gpu architecture")[0] == "hardware / build"
    assert pks.onednn_lever("heuristic fail: small M")[0] == "library heuristic"
    lever, action = pks.onednn_lever("something this tool has never seen")
    assert lever == "unclassified" and "read the gate" in action


_GATE_FILE = (
    "status_t xe_hp_systolic_t::pd_t::init(impl::engine_t *engine) {\n"
    "    VDISPATCH_GEMM(engine_ok, VERBOSE_UNSUPPORTED_DEVICE_FEATURE);\n"
    "    VDISPATCH_GEMM(limits_ok, VERBOSE_RUNTIMEDIM_UNSUPPORTED);\n"
    "    // comment\n"
    "    VDISPATCH_GEMM_SC(\n"
    "            set_default_formats(d->a_type()), VERBOSE_UNSUPPORTED_TAG);\n"
    "    VDISPATCH_GEMM(!use_nocopy(), VERBOSE_SKIP_PRIMITIVE_IMPL);\n"
    '    VDISPATCH_GEMM(arch_ok, VERBOSE_UNSUPPORTED_ARCH, "gpu");\n'
    "    return status::success;\n"
    "}\n"
    "\n"
    "status_t xe_hp_systolic_t::pd_t::set_default_formats(data_type_t dt) {\n"
    "    return status::unimplemented;\n"
    "}\n"
)


def _fake_checkout(root):
    """A oneDNN-shaped tree: enough of the real layout for every rule to have a target."""
    files = {
        "src/common/verbose.cpp": "// prints onednn_verbose lines\n",
        "src/gpu/gpu_impl_list.hpp": (
            "#ifndef GPU_GPU_IMPL_LIST_HPP\n"
            "#define GPU_GPU_IMPL_LIST_HPP\n"
            "#define GPU_INSTANCE(...) impl_list_item_t(__VA_ARGS__),\n"
            "#define GPU_INSTANCE_INTEL(...) DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))\n"
            "#define GPU_INSTANCE_NVIDIA(...) DNNL_GPU_NVIDIA_ONLY(GPU_INSTANCE(__VA_ARGS__))\n"
            "#define GPU_INSTANCE_GENERIC_SYCL(...) \\\n"
            "    DNNL_GPU_GENERIC_SYCL_ONLY(GPU_INSTANCE(__VA_ARGS__))\n"
            "#ifdef DNNL_DEV_MODE\n"
            "#define GPU_INSTANCE_INTEL_DEVMODE(...) DNNL_GPU_INTEL_ONLY(GPU_INSTANCE(__VA_ARGS__))\n"
            "#else\n"
            "#define GPU_INSTANCE_INTEL_DEVMODE(...)\n"
            "#endif\n"
            "#endif\n"
        ),
        "src/gpu/gpu_matmul_list.cpp": (
            "const std::map<pk_impl_key_t, std::vector<impl_list_item_t>>\n"
            "        impl_list_map REG_MATMUL_P({\n"
            "    {{forward}, {\n"
            "        GPU_INSTANCE_INTEL(intel::matmul::gemm_t)\n"
            "        GPU_INSTANCE_INTEL(intel::matmul::ref_t)\n"
            "        GPU_INSTANCE_NVIDIA(nvidia::cudnn_matmul_t)\n"
            "        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_matmul_t)\n"
            "        nullptr,\n"
            "    }},\n"
            "});\n"
        ),
        "src/gpu/gpu_gemm_list.cpp": (
            "        GPU_INSTANCE_INTEL_DEVMODE(intel::gemm::conv_t)\n"
            "        GPU_INSTANCE_INTEL(intel::gemm::xe_hp_systolic_t)\n"
            "        GPU_INSTANCE_INTEL(intel::gemm::gen_t)\n"
            "        GPU_INSTANCE_INTEL(intel::gemm::with_post_ops_t)\n"
        ),
        "src/gpu/intel/matmul/gemm.hpp": (
            '        DECLARE_COMMON_PD_T(gemm_pd_ ? gemm_pd_->name() : "gemm_t", gemm_t);\n'
        ),
        "src/gpu/intel/matmul/ref.hpp": '        DECLARE_COMMON_PD_T("ocl:ref:any", ref_t);\n',
        "src/gpu/intel/gemm/conv.hpp": '        DECLARE_COMMON_PD_T("conv:ir", conv_t);\n',
        "src/gpu/intel/gemm/jit.hpp": '        DECLARE_COMMON_PD_T("jit:gemm:any", gen_t);\n',
        "src/gpu/intel/gemm/jit.cpp": "// body of gen_t\n",
        "src/gpu/intel/gemm/with_post_ops.hpp": (
            '        DECLARE_COMMON_PD_T("ocl:with_po:any", with_post_ops_t);\n'
        ),
        "src/gpu/intel/gemm/jit_xe_hp_systolic.hpp": (
            '        DECLARE_COMMON_PD_T("jit:xe_hp:gemm:any", xe_hp_systolic_t);\n'
        ),
        # The cited gate must sit at line 79, as the synthetic dispatch line says.
        "src/gpu/intel/gemm/jit_xe_hp_systolic.cpp": "\n" * 74 + _GATE_FILE,
        "src/gpu/intel/gemm/jit/gen_kernel.cpp": (
            '    verbose_printf("info,gpu,gemm,consider:%s,score:%f\\n", e, s);\n'
        ),
        "src/gpu/intel/gemm/jit/selector/db/kernel.db": "{{'C', \"gemm\"}}\n",
        "src/gpu/nvidia/cudnn_matmul.hpp": (
            '        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_matmul_t);\n'
        ),
        "src/gpu/generic/sycl/ref_matmul.hpp": (
            '        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_matmul_t);\n'
        ),
        # A CPU implementation reusing a GPU name: engine scoping must skip it.
        "src/cpu/gemm/ref.hpp": '        DECLARE_COMMON_PD_T("jit:gemm:any", cpu_gen_t);\n',
        # A name computed at run time: the literal rule must report a gap, not a guess.
        "src/gpu/intel/conv/jit.hpp": "        DECLARE_COMMON_PD_T(name_.c_str(), conv_t);\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def test_checkout_is_identified_by_the_paths_the_library_cites(tmp_path):
    pks = _load("pull_kernel_source")
    other = tmp_path / "some_other_project"
    (other / "src").mkdir(parents=True)
    onednn = _fake_checkout(tmp_path / "onednn_src")
    roots = [other, onednn]
    assert pks.onednn_checkout(roots, ["src/gpu/intel/gemm/jit_xe_hp_systolic.cpp"]) == onednn
    # No rejection to cite: the file that prints the log is the anchor.
    assert pks.onednn_checkout(roots, []) == onednn
    assert pks.onednn_checkout([other], []) is None


def test_impl_name_maps_to_its_declaration_scoped_to_the_engine(tmp_path):
    pks = _load("pull_kernel_source")
    tree = pks._Tree(_fake_checkout(tmp_path / "onednn_src"))
    sites = pks.onednn_impl_sites(tree, "jit:gemm:any", ["gpu"])
    assert [p for p, _ in sites] == ["src/gpu/intel/gemm/jit.hpp", "src/gpu/intel/gemm/jit.cpp"]
    assert "gen_t" in sites[0][1] and "line 1" in sites[0][1]
    assert "cpu_gen_t" not in " ".join(why for _, why in sites)
    # A computed name is not mapped -- and not guessed either.
    assert pks.onednn_impl_sites(tree, "jit:ir:conv", ["gpu"]) == []
    decls = pks.onednn_declarations(tree, "gpu")
    computed = [d for d in decls if not d["literal"]]
    assert {d["expr"] for d in computed} == {
        'gemm_pd_ ? gemm_pd_->name() : "gemm_t"',
        "name_.c_str()",
    }


def test_build_conditions_are_read_from_the_macro_header(tmp_path):
    pks = _load("pull_kernel_source")
    tree = pks._Tree(_fake_checkout(tmp_path / "onednn_src"))
    conds = pks.onednn_build_conditions(tree, "gpu")
    assert conds["GPU_INSTANCE_INTEL_DEVMODE"] == "compiled out when !(DNNL_DEV_MODE)"
    assert conds["GPU_INSTANCE_NVIDIA"] == "built only for nvidia"
    # A continued #define must not inherit the include guard as its condition.
    assert conds["GPU_INSTANCE_GENERIC_SYCL"] == "built only for generic sycl"
    assert conds["GPU_INSTANCE_INTEL"] == "built only for intel"
    assert "GPU_INSTANCE" not in conds


def test_candidate_lists_carry_every_entrys_name_and_outcome(tmp_path):
    pks = _load("pull_kernel_source")
    tree = pks._Tree(_fake_checkout(tmp_path / "onednn_src"))
    run = _run(pks)
    decls = pks.onednn_declarations(tree, "gpu")

    path, rows = pks.onednn_candidates(tree, "gemm", "gpu", decls, run)
    assert path == "src/gpu/gpu_gemm_list.cpp"
    assert [(r["class"], r["name"], r["outcome"]) for r in rows] == [
        ("intel::gemm::conv_t", "conv:ir", "build-conditional"),
        ("intel::gemm::xe_hp_systolic_t", "jit:xe_hp:gemm:any", "rejected"),
        ("intel::gemm::gen_t", "jit:gemm:any", "selected"),
        ("intel::gemm::with_post_ops_t", "ocl:with_po:any", "not tried"),
    ]
    assert rows[0]["detail"] == "compiled out when !(DNNL_DEV_MODE)"
    assert "jit_xe_hp_systolic.cpp:79" in rows[1]["detail"]

    path, rows = pks.onednn_candidates(tree, "matmul", "gpu", decls, run)
    assert [(r["class"].split("::")[-1], r["outcome"]) for r in rows] == [
        ("gemm_t", "selected"),
        ("ref_t", "not tried"),
        ("cudnn_matmul_t", "not built"),
        ("ref_matmul_t", "not built"),
    ]
    # The wrapper's name is computed; the attribution says so and names the nested kind.
    assert rows[0]["name"] == "" and rows[0]["computed"].startswith("gemm_pd_")
    assert "by elimination" in rows[0]["detail"] and "nested `gemm`" in rows[0]["detail"]
    assert rows[2]["detail"] == "built only for nvidia"
    # An entry declared in a *different* directory than its namespace names is not matched.
    assert all(r["declared_at"].startswith("src/gpu/") for r in rows if r["declared_at"])


def test_gate_is_read_at_the_cited_line_with_position_helpers_and_what_follows(tmp_path):
    pks = _load("pull_kernel_source")
    tree = pks._Tree(_fake_checkout(tmp_path / "onednn_src"))
    g = pks.onednn_gate(tree, "src/gpu/intel/gemm/jit_xe_hp_systolic.cpp", "79")
    assert g["statement"].strip().startswith("VDISPATCH_GEMM_SC(")
    assert "set_default_formats(d->a_type())" in g["statement"]
    assert g["function"] == "xe_hp_systolic_t::pd_t::init"
    assert (g["index"], g["total"]) == (3, 5)
    assert g["helpers"] == [{"name": "set_default_formats", "from": 86, "to": 88}]
    assert g["remaining"] == ["VERBOSE_SKIP_PRIMITIVE_IMPL", "VERBOSE_UNSUPPORTED_ARCH"]
    # A line the tree does not have yields an empty, well-formed record.
    assert pks.onednn_gate(tree, "src/nowhere.cpp", "5")["remaining"] == []
    assert pks.onednn_gate(tree, "src/gpu/intel/gemm/jit.cpp", "999")["statement"] == ""


def test_sources_cover_winner_chain_gates_and_catalog_and_report_gaps(tmp_path):
    pks = _load("pull_kernel_source")
    tree = pks._Tree(_fake_checkout(tmp_path / "onednn_src"))
    files, gaps, analysis = pks.onednn_sources(tree, _run(pks))
    assert [(p, t) for p, _, t in files] == [
        ("src/gpu/intel/gemm/jit.hpp", "rule"),
        ("src/gpu/intel/gemm/jit.cpp", "rule"),
        ("src/gpu/gpu_gemm_list.cpp", "rule"),
        ("src/gpu/gpu_matmul_list.cpp", "rule"),
        ("src/gpu/intel/gemm/jit_xe_hp_systolic.cpp", "cited"),
        ("src/gpu/intel/gemm/jit_xe_hp_systolic.hpp", "rule"),
        ("src/gpu/intel/gemm/jit/gen_kernel.cpp", "rule"),
        ("src/gpu/intel/gemm/jit/selector/db/kernel.db", "rule"),
    ]
    whys = {p: w for p, w, _ in files}
    assert "unsupported format tag (line 79)" in whys["src/gpu/intel/gemm/jit_xe_hp_systolic.cpp"]
    assert "candidate order for `matmul`" in whys["src/gpu/gpu_matmul_list.cpp"]
    assert set(analysis["candidates"]) == {"gemm", "matmul"}
    gate = analysis["gates"][0]
    assert gate["lever"] == "operand layout" and gate["index"] == 3
    # The matmul-level name resolved to a gemm-level declaration: say that it is a wrapper.
    assert len(gaps) == 1 and "wrapper" in gaps[0] and "`matmul`" in gaps[0]

    # A selected implementation whose name is computed at run time is an explicit gap.
    computed = pks.parse_onednn_verbose(
        [
            _TEMPLATE,
            "onednn_verbose,v1,primitive,exec,gpu:0,convolution,jit:ir,forward,"
            "src:f16::blocked:acdb::f0,,,mb1_ic8oc8_ih8oh8kh3,0.2",
        ]
    )
    files, gaps, _ = pks.onednn_sources(tree, computed)
    assert files == []  # no convolution list in the stand-in tree, nothing to copy
    assert len(gaps) == 1 and "`jit:ir`" in gaps[0] and "computed at run time" in gaps[0]


def test_bundle_copies_source_and_writes_the_extended_provenance(tmp_path):
    pks = _load("pull_kernel_source")
    checkout = _fake_checkout(tmp_path / "onednn_src")
    harness = tmp_path / "h.py"
    harness.write_text('OP = "aten.linear.default"\n')
    out = pks.bundle_onednn(
        "aten::linear",
        "xpu:0",
        [],
        ["/build/aten/src/ATen/x.cpp"],
        _run(pks),
        harness,
        [["T", [4, 1024], "bfloat16"], ["T", [4096, 1024], "bfloat16"]],
        tmp_path / "pulled",
        roots=[checkout],
    )
    assert out == tmp_path / "pulled" / "aten_linear"
    assert (out / "harness.py").is_file()
    assert (out / "source/src/gpu/intel/gemm/jit.hpp").read_text().count("jit:gemm:any") == 1
    gate = (out / "source/src/gpu/intel/gemm/jit_xe_hp_systolic.cpp").read_text().splitlines()
    assert "VDISPATCH" in gate[78]
    assert _DISPATCH in (out / "verbose.log").read_text()
    repro = (out / "repro.py").read_text()
    assert '"aten"' in repro and '"linear"' in repro and "[4096, 1024]" in repro
    assert "ONEDNN_VERBOSE=all" in repro

    text = (out / "PROVENANCE.md").read_text()
    assert "# aten::linear" in text
    assert "`9.8.7+0123456789ab`" in text
    assert "Vendor(R) Card" in text and "driver_version:1.2.3" in text
    assert "| `matmul` | `jit:gemm:any` | `4x1024:1024x4096` |" in text
    assert text.count("**selected** `[reported]`") == 2  # one per candidate list
    assert "**rejected** `[reported]` | unsupported format tag at" in text
    assert "| gate `[cited]` | `src/gpu/intel/gemm/jit_xe_hp_systolic.cpp:79` |" in text
    assert "gate 3 of 5 in `xe_hp_systolic_t::pd_t::init`: it passed 2 checks" in text
    assert "| lever `[interpretation]` | **operand layout**" in text
    assert "`set_default_formats` (lines 86-88 of the same file)" in text
    assert (
        "2 more follow in the same function -- `VERBOSE_SKIP_PRIMITIVE_IMPL`, `VERBOSE_UNSUPPORTED_ARCH`"
        in text
    )
    assert "scored 1 distinct catalog strategies" in text
    assert "`source/src/gpu/gpu_gemm_list.cpp` (" in text and "`[rule]` -- candidate order" in text
    assert "`[cited]` -- rejected `jit:xe_hp:gemm:any`" in text
    assert "- gap: `jit:gemm:any` was selected for `matmul`" in text
    # Not a git checkout and no revision: the working tree is used and the bundle says so.
    assert "working tree" in text

    sel = json.loads((out / "selection.json").read_text())
    assert sel["library"] == {"name": "oneDNN", "version": "9.8.7", "commit": run_commit(pks)}
    assert sel["selected"][0]["implementation"] == "jit:gemm:any"
    assert sel["nested"][0]["kind"] == "gemm"
    assert sel["rejected"][0]["line"] == "79"
    assert sel["gates"][0]["remaining"] == [
        "VERBOSE_SKIP_PRIMITIVE_IMPL",
        "VERBOSE_UNSUPPORTED_ARCH",
    ]
    assert "statement" not in sel["gates"][0]
    assert [r["outcome"] for r in sel["candidates"]["gemm"]["rows"]] == [
        "build-conditional",
        "rejected",
        "selected",
        "not tried",
    ]
    assert sel["strategies"] == [
        {"strategy": "G gemm HHS TNN 16 16 aB wg 4x2 sys", "score": 4337337.487821}
    ]
    assert {f["confidence"] for f in sel["files"]} == {"rule", "cited"}


def run_commit(pks):
    return _run(pks).commit


def test_bundle_without_a_checkout_names_what_it_would_have_held(tmp_path):
    pks = _load("pull_kernel_source")
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    out = pks.bundle_onednn(
        "aten::linear", "xpu:0", [], [], _run(pks), None, [], tmp_path / "pulled", roots=[empty]
    )
    text = (out / "PROVENANCE.md").read_text()
    assert not any((out / "source").rglob("*.cpp"))
    assert "No oneDNN source tree was found" in text
    assert "the implementation declared as `jit:gemm:any`" in text
    assert "the candidate list for `gemm`" in text
    assert "`src/gpu/intel/gemm/jit_xe_hp_systolic.cpp` (rejected `jit:xe_hp:gemm:any`)" in text
    assert "/clone-repos" in text and "--search" in text
    assert "0123456789abcdef0123456789abcdef01234567" in text  # the revision to fetch
    # Sections 2 and 3 still carry what the library reported, with the lever interpretation.
    assert "No source tree, so the gates cannot be read" in text
    assert "lever: operand layout `[interpretation]`" in text
    assert (out / "verbose.log").is_file() and (out / "repro.py").is_file()
    assert json.loads((out / "selection.json").read_text())["files"] == []


def test_bundle_tolerates_being_handed_its_own_harness(tmp_path):
    pks = _load("pull_kernel_source")
    out = pks.bundle_onednn(
        "aten::linear", "xpu:0", [], [], _run(pks), None, [], tmp_path / "pulled", roots=[]
    )
    (out / "harness.py").write_text("# from a previous pull\n")
    again = pks.bundle_onednn(
        "aten::linear",
        "xpu:0",
        [],
        [],
        _run(pks),
        out / "harness.py",
        [],
        tmp_path / "pulled",
        roots=[],
    )
    assert again == out and (out / "harness.py").read_text() == "# from a previous pull\n"


def test_onednn_provenance_next_step_parses_against_the_real_cli(tmp_path):
    pks = _load("pull_kernel_source")
    kt = _load("kernel_trials")
    out = pks.bundle_onednn(
        "aten::linear", "xpu:0", [], [], _run(pks), None, [], tmp_path / "pulled", roots=[]
    )
    text = (out / "PROVENANCE.md").read_text()
    commands = re.findall(r"`python scripts/kernel_trials\.py ([^`]*)`", text)
    assert commands, text
    for command in commands:
        args = kt.build_parser().parse_args(command.split())
        assert args.cmd == "init"
        assert args.name == "aten_linear"
        assert args.baseline.endswith("harness.py")


# ---------------------------------------------------------------------------------------
# Resolving a provider op at all: the ops exist in a process only once whatever registers
# them has been imported. Without that the dispatcher dump is empty, every provider op is
# classed as a Python-registered custom op with no source, and `--bundle` writes nothing --
# which is indistinguishable from the provider not being installed, and is what kept
# `provider_patch` rejected at `source_present` for every candidate.
# ---------------------------------------------------------------------------------------


def _harness(path, providers):
    path.write_text(
        'OP = "_C.rms_norm.default"\nCALLS = 1\n'
        f"PROVIDERS = {providers!r}\n"
        "class Model:\n    pass\n\n\ndef get_inputs():\n    return []\n"
    )


def test_providers_come_from_the_harnesses_and_deepest_first(tmp_path):
    pks = _load("pull_kernel_source")
    _harness(tmp_path / "a.py", ["pkg", "pkg.ext._C"])
    _harness(tmp_path / "b.py", ["pkg.ext._C", "other"])
    # Deepest first: importing a package does not import the extension submodule that
    # carries the registrations, so the submodule has to be tried in its own right.
    assert pks.providers_from_harnesses(tmp_path) == ["pkg.ext._C", "other", "pkg"]


def test_providers_are_ignored_when_a_harness_declares_none(tmp_path):
    pks = _load("pull_kernel_source")
    (tmp_path / "plain.py").write_text('OP = "aten.linear.default"\n')
    assert pks.providers_from_harnesses(tmp_path) == []


def test_import_providers_reports_what_loaded_and_skips_what_cannot(tmp_path):
    pks = _load("pull_kernel_source")
    # A module this box does not have is not an error: the ops it would have registered
    # are then honestly unresolved, and the resolver still reports on everything else.
    assert pks.import_providers(["json", "no_such_module_for_this_test"]) == ["json"]


def test_resolution_imports_the_harnesses_providers_before_resolving(tmp_path, monkeypatch):
    """The wiring itself: a run must import them before it reads the dispatcher, or every
    provider op resolves as unregistered and no bundle is written for it."""
    pks = _load("pull_kernel_source")
    _harness(tmp_path / "a.py", ["pkg.ext._C"])

    order = []
    monkeypatch.setattr(pks, "import_providers", lambda names: order.append(list(names)) or [])
    stub = {"op": "x", "provider": "p", "share": None}
    monkeypatch.setattr(pks, "report", lambda *a, **k: order.append("resolved") or stub)
    monkeypatch.setattr(sys, "argv", ["pull_kernel_source.py", "--from-harnesses", str(tmp_path)])
    pks.main()

    assert order == [["pkg.ext._C"], "resolved"]
