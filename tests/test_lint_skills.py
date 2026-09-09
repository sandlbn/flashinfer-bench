"""Tests for the skills linter.

Two failure modes matter, and each rule gets a case for both: a violation the rewrite plan
cites that the linter must catch, and a legitimate constant, fact table or command the plan
says to keep that the linter must not flag. The positive snippets are quoted from the skills
as they stood when the plan was written; the negatives are from the passages the plan marks
"good" or "keep".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_skills.py"
_SPEC = importlib.util.spec_from_file_location("lint_skills", _SCRIPT)
lint = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = lint
_SPEC.loader.exec_module(lint)


def rules_of(text: str, path: Path | None = None, refs=None) -> set[str]:
    return {v.rule for v in lint.lint_text(text, path or Path("x/SKILL.md"), "x/SKILL.md", refs)}


def lines_of(text: str, rule: str) -> list[int]:
    return [
        v.line for v in lint.lint_text(text, Path("x/SKILL.md"), "x/SKILL.md") if v.rule == rule
    ]


TABLE_SEP = "| --- | --- | --- |"


# ---------------------------------------------------------------------------------------
# ENUM
# ---------------------------------------------------------------------------------------


class TestEnum:
    def test_catches_the_owners_example_fix_table(self):
        text = (
            "| | | |\n|---|---|---|\n"
            "| 1-3 | Repro, observe, read dispatch | what ran, and what was rejected |\n"
            "| Fix 1 | Weight layout | when dispatch says `unsupported format tag` |\n"
            "| Fix 5 | Tile/strategy catalog entry | the call is clean and the GEMM is still slow |\n"
        )
        assert lines_of(text, "ENUM") == [4, 5]

    def test_catches_fix_heading_and_fixes_range(self):
        text = (
            "# Fix 5: The tile/strategy the selector chose is wrong for your shape\n\n"
            "Fixes 1-4 change how you *call* oneDNN.\n"
        )
        assert lines_of(text, "ENUM") == [1, 3]

    def test_catches_counted_levers_and_try_then_then(self):
        assert "ENUM" in rules_of("The three levers are layout, caching and post-ops.\n")
        assert "ENUM" in rules_of("Try oneDNN, then Xe-Fuse, then SYCL.\n")

    def test_keeps_failure_table_fix_column_and_prose_use_of_fix(self):
        text = (
            "| Symptom | Cause | Fix |\n" + TABLE_SEP + "\n"
            "| Shape mismatch | Definition disagrees with reference | Fix the JSON |\n"
            "\nThe fix is labelling and sourcing, not removal.\n"
        )
        assert "ENUM" not in rules_of(text)


# ---------------------------------------------------------------------------------------
# ORDER
# ---------------------------------------------------------------------------------------


class TestOrder:
    def test_catches_try_first_then_last_resort_table(self):
        text = (
            "| op_type family | Try first | Then | Last resort |\n"
            "| --- | --- | --- | --- |\n"
            "| `gemm` | **oneDNN** | oneDNN **post-ops** | SYCL |\n"
        )
        violations = [v for v in lint.lint_text(text, Path("x"), "x") if v.rule == "ORDER"]
        assert [v.line for v in violations] == [1]
        assert "remedy-order column" in violations[0].message

    def test_catches_levers_in_order_of_value(self):
        assert {"ORDER", "ENUM"} <= rules_of("Levers, in order of value:\n\n1. **Vectorize.**\n")

    def test_catches_ranked_heading_and_first_move(self):
        assert "ORDER" in rules_of("## Where a kernel can come from, in order\n")
        text = (
            "The bundle's `source/` is the starting point. The fastest first\n"
            "move is usually a constant in the existing kernel.\n"
        )
        assert "ORDER" in rules_of(text)

    def test_catches_try_x_first_and_start_with(self):
        assert "ORDER" in rules_of("Try oneDNN post-ops first (`/optimize-onednn`).\n")
        assert "ORDER" in rules_of("Start with the vectorized load.\n")

    def test_keeps_validity_order_and_measurement_instruction(self):
        text = (
            "The rows are applied top to bottom because each is a precondition for the\n"
            "arithmetic below it. Check for this before concluding a oneDNN-backed solution\n"
            "is structurally slower than the vendor path.\n"
        )
        assert "ORDER" not in rules_of(text)

    def test_keeps_last_resort_in_a_sourced_flag_table(self):
        text = (
            "Source: <https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html>\n\n"
            "| Value | Prints | Use when |\n" + TABLE_SEP + "\n"
            "| `all` | everything | last resort |\n"
        )
        assert "ORDER" not in rules_of(text)


# ---------------------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------------------


class TestVerdict:
    def test_catches_because_column(self):
        text = (
            "| Family | Route to | Because |\n" + TABLE_SEP + "\n"
            "| `norm` | `/optimize-intel-kernels` | memory-bound; vectorized loads |\n"
        )
        assert lines_of(text, "VERDICT") == [1]

    def test_keeps_selected_when_and_regime_tables(self):
        text = (
            "| Regime | Test | What it means for the next step |\n" + TABLE_SEP + "\n"
            "| Spill-limited | `spill > 0` | remove the spill, re-measure, then classify |\n\n"
            "| Family | Selected when | Route |\n" + TABLE_SEP + "\n"
            "| gemm | `t_dev` moves with M | `/optimize-onednn` |\n"
        )
        assert "VERDICT" not in rules_of(text)

    def test_source_label_does_not_excuse_a_verdict_column(self):
        text = "Source: a probe\n\n| Family | Because |\n| --- | --- |\n| gemm | it is fast |\n"
        assert "VERDICT" in rules_of(text)


# ---------------------------------------------------------------------------------------
# MEASURE
# ---------------------------------------------------------------------------------------


class TestMeasure:
    @pytest.mark.parametrize(
        "text",
        [
            "use a hit ratio ≥ 85% of elements within tolerance, not elementwise",
            "expect a 2x speedup on decode shapes",
            "the strided read reaches 400 GB/s",
            "the launch floor is 12.5 us on this part",
            "each call costs 3 ms",
            "peak is 200 TFLOPS at bf16",
            "Within ~10x of tolerance, cosine similarity > 0.999",
        ],
    )
    def test_catches_stored_measurements(self, text):
        assert "MEASURE" in rules_of(text + "\n")

    def test_catches_tolerance_table(self):
        text = (
            "| dtype | atol | rtol |\n" + TABLE_SEP + "\n"
            "| float32 | 1e-5 | 1e-5 |\n| bfloat16 | 1e-2 | 5e-2 |\n"
        )
        assert lines_of(text, "MEASURE") == [3, 4]

    @pytest.mark.parametrize(
        "text",
        [
            "bf16/fp16 GEMM (default) uses a `256×256×32` tile with an 8×4 sub-group layout",
            "an 8 x 4 sub-group layout = 32 sub-groups = 512 work-items",
            "A regression at 100% substitution is a real result; 0% substitution is the lookup",
            "Guard the wide path on `hidden % width == 0` and aligned base pointers",
            "bfloat16's relative spacing is `2**-7`; a narrower mantissa cannot carry it",
            "`ApplyConfig` defaults to `max_atol=1e-2, max_rtol=1e-5` regardless of dtype",
            "`rms_norm_eps` (1e-5 and 1e-6 both occur)",
            "One DPAS is `M×16×K` with systolic depth 8 and `1 <= M <= 8`",
            "run 3 trials with `--num-trials 3`",
        ],
    )
    def test_keeps_legitimate_constants(self, text):
        assert "MEASURE" not in rules_of(text + "\n")

    def test_exempt_in_fenced_code(self):
        text = (
            "```\n<family>  <ratio>x per kernel  x  <share>% of device time  (measured: 40%)\n```\n"
        )
        assert "MEASURE" not in rules_of(text)

    def test_exempt_in_illustration_block_until_next_heading(self):
        text = (
            "## Illustration (one instance): Arc B580 / oneDNN 3.13 / M=1 GEMM"
            " — re-establish with: `python repro.py`\n\n"
            "The `ba` layout ran in 12 us against 30 us for `ab`, a 2.5x gap.\n\n"
            "## Read\n\nThe gap was 2.5x.\n"
        )
        assert lines_of(text, "MEASURE") == [7]
        assert "ILLUS" not in rules_of(text)

    def test_paragraph_illustration_ends_at_end_marker(self):
        text = (
            "Illustration (one instance): Arc B580 / oneDNN 3.13 / M=1 GEMM"
            " — re-establish with: `python repro.py`\n"
            "It ran in 12 us.\n\nEnd illustration.\n\nIt always runs in 12 us.\n"
        )
        assert lines_of(text, "MEASURE") == [6]

    def test_exempt_in_sourced_fact_table_only(self):
        sourced = (
            "Source: `scripts/calibrate_part.py`\n\n"
            "| Field | Value |\n| --- | --- |\n| `timing_floor_us` | 8 us |\n"
        )
        unsourced = "| Field | Value |\n| --- | --- |\n| `timing_floor_us` | 8 us |\n"
        assert "MEASURE" not in rules_of(sourced)
        assert "MEASURE" in rules_of(unsourced)

    def test_pragma_inline_and_line_above_require_a_reason(self):
        inline = "bf16 carries 8 mantissa bits, so 0.4% is one ULP. <!-- lint-skills: allow MEASURE format arithmetic -->\n"
        above = "<!-- lint-skills: allow MEASURE format arithmetic -->\nbf16: 0.4% is one ULP.\n"
        no_reason = "<!-- lint-skills: allow MEASURE -->\nbf16: 0.4% is one ULP.\n"
        assert "MEASURE" not in rules_of(inline)
        assert "MEASURE" not in rules_of(above)
        assert "MEASURE" in rules_of(no_reason)


# ---------------------------------------------------------------------------------------
# NAME
# ---------------------------------------------------------------------------------------


class TestName:
    def test_catches_part_name_across_a_line_break(self):
        text = (
            "- **Implementation** — `jit:gemm:any` is the generic Xe GEMM generator and is the correct\n"
            "  implementation on Battlemage. `jit:xe_hp:gemm:any` is the Xe-HP/PVC systolic path.\n"
        )
        assert {"NAME", "EXPECT"} <= rules_of(text)

    def test_catches_model_families_in_frontmatter(self):
        text = (
            "---\nname: optimize-ssm-scan\n"
            "description: Optimize SSD scan kernels — Mamba2, GDN, and the hybrid models built on"
            " them (Zamba, Granite-hybrid, Falcon-H, Jamba).\n---\n\n# Optimize\n"
        )
        assert lines_of(text, "NAME") == [3]

    @pytest.mark.parametrize(
        "text",
        [
            "Worked example in the dataset: `mamba_ssu/ssd_bc_contraction_c256_h64_s128`.",
            "`gqa_paged_decode_h5_kv1_d128_ps64` is the decode definition",
            "run it on meta-llama/Llama-3.1-8B first",
            "probed against Zyphra/Zamba2-2.7B",
            "3.11.x and 3.13.x behave identically",
            "requires oneDNN 3.13 or newer",
            "IP 20 = Battlemage `bmg`, 30 = Xe3.0 integrated, 35 = Crescent Island `cri`",
        ],
    )
    def test_catches_names_of_record(self, text):
        assert "NAME" in rules_of(text + "\n")

    @pytest.mark.parametrize(
        "text",
        [
            "Naming: `mamba_ssu_decode_h{n}_d{d}_s{s}_ng{g}` and `gdn_{decode,mtp,prefill}_qk{q}_v{v}_d{d}_k_last`",
            "`float8_e4m3fn` outputs use a hit ratio",
            "set `FIB_SYCL_LARGE_GRF=1` and measure",
            "python scripts/optimize_model_kernels_xpu.py --model <hf_repo_id> --device xpu:0",
            "`caps.vector_width(2)` gives the element count per widest access",
            "ls tmp/flashinfer-trace/definitions/gdn tmp/flashinfer-trace/definitions/mamba_ssu",
            "the `gdn` and `mamba_ssu` op_types; `sm_scale`, `eps`, the causal flag",
            "the definition name is `<op>_h<width>`",
        ],
    )
    def test_keeps_placeholders_templates_and_identifiers(self, text):
        assert "NAME" not in rules_of(text + "\n")

    def test_exempt_in_illustration_and_sourced_table(self):
        illustration = (
            "Illustration (one instance): Battlemage / oneDNN 3.13.0 / grouped fp8 scales"
            " — re-establish with: `tools/onednn/repro_grouped_scales.cpp`\n"
            "`set_scales` with `groups` returns wrong values on Battlemage.\n"
        )
        table = (
            "Source: `sycl -ze-...` and `caps.canonical_id`\n\n"
            "| IP | Part |\n| --- | --- |\n| 20 | Battlemage |\n"
        )
        assert "NAME" not in rules_of(illustration)
        assert "NAME" not in rules_of(table)


# ---------------------------------------------------------------------------------------
# EXPECT
# ---------------------------------------------------------------------------------------


class TestExpect:
    @pytest.mark.parametrize(
        "text",
        [
            "On a hybrid model the scan, not the GEMM, is usually where the time goes.",
            "Do not try to beat oneDNN's matmul — the wins are in how it is called.",
            "Most large gaps in eager model code are contractions written as\n"
            "broadcast-multiply-then-sum, and need no kernel.",
            "Routing each op on its own reaches the same dead end every time on this hardware",
            "which is faster depends on M: it can reverse between M=1 and mid-range M.",
            "It will not find a fusion or remove a materialisation",
            "The fusion wins only above a token count that depends on the part",
            "a miss is nearly free",
            "the reachable headroom for everything else is small by construction",
            "Then: is a naive SYCL kernel competitive? If yes, oneDNN is not the problem",
            "a missing baseline usually means a signature mismatch",
            "install the `Runnable` as the layer's forward, or expect no throughput win.",
            "so fp8 is slower than the same GEMM in fp16",
        ],
    )
    def test_catches_expectations(self, text):
        assert "EXPECT" in rules_of(text + "\n")

    @pytest.mark.parametrize(
        "text",
        [
            # optimize-intel-kernels "Measure what the access pattern allows", kept verbatim
            "Peak bandwidth is the wrong yardstick. Time three reads with the accelerator's timer: the\n"
            "whole tensor contiguously, the slice in the shape the kernel is obliged to touch, and the\n"
            "kernel itself. If the kernel matches the strided read, the gap to peak lives in the data\n"
            "layout — a serving-stack decision, not a kernel one.",
            "Confirm by measurement that a different tile is faster for your shape.",
            "check first if a oneDNN solution is slower than torch",
            "a kernel that is slower or wrong never passes",
            "It is far cheaper than a review round.",
            "stepwise checks with expected output",
            "| Expected architecture skip in dispatch output | Nothing |",
            "`F.linear` and `torch.matmul` on XPU **are** oneDNN matmuls",
        ],
    )
    def test_keeps_measurements_and_system_facts(self, text):
        assert "EXPECT" not in rules_of(text + "\n")


# ---------------------------------------------------------------------------------------
# STEPREF
# ---------------------------------------------------------------------------------------


class TestStepRef:
    @pytest.mark.parametrize(
        "text",
        [
            "Then: is a naive SYCL kernel competitive? (`optimize-intel-kernels` Step 4a).",
            "Add them as baselines (`/onboard-model-intel` Phase 5).",
            "see SKILL.md Step 4a for the harness",
            "Use as Phase 1 of /onboard-model.",
            "Publishing them is `onboard-model-intel`\nPhase 7 and `/submit-onboarding-prs`.",
            "`onboard-model-intel/SKILL.md` Phase 5 lists the providers",
        ],
    )
    def test_catches_cross_document_step_numbers(self, text):
        assert "STEPREF" in rules_of(text + "\n")

    @pytest.mark.parametrize(
        "text",
        [
            "recheck the Step 1 flags before collecting again.",
            "Re-run Step 2 for that model and update both",
            "a regression re-runs Stage 1",
            '`optimize-intel-kernels`, section "Measure what the access pattern allows"',
            '`/onboard-model-intel`, "Source the solution"',
        ],
    )
    def test_keeps_intra_document_and_named_section_references(self, text):
        assert "STEPREF" not in rules_of(text + "\n")


# ---------------------------------------------------------------------------------------
# UVRUN
# ---------------------------------------------------------------------------------------


class TestUvRun:
    def test_catches_uv_pip_inside_a_fence(self):
        text = (
            "```bash\nuv pip install torch --index-url https://download.pytorch.org/whl/xpu\n```\n"
        )
        assert lines_of(text, "UVRUN") == [2]

    def test_catches_uv_pip_recommended_in_prose(self):
        text = (
            "Subshells keep the working directory unchanged. In a `uv`-managed venv use\n"
            "`uv pip install` — `python -m pip` is not present.\n"
        )
        assert "UVRUN" in rules_of(text)

    def test_catches_uv_run_no_sync_workaround(self):
        assert "UVRUN" in rules_of("tools directly, or use `uv run --no-sync`.\n")

    def test_keeps_the_prohibition(self):
        text = (
            "### Never `uv run` in this venv\n\n"
            "`pyproject.toml` pins plain `torch` with no index override, so `uv run` re-syncs\n"
            "the venv to CUDA torch.\n\n"
            "| Never | `uv run`, `uv pip`, or `pip` resolving dependencies |\n"
        )
        assert "UVRUN" not in rules_of(text)
        assert "UVRUN" not in rules_of("Use `source .venv/bin/activate`, not `uv run`.\n")


# ---------------------------------------------------------------------------------------
# BROKENREF
# ---------------------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path):
    """A miniature repo: two skills, one reference file, one script, one tmp clone."""
    skills = tmp_path / ".claude" / "skills"
    (skills / "optimize-onednn" / "references").mkdir(parents=True)
    (skills / "optimize-onednn" / "SKILL.md").write_text("# x\n")
    (skills / "optimize-onednn" / "references" / "quantized-matmul.md").write_text("# q\n")
    (skills / "profile-intel").mkdir()
    (skills / "profile-intel" / "SKILL.md").write_text("# p\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "kernel_trials.py").write_text("")
    (tmp_path / "docs").mkdir()
    (tmp_path / "tmp" / "sycl-tla" / "examples").mkdir(parents=True)
    (tmp_path / "tmp" / "sycl-tla" / "examples" / "xe_gemm.cpp").write_text("")
    return tmp_path, skills, lint.ReferenceChecker(tmp_path, skills)


class TestBrokenRef:
    def test_catches_missing_link_path_and_skill(self, tree):
        root, skills, refs = tree
        path = skills / "optimize-onednn" / "SKILL.md"
        text = (
            "See [the catalog](references/strategy-selection.md).\n"
            "Quantized matmul is `references/quantized-matmul.md`.\n"
            "Then `scripts/bound_candidates.py`, then `/optimize-ssm-scan`.\n"
        )
        found = [v for v in lint.lint_text(text, path, "x", refs) if v.rule == "BROKENREF"]
        assert [v.line for v in found] == [1, 3]
        assert "strategy-selection.md" in found[0].message
        assert (
            "bound_candidates.py" in found[1].message and "/optimize-ssm-scan" in found[1].message
        )

    @pytest.mark.parametrize(
        "text",
        [
            "Quantized matmul is `references/quantized-matmul.md`.",
            "run `scripts/kernel_trials.py benchmark` and read `/profile-intel`",
            "the worked example is `../optimize-onednn/references/quantized-matmul.md`",
            "Upstream instructions: vLLM's `docs/getting_started/installation/gpu.md`, XPU section.",
            "the tutorial `examples/xe_gemm.cpp` in sycl-tla",
            "`tmp/sycl-tla/media/docs/cpp/xe_rearchitecture.md` explains the layout",
            "the bundle's `<bundle>/PROVENANCE.md` and `tools/kernel-harness/pulled/<op>/harness.py`",
            "see <https://github.com/uxlfoundation/oneDNN/issues> and [docs](https://docs.flashinfer.ai)",
            "files under `/dev/dri/renderD*` and `/home/sand/Projects/vllm-xpu-venv`",
            "`uv run --no-sync` and `--index-url https://download.pytorch.org/whl/xpu`",
            "compare `SKILL.md` against the plan",
        ],
    )
    def test_keeps_existing_external_and_placeholder_references(self, tree, text):
        root, skills, refs = tree
        path = skills / "optimize-onednn" / "SKILL.md"
        assert "BROKENREF" not in rules_of(text + "\n", path, refs)

    def test_possessive_on_previous_line_marks_an_external_path(self, tree):
        root, skills, refs = tree
        path = skills / "profile-intel" / "SKILL.md"
        text = "Upstream instructions: vLLM's\n`docs/getting_started/installation/gpu.md`, XPU section.\n"
        assert "BROKENREF" not in rules_of(text, path, refs)


# ---------------------------------------------------------------------------------------
# ILLUS
# ---------------------------------------------------------------------------------------


class TestIllustrationHeader:
    GOOD = (
        "Illustration (one instance): Arc B580 / oneDNN 3.13.0 / `[N,K]` weights, M=1..64"
        " — re-establish with: `python scripts/kernel_trials.py benchmark onednn-layout cand.py`"
    )

    def test_accepts_the_required_form_as_heading_bold_or_list_item(self):
        for wrapped in (self.GOOD, "## " + self.GOOD, "**" + self.GOOD + "**", "- " + self.GOOD):
            assert "ILLUS" not in rules_of(wrapped + "\n"), wrapped

    @pytest.mark.parametrize(
        "header",
        [
            "Illustration: layout vs M",
            "Illustration (one instance): Battlemage — the `ba` layout won at M=1",
            "Illustration (one instance): Battlemage / oneDNN 3.13 — re-establish with: `python repro.py`",
            "Illustration (one instance): Battlemage / oneDNN 3.13 / M=1 GEMM — re-establish with:",
        ],
    )
    def test_rejects_headers_missing_a_field(self, header):
        assert "ILLUS" in rules_of(header + "\n")

    def test_a_malformed_header_still_opens_a_block(self):
        text = "## Illustration: layout vs M\n\nBattlemage ran at 12 us.\n"
        assert rules_of(text) == {"ILLUS"}


# ---------------------------------------------------------------------------------------
# Baseline, CLI and self-consistency
# ---------------------------------------------------------------------------------------


class TestBaseline:
    def test_new_versus_known_versus_resolved_is_keyed_on_text_not_line(self, tmp_path):
        old = "Fix 1 is layout.\nStart with the vectorized load.\n"
        new = "# Intro\n\nA new paragraph.\n\nFix 1 is layout.\nStart with the vectorized load.\n"
        old_v = lint.lint_text(old, Path("x"), "x")
        baseline = tmp_path / "b.json"
        lint.write_baseline(old_v, baseline)
        loaded = lint.load_baseline(baseline)
        fresh, known, resolved = lint.split_against_baseline(
            lint.lint_text(new, Path("x"), "x"), loaded
        )
        assert (fresh, len(known), resolved) == ([], 2, 0)

        changed = "Fix 1 is layout.\n\nFix 1 is layout.\n\nThe vectorized load comes first.\n"
        fresh, known, resolved = lint.split_against_baseline(
            lint.lint_text(changed, Path("x"), "x"), loaded
        )
        assert [v.line for v in fresh] == [3]  # a second identical paragraph is new
        assert len(known) == 1 and resolved == 1  # "Start with" was fixed


class TestCli:
    def test_text_output_format_and_exit_code(self, tmp_path, capsys):
        f = tmp_path / "SKILL.md"
        f.write_text("# T\n\n| Family | Because |\n| --- | --- |\n| gemm | fast |\n")
        assert lint.main([str(f)]) == 1
        out = capsys.readouterr().out.splitlines()
        assert out[0].startswith(f"{f}:3: VERDICT  ")

    def test_clean_file_exits_zero(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text("# T\n\nRun `python scripts/kernel_trials.py benchmark <name> <candidate>`.\n")
        assert lint.main([str(f)]) == 0

    def test_json_output(self, tmp_path, capsys):
        f = tmp_path / "SKILL.md"
        f.write_text("Fix 1 is layout.\n")
        assert lint.main([str(f), "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["violations"][0]["rule"] == "ENUM"
        assert payload["violations"][0]["line"] == 1
        assert list(payload["counts"].values()) == [{"ENUM": 1}]

    def test_baseline_roundtrip_through_the_cli(self, tmp_path, capsys):
        f = tmp_path / "SKILL.md"
        f.write_text("Fix 1 is layout.\n")
        baseline = tmp_path / "baseline.json"
        assert lint.main([str(f), "--baseline", str(baseline)]) == 1
        assert lint.main([str(f), "--check-baseline", str(baseline)]) == 0
        f.write_text("Fix 1 is layout.\nStart with the vectorized load.\n")
        capsys.readouterr()
        assert lint.main([str(f), "--check-baseline", str(baseline)]) == 1
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 1 and ":2: ORDER" in out[0]

    def test_rules_filter_and_list(self, tmp_path, capsys):
        f = tmp_path / "SKILL.md"
        f.write_text("Fix 1 is layout.\nStart with the vectorized load.\n")
        assert lint.main([str(f), "--rules", "order"]) == 1
        assert capsys.readouterr().out.count("\n") == 1
        assert lint.main(["--list-rules"]) == 0
        assert "ENUM" in capsys.readouterr().out
        with pytest.raises(SystemExit):
            lint.main([str(f), "--rules", "NOPE"])

    def test_every_rule_has_a_plan_row(self):
        for rule in lint.RULES.values():
            assert rule.plan_row


class TestSelfConsistency:
    def test_the_linters_own_docstring_passes(self):
        refs = lint.ReferenceChecker(lint.REPO_ROOT, lint.SKILLS_ROOT)
        found = lint.lint_text(lint.__doc__, _SCRIPT, "scripts/lint_skills.py", refs)
        assert found == []

    def test_runs_over_the_real_skills_tree(self):
        if not lint.SKILLS_ROOT.is_dir():
            pytest.skip("no skills tree")
        found = lint.lint_paths([lint.SKILLS_ROOT])
        assert all(v.path.startswith(".claude/skills/") for v in found)
        assert all(v.rule in lint.RULES for v in found)
