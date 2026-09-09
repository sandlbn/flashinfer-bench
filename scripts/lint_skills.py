#!/usr/bin/env python3
"""Lint the agent skills against the shared conventions of the skills rewrite plan.

A skill is procedure text an agent executes. The conventions in
`.claude/skills/SKILLS-REWRITE-PLAN.md`, section 3, exist so that a skill stores a
*measurement and a way to reason from it* rather than a conclusion somebody reached once on
one part. This linter is the mechanical half of that: each rule below is one row of that
table, so a violation is caught by a command rather than by review.

    python scripts/lint_skills.py                       # lint .claude/skills, exit 1 on any hit
    python scripts/lint_skills.py --json                # machine-readable, for a driving agent
    python scripts/lint_skills.py --baseline FILE       # record the current violation set
    python scripts/lint_skills.py --check-baseline FILE # fail only on violations not in FILE
    python scripts/lint_skills.py --list-rules          # the rule table with its plan row

Output is one line per violation, `path:line: RULE  message`.

Rules, and the plan row each enforces
-------------------------------------

| Rule      | Plan row (section 3)           | What it catches                                   |
| --------- | ------------------------------ | ------------------------------------------------- |
| ENUM      | No solution-space enumerations | a numbered or counted list of remedies            |
| ORDER     | Ordering must be derived       | a remedy order stated instead of measured         |
| VERDICT   | No verdict columns             | a table column that holds conclusions             |
| MEASURE   | No stored measurements         | a number with a unit, or a tolerance literal      |
| NAME      | Names outside illustrations    | a part, model, definition or library version      |
| EXPECT    | No performance expectations    | a sentence that says what the agent will find     |
| STEPREF   | Cross-references               | another document cited by step or phase number    |
| UVRUN     | Environment                    | the package-manager command that breaks this box  |
| BROKENREF | Cross-references               | a linked file, path or skill that does not exist  |
| ILLUS     | Illustration header            | an illustration header not in the required form   |

The exact phrases and name lists each rule matches are the pattern tables in this file
(`ENUM_PATTERNS`, `ORDER_PATTERNS`, ... `PART_NAMES`, `MODEL_FAMILIES`); the tests quote
one caught and one kept example per rule. Phrases are matched across the hard-wrapped lines
of a paragraph, so a phrase split over a line break is still found.

Exemptions, and where each comes from
-------------------------------------

Every exemption is a marker you can grep for, not a guess about intent.

1. **Fenced code.** Commands and their output are facts about a tool. Exempts every rule
   except UVRUN: a forbidden command inside a fence is still an instruction.
2. **Illustration blocks.** A worked finding is required to carry its part, library version
   and shape family, so names, numbers and verdicts inside one are the point, not a defect.
   The block starts at a line beginning `Illustration (one instance):` (optionally as a
   heading, bold, or list item). If the header is a heading, the block ends at the next
   heading of the same or a higher level; otherwise at the next heading of any level, a
   horizontal rule, or a line reading `End illustration.` Exempts ENUM, ORDER, VERDICT,
   MEASURE, NAME and EXPECT. ILLUS checks that the header itself has the required form:

       Illustration (one instance): <part> / <library version> / <shape family> — re-establish with: <command>

3. **Fact tables with a stated source.** A table of facts about a tool, an API or the
   silicon is closed because its subject is closed (plan section 9, item 1). Mark it by a
   line `Source: <file or command>` immediately above the table (blank lines allowed).
   Exempts ENUM, ORDER, MEASURE and NAME inside that table only; a sourced table still may
   not carry a verdict column or a performance expectation.
4. **Line pragma.** `<!-- lint-skills: allow RULE[,RULE] <reason> -->` on the line itself or
   on the line above exempts those rules for that line. The reason is mandatory, so every
   exemption says why (an API default, a format's mantissa width, illustrative arithmetic).
5. **Boundary percentages.** `0%` and `100%` mean none and all, not a measurement, and are
   not matched by MEASURE.
6. **Comparisons the agent performs.** A comparative inside a conditional or a measurement
   instruction (`if ... is slower`, `confirm by measurement that ... is faster`) is the
   agent's own test, not a stored verdict, and is not matched by EXPECT.

Baselines
---------

`--baseline FILE` records every current violation keyed by (path, rule, line text), not by
line number, so an edit elsewhere in a file does not turn an old violation into a new one.
`--check-baseline FILE` then fails only on violations absent from the file, which lets the
rewrite proceed skill by skill while the linter runs on every change. Re-record the baseline
when a skill is finished so its count goes to zero and stays there.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
DEFAULT_EXCLUDE = ("SKILLS-REWRITE-PLAN.md",)

# ---------------------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Rule:
    id: str
    plan_row: str
    summary: str
    exempt_in_code: bool = True
    exempt_in_illustration: bool = True
    exempt_in_source_table: bool = False


RULES: dict[str, Rule] = {
    r.id: r
    for r in (
        Rule(
            "ENUM",
            "No solution-space enumerations",
            "closed list of remedies",
            exempt_in_source_table=True,
        ),
        Rule(
            "ORDER",
            "Ordering must be derived",
            "remedy order stated instead of measured",
            exempt_in_source_table=True,
        ),
        Rule("VERDICT", "No verdict columns", "table column holding conclusions"),
        Rule(
            "MEASURE",
            "No stored measurements",
            "measured value stored as a literal",
            exempt_in_source_table=True,
        ),
        Rule(
            "NAME",
            "Names outside illustrations",
            "part, model, definition or version named in procedure text",
            exempt_in_source_table=True,
        ),
        Rule("EXPECT", "No performance expectations", "tells the agent what it will find"),
        Rule(
            "STEPREF",
            "Cross-references",
            "cross-reference by step number",
            exempt_in_illustration=False,
        ),
        Rule(
            "UVRUN",
            "Environment",
            "package-manager command that breaks the venv",
            exempt_in_code=False,
            exempt_in_illustration=False,
        ),
        Rule(
            "BROKENREF",
            "Cross-references",
            "referenced file or skill does not exist",
            exempt_in_illustration=False,
        ),
        Rule(
            "ILLUS",
            "Illustration header",
            "illustration header not in the required form",
            exempt_in_illustration=False,
        ),
    )
}

# ---------------------------------------------------------------------------------------
# Patterns. Each table is the complete, greppable definition of what a rule matches.
# ---------------------------------------------------------------------------------------

Pattern = tuple[re.Pattern[str], str]

_NUMBER_WORDS = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"

ENUM_PATTERNS: tuple[Pattern, ...] = (
    (re.compile(r"\bFix(?:es)?\s+\d"), "numbered fix"),
    (re.compile(r"\blever\s+\d\b", re.I), "numbered lever"),
    (re.compile(rf"\b(?:the\s+)?{_NUMBER_WORDS}\s+levers\b", re.I), "counted levers"),
    (
        re.compile(rf"\bthe\s+{_NUMBER_WORDS}\s+(?:fixes|remedies|knobs|moves|tricks)\b", re.I),
        "counted remedies",
    ),
    (re.compile(r"\blevers?\s*,?\s+in\s+order\b", re.I), "ranked levers"),
    (
        re.compile(r"\btry\s+[\w`*-]+,?\s+then\s+[\w`*-]+,?\s+(?:and\s+)?then\s+[\w`*-]+", re.I),
        "try A then B then C",
    ),
)

ORDER_PATTERNS: tuple[Pattern, ...] = (
    (re.compile(r"\bstart\s+with\b", re.I), "start with"),
    (re.compile(r"\bfirst\s+moves?\b", re.I), "first move"),
    (re.compile(r"\blast\s+resort\b", re.I), "last resort"),
    (
        re.compile(r"\bin\s+order\s+of\s+(?:value|preference|priority|importance)\b", re.I),
        "ranked by value",
    ),
    (re.compile(r"\bin\s+(?:descending|ascending|rough)\s+order\b", re.I), "ranked order"),
    (re.compile(r"\btry\b[^.;|]{0,40}?\bfirst\b", re.I), "try X first"),
)
ORDER_HEADING = re.compile(r"^\s*#{1,6}\s.*\bin\s+order:?\s*$", re.I)
ORDER_HEADER_CELLS = frozenset({"try first", "first try", "then", "last resort"})

VERDICT_HEADER_CELLS = frozenset({"because", "why", "reason", "rationale"})

# A number followed by a unit. `x`/`×` must not be followed by another number (`256×256` is a
# tile, not a speedup) and the unit must end the token (`8x4` layouts, `us` in `status`).
MEASURE_PATTERNS: tuple[Pattern, ...] = (
    (
        re.compile(
            r"(?<![\w.])\d+(?:\.\d+)?\s?"
            r"(?:us|µs|μs|ms|ns|%|GB/s|TB/s|GiB|MiB|GB|MB|TFLOPS?|GFLOPS?|TFLOP/s|tok/s|tokens/s|x|×)"
            r"(?![\w/])(?!\s*\d)"
        ),
        "measured value",
    ),
)
# A tolerance in a table cell is a stored gate; in prose the derivation (dtype spacing, API
# default) sits next to it and cannot be checked mechanically, so only table rows are matched.
TOLERANCE_LITERAL = (re.compile(r"(?<![\w.])\d+(?:\.\d+)?e-\d+\b"), "tolerance literal")
MEASURE_BOUNDARY = re.compile(r"^(?:0|100)\s?%$")

PART_NAMES = (
    "Battlemage",
    "Ponte Vecchio",
    "PVC",
    "BMG",
    "CRI",
    "Crescent Island",
    "Lunar Lake",
    "Meteor Lake",
    "Arrow Lake",
    "Panther Lake",
    "Alchemist",
    "Data Center GPU Max",
    r"Flex\s+1[47]0",
    r"Arc\s+[AB]\d{3}",
    r"[AB]5[78]0",
    r"A7[57]0",
    r"Xe[23](?:\.\d)?",
    "Xe-HPG",
    "Xe-HPC",
    "Xe-LPG",
    r"[HAB]100",
    r"[HB]200",
    "GH200",
    "L40S",
    r"MI3[05]0X?",
    r"MI250X?",
)
MODEL_FAMILIES = (
    "Llama",
    "Qwen",
    "Mistral",
    "Mixtral",
    "DeepSeek",
    "Gemma",
    r"Phi-\d",
    "Granite",
    "Zamba",
    "Jamba",
    "Falcon",
    "Mamba",
    "Mamba2",
    "Nemotron",
    "GLM",
    "Kimi",
    "MiniMax",
    "OLMo",
    "SmolLM",
    "InternLM",
    "Baichuan",
    r"Yi-\d",
    "Command-R",
    "GPT-OSS",
    r"GPT-\d",
    "Grok",
)
HF_ORGS = (
    "meta-llama",
    "Qwen",
    "mistralai",
    "deepseek-ai",
    "ibm-granite",
    "Zyphra",
    "tiiuae",
    "ai21labs",
    "microsoft",
    "google",
    "openai",
    "nvidia",
    "allenai",
    "HuggingFaceTB",
    "state-spaces",
    "moonshotai",
    "zai-org",
    "THUDM",
    "MiniMaxAI",
    "xai-org",
    "unsloth",
    "RedHatAI",
    "neuralmagic",
)
NAME_PATTERNS: tuple[Pattern, ...] = (
    (re.compile(r"\b(?:" + "|".join(PART_NAMES) + r")\b"), "part name"),
    (re.compile(r"\b(?:" + "|".join(MODEL_FAMILIES) + r")\b"), "model family"),
    (
        re.compile(
            r"(?<![\w/.-])(?:"
            + "|".join(HF_ORGS)
            + r")/[A-Za-z0-9][A-Za-z0-9_.-]*"
            + r"|(?<![\w/.-])[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]*\d+(?:\.\d+)?[Bb]"
            r"(?:-[A-Za-z0-9_.]+)*\b"
        ),
        "HuggingFace id",
    ),
    (
        re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_(?:h|d|qk|kv|ps|ng)\d+(?:_[a-z0-9]+)*\b"),
        "definition name",
    ),
    (
        re.compile(
            r"\b(?:oneDNN|oneAPI|torch|PyTorch|triton|vLLM|SGLang|DPC\+\+|Level Zero|driver)"
            r"\s+v?\d+\.\d+(?:\.(?:\d+|x))?\b"
            r"|\b\d+\.\d+\.x\b"
        ),
        "library version",
    ),
)

EXPECT_PATTERNS: tuple[Pattern, ...] = (
    (
        re.compile(
            r"\b(?:usually|typically|in most cases|most of the time|almost always"
            r"|more often than not|tends? to|in practice|generally)\b",
            re.I,
        ),
        "hedge",
    ),
    (
        re.compile(r"\bmost\s+(?:large\s+|small\s+)?(?:gaps|wins|kernels|ops|of the time)\b", re.I),
        "hedge",
    ),
    (
        re.compile(
            r"\b(?:hard to beat|(?:do not|don't|never) try to beat|never in beating"
            r"|the wins? (?:is|are|come|comes) in|where the time goes|nearly free"
            r"|too small to repay|small by construction|same dead end|is not the problem"
            r"|the fastest first move|wins? only (?:above|below|when|where|for))\b",
            re.I,
        ),
        "verdict idiom",
    ),
    (
        re.compile(
            r"\b(?:is|are)\s+(?:already\s+)?the\s+(?:correct|right|fastest|best)\s+"
            r"(?:implementation|path|choice|kernel)\b",
            re.I,
        ),
        "declared winner",
    ),
    (
        re.compile(
            r"\b(?:is|are|runs?|becomes?)\s+(?:much\s+|far\s+|slightly\s+|often\s+|always\s+"
            r"|already\s+)?(?:faster|slower)\b",
            re.I,
        ),
        "comparative claim",
    ),
    (re.compile(r"\bwill not (?:find|win|help|beat|matter)\b", re.I), "predicted outcome"),
    (re.compile(r"\bexpects?\s+(?:no|a|an|to|that)\b", re.I), "expectation"),
)
# A comparative inside a clause the agent evaluates is its own test, not a stored verdict.
CLAUSE_BREAK = re.compile(r"[.;:|()—]")
CONDITIONAL = re.compile(
    r"\b(?:if|when|whether|where|unless|until|that|confirm|check|measure|compare|test)\b", re.I
)

# The cross-document half of a step reference: a backticked or slash-prefixed skill name, or a
# markdown file name. A bare hyphenated word ("re-run Stage 1") is not a document.
_SKILL_TOKEN = (
    r"`/?[a-z][a-z0-9]*(?:-[a-z0-9]+)+`|(?<![\w-])/[a-z][a-z0-9]*(?:-[a-z0-9]+)+"
    r"|`?[A-Za-z0-9_./-]+\.md`?"
)
STEPREF_PATTERNS: tuple[Pattern, ...] = (
    (
        re.compile(rf"(?:{_SKILL_TOKEN})\)?,?\s+(?:section\s+)?(?:Step|Phase|Stage)\s+\d"),
        "document + step number",
    ),
    (
        re.compile(rf"\b(?:Step|Phase|Stage)\s+\d[a-z]?\s+(?:of|in)\s+(?:{_SKILL_TOKEN})"),
        "step number + document",
    ),
)

UVRUN_PATTERN = re.compile(r"\buv\s+(?:run|pip)\b")
UVRUN_NEGATED = re.compile(
    r"\b(?:never|do not|don't|not|no|avoid|instead of)\b[\s|,:]*(?:use\s+|run\s+|call\s+)?"
    r"`?uv\s+(?:run|pip)",
    re.I,
)

MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
BACKTICK_PATH = re.compile(
    r"`((?:\.\./|\./)?[A-Za-z0-9_./-]+\.(?:md|mdx|py|cpp|hpp|cc|h|json|jsonl|toml|yaml|yml|sh|txt))"
    r"(?::\d+(?:-\d+)?)?`"
)
SLASH_SKILL = re.compile(r"(?<![\w/.-])/([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b(?!/)")
REPO_DIRS = frozenset(
    {"flashinfer_bench", "scripts", "tests", "tools", "docs", "examples", "web", ".claude"}
)
EXTERNAL_PREFIXES = ("tmp/", "/tmp/", "~", "http", "/home/", "/usr/", "/opt/", "/dev/")

ILLUSTRATION_START = re.compile(r"^\s*(?:#{1,6}\s*|[-*]\s+)?(?:\*\*)?Illustration\s*(?:\(|:)")
ILLUSTRATION_HEADER = re.compile(
    r"Illustration \(one instance\):\s*[^/—]+?/[^/—]+?/[^—]+?(?:—|–|\s-{1,2}\s)\s*"
    r"re-establish with:\s*\S"
)
END_ILLUSTRATION = re.compile(r"^\s*End illustration\.?\s*$", re.I)
SOURCE_LABEL = re.compile(r"^\s*\**Sources?:\**\s*\S")
PRAGMA = re.compile(r"<!--\s*lint-skills:\s*allow\s+([A-Z][A-Z,\s]*?)\s+(\S.*?)\s*-->")
HEADING = re.compile(r"^(#{1,6})\s")
HRULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
TABLE_ROW = re.compile(r"^\s*\|")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# ---------------------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass
class Line:
    number: int
    text: str
    in_code: bool = False
    in_frontmatter: bool = False
    in_illustration: bool = False
    in_source_table: bool = False
    is_table_header: bool = False
    heading_text: str = ""
    allowed: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    message: str
    text: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.rule, " ".join(self.text.split()))

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}  {self.message}"


def parse_document(text: str) -> list[Line]:
    """Annotate each line with the regions that decide which rules apply to it."""
    lines = [Line(i + 1, t) for i, t in enumerate(text.split("\n"))]

    in_code = False
    fence = ""
    in_frontmatter = lines[0].text.strip() == "---" if lines else False
    illus_end_level: int | None = None  # heading level that closes a heading illustration
    illus_paragraph = False  # a non-heading illustration, closed by any heading or rule
    heading_text = ""
    pending_allow: frozenset[str] = frozenset()

    for idx, line in enumerate(lines):
        stripped = line.text.strip()

        if in_frontmatter:
            line.in_frontmatter = True
            if idx > 0 and stripped == "---":
                in_frontmatter = False
            continue

        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match and not in_code:
            in_code, fence = True, fence_match.group(1)[0] * 3
            line.in_code = True
            continue
        if in_code:
            line.in_code = True
            if stripped.startswith(fence):
                in_code = False
            continue

        heading = HEADING.match(line.text)
        if heading:
            level = len(heading.group(1))
            heading_text = line.text[level:].strip()
            if illus_paragraph or (illus_end_level is not None and level <= illus_end_level):
                illus_paragraph, illus_end_level = False, None
        elif illus_paragraph and (HRULE.match(line.text) or END_ILLUSTRATION.match(line.text)):
            illus_paragraph = False
        line.heading_text = heading_text

        if ILLUSTRATION_START.match(line.text):
            if heading:
                illus_end_level = len(heading.group(1))
            else:
                illus_paragraph = True
        line.in_illustration = illus_paragraph or illus_end_level is not None

        # A pragma on its own line covers the next non-blank line; an inline one covers its own.
        allow = frozenset(
            r.strip() for m in PRAGMA.finditer(line.text) for r in m.group(1).split(",")
        )
        line.allowed = allow | pending_allow
        if allow and stripped.startswith("<!--"):
            pending_allow = allow
        elif stripped:
            pending_allow = frozenset()

        if TABLE_ROW.match(line.text) and idx + 1 < len(lines):
            line.is_table_header = bool(TABLE_SEPARATOR.match(lines[idx + 1].text))

    _mark_source_tables(lines)
    return lines


def _mark_source_tables(lines: list[Line]) -> None:
    """A table directly under a `Source:` line is a fact table with a stated source."""
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.in_code and SOURCE_LABEL.match(line.text):
            j = i + 1
            while j < len(lines) and not lines[j].text.strip():
                j += 1
            if j < len(lines) and TABLE_ROW.match(lines[j].text):
                while j < len(lines) and TABLE_ROW.match(lines[j].text):
                    lines[j].in_source_table = True
                    j += 1
                i = j
                continue
        i += 1


def iter_units(lines: Sequence[Line]) -> Iterator[list[Line]]:
    """Group lines into the units phrase patterns run over.

    A paragraph or a list item is one unit even when hard-wrapped; a heading, a table row, a
    frontmatter line and a code line are units of their own.
    """
    unit: list[Line] = []
    for line in lines:
        stripped = line.text.strip()
        standalone = (
            line.in_code
            or line.in_frontmatter
            or HEADING.match(line.text)
            or TABLE_ROW.match(line.text)
        )
        if not stripped or standalone or LIST_ITEM.match(line.text):
            if unit:
                yield unit
                unit = []
            if stripped:
                if standalone:
                    yield [line]
                else:
                    unit = [line]
            continue
        unit.append(line)
    if unit:
        yield unit


def _header_cells(text: str) -> list[str]:
    cells = [c.strip().strip("*").strip().lower() for c in text.strip().strip("|").split("|")]
    return [c for c in cells if c]


# ---------------------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Hit:
    rule: str
    label: str
    match: str
    offset: int  # into the unit text


def _find(patterns: Iterable[Pattern], text: str, rule: str) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[str] = set()
    for pattern, label in patterns:
        for m in pattern.finditer(text):
            if m.group(0).lower() not in seen:
                seen.add(m.group(0).lower())
                hits.append(Hit(rule, label, m.group(0), m.start()))
    return hits


def phrase_hits(text: str, table_row: bool = False) -> list[Hit]:
    """Every phrase-level finding in one unit of prose."""
    hits = _find(ENUM_PATTERNS, text, "ENUM")
    hits += _find(ORDER_PATTERNS, text, "ORDER")
    measure = MEASURE_PATTERNS + ((TOLERANCE_LITERAL,) if table_row else ())
    hits += [h for h in _find(measure, text, "MEASURE") if not MEASURE_BOUNDARY.match(h.match)]
    hits += _find(NAME_PATTERNS, text, "NAME")
    hits += [h for h in _find(EXPECT_PATTERNS, text, "EXPECT") if not _is_agents_own_test(text, h)]
    hits += _find(STEPREF_PATTERNS, text, "STEPREF")
    return hits


def _is_agents_own_test(text: str, hit: Hit) -> bool:
    if hit.label != "comparative claim":
        return False
    breaks = [m.end() for m in CLAUSE_BREAK.finditer(text, 0, hit.offset)]
    clause = text[breaks[-1] if breaks else 0 : hit.offset]
    return bool(CONDITIONAL.search(clause))


HINTS = {
    "ENUM": "a list is allowed only when its subject is closed and the heading says what closes it",
    "ORDER": "an order appears only as the output of a classification performed in this run",
    "MEASURE": "name the command that measures it, or put it in an Illustration block or a Source: table",
    "NAME": "use a placeholder or the command that discovers it, or an Illustration block",
    "EXPECT": "state the measurement the agent performs, or move it to an Illustration block",
    "STEPREF": "cite the skill name and the section heading",
}
WHAT = {
    "ENUM": "solution-space enumeration",
    "ORDER": "remedy order stated, not derived from a measurement",
    "MEASURE": "stored measurement",
    "NAME": "named instance in procedure text",
    "EXPECT": "performance expectation",
    "STEPREF": "cross-reference by step number",
}


def _message(rule: str, hits: Sequence[Hit]) -> str:
    parts = ", ".join(f"{h.label} {h.match!r}" for h in hits[:4])
    more = f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""
    return f"{WHAT[rule]}: {parts}{more}; {HINTS[rule]}"


def line_findings(line: Line) -> Iterator[tuple[str, str]]:
    """Structural checks that apply to one line: table headers, headings, commands."""
    if ORDER_HEADING.match(line.text):
        yield (
            "ORDER",
            f"{WHAT['ORDER']}: ranked heading {line.text.strip('# ').strip()!r}; {HINTS['ORDER']}",
        )
    if line.is_table_header:
        cells = _header_cells(line.text)
        ordered = [c for c in cells if c in ORDER_HEADER_CELLS]
        if ordered:
            yield (
                "ORDER",
                f"{WHAT['ORDER']}: remedy-order column {_quote(ordered)}; {HINTS['ORDER']}",
            )
        verdict = [c for c in cells if c in VERDICT_HEADER_CELLS]
        if verdict:
            yield (
                "VERDICT",
                f"verdict column {_quote(verdict)}: carry the measurement that selects the row "
                "('Selected when'), not the conclusion",
            )
    if UVRUN_PATTERN.search(line.text) and not (
        UVRUN_NEGATED.search(line.text) or re.search(r"\bnever\b", line.heading_text, re.I)
    ):
        yield "UVRUN", "`uv run` / `uv pip` replaces torch-xpu; use `source .venv/bin/activate`"
    if ILLUSTRATION_START.match(line.text) and not ILLUSTRATION_HEADER.search(line.text):
        yield (
            "ILLUS",
            "illustration header must read 'Illustration (one instance): <part> / "
            "<library version> / <shape family> — re-establish with: <command>'",
        )


def _quote(items: Sequence[str]) -> str:
    return ", ".join(repr(i) for i in items[:4]) + (
        f" (+{len(items) - 4} more)" if len(items) > 4 else ""
    )


class ReferenceChecker:
    """Resolve links, backticked paths and `/skill` mentions against the tree."""

    def __init__(self, repo_root: Path, skills_root: Path):
        self.repo_root = repo_root
        self.skills_root = skills_root
        self.skills = (
            {p.name for p in skills_root.iterdir() if (p / "SKILL.md").is_file()}
            if skills_root.is_dir()
            else set()
        )
        self.clones = [p for p in (repo_root / "tmp").glob("*") if p.is_dir()]
        self._basenames: set[str] | None = None

    def _known_basenames(self) -> set[str]:
        """Names of every markdown file in the repo, for bare-name references like `SKILL.md`."""
        if self._basenames is None:
            skip = {"tmp", ".venv", "node_modules", ".git"}
            self._basenames = {
                p.name
                for p in self.repo_root.rglob("*.md")
                if not (set(p.relative_to(self.repo_root).parts[:-1]) & skip)
            }
        return self._basenames

    def check(self, path: Path, line: Line, previous_text: str) -> Iterator[tuple[str, str]]:
        missing: list[str] = []
        for m in MD_LINK.finditer(line.text):
            target = m.group(1).split("#", 1)[0]
            if target and self._is_internal(target) and not self._exists(path, target):
                missing.append(target)
        for m in BACKTICK_PATH.finditer(line.text):
            target = m.group(1)
            before = line.text[: m.start()].rstrip()
            possessive = before.endswith("'s") or (
                not before and previous_text.rstrip().endswith("'s")
            )
            if possessive or not self._is_internal(target):
                continue
            if not self._exists(path, target):
                missing.append(target)
        for m in SLASH_SKILL.finditer(line.text):
            if m.group(1) not in self.skills:
                missing.append("/" + m.group(1))
        if missing:
            yield "BROKENREF", f"does not exist: {_quote(missing)}"

    def _is_internal(self, target: str) -> bool:
        if any(ch in target for ch in "<>{}*$") or target.startswith(EXTERNAL_PREFIXES):
            return False
        first = target.split("/", 1)[0]
        if first in ("..", ".", "references") or first in REPO_DIRS or first in self.skills:
            return True
        return "/" not in target and target.endswith((".md", ".mdx"))

    def _exists(self, path: Path, target: str) -> bool:
        if "/" not in target and target in self._known_basenames():
            return True
        candidates = [
            path.parent / target,
            self.skills_root / target,
            self.repo_root / target,
            self.repo_root / ".claude" / "skills" / target,
        ]
        candidates.extend(clone / target for clone in self.clones)
        return any(c.exists() for c in candidates)


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------


def _applies(rule: Rule, line: Line) -> bool:
    if rule.id in line.allowed:
        return False
    if line.in_code and rule.exempt_in_code:
        return False
    if line.in_illustration and rule.exempt_in_illustration:
        return False
    if line.in_source_table and rule.exempt_in_source_table:
        return False
    return True


def lint_text(
    text: str,
    path: Path,
    display: str,
    refs: ReferenceChecker | None = None,
    rules: Iterable[str] | None = None,
) -> list[Violation]:
    selected = set(rules) if rules else set(RULES)
    lines = parse_document(text)
    findings: dict[tuple[int, str], list[Hit]] = {}
    out: list[Violation] = []

    for unit in iter_units(lines):
        if unit[0].in_code:
            continue
        pieces = [ln.text.strip() for ln in unit]
        starts: list[int] = []
        pos = 0
        for piece in pieces:
            starts.append(pos)
            pos += len(piece) + 1
        table_row = bool(TABLE_ROW.match(unit[0].text))
        for hit in phrase_hits(" ".join(pieces), table_row=table_row):
            index = max(i for i, s in enumerate(starts) if s <= hit.offset)
            findings.setdefault((unit[index].number - 1, hit.rule), []).append(hit)

    # One violation per (line, rule): phrase findings and structural findings merge.
    messages: dict[tuple[int, str], list[str]] = {}
    for (line_idx, rule_id), hits in findings.items():
        messages.setdefault((line_idx, rule_id), []).append(_message(rule_id, hits))
    for idx, line in enumerate(lines):
        structural = list(line_findings(line))
        if refs is not None:
            structural.extend(refs.check(path, line, lines[idx - 1].text if idx else ""))
        for rule_id, message in structural:
            messages.setdefault((idx, rule_id), []).append(message)

    for (line_idx, rule_id), texts in messages.items():
        line = lines[line_idx]
        if rule_id in selected and _applies(RULES[rule_id], line):
            out.append(Violation(display, line.number, rule_id, " | ".join(texts), line.text))

    out.sort(key=lambda v: (v.line, v.rule))
    return out


def iter_files(paths: Sequence[Path], exclude: Sequence[str]) -> Iterator[Path]:
    for p in paths:
        files: Iterable[Path] = sorted(p.rglob("*.md")) if p.is_dir() else [p]
        for f in files:
            if any(f.match(pattern) or f.name == pattern for pattern in exclude):
                continue
            yield f


def lint_paths(
    paths: Sequence[Path],
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    rules: Iterable[str] | None = None,
    repo_root: Path = REPO_ROOT,
    skills_root: Path = SKILLS_ROOT,
) -> list[Violation]:
    refs = ReferenceChecker(repo_root, skills_root)
    out: list[Violation] = []
    for f in iter_files(paths, exclude):
        try:
            display = str(f.resolve().relative_to(repo_root))
        except ValueError:
            display = str(f)
        out.extend(lint_text(f.read_text(encoding="utf-8"), f.resolve(), display, refs, rules))
    out.sort(key=lambda v: (v.path, v.line, v.rule))
    return out


def write_baseline(violations: Sequence[Violation], file: Path) -> None:
    entries = [{"path": p, "rule": r, "text": t} for p, r, t in sorted(v.key for v in violations)]
    file.write_text(json.dumps({"version": 1, "entries": entries}, indent=1) + "\n")


def load_baseline(file: Path) -> Counter[tuple[str, str, str]]:
    data = json.loads(file.read_text())
    return Counter((e["path"], e["rule"], e["text"]) for e in data["entries"])


def split_against_baseline(
    violations: Sequence[Violation], baseline: Counter[tuple[str, str, str]]
) -> tuple[list[Violation], list[Violation], int]:
    """Return (new, known, resolved_count). Multiplicity counts: a second identical line is new."""
    remaining = Counter(baseline)
    new: list[Violation] = []
    known: list[Violation] = []
    for v in violations:
        if remaining[v.key] > 0:
            remaining[v.key] -= 1
            known.append(v)
        else:
            new.append(v)
    return new, known, sum(remaining.values())


def per_file_counts(violations: Sequence[Violation]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for v in violations:
        counts.setdefault(v.path, Counter())[v.rule] += 1
    return counts


def list_rules() -> str:
    rows = ["RULE       PLAN ROW (section 3)             SUMMARY"]
    for r in RULES.values():
        rows.append(f"{r.id:<10} {r.plan_row:<32} {r.summary}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "paths", nargs="*", type=Path, help="files or directories (default: .claude/skills)"
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--baseline", type=Path, metavar="FILE", help="write the violation set to FILE")
    ap.add_argument(
        "--check-baseline", type=Path, metavar="FILE", help="fail only on violations not in FILE"
    )
    ap.add_argument("--rules", help="comma-separated subset of rule ids to run")
    ap.add_argument(
        "--exclude",
        action="append",
        default=list(DEFAULT_EXCLUDE),
        help="file name or glob to skip (repeatable)",
    )
    ap.add_argument("--list-rules", action="store_true", help="print the rule table and exit")
    ap.add_argument("--summary", action="store_true", help="also print per-file counts")
    args = ap.parse_args(argv)

    if args.list_rules:
        print(list_rules())
        return 0

    rules = None
    if args.rules:
        rules = [r.strip().upper() for r in args.rules.split(",") if r.strip()]
        unknown = [r for r in rules if r not in RULES]
        if unknown:
            ap.error(f"unknown rule(s): {', '.join(unknown)}; see --list-rules")

    paths = args.paths or [SKILLS_ROOT]
    violations = lint_paths(paths, exclude=args.exclude, rules=rules)

    if args.baseline:
        write_baseline(violations, args.baseline)

    new, known, resolved = violations, [], 0
    if args.check_baseline:
        new, known, resolved = split_against_baseline(
            violations, load_baseline(args.check_baseline)
        )

    if args.json:
        payload = {
            "violations": [dataclasses.asdict(v) for v in violations],
            "new": [dataclasses.asdict(v) for v in new],
            "counts": {p: dict(c) for p, c in per_file_counts(violations).items()},
            "baseline": None
            if not args.check_baseline
            else {"file": str(args.check_baseline), "known": len(known), "resolved": resolved},
        }
        print(json.dumps(payload, indent=1))
    else:
        for v in new:
            print(v.format())
        if args.summary:
            for path, counts in per_file_counts(violations).items():
                detail = ", ".join(f"{r} {n}" for r, n in sorted(counts.items()))
                print(f"{path}: {sum(counts.values())}  ({detail})", file=sys.stderr)
        if args.check_baseline:
            print(
                f"{len(violations)} violation(s): {len(known)} in baseline, {len(new)} new, "
                f"{resolved} baseline entr{'y' if resolved == 1 else 'ies'} resolved",
                file=sys.stderr,
            )
        elif violations:
            print(f"{len(violations)} violation(s)", file=sys.stderr)
        if args.baseline:
            print(f"baseline written: {args.baseline} ({len(violations)} entries)", file=sys.stderr)

    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
