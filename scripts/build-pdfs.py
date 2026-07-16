#!/usr/bin/env python3
"""Build professional, shareable PDFs from the Markdown documentation.

    python3 scripts/build-pdfs.py          # -> docs/pdf/*.pdf

Requires pandoc + xelatex (macOS: `brew install pandoc`, MacTeX/BasicTeX).

Produces:
  QF-AgentOS-Documentation.pdf   the complete manual (cover + TOC + every doc)
  QF-AgentOS-Findings.pdf        the standalone flagship report
  <Doc>.pdf                      one per source document

Markdown is preprocessed so the PDFs stand alone: badge images are dropped,
relative .md links are rewritten to absolute GitHub URLs (clickable, not broken),
and Unicode maths/arrows are mapped to LaTeX maths so nothing renders blank.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "pdf"
BUILD = OUT / ".build"
REPO = "https://github.com/yazhsab/qf-agentos"
BLOB = f"{REPO}/blob/main"

AUTHOR = "QF-AgentOS contributors"
SUBTITLE = "An honest benchmark and evidence harness for quantum finance"

# (source, chapter title) — order defines the manual.
CHAPTERS: list[tuple[Path, str]] = [
    (ROOT / "README.md", "Overview"),
    (DOCS / "FINDINGS.md", "Findings — does quantum help?"),
    (DOCS / "PROBLEM-FAMILIES.md", "Problem families"),
    (DOCS / "CLI.md", "CLI reference"),
    (DOCS / "API.md", "REST API reference"),
    (DOCS / "CONFIGURATION.md", "Configuration"),
    (DOCS / "ARCHITECTURE.md", "Architecture"),
    (DOCS / "real-qpu.md", "Running on real quantum hardware"),
    (DOCS / "research-note-quantum-extras.md", "Research note — QAE & tensor networks"),
]

# Text fonts lack these glyphs; map them to LaTeX maths so they never render blank.
LATEX_HEADER = r"""
\usepackage{newunicodechar}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{⇒}{\ensuremath{\Rightarrow}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{ε}{\ensuremath{\epsilon}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{χ}{\ensuremath{\chi}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{▼}{\textbullet}
\newunicodechar{⚠}{\textbf{!}}
\newunicodechar{✅}{\textbf{[done]}}
\newunicodechar{⛔}{\textbf{[todo]}}

% Running headers + page numbers
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\nouppercase{\leftmark}}
\fancyhead[R]{\small\textit{QF-AgentOS}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\small\thepage}\renewcommand{\headrulewidth}{0pt}}

% Tables: smaller type + generous leading so 6-column tables actually fit.
\usepackage{array}
\usepackage{ragged2e}
\usepackage{etoolbox}
% longtable only — \maketitle typesets the author block in a `tabular`, so hooking
% that environment would silently shrink the author line on every cover page.
\AtBeginEnvironment{longtable}{\footnotesize}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}

% Long snake_case identifiers (collateral_allocation) are single unbreakable words
% and would overflow their column into the neighbour. Allow a break after each
% underscore so they wrap cleanly instead of colliding.
\let\qforigunderscore\_
\renewcommand{\_}{\qforigunderscore\allowbreak}

% Let TeX stretch rather than overflow the margin.
\setlength{\emergencystretch}{3em}
\sloppy
"""

PANDOC_COMMON = [
    "--pdf-engine=xelatex",
    "--highlight-style=tango",
    "-V",
    "mainfont=Charter",
    "-V",
    "monofont=Menlo",
    "-V",
    "monofontoptions=Scale=0.82",
    "-V",
    "fontsize=10pt",
    "-V",
    "geometry:a4paper,margin=2.4cm",
    "-V",
    "colorlinks=true",
    "-V",
    "linkcolor=[HTML]{1F4E79}",
    "-V",
    "urlcolor=[HTML]{1F4E79}",
    "-V",
    "toccolor=black",
]


def version() -> str:
    """The installed package version (the source has an unknown-version fallback)."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from importlib.metadata import version as _v

        return _v("qf-agentos")
    except Exception:
        pyproject = (ROOT / "pyproject.toml").read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.M)
        return m.group(1) if m else "0.0.0"


def preprocess(md: str, source: Path, *, title: str | None = None) -> str:
    """Make a Markdown doc stand alone inside a PDF."""
    # Drop badge images (external fetches, and they look wrong in print).
    md = re.sub(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)\s*$", "", md, flags=re.M)
    md = re.sub(r"^!\[[^\]]*\]\([^)]*\)\s*$", "", md, flags=re.M)
    md = md.replace("️", "")  # emoji variation selector

    def fix_link(m: re.Match[str]) -> str:
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        resolved = (source.parent / target).resolve()
        try:
            rel = resolved.relative_to(ROOT)
        except ValueError:
            return m.group(0)
        return f"[{label}]({BLOB}/{rel})"

    md = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix_link, md)

    if title is not None:  # manual mode
        md = re.sub(r"^#\s+.*$", f"# {title}", md, count=1, flags=re.M)
        # The manual numbers sections itself (--number-sections). Some docs hand-number
        # their headings ("## 1. The leaderboard"), which would render "2.1 1. The
        # leaderboard". Drop the hand-written prefix and let LaTeX be the only source.
        md = re.sub(r"^(#{2,6})\s+\d+\.\s+", r"\1 ", md, flags=re.M)
    return md.strip() + "\n"


def run_pandoc(md_text: str, out_pdf: Path, meta: list[str], extra: list[str]) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD / (out_pdf.stem + ".md")
    hdr = BUILD / "header.tex"
    src.write_text(md_text)
    hdr.write_text(LATEX_HEADER)
    cmd = [
        "pandoc",
        str(src),
        "-o",
        str(out_pdf),
        "-H",
        str(hdr),
        *PANDOC_COMMON,
        *meta,
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    missing = [line for line in proc.stderr.splitlines() if "Missing character" in line]
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"pandoc failed for {out_pdf.name}")
    if missing:
        # A missing glyph renders as a blank — never ship that silently.
        for line in dict.fromkeys(missing):
            print(f"  !! {line.strip()}", file=sys.stderr)
        raise SystemExit(f"{out_pdf.name}: {len(missing)} missing glyph(s) — fix LATEX_HEADER")
    print(f"  ok  {out_pdf.relative_to(ROOT)}  ({out_pdf.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found — install pandoc and a TeX distribution.")
    OUT.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    ver = version()
    print(f"QF-AgentOS docs -> PDF (v{ver}, {today})")

    # 1. The complete manual: cover + TOC + every doc as a chapter.
    parts = [preprocess(p.read_text(), p, title=t) for p, t in CHAPTERS]
    manual_meta = [
        "-V",
        "title=QF-AgentOS",
        "-V",
        f"subtitle={SUBTITLE}",
        "-V",
        f"author={AUTHOR}",
        "-V",
        f"date=Version {ver} — {today}",
        "-V",
        "documentclass=report",
    ]
    run_pandoc(
        "\n\n\\newpage\n\n".join(parts),
        OUT / "QF-AgentOS-Documentation.pdf",
        manual_meta,
        ["--toc", "--toc-depth=2", "--top-level-division=chapter", "--number-sections"],
    )

    # 2. The flagship standalone report + one PDF per document.
    standalone = [(DOCS / "FINDINGS.md", "QF-AgentOS-Findings", "Findings — does quantum help?")]
    standalone += [
        (p, p.stem if p.name != "README.md" else "QF-AgentOS-Overview", None)
        for p, _ in CHAPTERS
        if p.name != "FINDINGS.md"
    ]
    for path, stem, title in standalone:
        text = preprocess(path.read_text(), path)
        head = re.search(r"^#\s+(.*)$", text, flags=re.M)
        doc_title = title or (head.group(1) if head else stem)
        text = re.sub(r"^#\s+.*$", "", text, count=1, flags=re.M).strip() + "\n"
        meta = [
            "-V",
            f"title={doc_title}",
            "-V",
            f"subtitle=QF-AgentOS — {SUBTITLE}",
            "-V",
            f"author={AUTHOR}",
            "-V",
            f"date=Version {ver} — {today}",
        ]
        run_pandoc(text, OUT / f"{stem}.pdf", meta, ["--toc", "--toc-depth=2"])

    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"\nWrote {len(list(OUT.glob('*.pdf')))} PDFs to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
