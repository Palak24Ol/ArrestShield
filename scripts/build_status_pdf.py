"""Render docs/ML_STATUS.md to the submitted status PDF.

The PDF is generated from the markdown so the two cannot drift. The previous
hand-made PDF claimed "ML COMPONENT - Complete" on its front page while its own
later page listed the reasons production was blocked; generating from a single
source removes that failure mode.

Build-time only. reportlab is deliberately not added to requirements/base.txt.

    python -m pip install reportlab
    python scripts/build_status_pdf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#d4d4d4")
BAND = colors.HexColor("#f2f2f2")
ALERT = colors.HexColor("#8a1c1c")


# The built-in Type 1 fonts lack these glyphs and render them as replacement
# characters, so fold them to ASCII before any text reaches reportlab.
SMART_PUNCTUATION = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", " ": " ",
    "→": "->", "±": "+/-", "×": "x", "≤": "<=",
    "≥": ">=", "σ": "sigma", "•": "-",
}


def escape(text: str) -> str:
    for source, replacement in SMART_PUNCTUATION.items():
        text = text.replace(source, replacement)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    """Convert the inline markdown used in ML_STATUS.md to reportlab markup."""
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=25, textColor=INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=INK, spaceBefore=16, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=INK, spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=INK,
            leftIndent=11, bulletIndent=2, spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9.5, leading=13.5, textColor=ALERT,
            leftIndent=8, spaceAfter=6,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Normal"], fontName="Courier",
            fontSize=8, leading=10.5, textColor=INK, leftIndent=8, spaceAfter=1,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.5, textColor=INK,
        ),
        "cellhead": ParagraphStyle(
            "cellhead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10.5, textColor=INK,
        ),
    }


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def make_table(rows: list[list[str]], styles, available: float) -> Table:
    header, *body = rows
    columns = len(header)
    data = [[Paragraph(inline(cell), styles["cellhead"]) for cell in header]]
    for row in body:
        row = (row + [""] * columns)[:columns]
        data.append([Paragraph(inline(cell), styles["cell"]) for cell in row])

    first = min(available * 0.42, available / columns * 1.9) if columns > 2 else available / columns
    rest = (available - first) / (columns - 1) if columns > 1 else available
    widths = [first] + [rest] * (columns - 1)

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BAND),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                ("GRID", (0, 0), (-1, -1), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def render(markdown: str, styles, available: float) -> list:
    story: list = []
    lines = markdown.splitlines()
    index = 0
    code_buffer: list[str] = []
    in_code = False

    while index < len(lines):
        line = lines[index]

        if line.strip().startswith("```"):
            if in_code:
                story.append(Spacer(1, 3))
                for entry in code_buffer:
                    story.append(Paragraph(escape(entry) or "&nbsp;", styles["code"]))
                story.append(Spacer(1, 7))
                code_buffer, in_code = [], False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_buffer.append(line)
            index += 1
            continue

        stripped = line.strip()

        if stripped.startswith("|") and index + 1 < len(lines) and set(
            lines[index + 1].strip().replace("|", "").replace(" ", "")
        ) <= {"-", ":"} and lines[index + 1].strip().startswith("|"):
            rows = [split_row(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_row(lines[index]))
                index += 1
            story.append(Spacer(1, 3))
            story.append(make_table(rows, styles, available))
            story.append(Spacer(1, 9))
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), styles["title"]))
        elif stripped.startswith("## "):
            story.append(
                KeepTogether([
                    Paragraph(inline(stripped[3:]), styles["h2"]),
                    HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=5),
                ])
            )
        elif stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), styles["h3"]))
        elif stripped.startswith("> "):
            story.append(Paragraph(inline(stripped[2:]), styles["quote"]))
        elif stripped == ">":
            story.append(Spacer(1, 2))
        elif re.match(r"^[-*] ", stripped):
            story.append(Paragraph(inline(stripped[2:]), styles["bullet"], bulletText="•"))
        elif re.match(r"^\d+\. ", stripped):
            number, _, rest = stripped.partition(". ")
            story.append(Paragraph(inline(rest), styles["bullet"], bulletText=f"{number}."))
        elif stripped.startswith("---"):
            story.append(Spacer(1, 4))
        elif stripped:
            story.append(Paragraph(inline(stripped), styles["body"]))
        else:
            story.append(Spacer(1, 3))
        index += 1
    return story


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "docs/ML_STATUS.md")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs/ArrestShield_ML_Status_and_Testing_Guide.pdf",
    )
    parser.add_argument("--title", default="ArrestShield ML Status")
    parser.add_argument(
        "--footer",
        default="ArrestShield - research prototype - not production ready",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    markdown = source.read_text(encoding="utf-8")
    styles = build_styles()

    margin = 18 * mm
    available = A4[0] - 2 * margin
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=args.title,
        author="ArrestShield",
        subject="Research prototype status and limitations",
    )

    def furniture(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 10 * mm, args.footer)
        canvas.drawRightString(A4[0] - margin, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story = render(markdown, styles, available)
    story.insert(
        1,
        Paragraph(
            f"Generated from {source.relative_to(PROJECT_ROOT).as_posix()}. "
            "Do not edit this PDF directly; edit the markdown and rebuild.",
            styles["subtitle"],
        ),
    )
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    size = output.stat().st_size
    print(f"Wrote {output} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
