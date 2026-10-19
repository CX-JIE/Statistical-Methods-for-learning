"""Execute the report notebook once and write both readable forms of the report.

`pixi run --locked checkpoint2` calls this script. It runs `checkpoint2.ipynb` in a fresh
kernel and writes:

  output/checkpoint2.html  the rendered notebook (the format the checklist asks for)
  output/report.md         the same report as plain Markdown, no code and no base64,
                           linking to the four PNG files the notebook writes

The HTML is produced by nbconvert's HTMLExporter, i.e. the same exporter as
`jupyter nbconvert --to html --execute`, so the graded artefact is unchanged.
"""

from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
NOTEBOOK = ROOT / "checkpoint2.ipynb"

# The notebook draws exactly these four figures, in this order, and saves each one as a PNG.
# The Markdown report links to those files instead of embedding a second copy as base64.
FIGURES = [
    "cp2_lasso_coefficients.png",
    "cp2_gam_partial_effects.png",
    "cp2_roc_oof.png",
    "cp2_fold_auc.png",
]

PREAMBLE = (
    "> This is the plain-Markdown copy of the Checkpoint 2 report, written by the same\n"
    "> `pixi run --locked checkpoint2` command that renders `checkpoint2.html`. The HTML file is the\n"
    "> version the checklist asks for; this file is the same text without code cells, so it reads\n"
    "> start to finish in an editor or on the repository page. Tables are Markdown tables and the\n"
    "> four figures are the PNG files in this folder.\n"
)


class _TableParser(HTMLParser):
    """Turn the HTML that pandas Styler emits into rows of plain text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.caption = ""
        self.header: list[str] = []
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_caption = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "caption":
            self._in_caption = True
        elif tag == "tr":
            self._row = []
        elif tag in {"th", "td"}:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "caption":
            self._in_caption = False
        elif tag in {"th", "td"} and self._cell is not None and self._row is not None:
            text = " ".join("".join(self._cell).split())
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                if not self.header:
                    self.header = self._row
                else:
                    self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self.caption += data
        elif self._cell is not None:
            self._cell.append(data)


def _cell_text(text: str) -> str:
    return text.replace("|", "\\|") or " "


def html_table_to_markdown(markup: str) -> str:
    """Convert one pandas Styler table to a caption plus a Markdown pipe table."""
    parser = _TableParser()
    parser.feed(markup)
    if not parser.header:
        return ""

    width = len(parser.header)
    lines = []
    caption = " ".join(parser.caption.split())
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join(_cell_text(c) for c in parser.header) + " |")
    lines.append("| " + " | ".join("---" for _ in parser.header) + " |")
    for row in parser.rows:
        row = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(_cell_text(c) for c in row) + " |")
    return "\n".join(lines)


_STYLER_ID = re.compile(r"T_[0-9a-f]{5,}")


def stabilise_cell_ids(nb: nbformat.NotebookNode) -> None:
    """Fix the id of every cell.

    nbformat gives a cell without an id a random one, and the HTML exporter writes that id into
    the page, so two runs on identical data would otherwise differ. Numbering them in order makes
    the rendered report reproducible byte for byte.
    """
    for index, cell in enumerate(nb.cells):
        cell["id"] = f"cp2-cell-{index:03d}"


def stabilise_styler_ids(nb: nbformat.NotebookNode) -> int:
    """Give every pandas Styler table a fixed id.

    pandas draws a random id (for example `T_54bf9`) for each styled table, which would make the
    rendered HTML differ between two runs on identical data. Replacing them with ids numbered in
    cell order makes the report reproducible byte for byte.
    """
    counter = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "text/html" not in data:
                continue
            markup = data["text/html"]
            for old in dict.fromkeys(_STYLER_ID.findall(markup)):
                counter += 1
                markup = markup.replace(old, f"T_cp2_{counter:03d}")
            data["text/html"] = markup
    return counter


def notebook_to_markdown(nb: nbformat.NotebookNode) -> str:
    blocks: list[str] = [PREAMBLE]
    figure = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            blocks.append(cell.source.strip())
            continue
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                text = output.get("text", "").rstrip()
                if text:
                    blocks.append("```text\n" + text + "\n```")
            elif kind in {"execute_result", "display_data"}:
                data = output.get("data", {})
                if "image/png" in data:
                    name = FIGURES[figure] if figure < len(FIGURES) else f"figure-{figure + 1}.png"
                    figure += 1
                    blocks.append(f"![Figure {figure}]({name})")
                elif "text/html" in data:
                    table = html_table_to_markdown(data["text/html"])
                    if table:
                        blocks.append(table)
                elif "text/plain" in data:
                    text = str(data["text/plain"]).strip()
                    if text:
                        blocks.append("```text\n" + text + "\n```")

    if figure != len(FIGURES):
        raise RuntimeError(f"expected {len(FIGURES)} figures in the report, found {figure}")

    return "\n\n".join(part for part in blocks if part) + "\n"


def main() -> None:
    start = time.time()
    OUT.mkdir(exist_ok=True)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    stabilise_cell_ids(nb)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    tables = stabilise_styler_ids(nb)

    html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    (OUT / "checkpoint2.html").write_text(html, encoding="utf-8")

    markdown = notebook_to_markdown(nb)
    (OUT / "report.md").write_text(markdown, encoding="utf-8")
    print(
        f"wrote output/checkpoint2.html and output/report.md "
        f"({len(markdown):,} characters, {tables} styled tables) in "
        f"{(time.time() - start) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
