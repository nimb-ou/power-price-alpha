"""Build `.ipynb` lessons from percent-format `.py` sources.

Notebooks are terrible in git: JSON diffs, embedded outputs, merge conflicts on
execution counts. So the course sources live in `notebooks/_src/*.py` in the
standard "percent" format, and the `.ipynb` files are generated from them with
no outputs.

    # %% [markdown]
    # # Heading
    # prose

    # %%
    code()

Usage:  python tools/py2nb.py            # rebuild every lesson
        python tools/py2nb.py 01 02      # rebuild specific ones
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def parse(text: str) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    kind = "code"
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            buf.clear()
            return
        if kind == "markdown":
            # Strip the leading "# " that percent format uses for prose.
            lines = [ln[2:] if ln.startswith("# ") else ln.removeprefix("#") for ln in body.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
        buf.clear()

    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        else:
            buf.append(line)
    flush()
    return cells


def build(src: Path) -> Path:
    nb = nbformat.v4.new_notebook(cells=parse(src.read_text()))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    dest = OUT / f"{src.stem}.ipynb"
    nbformat.write(nb, dest)
    return dest


def main(argv: list[str]) -> None:
    sources = sorted(SRC.glob("*.py"))
    if argv:
        sources = [s for s in sources if any(s.stem.startswith(a) for a in argv)]
    if not sources:
        raise SystemExit(f"no lesson sources matched in {SRC}")
    for src in sources:
        print(f"{src.relative_to(ROOT)} -> {build(src).relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
