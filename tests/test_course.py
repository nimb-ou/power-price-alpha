"""Course integrity.

The lessons are half the value of this repo, and they are the half nothing else
checks. A renamed file leaves a "Next:" link pointing at nothing, a regenerated
notebook can drift from its source, and neither shows up in a normal test run.

These tests are cheap and they keep the course navigable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
NOTEBOOKS = ROOT / "notebooks"
COURSE = ROOT / "course"

# `01-ingest-prices.ipynb` or `course/00-gb-power-market.md`
LINK = re.compile(r"`((?:course/)?\d{2}-[a-z0-9-]+\.(?:ipynb|md))`")


def lesson_sources() -> list[Path]:
    return sorted(SRC.glob("*.py"))


def markdown_lessons() -> list[Path]:
    return sorted(COURSE.glob("*.md"))


def all_lesson_files() -> list[Path]:
    return lesson_sources() + markdown_lessons()


def test_there_are_lessons() -> None:
    assert len(lesson_sources()) >= 10
    assert len(markdown_lessons()) >= 2


def test_every_source_has_a_generated_notebook() -> None:
    """`make notebooks` must have been run after adding a lesson."""
    for source in lesson_sources():
        generated = NOTEBOOKS / f"{source.stem}.ipynb"
        assert generated.exists(), f"{generated.name} missing — run `make notebooks`"


def test_no_orphan_notebooks() -> None:
    """A notebook with no source would be hand-edited and lost on the next build."""
    sources = {p.stem for p in lesson_sources()}
    for notebook in NOTEBOOKS.glob("*.ipynb"):
        assert notebook.stem in sources, f"{notebook.name} has no source in notebooks/_src"


@pytest.mark.parametrize("lesson", all_lesson_files(), ids=lambda p: p.name)
def test_cross_references_resolve(lesson: Path) -> None:
    """Every `NN-name.ipynb` or `course/NN-name.md` reference must exist."""
    for target in LINK.findall(lesson.read_text()):
        candidates = [ROOT / target, NOTEBOOKS / target, COURSE / target]
        assert any(c.exists() for c in candidates), f"{lesson.name} links to missing {target}"


@pytest.mark.parametrize("lesson", all_lesson_files(), ids=lambda p: p.name)
def test_lesson_has_a_title(lesson: Path) -> None:
    text = lesson.read_text()
    assert re.search(r"#\s*#?\s*Lesson \d{2} —", text), f"{lesson.name} has no lesson title"


@pytest.mark.parametrize("lesson", lesson_sources(), ids=lambda p: p.name)
def test_notebook_sources_are_valid_percent_format(lesson: Path) -> None:
    """Every cell marker must be a recognised one, or py2nb silently merges cells."""
    for line in lesson.read_text().splitlines():
        if line.startswith("# %%"):
            assert line.rstrip() in ("# %%", "# %% [markdown]"), (
                f"{lesson.name}: unrecognised cell marker {line!r}"
            )


@pytest.mark.parametrize("lesson", lesson_sources(), ids=lambda p: p.name)
def test_generated_notebooks_have_no_outputs(lesson: Path) -> None:
    """Committed notebooks must be clean: no outputs, no execution counts."""
    notebook = json.loads((NOTEBOOKS / f"{lesson.stem}.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{lesson.stem}.ipynb has stored outputs"
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("lesson", lesson_sources(), ids=lambda p: p.name)
def test_generated_notebook_matches_its_source(lesson: Path) -> None:
    """Catches editing a lesson source and forgetting `make notebooks`.

    Rebuilds the notebook in memory from the source and compares cell contents.
    Without this, a committed .ipynb can silently drift from the .py that is
    supposed to be its single source of truth — and the .ipynb is what a reader
    actually opens.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    from py2nb import parse  # noqa: PLC0415

    # nbformat allows `source` to be a string or a list of lines, and which one
    # you get depends on whether nbformat or nbconvert wrote the file last. So
    # compare normalised text rather than the raw JSON shape.
    def text(cell: object) -> str:
        source = cell["source"] if isinstance(cell, dict) else cell.source  # type: ignore[index]
        return "".join(source) if isinstance(source, list) else str(source)

    expected = [text(c) for c in parse(lesson.read_text())]
    actual = [
        text(c) for c in json.loads((NOTEBOOKS / f"{lesson.stem}.ipynb").read_text())["cells"]
    ]
    assert actual == expected, f"{lesson.stem}.ipynb is stale — run `make notebooks`"


def test_lesson_numbers_are_unique() -> None:
    numbers = [p.name[:2] for p in all_lesson_files()]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    assert not duplicates, f"duplicate lesson numbers: {sorted(duplicates)}"


def test_readme_points_at_the_first_lesson() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "course/00-gb-power-market.md" in readme
