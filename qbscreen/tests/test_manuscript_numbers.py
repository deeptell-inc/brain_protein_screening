"""The manuscript's numbers are generated, never typed: the committed macro files
must equal what the generator produces from the stored results."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / "calibration_results" / "manuscript_numbers.json").exists(),
                    reason="calibration results not present")
def test_numbers_tex_is_current(monkeypatch):
    monkeypatch.chdir(ROOT)
    from qbscreen.make_numbers_tex import render
    text, table = render()
    d = ROOT / "manuscript" / "magnetobio_v4"
    assert (d / "numbers.tex").read_text() == text, "re-run qbscreen/make_numbers_tex.py"
    assert (d / "table_competence.tex").read_text() == table, "re-run qbscreen/make_numbers_tex.py"


def test_no_hand_typed_numbers_in_abstract():
    """Quantitative claims in the abstract go through macros."""
    import re
    tex = (ROOT / "manuscript" / "magnetobio_v4" / "main.tex").read_text()
    abstract = tex[tex.index("\\begin{abstract}"):tex.index("\\end{abstract}")]
    # allow the model constants that define the question (50 µT) but no results
    stray = [m for m in re.findall(r"\d+(?:\.\d+)?", abstract) if m not in {"2"}]
    assert not stray, f"hand-typed numbers in abstract: {stray}"
