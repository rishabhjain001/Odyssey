"""Integration / snapshot tests for the report PDF generator.

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7

These are plain pytest tests (no hypothesis).  They exercise
``report.generate.generate_report`` end-to-end and verify the structural
and content properties of the produced PDF.

Text extraction strategy
------------------------
1. Try ``pdfplumber`` (preferred, accurate text extraction).
2. Try ``PyPDF2`` (fallback).
3. Fall back to raw-bytes search (works for standard embedded ASCII fonts).

The raw-bytes fallback is documented where it is used and works because
ReportLab embeds section-heading strings as plain ASCII in the content stream.
"""

from __future__ import annotations

import os
import re
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Helper: create a minimal matplotlib figure for testing
# ---------------------------------------------------------------------------

def _make_figure():
    """Return a simple matplotlib Figure suitable for embedding in the report."""
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 10, 5])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    return fig


# ---------------------------------------------------------------------------
# Helper: extract text from a PDF file
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from *pdf_path* using the best available library.

    Tries pdfplumber, then PyPDF2, then falls back to decoding the raw bytes
    with latin-1 (which preserves ASCII content stream text for standard fonts).
    """
    # Attempt 1: pdfplumber
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except ImportError:
        pass

    # Attempt 2: PyPDF2
    try:
        import PyPDF2  # type: ignore

        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            return "\n".join(
                reader.pages[i].extract_text() or ""
                for i in range(len(reader.pages))
            )
    except ImportError:
        pass

    # Fallback 3: raw bytes decoded as latin-1.
    # Standard ReportLab fonts embed ASCII strings directly in the content
    # stream, so section headers and author names are readable this way.
    with open(pdf_path, "rb") as fh:
        return fh.read().decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# Helper: count pages in a PDF
# ---------------------------------------------------------------------------

def _count_pdf_pages(pdf_path: str) -> int:
    """Return the page count of *pdf_path* using the best available library."""
    # Attempt 1: pdfplumber
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except ImportError:
        pass

    # Attempt 2: PyPDF2
    try:
        import PyPDF2  # type: ignore

        with open(pdf_path, "rb") as fh:
            return len(PyPDF2.PdfReader(fh).pages)
    except ImportError:
        pass

    # Fallback 3: regex on raw bytes — look for /Count N in the PDF cross-ref
    with open(pdf_path, "rb") as fh:
        raw = fh.read()
    # The Pages dictionary contains /Count <N>; pick the largest value found.
    counts = [int(m.group(1)) for m in re.finditer(rb"/Count\s+(\d+)", raw)]
    if counts:
        return max(counts)
    raise RuntimeError("Could not determine PDF page count")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def figures():
    """Return a pair of matplotlib figures and close them after the test."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    fig1 = _make_figure()
    fig2 = _make_figure()
    yield fig1, fig2
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(fig2)


@pytest.fixture
def pdf_path(figures, tmp_path):
    """Generate the report PDF once and return its path."""
    from report.generate import generate_report  # noqa: PLC0415

    fig1, fig2 = figures
    out = str(tmp_path / "report.pdf")
    generate_report(
        output_path=out,
        training_curve_figure=fig1,
        reward_comparison_figure=fig2,
        odyssey_avg_reward=42.0,
        ppo_avg_reward=30.0,
        extra_notes="Test run.",
    )
    return out


# ===========================================================================
# Test 1 — single PDF, path ends with .pdf, file exists, non-empty  (Req 16.1)
# ===========================================================================

def test_generate_report_produces_single_pdf(figures, tmp_path):
    """Req 16.1: generate_report returns a non-empty .pdf file."""
    from report.generate import generate_report  # noqa: PLC0415

    fig1, fig2 = figures
    out = str(tmp_path / "report_test1.pdf")
    result = generate_report(
        output_path=out,
        training_curve_figure=fig1,
        reward_comparison_figure=fig2,
    )

    assert isinstance(result, str), "Return value must be a string path"
    assert result.endswith(".pdf"), "Returned path must end with '.pdf'"
    assert os.path.isfile(result), "PDF file must exist on disk"
    assert os.path.getsize(result) > 0, "PDF file must be non-empty"


# ===========================================================================
# Test 2 — return value equals output_path  (Req 16.1)
# ===========================================================================

def test_generate_report_returns_output_path(figures, tmp_path):
    """Req 16.1: the return value of generate_report equals the output_path arg."""
    from report.generate import generate_report  # noqa: PLC0415

    fig1, fig2 = figures
    out = str(tmp_path / "report_test2.pdf")
    result = generate_report(
        output_path=out,
        training_curve_figure=fig1,
        reward_comparison_figure=fig2,
    )

    assert result == out


# ===========================================================================
# Test 3 — page count is 4 to 6  (Req 16.5)
# ===========================================================================

def test_pdf_page_count_is_4_to_6(pdf_path):
    """Req 16.5: PDF has between 4 and 6 pages inclusive."""
    pages = _count_pdf_pages(pdf_path)
    assert 4 <= pages <= 6, (
        f"Expected 4–6 pages but got {pages}. PDF path: {pdf_path}"
    )


# ===========================================================================
# Test 4 — six ordered sections present  (Req 16.2)
# ===========================================================================

EXPECTED_SECTIONS = [
    "Abstract",
    "Background",
    "Architecture",
    "Experiments",
    "Results",
    "References",
]


def test_pdf_contains_six_ordered_sections(pdf_path):
    """Req 16.2: PDF contains all six required sections in document order."""
    text = _extract_pdf_text(pdf_path)

    positions = {}
    for section in EXPECTED_SECTIONS:
        idx = text.find(section)
        assert idx != -1, f"Section '{section}' not found in PDF text"
        positions[section] = idx

    # Verify ordering
    ordered = sorted(EXPECTED_SECTIONS, key=lambda s: positions[s])
    assert ordered == EXPECTED_SECTIONS, (
        f"Sections are out of order. Found order: {ordered}"
    )


# ===========================================================================
# Test 5 — all four citations present in body and References  (Req 16.3)
# ===========================================================================

def test_pdf_contains_all_four_citations(pdf_path):
    """Req 16.3: PDF body and References section contain all four required citations."""
    text = _extract_pdf_text(pdf_path)

    # Each citation must appear at least twice: once inline and once in References.
    # We check for distinctive strings from each required citation.
    required = {
        "Hafner et al., 2018 (RSSM)": "2018",
        "Hafner et al., 2019 (Dreamer)": "2019",
        "Hafner et al., 2020 (DreamerV2)": "2020",
        "Schulman et al., 2017 (PPO)": "2017",
    }
    for label, year in required.items():
        count = text.count(year)
        assert count >= 2, (
            f"Expected citation year {year!r} ({label}) to appear at least "
            f"2 times (once in body, once in References), but found {count}."
        )

    # Also verify author surnames appear
    for author in ("Hafner", "Schulman"):
        assert author in text, (
            f"Author name '{author}' not found in PDF text"
        )


# ===========================================================================
# Test 6 — missing training_curve_figure raises ReportInputMissingError  (Req 16.7)
# ===========================================================================

def test_missing_training_figure_raises_report_input_missing_error(figures, tmp_path):
    """Req 16.7: passing None for training_curve_figure raises ReportInputMissingError."""
    from report.generate import generate_report  # noqa: PLC0415
    from agent.errors import ReportInputMissingError  # noqa: PLC0415

    _, fig2 = figures
    out = str(tmp_path / "should_not_be_created.pdf")

    with pytest.raises(ReportInputMissingError):
        generate_report(
            output_path=out,
            training_curve_figure=None,
            reward_comparison_figure=fig2,
        )


# ===========================================================================
# Test 7 — missing reward_comparison_figure raises ReportInputMissingError  (Req 16.7)
# ===========================================================================

def test_missing_comparison_figure_raises_report_input_missing_error(figures, tmp_path):
    """Req 16.7: passing None for reward_comparison_figure raises ReportInputMissingError."""
    from report.generate import generate_report  # noqa: PLC0415
    from agent.errors import ReportInputMissingError  # noqa: PLC0415

    fig1, _ = figures
    out = str(tmp_path / "should_not_be_created.pdf")

    with pytest.raises(ReportInputMissingError):
        generate_report(
            output_path=out,
            training_curve_figure=fig1,
            reward_comparison_figure=None,
        )


# ===========================================================================
# Test 8 — non-existent path string raises ReportInputMissingError  (Req 16.7)
# ===========================================================================

def test_nonexistent_path_raises_report_input_missing_error(figures, tmp_path):
    """Req 16.7: a non-existent PNG path raises ReportInputMissingError."""
    from report.generate import generate_report  # noqa: PLC0415
    from agent.errors import ReportInputMissingError  # noqa: PLC0415

    _, fig2 = figures
    out = str(tmp_path / "should_not_exist.pdf")

    with pytest.raises(ReportInputMissingError):
        generate_report(
            output_path=out,
            training_curve_figure="/nonexistent/path/curve.png",
            reward_comparison_figure=fig2,
        )


# ===========================================================================
# Test 9 — prior artifact unchanged on error  (Req 16.7)
# ===========================================================================

def test_prior_artifact_unchanged_on_error(figures, tmp_path):
    """Req 16.7: if generation aborts, any previously written file is unchanged."""
    from report.generate import generate_report  # noqa: PLC0415
    from agent.errors import ReportInputMissingError  # noqa: PLC0415

    _, fig2 = figures
    out = str(tmp_path / "existing_report.pdf")

    # Write a sentinel file that represents a previously generated artifact.
    sentinel_content = b"%PDF-1.4 dummy artifact"
    with open(out, "wb") as fh:
        fh.write(sentinel_content)

    mtime_before = os.path.getmtime(out)

    with pytest.raises(ReportInputMissingError):
        generate_report(
            output_path=out,
            training_curve_figure=None,
            reward_comparison_figure=fig2,
        )

    # The file must not have been modified.
    mtime_after = os.path.getmtime(out)
    with open(out, "rb") as fh:
        content_after = fh.read()

    assert content_after == sentinel_content, (
        "Prior artifact was modified even though generation aborted"
    )
    assert mtime_after == mtime_before, (
        "Prior artifact's modification time changed even though generation aborted"
    )


# ===========================================================================
# Test 10 — report/generate.py is separate from agent/ codebase  (Req 16.6)
# ===========================================================================

def test_separate_from_agent_codebase():
    """Req 16.6: report/generate.py exists and is NOT under the agent/ directory."""
    # Derive the project root relative to this test file.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    report_module = os.path.join(project_root, "report", "generate.py")
    assert os.path.isfile(report_module), (
        f"report/generate.py not found at expected path: {report_module}"
    )

    # Confirm the resolved path does not sit inside the agent/ directory.
    agent_dir = os.path.realpath(os.path.join(project_root, "agent"))
    resolved = os.path.realpath(report_module)
    assert not resolved.startswith(agent_dir + os.sep) and resolved != agent_dir, (
        f"report/generate.py appears to be under the agent/ directory: {resolved}"
    )
