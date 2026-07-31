"""Run the Python in the documentation and check the outputs it claims.

The guide says "running the page top to bottom reproduces every number shown", and
the algorithm pages quote figures too. That promise is only worth making if
something checks it, because a docs example is exactly the kind of code nobody
runs again after writing it.

Each page's ``python`` blocks are concatenated in order and executed in one
namespace, so a block may use names an earlier one bound — which is what makes the
guide readable. Any line of the form ``# <expected>`` immediately after a
``print(...)`` is compared against what that print actually produced.

    uv run python scripts/check_docs.py
    uv run python scripts/check_docs.py docs/guide.md
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: A fenced block whose language is python. Blocks in other languages — the bash
#: install lines, the mermaid diagrams — are skipped.
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def blocks(text: str) -> list[str]:
    """Return every ``python`` fenced block of a page, in order."""
    return [match.group(1) for match in BLOCK.finditer(text)]


def expected(block: str) -> list[str]:
    """Return the output a block claims, in order.

    A claim is a run of unindented ``# ...`` lines that directly follows a line
    containing ``print(``. That rule is what lets the pages interleave several
    prints with their outputs, while leaving an ordinary explanatory comment —
    which never sits under a ``print`` — alone.
    """
    claimed: list[str] = []
    in_claim = False
    for line in block.splitlines():
        if line.startswith("#"):
            if in_claim:
                claimed.append(line[2:] if line.startswith("# ") else line[1:])
            continue
        in_claim = "print(" in line
    return claimed


def run_page(path: Path) -> list[str]:
    """Execute one page's blocks in a shared namespace, returning any complaints."""
    namespace: dict[str, object] = {"__name__": "__docs__"}
    problems: list[str] = []

    for number, block in enumerate(blocks(path.read_text()), start=1):
        claimed = expected(block)
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(block, f"{path}:block{number}", "exec"), namespace)
        except Exception as error:
            problems.append(f"{path.name} block {number}: raised {error!r}")
            continue

        if not claimed:
            continue
        printed = buffer.getvalue().splitlines()
        if printed != claimed:
            problems.append(
                f"{path.name} block {number}: printed {printed!r}, "
                f"page claims {claimed!r}"
            )
    return problems


def main(argv: list[str]) -> int:
    """Check the pages named on the command line, or every page under docs/."""
    if argv:
        pages = [Path(name).resolve() for name in argv]
    else:
        pages = sorted((ROOT / "docs").rglob("*.md"))

    problems: list[str] = []
    # The quickstart saves a figure, and other pages may grow the habit. Running
    # from a scratch directory keeps what they write out of the repository.
    with tempfile.TemporaryDirectory() as scratch:
        here = Path.cwd()
        os.chdir(scratch)
        try:
            for page in pages:
                found = run_page(page)
                print(f"{'FAIL' if found else 'ok':4s} {page.relative_to(ROOT)}")
                problems.extend(found)
        finally:
            os.chdir(here)

    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
