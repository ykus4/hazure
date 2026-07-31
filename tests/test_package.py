"""Guards on the package itself, rather than on anything it computes.

These exist because the failures they catch are invisible until a release: a
version string that disagrees with the distribution's, a name promised in
``__all__`` that nobody exports, or a module that reaches for pandas at import
time and so quietly breaks the "narwhals and numpy only" promise the README makes.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import pytest

import hazure

MODULES = (
    "hazure.calibration",
    "hazure.compose",
    "hazure.datasets",
    "hazure.detection",
    "hazure.ensemble",
    "hazure.evaluation",
    "hazure.events",
    "hazure.features",
    "hazure.methods",
    "hazure.scoring",
    "hazure.streaming",
    "hazure.thresholds",
)

OPTIONAL = (
    "matplotlib",
    "numba",
    "pandas",
    "polars",
    "pyarrow",
    "ruptures",
    "scipy",
    "sklearn",
    "statsmodels",
    "stumpy",
)


def test_dunder_version_matches_the_distribution() -> None:
    assert hazure.__version__ == version("hazure")


def test_every_promised_name_exists() -> None:
    missing = [name for name in hazure.__all__ if not hasattr(hazure, name)]
    assert missing == []


def test_all_has_no_duplicates() -> None:
    assert len(hazure.__all__) == len(set(hazure.__all__))


@pytest.mark.parametrize("module", MODULES)
def test_submodule_all_is_reexported_from_the_package(module: str) -> None:
    """Names a subject module publishes are reachable from ``hazure`` itself.

    ``hazure.datasets`` is deliberately exempt: it is sample data for trying
    things out, not part of the detection API, and ``compare`` is too generic a
    name to put in the top-level namespace.
    """
    imported = __import__(module, fromlist=["__all__"])
    if module == "hazure.datasets":
        pytest.skip("datasets is documented as reachable only through its module")
    unexported = [name for name in imported.__all__ if name not in set(hazure.__all__)]
    assert unexported == []


def test_importing_hazure_pulls_in_nothing_optional() -> None:
    """The core promise, checked in a fresh interpreter.

    Checking ``sys.modules`` in *this* process would prove nothing: the test suite
    has already imported pandas. So this asks a subprocess, which has not.
    """
    names = ", ".join(repr(name) for name in OPTIONAL)
    script = (
        "import sys, hazure\n"
        f"leaked = {{{names}}} & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"
