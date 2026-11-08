"""The validation corpus as a release gate.

Every case must keep its independent bounds, its check() verdict, and its
recorded values (``corpus/expected-v1.json``); re-record with
``uv run python corpus/run.py --write`` only after an explained change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
sys.path.insert(0, str(CORPUS))

from cases import CASES, CORPUS_VERSION  # noqa: E402
from run import EXPECTED, compare, run_case  # noqa: E402

RECORD = json.loads(EXPECTED.read_text(encoding="utf-8"))


def test_record_covers_exactly_the_registered_cases():
    assert RECORD["schema"] == "mbsd.validation_corpus"
    assert RECORD["schema_version"] == CORPUS_VERSION
    assert set(RECORD["cases"]) == set(CASES)
    for key in ("mbsd", "numpy", "scipy", "python", "platform", "recorded"):
        assert RECORD["provenance"][key]


def test_corpus_covers_passing_and_failing_verdicts_in_both_dimensions():
    verdicts = {(c["dimension"], c["check"]) for c in RECORD["cases"].values()}
    assert {(2, True), (2, False), (3, True), (3, False)} <= verdicts


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_corpus_case(case_id):
    problems = compare(run_case(case_id), RECORD["cases"][case_id])
    assert not problems, "\n".join(problems)
