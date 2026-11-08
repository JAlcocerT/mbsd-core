"""Run the validation corpus; ``--write`` regenerates ``expected-v1.json``.

    uv run python corpus/run.py            # run and compare with the record
    uv run python corpus/run.py --write    # record after an explained change
    uv run python corpus/run.py --only planar.rack_pinion
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

import mbsd  # noqa: E402
from cases import CASES, CORPUS_VERSION  # noqa: E402

EXPECTED = HERE / f"expected-v{CORPUS_VERSION}.json"
VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-12


def run_case(case_id: str) -> dict:
    case = CASES[case_id]
    start = time.perf_counter()
    out = case.run()
    return {
        "id": case.id,
        "family": case.family,
        "dimension": case.dimension,
        "description": case.description,
        "reference": case.reference,
        "settings": case.settings,
        "errors": {k: {"value": float(v), "bound": float(b)} for k, (v, b) in out["errors"].items()},
        "values": {k: float(v) for k, v in out["values"].items()},
        "check": out["check"],
        "seconds": round(time.perf_counter() - start, 3),
    }


def compare(record: dict, expected: dict | None) -> list[str]:
    """Return problems: a broken independent bound, a drifted value, a changed verdict."""
    problems = [f"{k}: {e['value']:.3e} > bound {e['bound']:.1e}"
                for k, e in record["errors"].items() if not e["value"] <= e["bound"]]
    if expected is None:
        return problems + ["not in the expected record"]
    for key in ("reference", "settings", "check"):
        if record[key] != expected[key]:
            problems.append(f"{key} changed: {expected[key]!r} -> {record[key]!r}")
    for k, e in record["errors"].items():
        if expected["errors"].get(k, {}).get("bound") != e["bound"]:
            problems.append(f"bound of {k} changed without re-recording")
    if set(record["values"]) != set(expected["values"]):
        problems.append("recorded value names changed")
    for k, v in record["values"].items():
        old = expected["values"].get(k)
        if old is not None and not np.isclose(v, old, rtol=VALUE_RTOL, atol=VALUE_ATOL):
            problems.append(f"value {k} drifted: {old!r} -> {v!r}")
    return problems


def provenance() -> dict:
    return {
        "mbsd": mbsd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "recorded": _dt.date.today().isoformat(),
        "value_rtol": VALUE_RTOL,
        "value_atol": VALUE_ATOL,
    }


def load_expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8")) if EXPECTED.exists() else {"cases": {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    expected = load_expected()
    records, failures = {}, 0
    for case_id in args.only or sorted(CASES):
        record = run_case(case_id)
        records[case_id] = record
        problems = compare(record, None if args.write else expected["cases"].get(case_id))
        if args.write:
            problems = [p for p in problems if "not in the expected" not in p]
        failures += bool(problems)
        status = "ok " if not problems else "FAIL"
        print(f"{status} {case_id:48s} {record['seconds']:6.2f}s check={record['check']}")
        for problem in problems:
            print(f"       {problem}")
    if args.write:
        if failures:
            print("refusing to record: independent bounds fail", file=sys.stderr)
            return 1
        cases = {**expected["cases"], **records} if args.only else records
        payload = {
            "schema": "mbsd.validation_corpus",
            "schema_version": CORPUS_VERSION,
            "provenance": provenance(),
            "cases": {k: {key: val for key, val in v.items() if key != "seconds"} for k, v in sorted(cases.items())},
        }
        EXPECTED.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(cases)} cases in {EXPECTED.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
