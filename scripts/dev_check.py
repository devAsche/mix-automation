"""One-shot verification for every change: ruff (strict subset + full-config count) and pytest.

Prints counts and failures only, so the output stays short enough to paste into a report.

    .venv\\Scripts\\python.exe scripts\\dev_check.py            # all checks
    .venv\\Scripts\\python.exe scripts\\dev_check.py --no-tests # ruff only

Exit code is non-zero when the strict ruff subset has findings, the full-config ruff count
rises above the baseline, or pytest fails.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SCRIPTS = ROOT / ".venv" / "Scripts"
# Full-config `ruff check src tests` findings inherited from the Gemini era (2026-11-26).
# Changes must not raise this number; lower it when existing findings get fixed.
FULL_RUFF_BASELINE = 129
TARGETS = ["src", "tests"]


def _ruff_exe() -> str:
    exe = VENV_SCRIPTS / "ruff.exe"
    return str(exe) if exe.exists() else "ruff"


def _ruff_findings(extra: list[str]) -> list[dict]:
    cmd = [_ruff_exe(), "check", *TARGETS, "--output-format", "json", "--no-cache", *extra]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
        raise SystemExit("ruff output could not be parsed") from None


def _rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return path


def check_ruff() -> bool:
    ok = True
    strict = _ruff_findings(["--select", "E,F,W,I", "--ignore", "E501"])
    print(f"ruff strict (E,F,W,I -E501): {len(strict)} findings")
    for f in strict[:30]:
        loc = f.get("location") or {}
        print(f"  {_rel(f['filename'])}:{loc.get('row')} {f.get('code')} {f.get('message')}")
    if strict:
        ok = False

    full = _ruff_findings([])
    delta = len(full) - FULL_RUFF_BASELINE
    sign = "+" if delta > 0 else ""
    print(f"ruff full config: {len(full)} findings (baseline {FULL_RUFF_BASELINE}, {sign}{delta})")
    if delta > 0:
        ok = False
        by_code = Counter(f.get("code") for f in full)
        print("  top codes: " + ", ".join(f"{c}={n}" for c, n in by_code.most_common(8)))
    return ok


def check_pytest() -> bool:
    py = VENV_SCRIPTS / "python.exe"
    cmd = [str(py if py.exists() else sys.executable), "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rfE"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = proc.stdout.splitlines()
    failures = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    summary = next((ln for ln in reversed(lines) if re.search(r"\b(passed|failed|error)", ln)), "(no summary)")
    print(f"pytest: {summary.strip('= ')}")
    for ln in failures[:30]:
        print(f"  {ln}")
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-tests", action="store_true", help="skip pytest")
    args = ap.parse_args()
    ok = check_ruff()
    if not args.no_tests:
        ok = check_pytest() and ok
    print("RESULT: " + ("OK" if ok else "NG"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
