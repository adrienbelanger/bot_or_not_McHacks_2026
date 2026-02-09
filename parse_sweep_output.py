#!/usr/bin/env python3
"""Parse sweep runner output to extract best booster parameters.

Usage:
  python parse_sweep_output.py output_144_iters.txt
  python parse_sweep_output.py output_144_iters.txt --out-dir artifacts --top-n 10
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

RUN_START_RE = re.compile(r"^\[(\d+)/(\d+)\] Running (\S+) \.\.\.$")
OOF_RE = re.compile(
    r"Building OOF first-stage features.*seeds=\[(.*?)\], folds=(\d+), epochs=(\d+)\)\.\.\."
)
ENSEMBLE_AGG_RE = re.compile(r"Ensemble aggregation: (\w+)")
ENSEMBLE_THRESH_RE = re.compile(r"Ensemble selected threshold: ([0-9.]+)")
ENSEMBLE_SCORE_RE = re.compile(
    r"Ensemble test score: (\d+) \(TP=(\d+), FN=(\d+), FP=(\d+), accounts=(\d+)\)"
)
SEED_SCORE_RE = re.compile(r"Seed score mean/std: ([0-9.]+) / ([0-9.]+)")
PROFILE_MODE_RE = re.compile(r"Second-stage profile mode: (.+)$")
PROFILE_SELECTED_RE = re.compile(r"Second-stage selected profile: (.+)$")
BLEND_ALPHA_RE = re.compile(r"Second-stage blend alpha \(CatBoost weight\): ([0-9.]+)")
SECOND_THRESH_RE = re.compile(r"Second-stage threshold: ([0-9.]+)")
SECOND_SCORE_RE = re.compile(r"Second-stage test score: (\d+)/(\d+)")
SECOND_CONF_RE = re.compile(r"Second-stage confusion components -> TP=(\d+), FN=(\d+), FP=(\d+)")
SUMMARY_RE = re.compile(
    r"^(week_sweep_\d+): booster=([0-9.]+), ensemble=([0-9.]+), gain=([-0-9.]+), dur=([0-9.]+)s"
)


def _parse_seeds(raw: str) -> List[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_output(path: Path) -> Dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Some logs wrap long lines with a trailing backslash and newline. Re-join them.
    text: List[str] = []
    carry: Optional[str] = None
    for line in raw_lines:
        if carry is not None:
            line = carry + line
            carry = None
        if line.endswith("\\"):
            carry = line[:-1]
            continue
        text.append(line)
    if carry is not None:
        text.append(carry)

    runs: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    in_candidates = False

    def finalize_current() -> None:
        nonlocal current
        if not current:
            return
        runs.append(current)
        current = None

    for line in text:
        line = line.rstrip("\n")

        m = RUN_START_RE.match(line)
        if m:
            # start a new run, but don't finalize until summary line
            current = {
                "run_index": int(m.group(1)),
                "run_total": int(m.group(2)),
                "run_name": m.group(3),
                "candidates": [],
            }
            in_candidates = False
            continue

        if current is None:
            # still allow summary parsing if run start missing
            sm = SUMMARY_RE.match(line)
            if sm:
                current = {"run_name": sm.group(1), "candidates": []}
            else:
                continue

        # candidate table parsing
        if line.strip().startswith("Second-stage candidate report"):
            in_candidates = True
            continue
        if in_candidates:
            if not line.strip() or line.strip().startswith("Baseline account-level"):
                in_candidates = False
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "profile":
                continue
            if parts[0] in {"legacy", "regularized"} and len(parts) >= 11:
                cand = {
                    "profile": parts[0],
                    "alpha": float(parts[1]),
                    "threshold": float(parts[2]),
                    "val_score": float(parts[3]),
                    "val_tp_accounts": int(parts[4]),
                    "val_fn_accounts": int(parts[5]),
                    "val_fp_accounts": int(parts[6]),
                    "test_score": float(parts[7]),
                    "test_tp_accounts": int(parts[8]),
                    "test_fn_accounts": int(parts[9]),
                    "test_fp_accounts": int(parts[10]),
                }
                current.setdefault("candidates", []).append(cand)
                continue
            # not a candidate row; fall back to normal parsing
            in_candidates = False

        m = OOF_RE.search(line)
        if m:
            current["oof_seeds"] = _parse_seeds(m.group(1))
            current["folds"] = int(m.group(2))
            current["epochs"] = int(m.group(3))
            continue

        m = ENSEMBLE_AGG_RE.search(line)
        if m:
            current["ensemble_agg"] = m.group(1)
            continue
        m = ENSEMBLE_THRESH_RE.search(line)
        if m:
            current["ensemble_threshold"] = float(m.group(1))
            continue
        m = ENSEMBLE_SCORE_RE.search(line)
        if m:
            current["ensemble_score"] = float(m.group(1))
            current["ensemble_tp"] = int(m.group(2))
            current["ensemble_fn"] = int(m.group(3))
            current["ensemble_fp"] = int(m.group(4))
            current["ensemble_accounts"] = int(m.group(5))
            continue

        m = SEED_SCORE_RE.search(line)
        if m:
            current["seed_score_mean"] = float(m.group(1))
            current["seed_score_std"] = float(m.group(2))
            continue

        m = PROFILE_MODE_RE.search(line)
        if m:
            current["profile_mode"] = m.group(1)
            continue
        m = PROFILE_SELECTED_RE.search(line)
        if m:
            current["selected_profile"] = m.group(1)
            continue
        m = BLEND_ALPHA_RE.search(line)
        if m:
            current["blend_alpha"] = float(m.group(1))
            continue
        m = SECOND_THRESH_RE.search(line)
        if m:
            current["second_threshold"] = float(m.group(1))
            continue
        m = SECOND_SCORE_RE.search(line)
        if m:
            current["booster_score"] = float(m.group(1))
            current["booster_max"] = int(m.group(2))
            continue
        m = SECOND_CONF_RE.search(line)
        if m:
            current["second_tp"] = int(m.group(1))
            current["second_fn"] = int(m.group(2))
            current["second_fp"] = int(m.group(3))
            continue

        m = SUMMARY_RE.match(line)
        if m:
            current["run_name"] = m.group(1)
            current["booster_score"] = float(m.group(2))
            current["ensemble_score"] = float(m.group(3))
            current["gain"] = float(m.group(4))
            current["duration_s"] = float(m.group(5))
            finalize_current()
            continue

    # handle incomplete run if present
    if current is not None:
        incomplete.append(current)

    return {"runs": runs, "incomplete": incomplete}


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_index",
        "run_total",
        "run_name",
        "booster_score",
        "booster_max",
        "ensemble_score",
        "gain",
        "duration_s",
        "oof_seeds",
        "folds",
        "epochs",
        "ensemble_agg",
        "ensemble_threshold",
        "ensemble_tp",
        "ensemble_fn",
        "ensemble_fp",
        "profile_mode",
        "selected_profile",
        "blend_alpha",
        "second_threshold",
        "second_tp",
        "second_fn",
        "second_fp",
        "seed_score_mean",
        "seed_score_std",
        "candidates",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            if "oof_seeds" in flat:
                flat["oof_seeds"] = ",".join(str(s) for s in flat["oof_seeds"])
            if "candidates" in flat:
                flat["candidates"] = json.dumps(flat["candidates"], ensure_ascii=True)
            writer.writerow({k: flat.get(k, "") for k in fieldnames})


def select_best(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not runs:
        return None

    def key(r: Dict[str, Any]) -> tuple:
        booster = r.get("booster_score", float("-inf"))
        gain = r.get("gain", float("-inf"))
        fp = r.get("second_fp", float("inf"))
        return (booster, gain, -fp)

    return max(runs, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse booster sweep output logs")
    parser.add_argument("input", type=Path, help="Path to sweep output txt")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    results = parse_output(args.input)
    runs = results["runs"]
    incomplete = results["incomplete"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "booster_sweep_parsed.csv"
    write_csv(runs, csv_path)

    best = select_best(runs)
    best_path = args.out_dir / "booster_sweep_best.json"
    if best is not None:
        best_path.write_text(json.dumps(best, indent=2, ensure_ascii=True), encoding="utf-8")

    incomplete_path = args.out_dir / "booster_sweep_incomplete.json"
    if incomplete:
        incomplete_path.write_text(json.dumps(incomplete, indent=2, ensure_ascii=True), encoding="utf-8")

    # Print quick summary
    print(f"Parsed runs: {len(runs)}")
    if incomplete:
        print(f"Incomplete runs: {len(incomplete)} (saved to {incomplete_path})")
    print(f"CSV: {csv_path}")
    if best is not None:
        print(f"Best: {best.get('run_name')} booster={best.get('booster_score')} gain={best.get('gain')} profile={best.get('selected_profile')} threshold={best.get('second_threshold')} seeds={best.get('oof_seeds')} folds={best.get('folds')} epochs={best.get('epochs')}")

    # top-N by booster score
    top_n = sorted(runs, key=lambda r: (r.get("booster_score", -1), r.get("gain", -1)), reverse=True)[: args.top_n]
    print("Top runs:")
    for r in top_n:
        print(
            f"  {r.get('run_name')} booster={r.get('booster_score')} gain={r.get('gain')} "
            f"profile={r.get('selected_profile')} thr={r.get('second_threshold')} seeds={r.get('oof_seeds')} folds={r.get('folds')} epochs={r.get('epochs')}"
        )


if __name__ == "__main__":
    main()
