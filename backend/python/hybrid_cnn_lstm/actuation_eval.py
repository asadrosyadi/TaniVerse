"""
Actuation Performance Evaluation -- manuscript sec. 3.8 / Eq. 13.

    SR = ( N_correct / N_total ) x 100%

An actuation event is "correct" when the system's response matches the rule-based
agronomic ground truth:
  * a detected pest      -> UV trap / ultrasonic repeller  (Hispa -> both)
  * elevated VPD (>= 1.5 kPa, plant water-stress threshold) -> irrigation
  * otherwise            -> continue monitoring

Input is the JSON run-log written by algorithm1.py (`algorithm1_run_log.json`),
so the pipeline is: algorithm1.py -> run log -> this script -> SR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VPD_STRESS_KPA = 1.5           # sec. 4.2: VPD >= 1.5 kPa => plant water stress


def expected_action(entry: dict) -> str:
    """Rule-based agronomic ground truth for one timestep."""
    pest = entry.get("pest", {})
    pest_label = pest.get("label", "No pest detected")
    pest_class = (pest.get("class", "") or "").lower()
    vpd = float(entry.get("vpd_kpa", entry.get("microclimate", {}).get("vpd", 0.0)))
    status = entry.get("health", {}).get("status", "")

    if pest_label != "No pest detected":
        if pest_class == "hispa":
            return "activate_ultrasonic_repeller_and_uv_lamp"
        return "recommend_selective_insecticide_spraying"
    if vpd >= VPD_STRESS_KPA or status == "High Stress":
        return "irrigation_notification"
    return "continue_monitoring"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Hitung actuation success rate (Eq. 13)")
    ap.add_argument("--run-log", type=str, default=str(here / "algorithm1_run_log.json"))
    ap.add_argument("--out", type=str, default=str(here / "actuation_success_rate.json"))
    args = ap.parse_args()

    log = json.loads(Path(args.run_log).read_text(encoding="utf-8"))
    if not log:
        raise SystemExit("run log kosong -- jalankan algorithm1.py dulu")

    total = len(log)
    correct = 0
    per_kind: dict[str, list[int]] = {}
    mismatches = []
    for e in log:
        want = expected_action(e)
        got = e.get("actuation", {}).get("action", "")
        ok = int(want == got)
        correct += ok
        per_kind.setdefault(want, [0, 0])
        per_kind[want][0] += ok
        per_kind[want][1] += 1
        if not ok:
            mismatches.append({"timestamp": e.get("timestamp"), "expected": want, "got": got})

    sr = 100.0 * correct / total
    result = {
        "N_total": total,
        "N_correct": correct,
        "success_rate_percent": round(sr, 2),
        "per_action": {k: {"correct": v[0], "total": v[1],
                           "rate_percent": round(100.0 * v[0] / v[1], 2)}
                       for k, v in per_kind.items()},
        "mismatches": mismatches,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Actuation events (N_total)  : {total}")
    print(f"Correct (N_correct)         : {correct}")
    print(f"Actuation success rate (SR) : {sr:.2f}%")
    for k, v in result["per_action"].items():
        print(f"  {k:<42} {v['correct']}/{v['total']}  ({v['rate_percent']:.1f}%)")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
