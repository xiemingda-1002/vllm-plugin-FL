"""Average rounds 2/3 only if all rounds returned 1024 successful fixed outputs."""

import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
round_numbers = (
    [int(sys.argv[3])]
    if len(sys.argv) == 4 and sys.argv[2] == "--check-round"
    else (1, 2, 3)
)
for n in round_numbers:
    paths = list((root / f"round{n}").rglob("benchmark_summary.json"))
    if len(paths) != 1:
        raise SystemExit(
            f"Round {n}: expected one benchmark_summary.json, found {len(paths)}"
        )
    row = json.loads(paths[0].read_text())
    for key, value in {
        "Total Requests": 1024,
        "Success Requests": 1024,
        "Failed Requests": 0,
    }.items():
        if float(row[key]) != value:
            raise SystemExit(f"Round {n}: invalid {key}={row[key]}; inspect {paths[0]}")
    if not math.isclose(float(row["Avg Output Tokens"]), 1024, rel_tol=0, abs_tol=1e-9):
        raise SystemExit(
            f"Round {n}: output length differs from 1024; inspect raw records"
        )
    if (
        not math.isfinite(float(row["Output Throughput (tok/s)"]))
        or float(row["Output Throughput (tok/s)"]) <= 0
    ):
        raise SystemExit(f"Round {n}: invalid throughput")
    rows.append(
        {
            "round": n,
            "output_tokens_per_second": float(row["Output Throughput (tok/s)"]),
            "source": str(paths[0]),
        }
    )
if len(round_numbers) == 1:
    print(f"Round {round_numbers[0]} passed aggregate success/output-length checks.")
    raise SystemExit(0)
a, b = (x["output_tokens_per_second"] for x in rows[1:])
mean = (a + b) / 2
variation = abs(a - b) / mean if mean > 0 else float("inf")
result = {
    "rounds": rows,
    "round_2_3_mean_output_tokens_per_second": mean,
    "relative_difference": variation,
    "stable_below_1_percent": variation < 0.01,
    "difference_formula": "abs(round2-round3)/mean(round2,round3)",
    "note": "Synthetic fixed-output load; length finish is expected. TP2/crop configuration retained; not exact H100 reproduction.",
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["stable_below_1_percent"]:
    raise SystemExit(
        "Rounds 2/3 differ by >=1%; keep results and repeat the case before accepting the mean."
    )
