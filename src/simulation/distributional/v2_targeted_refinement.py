"""High-inner-resample diagnostic for scenario 16 of the V2 simulation.

The script reads a user-supplied formal run and writes only to an explicitly
supplied new or empty output directory. It does not modify frozen results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import v2_mixture_simulation as sim


SCENARIO_ID = 16
FORMAL_INNER_RESAMPLES = 1999
ADAPTIVE_INNER_RESAMPLES = 7999
HISTORICAL_SEED_NAMESPACE = 9132602


def load_formal_inputs(formal_input: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = formal_input.expanduser().resolve()
    counts_path = root / "outer_station_counts.parquet"
    replicates_path = root / f"scenario_{SCENARIO_ID:02d}_replicates.parquet"
    missing = [path.name for path in (counts_path, replicates_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Adaptive refinement requires saved formal outer counts and scenario 16 "
            f"replicates. Missing: {', '.join(missing)}"
        )
    counts = pd.read_parquet(counts_path)
    counts = counts[counts.scenario_id.eq(SCENARIO_ID)]
    saved = pd.read_parquet(replicates_path)
    saved = saved[saved.method.eq("percentile_mc")].set_index(["replicate", "metric"])
    if counts.empty or saved.empty:
        raise ValueError("Formal input has no percentile-MC scenario 16 records")
    return counts, saved


def run(args: argparse.Namespace) -> None:
    counts, saved = load_formal_inputs(args.formal_input)
    out = sim.prepare_output(args.output)
    scenario = sim.scenario_list()[SCENARIO_ID - 1]
    records = []
    prefix_error = 0.0
    for row in counts.itertuples(index=False):
        count = np.array([getattr(row, f"type_{j}") for j in range(6)])
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, SCENARIO_ID, row.replicate, 1])
        )
        boot = rng.multinomial(scenario["n"], count / scenario["n"], size=ADAPTIVE_INNER_RESAMPLES)
        values = sim.stats(boot, scenario["profile"])[0]
        low_prefix, high_prefix = np.quantile(
            values[:FORMAL_INNER_RESAMPLES], [0.025, 0.975], axis=0, method="linear"
        )
        low, high = np.quantile(values, [0.025, 0.975], axis=0, method="linear")
        for j, metric in enumerate(sim.METRICS):
            saved_row = saved.loc[(row.replicate, metric)]
            prefix_error = max(
                prefix_error,
                abs(low_prefix[j] - saved_row.lower),
                abs(high_prefix[j] - saved_row.upper),
            )
            records.append(
                {
                    "replicate": row.replicate,
                    "metric": metric,
                    "truth": saved_row.truth,
                    "truth_unique": saved_row.truth_unique,
                    "lower_1999": saved_row.lower,
                    "upper_1999": saved_row.upper,
                    "lower_7999": low[j],
                    "upper_7999": high[j],
                    "included_1999": bool(
                        (saved_row.lower <= saved_row.truth + sim.TOL)
                        and (saved_row.upper >= saved_row.truth - sim.TOL)
                    ),
                    "included_7999": bool(
                        (low[j] <= saved_row.truth + sim.TOL)
                        and (high[j] >= saved_row.truth - sim.TOL)
                    ),
                }
            )
        if row.replicate % 500 == 0:
            print(f"Adaptive scenario 16: {row.replicate}/{len(counts)}", flush=True)
    if prefix_error >= 1e-10:
        raise AssertionError("The first 1,999 inner resamples do not reproduce the formal intervals")
    frame = pd.DataFrame(records)
    frame.to_parquet(out / "adaptive_scenario16_replicates.parquet", index=False)
    summary = []
    for metric, group in frame.groupby("metric"):
        delta = group.included_7999.to_numpy(dtype=float) - group.included_1999.to_numpy(dtype=float)
        coverage = group.included_7999.mean()
        n = len(group)
        summary.append(
            {
                "scenario_id": SCENARIO_ID,
                "metric": metric,
                "outer_datasets": n,
                "coverage_1999": group.included_1999.mean(),
                "coverage_7999": coverage,
                "coverage_7999_mcse": np.sqrt(coverage * (1 - coverage) / n),
                "paired_coverage_change": delta.mean(),
                "paired_change_mcse": delta.std(ddof=1) / np.sqrt(n),
                "inclusion_changed_count": int((delta != 0).sum()),
                "mean_length_7999": float((group.upper_7999 - group.lower_7999).mean()),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(out / "adaptive_scenario16_summary.csv", index=False)
    sim.write_json(
        out / "adaptive_scenario16_checks.json",
        {
            "status": "PASS",
            "prefix_max_error": prefix_error,
            "outer_datasets": int(len(counts)),
            "inner_resamples": ADAPTIVE_INNER_RESAMPLES,
            "seed_namespace": args.seed,
            "research_status": "Post-result diagnostic; formal outputs remain unchanged.",
        },
    )
    print(summary_frame.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-input",
        type=Path,
        required=True,
        help="Directory containing a compatible formal run with saved parquet replicate files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory for diagnostic outputs.")
    parser.add_argument("--seed", type=int, default=HISTORICAL_SEED_NAMESPACE)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
