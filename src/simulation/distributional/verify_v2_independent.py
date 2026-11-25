"""Independent compact checks for the V2 known-truth simulation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

import v2_mixture_simulation as sim


CODE_DIR = Path(__file__).resolve().parent
PRODUCTION = CODE_DIR / "v2_mixture_simulation.py"
Y = np.array([0.0, 0.003, 0.006, 1.5, 6.0, 7.2])
R = np.array([False, False, False, True, True, True])
PU = np.array([0.20, 0.50, 0.30])
PR = np.array([0.03, 0.22, 0.75])
PROFILES = {
    "opposed_sides": (
        np.array([1.0, 2.0, 4.0, 3.0, 2.0, 1.0]),
        np.array([4.0, 2.0, 1.0, 1.0, 2.0, 4.0]),
    ),
    "aligned_sides": (
        np.array([1.0, 2.0, 4.0, 1.4, 2.8, 5.6]),
        np.array([1.0, 2.0, 4.0, 1.4, 2.8, 5.6]),
    ),
}
TOL = 1e-10
MEDIAN_TOL = 1e-12


def load_production():
    spec = importlib.util.spec_from_file_location("v2_production", PRODUCTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the production simulation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probability_ref(p: float) -> np.ndarray:
    return np.r_[((1.0 - p) * PU), (p * PR)]


def normalize_ref(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    return values / total


def hybrid_weights_ref(masses: np.ndarray, profile: str) -> np.ndarray:
    masses = normalize_ref(np.asarray(masses, dtype=float))
    minus, plus = PROFILES[profile]
    target = 0.5 * (normalize_ref(masses * minus) + normalize_ref(masses * plus))
    m10 = np.zeros(6, dtype=float)
    m01 = np.zeros(6, dtype=float)
    for mask in (R, ~R):
        station_mass = masses[mask].sum()
        target_mass = target[mask].sum()
        if station_mass > 0:
            m10[mask] = masses[mask] * target_mass / station_mass
        if target_mass > 0:
            m01[mask] = target[mask] * station_mass / target_mass
    return np.vstack([masses, m10, m01, target])


def median_ref(weights: np.ndarray) -> tuple[float, float, float]:
    weights = normalize_ref(weights)
    cumulative = np.cumsum(weights)
    hits = [
        i
        for i, (mass, cum) in enumerate(zip(weights, cumulative))
        if mass > 0 and abs(cum - 0.5) <= MEDIAN_TOL
    ]
    if hits:
        index = hits[0]
        following = [j for j in range(index + 1, len(Y)) if weights[j] > 0]
        if not following:
            raise AssertionError("exact half hit has no upper support")
        lo, hi = Y[index], Y[following[0]]
    else:
        index = int(np.flatnonzero(cumulative >= 0.5)[0])
        lo = hi = Y[index]
    return float((lo + hi) / 2.0), float(lo), float(hi)


def truth_ref(p: float, profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = hybrid_weights_ref(probability_ref(p), profile)
    medians = []
    lower = []
    upper = []
    for row in weights:
        median, lo, hi = median_ref(row)
        medians.append(median)
        lower.append(lo)
        upper.append(hi)
    m00, m10, m01, m11 = medians
    gap = m11 - m00
    composition = 0.5 * ((m10 - m00) + (m11 - m01))
    within = 0.5 * ((m01 - m00) + (m11 - m10))
    values = np.array([m00, m10, m01, m11, gap, composition, within, composition - within])
    return values, np.array(lower), np.array(upper)


def critical_mass_ref() -> float:
    minus, plus = PROFILES["opposed_sides"]
    a = float(minus[R] @ PR) / float(minus[~R] @ PU)
    b = float(plus[R] @ PR) / float(plus[~R] @ PU)
    return 1.0 / (1.0 + math.sqrt(a * b))


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def exhaustive_bootstrap_percentiles(counts: np.ndarray, n: int) -> tuple[float, float]:
    counts = np.asarray(counts, dtype=int)
    empirical = counts / float(n)
    mass_by_value = {float(value): 0.0 for value in Y}
    factorial_n = math.factorial(n)
    for composition in compositions(n, len(Y)):
        probability = factorial_n
        for value in composition:
            probability /= math.factorial(value)
        for observed, count in zip(empirical, composition):
            probability *= observed**count
        if probability == 0.0:
            continue
        cumulative = np.cumsum(composition)
        index = int(np.flatnonzero(cumulative >= n // 2 + 1)[0])
        mass_by_value[float(Y[index])] += probability
    running = 0.0
    lower = None
    upper = None
    for value in Y:
        running += mass_by_value[float(value)]
        if lower is None and running >= 0.025 - 1e-15:
            lower = float(value)
        if upper is None and running >= 0.975 - 1e-15:
            upper = float(value)
    if lower is None or upper is None:
        raise AssertionError("exact bootstrap CDF did not reach target quantile")
    return lower, upper


def check_truth_and_zero_counts(prod) -> dict:
    max_scalar_error = 0.0
    max_endpoint_error = 0.0
    max_stats_error = 0.0
    for profile in PROFILES:
        for p in [0.45, 0.49, 0.50, 0.51, 0.55, critical_mass_ref()]:
            ref, ref_lo, ref_hi = truth_ref(p, profile)
            scalar, scalar_endpoints = prod.scalar_truth(p, profile)
            values, lo, hi, empty, _, _ = prod.stats(np.asarray([probability_ref(p)]), profile)
            max_scalar_error = max(max_scalar_error, float(np.max(np.abs(scalar - ref))))
            max_endpoint_error = max(
                max_endpoint_error,
                float(np.max(np.abs(scalar_endpoints[:, 0] - ref_lo))),
                float(np.max(np.abs(scalar_endpoints[:, 1] - ref_hi))),
            )
            max_stats_error = max(max_stats_error, float(np.max(np.abs(values[0] - ref))))
            if bool(empty[0]):
                raise AssertionError("positive-probability truth incorrectly marked empty")
    for weights in [
        np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0]),
        np.array([0.0, 0.4, 0.0, 0.6, 0.0, 0.0]),
    ]:
        ref = np.array(median_ref(weights))
        got = np.array(prod.quantiles(weights), dtype=float).reshape(-1)
        if not np.allclose(got, ref, atol=TOL, rtol=0.0):
            raise AssertionError("zero-count quantile mismatch")
    return {
        "max_scalar_truth_abs_error": max_scalar_error,
        "max_truth_endpoint_abs_error": max_endpoint_error,
        "max_stats_truth_abs_error": max_stats_error,
    }


def check_critical_mass(prod) -> dict:
    critical = critical_mass_ref()
    scenarios = prod.scenario_list()
    observed = sorted({float(row["critical_p_R_opposed"]) for row in scenarios})
    scenario_p = [float(row["p_R"]) for row in scenarios if row["p_label"] == "activity_critical"]
    if len(observed) != 1 or abs(observed[0] - critical) > TOL:
        raise AssertionError("critical mass mismatch")
    if len(scenario_p) != 4 or any(abs(value - critical) > TOL for value in scenario_p):
        raise AssertionError("critical scenarios mismatch")
    return {"critical_p_R": critical, "scenario_count": len(scenarios), "critical_scenario_count": len(scenario_p)}


def check_exhaustive_small_n(prod) -> dict:
    counts = np.array([2, 1, 0, 1, 2, 1], dtype=int)
    n = int(counts.sum())
    sign, exact, k, bound = prod.exact_st_intervals(np.atleast_2d(counts), n)
    reference = exhaustive_bootstrap_percentiles(counts, n)
    if not np.allclose(exact[0], reference, atol=TOL, rtol=0.0):
        raise AssertionError("small-n exact bootstrap mismatch")
    return {
        "n": n,
        "sign_k": int(k),
        "sign_coverage_lower_bound": float(bound),
        "exact_bootstrap_interval": exact[0].tolist(),
    }


def check_frozen_summaries(formal_dir: Path, prod) -> dict:
    root = formal_dir.expanduser().resolve()
    required = ["scenarios.csv", "population_truth.csv", "performance_summary.csv", "adaptive_scenario16_summary.csv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen public summary files: " + ", ".join(missing))
    scenarios = pd.read_csv(root / "scenarios.csv")
    truth = pd.read_csv(root / "population_truth.csv")
    summary = pd.read_csv(root / "performance_summary.csv")
    adaptive = pd.read_csv(root / "adaptive_scenario16_summary.csv")
    expected = pd.DataFrame(prod.scenario_list()).sort_values("scenario_id").reset_index(drop=True)
    observed = scenarios.sort_values("scenario_id").reset_index(drop=True)
    if len(observed) != 24 or not np.allclose(observed["p_R"], expected["p_R"], atol=TOL, rtol=0.0):
        raise AssertionError("frozen scenario design does not match the implementation")
    maximum_truth_error = 0.0
    for scenario in prod.scenario_list():
        values, lower, upper = truth_ref(float(scenario["p_R"]), scenario["profile"])
        rows = truth[truth["scenario_id"].eq(scenario["scenario_id"])].set_index("metric")
        for index, metric in enumerate(sim.METRICS):
            if metric not in rows.index:
                raise AssertionError("frozen truth is missing a metric")
            maximum_truth_error = max(maximum_truth_error, abs(float(rows.loc[metric, "truth"]) - values[index]))
            if index < 4:
                maximum_truth_error = max(
                    maximum_truth_error,
                    abs(float(rows.loc[metric, "median_set_lower"]) - lower[index]),
                    abs(float(rows.loc[metric, "median_set_upper"]) - upper[index]),
                )
    selected = summary[(summary["method"] == "percentile_mc") & (summary["metric"] == "within")]
    if len(selected) != 24 or not (selected["replicates"] == 2000).all():
        raise AssertionError("frozen within-summary rows are incomplete")
    adaptive_within = adaptive[(adaptive["scenario_id"] == 16) & (adaptive["metric"] == "within")]
    if len(adaptive_within) != 1:
        raise AssertionError("frozen adaptive scenario 16 summary is incomplete")
    scenario16_path = root / "scenario_16_replicates.parquet"
    if not scenario16_path.is_file():
        raise FileNotFoundError("Missing saved formal scenario 16 replicate parquet")
    scenario16 = pd.read_parquet(scenario16_path)
    percentile = scenario16[scenario16["method"] == "percentile_mc"].copy()
    expected_metrics = [
        "m00",
        "m10",
        "m01",
        "m11",
        "gap",
        "composition",
        "within",
        "composition_minus_within",
    ]
    if set(percentile["metric"]) != set(expected_metrics):
        raise AssertionError("scenario 16 percentile replicate metrics are incomplete")
    wide = percentile.pivot(index="replicate", columns="metric", values="estimate")
    closure = wide["composition"] + wide["within"] - wide["gap"]
    scenario = prod.scenario_list()[15]
    reference, _, _ = truth_ref(float(scenario["p_R"]), scenario["profile"])
    stored_truth = percentile.groupby("metric")["truth"].first()
    truth_error = max(
        abs(float(stored_truth[metric]) - reference[index])
        for index, metric in enumerate(expected_metrics)
    )
    if len(wide) != 2000:
        raise AssertionError("scenario 16 does not contain 2,000 percentile replicates")
    if float(np.max(np.abs(closure))) > 1e-10:
        raise AssertionError("scenario 16 decomposition does not close")
    if truth_error > 1e-10:
        raise AssertionError("scenario 16 truth differs from the independent reference")
    if not np.isfinite(percentile[["estimate", "lower", "upper", "truth"]].to_numpy()).all():
        raise AssertionError("scenario 16 contains nonfinite values")
    if not (percentile["lower"] <= percentile["upper"] + TOL).all():
        raise AssertionError("scenario 16 has unordered interval endpoints")
    return {
        "frozen_scenario_count": int(len(observed)),
        "frozen_truth_rows": int(len(truth)),
        "frozen_performance_rows": int(len(summary)),
        "max_truth_error_vs_independent_reference": float(maximum_truth_error),
        "scenario16_within_coverage_1999": float(adaptive_within.iloc[0]["coverage_1999"]),
        "scenario16_within_coverage_7999": float(adaptive_within.iloc[0]["coverage_7999"]),
        "scenario16_replicate_check": {
            "status": "CHECKED",
            "percentile_replicates": int(len(wide)),
            "max_estimate_decomposition_closure_abs": float(np.max(np.abs(closure))),
            "max_truth_error_vs_independent_reference": float(truth_error),
        },
    }


def write_report(path: Path, report: dict) -> None:
    output = path.expanduser().resolve()
    frozen = sim.FROZEN_FORMAL_DIR.resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError("--output must not write into the preserved formal-result directory")
    if output.exists():
        raise FileExistsError("--output must name a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, default=sim.FROZEN_FORMAL_DIR)
    parser.add_argument("--output", type=Path, help="Optional new JSON report path.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    production = load_production()
    report = {
        "status": "PASS",
        "truth_and_zero_count_checks": check_truth_and_zero_counts(production),
        "critical_truth_check": check_critical_mass(production),
        "exhaustive_small_n_check": check_exhaustive_small_n(production),
        "frozen_summary_check": check_frozen_summaries(args.formal_dir, production),
    }
    if args.output:
        write_report(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
