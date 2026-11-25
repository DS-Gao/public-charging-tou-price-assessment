"""Known-truth finite-station-type simulation for A3 quantile functionals.

This public version writes only to an explicitly supplied empty output
directory. It never writes into the preserved formal-result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parents[2]
FROZEN_FORMAL_DIR = (
    PACKAGE_ROOT
    / "data"
    / "reference"
    / "historical"
    / "window_and_mixture_validation"
    / "v2"
    / "formal"
)
Y = np.array([0.0, 0.003, 0.006, 1.5, 6.0, 7.2])
R = np.array([False, False, False, True, True, True])
PU = np.array([0.2, 0.5, 0.3])
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
METRICS = [
    "m00",
    "m10",
    "m01",
    "m11",
    "gap",
    "composition",
    "within",
    "composition_minus_within",
]
TOL = 1e-12


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    frozen = FROZEN_FORMAL_DIR.resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError("--output must not be the preserved formal-result directory or a child of it")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("--output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def probability(p: float) -> np.ndarray:
    return np.r_[(1 - p) * PU, p * PR]


def scenario_list() -> list[dict]:
    minus, plus = PROFILES["opposed_sides"]
    a = (minus[R] @ PR) / (minus[~R] @ PU)
    b = (plus[R] @ PR) / (plus[~R] @ PU)
    critical = 1 / (1 + np.sqrt(a * b))
    result = []
    for profile in PROFILES:
        for n in [175, 701]:
            for j, p in enumerate([0.45, 0.49, 0.50, 0.51, 0.55, critical]):
                result.append(
                    {
                        "scenario_id": len(result) + 1,
                        "profile": profile,
                        "n": n,
                        "p_R": float(p),
                        "p_label": f"{p:.2f}" if j < 5 else "activity_critical",
                        "critical_p_R_opposed": float(critical),
                    }
                )
    return result


def normalize(array: np.ndarray) -> np.ndarray:
    return array / array.sum(axis=-1, keepdims=True)


def hybrid_weights(counts: np.ndarray, profile: str):
    counts = np.atleast_2d(counts).astype(float)
    minus, plus = PROFILES[profile]
    st = normalize(counts)
    target = 0.5 * (normalize(counts * minus) + normalize(counts * plus))
    group_st = st[:, R].sum(axis=1)
    group_target = target[:, R].sum(axis=1)
    m10 = np.zeros_like(st)
    m01 = np.zeros_like(st)
    for mask in [R, ~R]:
        ps = st[:, mask].sum(axis=1, keepdims=True)
        pt = target[:, mask].sum(axis=1, keepdims=True)
        m10[:, mask] = np.divide(
            st[:, mask] * pt, ps, out=np.zeros_like(st[:, mask]), where=ps > 0
        )
        m01[:, mask] = np.divide(
            target[:, mask] * ps, pt, out=np.zeros_like(st[:, mask]), where=pt > 0
        )
    empty = (group_st <= 0) | (group_st >= 1)
    return np.stack([st, m10, m01, target], axis=1), empty, group_st, group_target


def quantiles(weights: np.ndarray):
    """Return median midpoint and median-set endpoints on positive type support."""
    shape = weights.shape[:-1]
    w = normalize(weights.reshape(-1, len(Y)))
    cumulative = np.cumsum(w, axis=1)
    hits = (np.abs(cumulative - 0.5) <= TOL) & (w > 0)
    has_hit = hits.any(axis=1)
    first = np.argmax(cumulative >= 0.5, axis=1)
    hit_index = np.argmax(hits, axis=1)
    lower = np.where(has_hit, hit_index, first)
    next_positive = (np.arange(len(Y))[None, :] > lower[:, None]) & (w > 0)
    upper = np.where(has_hit, np.argmax(next_positive, axis=1), lower)
    assert np.all(~has_hit | next_positive.any(axis=1))
    lo = Y[lower].reshape(shape)
    hi = Y[upper].reshape(shape)
    return (lo + hi) / 2, lo, hi


def stats(counts: np.ndarray, profile: str):
    weights, empty, ps, pt = hybrid_weights(counts, profile)
    med, lo, hi = quantiles(weights)
    m00, m10, m01, m11 = med.T
    gap = m11 - m00
    composition = 0.5 * (m10 - m00 + m11 - m01)
    composition = np.where(empty, 0.0, composition)
    within = gap - composition
    metrics = np.column_stack([med, gap, composition, within, composition - within])
    closure = np.max(np.abs(metrics[:, 5] + metrics[:, 6] - metrics[:, 4]))
    assert closure < 1e-10
    return metrics, lo, hi, empty, ps, pt


def scalar_truth(p: float, profile: str):
    """Independent small-support scalar formula, without production helpers."""
    masses = probability(p)
    minus, plus = PROFILES[profile]
    exposure = [
        0.5
        * (
            masses[i] * minus[i] / sum(masses * minus)
            + masses[i] * plus[i] / sum(masses * plus)
        )
        for i in range(6)
    ]
    distributions = []
    endpoints = []
    for g_source, within_source in [
        (masses, masses),
        (exposure, masses),
        (masses, exposure),
        (exposure, exposure),
    ]:
        output = []
        for i in range(6):
            members = range(3) if i < 3 else range(3, 6)
            output.append(
                sum(g_source[k] for k in members)
                * within_source[i]
                / sum(within_source[k] for k in members)
            )
        cumulative = 0.0
        for i, value in enumerate(Y):
            cumulative += output[i]
            if abs(cumulative - 0.5) <= TOL:
                pair = (float(value), float(Y[i + 1]))
                break
            if cumulative > 0.5:
                pair = (float(value), float(value))
                break
        endpoints.append(pair)
        distributions.append(sum(pair) / 2)
    a, b, c, d = distributions
    gap = d - a
    composition = (b - a + d - c) / 2
    within = (c - a + d - b) / 2
    return np.array(distributions + [gap, composition, within, composition - within]), np.array(endpoints)


def exact_st_intervals(counts: np.ndarray, n: int):
    if n < 7 or n % 2 == 0:
        raise ValueError("This finite-endpoint benchmark requires an odd sample size >= 7.")
    counts = np.atleast_2d(counts)
    cumulative = np.cumsum(counts, axis=1)
    k = int(binom.ppf(0.025, n, 0.5))
    while binom.cdf(k, n, 0.5) <= 0.025:
        k += 1
    while binom.cdf(k - 1, n, 0.5) > 0.025:
        k -= 1
    sign_lo = Y[np.argmax(cumulative >= k, axis=1)]
    sign_hi = Y[np.argmax(cumulative >= n - k + 1, axis=1)]
    bootstrap_cdf = binom.sf(n // 2, n, np.clip(cumulative / n, 0, 1))
    exact_lo = Y[np.argmax(bootstrap_cdf >= 0.025, axis=1)]
    exact_hi = Y[np.argmax(bootstrap_cdf >= 0.975, axis=1)]
    return (
        np.column_stack([sign_lo, sign_hi]),
        np.column_stack([exact_lo, exact_hi]),
        k,
        float(1 - 2 * binom.cdf(k - 1, n, 0.5)),
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, z in frame.groupby(["scenario_id", "method", "metric"], sort=True):
        truth = float(z.truth.iloc[0])
        error = z.estimate.to_numpy() - truth
        covered = (z.lower.to_numpy() <= truth + TOL) & (z.upper.to_numpy() >= truth - TOL)
        length = z.upper.to_numpy() - z.lower.to_numpy()
        record = {
            key: z[key].iloc[0]
            for key in ["scenario_id", "profile", "n", "p_R", "p_label", "method", "metric", "truth", "truth_unique"]
        }
        count = len(z)
        coverage = float(covered.mean())
        wilson_center = (coverage + 1.96**2 / (2 * count)) / (1 + 1.96**2 / count)
        wilson_half = 1.96 * np.sqrt(coverage * (1 - coverage) / count + 1.96**2 / (4 * count**2)) / (1 + 1.96**2 / count)
        record.update(
            {
                "replicates": count,
                "bias": float(error.mean()),
                "bias_mcse": float(error.std(ddof=1) / np.sqrt(count)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "coverage": coverage,
                "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / count)),
                "coverage_mc_interval_low": wilson_center - wilson_half,
                "coverage_mc_interval_high": wilson_center + wilson_half,
                "mean_interval_length": float(length.mean()),
                "length_mcse": float(length.std(ddof=1) / np.sqrt(count)),
                "zero_length_fraction": float((length <= TOL).mean()),
            }
        )
        if z.metric.iloc[0] in METRICS[:4]:
            a = float(z.median_set_lower.iloc[0])
            b = float(z.median_set_upper.iloc[0])
            estimate = z.estimate.to_numpy()
            record.update(
                {
                    "median_set_lower": a,
                    "median_set_upper": b,
                    "mean_distance_to_median_set": float(np.mean(np.maximum(a - estimate, 0) + np.maximum(estimate - b, 0))),
                    "median_set_intersection_coverage": float(((z.lower <= b + TOL) & (z.upper >= a - TOL)).mean()),
                    "entire_median_set_coverage": float(((z.lower <= a + TOL) & (z.upper >= b - TOL)).mean()),
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    out = prepare_output(args.output)
    design_path = args.design.expanduser().resolve()
    if not design_path.is_file():
        raise FileNotFoundError(f"Design file not found: {design_path}")
    spec = scenario_list()
    pd.DataFrame(spec).to_csv(out / "scenarios.csv", index=False)
    design_hash = hashlib.sha256(design_path.read_bytes()).hexdigest()
    manifest = {
        "status": "RUNNING",
        "phase": args.phase,
        "outer_replicates": args.replicates,
        "inner_replicates": args.bootstrap,
        "seed_namespace": args.seed,
        "scenario_count": len(spec),
        "design_sha256": design_hash,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resampling": "iid station types, one layer, all observations supported",
    }
    write_json(out / "simulation_manifest.json", manifest)
    truth_rows = []
    all_frames = []
    progress = []
    counts_records = []
    total_empty = 0
    largest_truth_error = 0.0
    for s in spec:
        sid = s["scenario_id"]
        n = s["n"]
        profile = s["profile"]
        p = s["p_R"]
        truth, truth_lo, truth_hi, _, _, _ = stats(probability(p), profile)
        truth = truth[0]
        truth_lo = truth_lo[0]
        truth_hi = truth_hi[0]
        check, endpoints = scalar_truth(p, profile)
        largest_truth_error = max(
            largest_truth_error,
            float(np.max(abs(check - truth))),
            float(np.max(abs(endpoints[:, 0] - truth_lo))),
            float(np.max(abs(endpoints[:, 1] - truth_hi))),
        )
        assert largest_truth_error < 1e-10
        unique = np.r_[
            abs(truth_hi - truth_lo) < TOL,
            (abs(truth_hi[0] - truth_lo[0]) < TOL) & (abs(truth_hi[3] - truth_lo[3]) < TOL),
            np.repeat(np.all(abs(truth_hi - truth_lo) < TOL), 3),
        ]
        for j, metric in enumerate(METRICS):
            truth_rows.append(
                {
                    **s,
                    "metric": metric,
                    "truth": truth[j],
                    "truth_unique": bool(unique[j]),
                    "median_set_lower": truth_lo[j] if j < 4 else None,
                    "median_set_upper": truth_hi[j] if j < 4 else None,
                }
            )
        outer_rng = np.random.default_rng(np.random.SeedSequence([args.seed, sid, 0]))
        counts = outer_rng.multinomial(n, probability(p), size=args.replicates)
        count_frame = pd.DataFrame(counts, columns=[f"type_{i}" for i in range(6)])
        count_frame["scenario_id"] = sid
        count_frame["replicate"] = np.arange(1, args.replicates + 1)
        counts_records.append(count_frame)
        estimates = stats(counts, profile)[0]
        lower = np.empty_like(estimates)
        upper = np.empty_like(estimates)
        for i, count in enumerate(counts):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, sid, i + 1, 1]))
            boot = rng.multinomial(n, count / n, size=args.bootstrap)
            boot_stats, _, _, empty, _, _ = stats(boot, profile)
            total_empty += int(empty.sum())
            lower[i], upper[i] = np.quantile(boot_stats, [0.025, 0.975], axis=0, method="linear")
        sign, exact, k, bound = exact_st_intervals(counts, n)
        frames = []
        for j, metric in enumerate(METRICS):
            frames.append(
                pd.DataFrame(
                    {
                        "replicate": np.arange(1, args.replicates + 1),
                        "metric": metric,
                        "method": "percentile_mc",
                        "estimate": estimates[:, j],
                        "lower": lower[:, j],
                        "upper": upper[:, j],
                        "truth": truth[j],
                        "truth_unique": bool(unique[j]),
                        "median_set_lower": truth_lo[j] if j < 4 else np.nan,
                        "median_set_upper": truth_hi[j] if j < 4 else np.nan,
                    }
                )
            )
        for method, interval in [("sign_order_statistic", sign), ("percentile_exact_ST", exact)]:
            z = frames[0].copy()
            z["method"] = method
            z["lower"] = interval[:, 0]
            z["upper"] = interval[:, 1]
            frames.append(z)
        frame = pd.concat(frames, ignore_index=True)
        for key, value in s.items():
            frame[key] = value
        assert np.isfinite(frame[["estimate", "lower", "upper", "truth"]].to_numpy()).all()
        assert (frame.lower <= frame.upper + TOL).all()
        frame.to_parquet(out / f"scenario_{sid:02d}_replicates.parquet", index=False)
        all_frames.append(frame)
        elapsed = time.perf_counter() - start
        progress.append({**s, "status": "COMPLETE", "elapsed_seconds": elapsed, "sign_k": k, "sign_coverage_lower_bound": bound})
        pd.DataFrame(progress).to_csv(out / "progress.csv", index=False)
        print(f"{args.phase}: scenario {sid}/{len(spec)}; n={n}, p_R={p:.6f}, {profile}; elapsed {elapsed:.1f}s", flush=True)
    frame = pd.concat(all_frames, ignore_index=True)
    summarize(frame).to_csv(out / "performance_summary.csv", index=False)
    pd.DataFrame(truth_rows).to_csv(out / "population_truth.csv", index=False)
    pd.concat(counts_records, ignore_index=True).to_parquet(out / "outer_station_counts.parquet", index=False)
    manifest.update(
        {
            "status": "COMPLETE",
            "outer_datasets": len(spec) * args.replicates,
            "metric_method_rows": len(frame),
            "runtime_seconds": time.perf_counter() - start,
            "nonfinite_or_failed_outer_datasets": 0,
            "single_group_inner_draw_count": total_empty,
            "max_independent_truth_error": largest_truth_error,
            "note": "Performance is not a pass criterion. Nonunique targets are midpoint-convention results; see median-set metrics.",
        }
    )
    write_json(out / "simulation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory for generated results.")
    parser.add_argument("--design", type=Path, default=CODE_DIR / "simulation_design.md")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.replicates < 2 or arguments.bootstrap < 99:
        raise ValueError("--replicates must be >= 2 and --bootstrap must be >= 99")
    run(arguments)
