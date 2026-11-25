"""Lightweight point summaries only. No calibration fit, bootstrap or simulation.

Run from any directory: python scripts/build_a1_a2.py --output PATH
Default output is this evidence package. Frozen inputs and original functions
are resolved relative to this script; no historical project path is required.

To render only the paired table from the frozen overall CSV, without
recalculating or writing any scientific result:
python scripts/build_a1_a2.py --export-a2-main-paired-only
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "9792047edc886e66f7c40cf237c003bc5998193a735bd75661a076c3d9c2145d"
METRICS = ["mae", "me", "p90_abs_error", "label_disagreement"]
CONFIGS = ["E", "CAL", "E_plus_S", "LOCAL"]
GROUPS = ["all_local", "complete", "additional_local"]
GROUP_LABELS = {"all_local": r"All $A_f$", "complete": r"Complete $C_f$",
                "additional_local": r"Added $A_f\setminus C_f$"}
PAIRS = [("E", "LOCAL"), ("E", "E_plus_S"), ("E_plus_S", "LOCAL"), ("CAL", "LOCAL")]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(x, digits=6):
    if x == 0:
        return "0"
    if abs(x) < 0.5 * 10 ** (-digits):
        m, e = f"{x:.4e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{x:.{digits}f}"


def _paired_table_tex(rows):
    """Render the five-column paired table from already-frozen summary rows."""
    expected = [(source, target) for source, target in PAIRS]
    keyed = {(r["source"], r["target"]): r for r in rows}
    assert list(keyed) == expected, "Frozen paired summary has an unexpected comparison order."
    assert len(rows) == len(expected)
    eval_ns = {int(r["eval_n"]) for r in rows}
    layer_ns = {int(r["n_layers"]) for r in rows}
    assert eval_ns == {5846} and layer_ns == {10}, (eval_ns, layer_ns)
    base = float(keyed[("E", "LOCAL")]["mean_paired_abs_loss_reduction"])
    assert base > 0

    def display_name(name):
        return name.replace("E_plus_S", "E+S")

    body = []
    for source, target in expected:
        r = keyed[(source, target)]
        net = float(r["mean_paired_abs_loss_reduction"])
        mean_abs = float(r["mean_abs_paired_loss_difference"])
        signs = f'{int(r["positive_layers"])}/{int(r["negative_layers"])}/{int(r["zero_layers"])}'
        relative = 100 * net / base
        body.append(" & ".join([
            display_name(source) + r"$\to$" + display_name(target),
            fmt(net, 8),
            fmt(mean_abs, 8),
            signs,
            f"{relative:.3f}",
        ]) + r" \\")

    header = (r"\shortstack[l]{Input\\comparison} & "
              r"\shortstack[r]{Net paired\\loss reduction\\(RMB/kWh)} & "
              r"\shortstack[r]{Mean absolute\\paired loss\\difference\\(RMB/kWh)} & "
              r"\shortstack{Positive /\\negative /\\zero layers} & "
              r"\shortstack[r]{Relative to\\E$\to$LOCAL\\(\%)}")
    note = (r"All four comparisons use the same subsequent observed total-price target "
            r"$Y=T^{*,v}$ on $5{,}846$ evaluation units. Stations receive equal ST weights "
            r"within each layer and the ten layers are weighted equally. For $A\to B$, "
            r"$d_i=|e_i^A|-|e_i^B|$; positive values favour the second input. The mean "
            r"absolute paired loss difference is the within-layer ST-weighted mean of "
            r"$|d_i|$, followed by the equal ten-layer mean. Relative to E$\to$LOCAL is "
            r"each net reduction divided by the unrounded E$\to$LOCAL net reduction. "
            r"E$\to$E+S and E+S$\to$LOCAL are consecutive comparisons, whereas "
            r"CAL$\to$LOCAL is an independent reference comparison. No new confidence "
            r"intervals are estimated; the existing E$\to$LOCAL interval remains in "
            r"Table~\ref{tab:r42-crosswindow}.")
    return (r"\begin{table}[htbp]\centering" + "\n"
            + r"\begingroup\linespread{1}\fontsize{9.5}{11.4}\selectfont" + "\n"
            + r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.12}" + "\n"
            + r"\caption{Paired loss reductions and layer-level heterogeneity across historical inputs.}" + "\n"
            + r"\label{tab:r420-a2-paired}" + "\n"
            + r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X>{\raggedleft\arraybackslash}X>{\raggedleft\arraybackslash}X>{\centering\arraybackslash}X>{\raggedleft\arraybackslash}X@{}}" + "\n"
            + r"\toprule " + header + r"\\\midrule" + "\n"
            + "\n".join(body) + "\n"
            + r"\bottomrule\end{tabularx}" + "\n"
            + r"\SharedNote{" + note + "}" + "\n"
            + r"\endgroup\end{table}" + "\n")


def write_paired_table_from_frozen_summary(summary_path, output_path):
    """Write only the paired table from frozen CSV; do not recalculate or write CSVs."""
    summary_path = Path(summary_path)
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_paired_table_tex(rows), encoding="utf-8")
    return output_path


def layer_label(r):
    return ("Jan" if r["month"] == "2025-01" else "Jul") + f" {int(r['boundary_hour']):02d}:00"


def table(caption, label, columns, header, body, note, long=False):
    if long:
        return (r"\begingroup\CompactTableBodyStyle\setlength{\tabcolsep}{4pt}" + "\n"
                + r"\begin{longtable}{" + columns + "}\n"
                + rf"\caption{{{caption}}}\label{{{label}}}\\" + "\n"
                + r"\toprule " + header + r"\\\midrule\endfirsthead" + "\n"
                + rf"\multicolumn{{{header.count('&')+1}}}{{l}}{{\textit{{Continued}}}}\\" + "\n"
                + r"\toprule " + header + r"\\\midrule\endhead" + "\n"
                + body + "\n" + r"\bottomrule\end{longtable}\endgroup" + "\n"
                + r"\SharedNote{" + note + "}\n")
    return (r"\begin{table}[htbp]\centering\CompactTableBodyStyle" + "\n"
            + r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.12}" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
            + r"\begin{tabular}{" + columns + "}\n"
            + r"\toprule " + header + r"\\\midrule" + "\n"
            + body + "\n" + r"\bottomrule\end{tabular}" + "\n"
            + r"\SharedNote{" + note + "}\n" + r"\end{table}" + "\n")


def make_tables(out, a1_over, a1_layer, a2_over, a2_layer, paired_over, paired_layer):
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    note1 = (r"Counts are station--month--boundary units. Availability is $N_{\rm obs}/N_f$ "
             r"using pooled counts. Errors use each group's intersection with the unchanged $\mathcal V$; "
             r"stations are equally weighted within each group--layer and all ten layers receive equal weight. "
             r"$\Delta={\rm MAE}_{E}-{\rm MAE}_{LOCAL}$ is a paired absolute-loss reduction. "
             r"Errors are in RMB/kWh. P90 is the mean of ten layer P90s. "
             r"These revision-added comparisons are descriptive; intervals were not estimated. "
             r"The complete and added groups are different observed populations.")
    def body1(rows, group=True):
        return "\n".join(" & ".join([
            GROUP_LABELS[r["group"]] if group else layer_label(r),
            str(r["entry_n"]), str(r["eval_n"]), f'{100*r["observability"]:.2f}',
            fmt(r["E_mae"]), fmt(r["LOCAL_mae"]), fmt(r["E_to_LOCAL_gain"]),
            fmt(r["LOCAL_p90_abs_error"])]) + r" \\" for r in rows)
    header1 = r"Group & $N_f$ & $N_{\rm obs}$ & Avail. (\%) & E MAE & LOCAL MAE & $\Delta$ & P90"
    (tables / "A1_main_support.tex").write_text(table(
        "Formation support, later observability and conditional performance.",
        "epkg:a1:main", "@{}lrrrrrrr@{}", header1, body1(a1_over), note1), encoding="utf-8")
    text = []
    for group in GROUPS:
        rr = [r for r in a1_layer if r["group"] == group]
        text.append(table("Layer details: " + GROUP_LABELS[group] + ".",
                          "epkg:a1:layers:" + group, "@{}lrrrrrrr@{}",
                          header1.replace("Group", "Layer"),
                          body1(rr, False), note1.replace(
                              "P90 is the mean of ten layer P90s.", "P90 is computed within the displayed layer.")
                          + r" Missing later targets are not assigned errors."))
    (tables / "A1_SI1_layers.tex").write_text("\n".join(text), encoding="utf-8")
    note2 = (r"All inputs use the original $Y=T^{*,v}$ and $\mathcal V$ ($n=5{,}846$); "
             r"weights are equal within each layer and then equal over ten layers. "
             r"E+S is $E^{*,f}+S^{*,f}$, checked against the stored component-sum contrast; "
             r"LOCAL is $T^{*,f}$. CAL retains its original formation-only leave-one-station-out adjustment. "
             r"ME is input minus target. Monetary errors are in RMB/kWh. "
             r"P90 is the mean of layer P90s. This is a descriptive point panel; no intervals were estimated "
             r"for E+S or its paired comparisons. Existing E/CAL/LOCAL ranges remain in the original table.")
    body2 = "\n".join(" & ".join([
        r["configuration"].replace("E_plus_S", "E+S"), fmt(r["mae"]), fmt(r["me"]),
        fmt(r["p90_abs_error"]), f'{100*r["label_disagreement"]:.4f}']) + r" \\" for r in a2_over)
    (tables / "A2_main_inputs.tex").write_text(table(
        "Descriptive comparison of historical input content.",
        "epkg:a2:main", "@{}lrrrr@{}",
        r"Input & MAE & ME & Mean layer P90 & Label disag. (\%)", body2,
        note2 + r" Label disagreement uses the original 0.03 RMB/kWh tolerance."), encoding="utf-8")
    (tables / "A2_main_paired.tex").write_text(_paired_table_tex(paired_over), encoding="utf-8")
    body4 = "\n".join(" & ".join([
        layer_label(r), r["configuration"].replace("E_plus_S", "E+S"), str(r["eval_n"]),
        fmt(r["mae"]), fmt(r["me"]), fmt(r["p90_abs_error"]),
        f'{100*r["label_disagreement"]:.4f}']) + r" \\" for r in a2_layer)
    (tables / "A2_SI1_layers.tex").write_text(table(
        "Layer-specific historical input performance, with the added E+S input.",
        "epkg:a2:layers", "@{}llrrrrr@{}",
        r"Layer & Input & $n$ & MAE & ME & P90 & Label disag. (\%)", body4,
        note2.replace("P90 is the mean of layer P90s.", "P90 is the displayed layer's quantile."),
        long=True), encoding="utf-8")
    paired_wide = []
    for layer in sorted({r["layer_order"] for r in paired_layer}):
        rr = [r for r in paired_layer if r["layer_order"] == layer]
        values = {(r["source"], r["target"]): r["mean_paired_abs_loss_reduction"] for r in rr}
        paired_wide.append(" & ".join([layer_label(rr[0]), str(rr[0]["eval_n"]),
                            *[fmt(values[p], 8) for p in PAIRS]]) + r" \\")
    (tables / "A2_SI1_paired.tex").write_text(table(
        "Layer-specific paired absolute-loss reductions.",
        "epkg:a2:layerpairs", "@{}lrrrrr@{}",
        r"Layer & $n$ & E$-$LOCAL & E$-$(E+S) & (E+S)$-$LOCAL & CAL$-$LOCAL",
        "\n".join(paired_wide),
        r"Each difference is a within-unit paired absolute-loss difference averaged equally within the layer. "
        r"Positive values favour the second input. Units: RMB/kWh. Intervals: not estimated. "
        r"The (E+S)$-$LOCAL difference is positive in six layers and negative in four; "
        r"a small overall net difference does not imply uniform station-level irrelevance of the residual."),
        encoding="utf-8")


def main():
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT)
    args = ap.parse_args()
    out = args.output.resolve()
    f = ROOT / "inputs/module_b_row_level_inputs.csv.gz"
    assert sha(f) == EXPECTED_SHA, "Input identity mismatch; stop without substituting versions."
    # Round-trip parser preserves the saved IEEE values and exact quantile ties.
    df = pd.read_csv(f, dtype={"station_id": str, "month": str}, float_precision="round_trip")
    assert df.shape == (6883, 76)
    assert not df.duplicated(["month", "boundary_hour", "station_id"]).any()
    assert df[["month", "boundary_hour", "station_id"]].notna().all().all()
    stats_path = ROOT / "scripts/vendor/module_b_stats.py"
    spec = importlib.util.spec_from_file_location("original_module_b", stats_path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    for flag in ["V", "O_v", "C_f", "input_all_available", "validation_y_available",
                 "formation_y_input_available", "all_formation_segments", "formation_price_available",
                 "formation_before_endpoint_available", "formation_after_endpoint_available",
                 "validation_before_endpoint_available", "validation_after_endpoint_available"]:
        assert df[flag].notna().all() and df[flag].isin([True, False]).all(), flag
    assert df.V.equals(df.O_v) and df.O_v.equals(df.validation_y_available)
    assert df.formation_y_input_available.all() and df.input_all_available.all()
    assert df.formation_price_available.all()
    assert df.C_f.equals(df.n_native_segments.eq(8) & df.all_formation_segments)
    assert df.groupby(["month", "station_id"]).C_f.nunique().eq(1).all()
    assert (df[["formation_n_sessions_before", "formation_n_sessions_after"]] >= 10).all().all()
    assert (df[["formation_n_tariff_days_before", "formation_n_tariff_days_after"]] >= 3).all().all()
    assert df.formation_before_endpoint_available.all() and df.formation_after_endpoint_available.all()
    assert df.O_v.equals(df.validation_before_endpoint_available & df.validation_after_endpoint_available)
    finite_formation = ["E", "CAL", "LOCAL", "formation_E_star", "formation_S_star",
                        "formation_T_star", "formation_T_cons_star", "cal_value", "expected_direction"]
    assert np.isfinite(df[finite_formation].to_numpy(float)).all()
    assert np.isfinite(df.loc[df.V, "Y"]).all()
    assert df.loc[~df.V, "Y"].isna().all()
    assert int(df.V.sum()) == 5846
    df["E_plus_S"] = df.formation_E_star + df.formation_S_star
    algebra = {
        "E_equals_formation_E_star": float((df.E-df.formation_E_star).abs().max()),
        "LOCAL_equals_formation_T_star": float((df.LOCAL-df.formation_T_star).abs().max()),
        "E_plus_S_equals_formation_T_cons_star": float((df.E_plus_S-df.formation_T_cons_star).abs().max()),
        "CAL_equals_E_plus_saved_calibration": float((df.CAL-df.E-df.cal_value).abs().max()),
        "E_endpoint_identity": float((df.E-df.expected_direction*(df.formation_E_after-df.formation_E_before)).abs().max()),
        "S_endpoint_identity": float((df.formation_S_star-df.expected_direction*(df.formation_S_after-df.formation_S_before)).abs().max()),
        "LOCAL_endpoint_identity": float((df.LOCAL-df.expected_direction*(df.formation_T_after-df.formation_T_before)).abs().max()),
        "Y_endpoint_identity": float((df.loc[df.V,"Y"]-df.loc[df.V,"expected_direction"]*
            (df.loc[df.V,"validation_after_total_price_observed"]-df.loc[df.V,"validation_before_total_price_observed"])).abs().max()),
    }
    assert max(algebra.values()) <= 1e-12, algebra
    layer_map = df[["layer_order", "month", "boundary_hour", "expected_direction"]].drop_duplicates().sort_values("layer_order")
    # Historical layer IDs retain gaps for unselected July boundaries.
    assert layer_map.layer_order.tolist() == [1,2,3,4,5,6,7,8,11,12]
    assert layer_map.month.value_counts().to_dict() == {"2025-01": 6, "2025-07": 4}
    assert layer_map.boundary_hour.tolist() == [7,10,13,17,22,23,7,10,22,23]
    assert layer_map.expected_direction.tolist() == [1,1,-1,1,-1,-1,1,1,-1,-1]
    counts = df.groupby("layer_order").station_id.transform("size")-1
    assert np.array_equal(df.cal_n_other_Af, counts)
    a1, a2, paired, all_group_metrics = [], [], [], []
    for group_name in GROUPS:
        mask = np.ones(len(df), dtype=bool) if group_name == "all_local" else (
            df.C_f if group_name == "complete" else ~df.C_f)
        for layer in layer_map.itertuples(index=False):
            entry = df.loc[mask & df.layer_order.eq(layer.layer_order)]
            panel = entry.loc[entry.V]
            # Empty/nonfinite cells must stop, never be silently omitted.
            assert len(entry) > 0 and len(panel) > 0, (group_name, layer.layer_order)
            assert np.isfinite(panel[CONFIGS + ["Y"]].to_numpy(float)).all()
            w = stats.weights_for_panel(entry, panel, "ST")
            assert np.allclose(w, np.ones(len(panel))/len(panel), atol=0, rtol=0)
            common = dict(group=group_name, layer_order=int(layer.layer_order), month=layer.month,
                          boundary_hour=int(layer.boundary_hour), entry_n=len(entry), eval_n=len(panel),
                          observable_n=int(entry.O_v.sum()), missing_target_n=int((~entry.O_v).sum()),
                          observability=float(entry.O_v.sum()/len(entry)))
            row = dict(common)
            metrics = {}
            for config in CONFIGS:
                m = stats.metric_row(panel, config, w, "layer", layer.layer_order, layer.month, group_name)
                assert m["status"] == "ok" and all(np.isfinite(m[x]) for x in METRICS)
                metrics[config] = m
                all_group_metrics.append(dict(common, configuration=config, **{x:m[x] for x in METRICS}))
                if config in ["E", "LOCAL"]:
                    row.update({config+"_"+x: m[x] for x in METRICS})
                if group_name == "all_local":
                    a2.append(dict(common, configuration=config, **{x:m[x] for x in METRICS},
                                   calibration_peer_n=len(entry)-1, interval_status="not estimated in this package"))
            row["E_to_LOCAL_gain"] = metrics["E"]["mae"] - metrics["LOCAL"]["mae"]
            paired_direct = np.dot(w, np.abs(panel.E-panel.Y)-np.abs(panel.LOCAL-panel.Y))
            assert abs(row["E_to_LOCAL_gain"]-paired_direct) <= 1e-14
            row["interval_status"] = "not estimated"
            a1.append(row)
            if group_name == "all_local":
                for source, target in PAIRS:
                    diff = (np.abs(panel[source]-panel.Y)-np.abs(panel[target]-panel.Y)).to_numpy(float)
                    paired.append(dict(layer_order=int(layer.layer_order), month=layer.month,
                        boundary_hour=int(layer.boundary_hour), eval_n=len(panel), source=source, target=target,
                        mean_paired_abs_loss_reduction=float(np.dot(w,diff)),
                        mean_abs_paired_loss_difference=float(np.dot(w,np.abs(diff))),
                        interval_status="not estimated"))
    a1_over = []
    a1_metric_cols = [x for x in a1[0] if x.startswith(("E_", "LOCAL_"))]
    for group in GROUPS:
        rr = [r for r in a1 if r["group"] == group]
        r = dict(group=group, n_layers=10, entry_n=sum(x["entry_n"] for x in rr),
                 eval_n=sum(x["eval_n"] for x in rr), observable_n=sum(x["observable_n"] for x in rr),
                 missing_target_n=sum(x["missing_target_n"] for x in rr))
        r["observability"] = r["observable_n"]/r["entry_n"]
        r.update({x:float(np.mean([t[x] for t in rr])) for x in a1_metric_cols})
        r["interval_status"]="not estimated"
        a1_over.append(r)
    a2_over = []
    for config in CONFIGS:
        rr = [r for r in a2 if r["configuration"] == config]
        a2_over.append(dict(configuration=config, eval_n=5846, n_layers=10,
                            **{x:float(np.mean([t[x] for t in rr])) for x in METRICS},
                            interval_status="not estimated in this package"))
    paired_over = []
    for source, target in PAIRS:
        rr = [r for r in paired if r["source"] == source and r["target"] == target]
        paired_over.append(dict(source=source, target=target, eval_n=5846, n_layers=10,
            mean_paired_abs_loss_reduction=float(np.mean([r["mean_paired_abs_loss_reduction"] for r in rr])),
            mean_abs_paired_loss_difference=float(np.mean([r["mean_abs_paired_loss_difference"] for r in rr])),
            positive_layers=sum(r["mean_paired_abs_loss_reduction"]>0 for r in rr),
            negative_layers=sum(r["mean_paired_abs_loss_reduction"]<0 for r in rr),
            zero_layers=sum(r["mean_paired_abs_loss_reduction"]==0 for r in rr),
            interval_status="not estimated"))
    # Compare already-checked prior points; extend them with original-function ME/labels.
    prior = json.loads((ROOT / "inputs/prior_review/independent_point_verification.json").read_text(encoding="utf-8"))
    mapping = dict(all_local="all", complete="complete", additional_local="local_only")
    comparisons = []
    for r in all_group_metrics:
        old = next(p for p in prior["layer_rows"] if p["group"]==mapping[r["group"]] and p["layer"]==r["layer_order"])
        assert r["entry_n"]==old["entry_n"] and r["eval_n"]==old["eval_n"]
        for new, old_suffix in [("mae", "mae"), ("p90_abs_error", "p90")]:
            expected=old[r["configuration"]+"_"+old_suffix]
            comparisons.append(dict(reference="prior_independent_point", group=r["group"],
                layer_order=r["layer_order"], configuration=r["configuration"], metric=new,
                reference_value=expected, package_value=r[new], absolute_difference=abs(expected-r[new])))
    ref = pd.read_csv(ROOT/"inputs/reference/module_b_point_estimates.csv", float_precision="round_trip")
    for r in a2 + a2_over:
        if r["configuration"] == "E_plus_S":
            continue
        selector = (ref.group.eq("all") & ref.weight.eq("ST") & ref.configuration.eq(r["configuration"]))
        if "layer_order" in r:
            selector &= ref.scope.eq("layer") & ref.layer_order.eq(str(r["layer_order"]))
        else:
            selector &= ref.scope.eq("overall_equal_10_layers")
        old = ref.loc[selector]
        assert len(old)==1, (r, len(old))
        assert int(old.iloc[0]["n"])==r["eval_n"]
        for metric in METRICS:
            expected=float(old.iloc[0][metric])
            comparisons.append(dict(reference="published_core_point", group="all_local",
                layer_order=r.get("layer_order","ALL"), configuration=r["configuration"], metric=metric,
                reference_value=expected, package_value=r[metric], absolute_difference=abs(expected-r[metric])))
    assert max(r["absolute_difference"] for r in comparisons) <= 1e-12
    assert [(r["entry_n"],r["eval_n"],r["missing_target_n"]) for r in a1_over] == [(6883,5846,1037),(3524,3305,219),(3359,2541,818)]
    pmap = {(r["source"],r["target"]):r for r in paired_over}
    ratio = 100*pmap["E_plus_S","LOCAL"]["mean_paired_abs_loss_reduction"]/pmap["E","LOCAL"]["mean_paired_abs_loss_reduction"]
    assert abs(ratio-0.2550213088832746) <= 1e-10
    assert (pmap["E_plus_S","LOCAL"]["positive_layers"],pmap["E_plus_S","LOCAL"]["negative_layers"]) == (6,4)
    for name, rows in [("A1/overall",a1_over),("A1/by_layer",a1),("A2/overall",a2_over),
                       ("A2/by_layer",a2),("A2/paired_overall",paired_over),("A2/paired_by_layer",paired)]:
        write_rows(out/"results"/(name+".csv"), rows)
        dump(out/"results"/(name+".json"), rows)
    write_rows(out/"results/A1_A2_reference_comparison.csv",comparisons)
    write_rows(out/"results/layer_index.csv",layer_map.to_dict(orient="records"))
    dump(out/"results/A2/net_comparison.json",{
        "E_plus_S_to_LOCAL_net_MAE_reduction":pmap["E_plus_S","LOCAL"]["mean_paired_abs_loss_reduction"],
        "net_share_of_E_to_LOCAL_reduction_pct":ratio,
        "interpretation":"Specified input order, absolute loss, ten-layer weights and observed Y; not a causal or variance contribution.",
        "positive_layers":6,"negative_layers":4,"interval_status":"not estimated"})
    check = {
        "status":"POINT_SUMMARIES_RECALCULATED_AND_COMPARED",
        "input_sha256":sha(f),"shape":[6883,76],"unique_key":["month","boundary_hour","station_id"],
        "V_equals_O_v_in_this_input_only":True,"V_n":5846,
        "formation_stations":int(df.station_id.nunique()),
        "no_duplicate_keys":True,"no_empty_required_group_layers":True,
        "formation_required_fields_finite":True,"evaluation_required_fields_finite":True,
        "missing_Y_outside_V":1037,"missing_Y_inside_V":0,
        "support_flag_checks":"local endpoint count/date thresholds; C_f from eight-segment summary; consistent station-month C_f; O_v from endpoint flags",
        "native_segment_raw_support_not_rebuilt":True,
        "formation_only_calibration_preserved":True,"calibration_refit":False,
        "algebra_max_absolute_errors":algebra,
        "original_core_comparison":{"metrics":METRICS,"configurations":["E","CAL","LOCAL"],
            "scopes":"ten layers and overall; ST/all", "comparison_count":132,
            "max_absolute_difference":max(r["absolute_difference"] for r in comparisons if r["reference"]=="published_core_point")},
        "prior_point_comparison":{"comparison_count":240,
            "max_absolute_difference":max(r["absolute_difference"] for r in comparisons if r["reference"]=="prior_independent_point")},
        "newly_completed_metrics":["ME","label_disagreement","formal paired-loss tables and metadata"],
        "recalculated_for_delivery_validation":["existing A1/A2 MAE/P90 points","paired loss reductions"],
        "tolerance":1e-12, "quantile_source_sha256":sha(stats_path),
        "quantile_convention":"original weighted_quantile_tie; merge exact ties; cumulative exact-hit midpoint, tolerance 1e-12",
        "aggregation":"station-equal within group-layer; equal mean across same ten layers",
        "counts_and_observability":"pooled unit counts; observable_n/entry_n",
        "complete_January_eval_n":[r["eval_n"] for r in a1 if r["group"]=="complete" and r["month"]=="2025-01"],
        "A2_net_share_pct":ratio,
        "new_intervals":False,"new_bootstrap":False,"simulation_rerun":False,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
        "unresolved_substantive_inconsistencies":[],
    }
    dump(out/"results/A1_A2_VERIFICATION.json",check)
    make_tables(out,a1_over,a1,a2_over,a2,paired_over,paired)
    print(json.dumps({"A1":a1_over,"A2":a2_over,"net_share_pct":ratio,
                      "core_max_diff":check["original_core_comparison"]["max_absolute_difference"]},indent=2))


if __name__ == "__main__":
    if "--export-a2-main-paired-only" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Render the paired table from the frozen summary only.")
        parser.add_argument("--export-a2-main-paired-only", action="store_true")
        parser.add_argument("--output", type=Path, default=ROOT,
                            help="Evidence-package output folder; only tables/A2_main_paired.tex is written.")
        args = parser.parse_args()
        source = ROOT / "results/A2/paired_overall.csv"
        before = sha(source)
        destination = write_paired_table_from_frozen_summary(source, args.output / "tables/A2_main_paired.tex")
        assert sha(source) == before
        print(json.dumps({"mode": "frozen_summary_table_export_only", "source": str(source),
                          "source_sha256": before, "output": str(destination),
                          "scientific_results_recalculated": False}, indent=2))
    else:
        main()
