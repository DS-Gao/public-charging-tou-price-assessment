"""Index and reaggregate frozen formal outputs. Generates no random samples."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
from build_a1_a2 import write_rows, dump, table

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"results/A3/frozen"
FORMAL=FROZEN/"reconstruction/formal"
SHA="21357f692d59811a5705bfb7b4b415b7bda5239ff84188a7b92af371dced9bc6"
PROCESSES=["iid_low","iid_high","multi_price","shared_day","selective_support"]
LAYOUTS=[(20,5),(80,5),(80,20)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metric_dictionary():
    # Line numbers refer to the byte-frozen historical reconstruction_simulation.py.
    d={}
    def add(name,unit,definition,denominator,line,undefined="none in frozen formal output"):
        d[name]=dict(metric=name,unit=unit,replicate_definition=definition,
                     within_replicate_denominator=denominator,source_file="frozen/source_code/reconstruction_simulation.py",
                     source_line=line,undefined_rule=undefined,
                     summary="mean of finite replicate metrics; MCSE=sample SD(ddof=1)/sqrt(n_finite); n_nonfinite retained")
    add("n_A","stations","|A|","fixed panel of 180",149)
    add("A_coverage","proportion","|A|/180","fixed panel of 180",149)
    add("n_true_R","stations","|A intersect {true E*>0.01}|","A",150)
    add("n_est_R","stations","|A intersect {estimated E*>0.01}|","A",150)
    add("true_R_prevalence","proportion","mean(true E*>0.01) on A","A",151)
    add("estimated_R_prevalence","proportion","mean(estimated E*>0.01) on A","A",151)
    add("electricity_category_error","proportion","mean(estimated vs true E three-way category differs); thresholds +/-0.01","A",152)
    add("total_category_error","proportion","mean(estimated vs true T three-way category differs); thresholds +/-0.03","A",153)
    add("R_sensitivity","proportion","mean(estimated E*>0.01) on supported true R","A intersect true R",154,"NA if no supported true R")
    add("R_precision","proportion","mean(true E*>0.01) on supported estimated R","A intersect estimated R",155,"NA if no supported estimated R")
    add("near_R_threshold_error","proportion","mean(estimated R indicator != true R indicator), abs(true E*-0.01)<=0.003","supported near-threshold stations",157,"NA if no near-threshold stations")
    add("far_R_threshold_error","proportion","same indicator error, abs(true E*-0.01)>0.003","supported far-threshold stations",159,"NA if no far-threshold stations")
    for prefix,error,unit in [
        ("E","estimated E* - true E*","RMB/kWh"),
        ("T","estimated T* - true T*","RMB/kWh"),
        ("D30","30*((estimated T*-estimated E*)-(true T*-true E*))","reference RMB"),
        ("residual","(estimated T*-estimated E*-estimated S*)-(true T*-true E*-true S*)","RMB/kWh")]:
        for suffix,formula,line in [("bias","mean(error)",163),("mae","mean(abs(error))",164),("mse","mean(error^2)",165)]:
            add(prefix+"_"+suffix, f"({unit})^2" if suffix=="mse" else unit,
                formula+" with error="+error,"A",line)
    for name,formula,line in [
        ("FIQR_truth_R","F(true prices, A intersect true R)",169),
        ("FIQR_price_error_fixed_true_R","F(estimated prices, A intersect true R)-F(true prices, A intersect true R)",170),
        ("FIQR_combined_error","F(estimated prices, A intersect estimated R)-F(true prices, A intersect true R)",171),
        ("FIQR_price_error_given_est_R","F(estimated prices, A intersect estimated R)-F(true prices, A intersect estimated R)",172),
        ("FIQR_membership_change","F(true prices, A intersect estimated R)-F(true prices, A intersect true R)",173),
        ("FIQR_combined_absolute_error","abs(FIQR_combined_error)",177),
        ("FIQR_fixed_R_absolute_error","abs(FIQR_price_error_fixed_true_R)",178)]:
        add(name,"RMB/kWh",formula+"; F=IQR(T*)-IQR(E*), NumPy type 7",
            "specified supported R group",line,"NA when any required F group has fewer than 2 members")
    for name,formula,denom,line in [
        ("full_true_mean_abs_B30","mean(abs(30*true T*))","all 180 latent stations",179),
        ("supported_true_mean_abs_B30","mean(abs(30*true T*))","A",180),
        ("estimated_mean_abs_B30","mean(abs(30*estimated T*))","A",181),
        ("amount_reconstruction_error_on_A","estimated_mean_abs_B30-supported_true_mean_abs_B30","A",184),
        ("amount_support_composition_difference","supported_true_mean_abs_B30-full_true_mean_abs_B30","A vs full panel",185),
        ("amount_combined_difference_from_full","estimated_mean_abs_B30-full_true_mean_abs_B30","A vs full panel",186)]:
        add(name,"reference RMB",formula,denom,line)
    add("true_R_retention","proportion","mean(true T*>0.03) on supported true R","A intersect true R",189,"NA if no supported true R")
    add("estimated_R_retention","proportion","mean(estimated T*>0.03) on supported estimated R","A intersect estimated R",190,"NA if no supported estimated R")
    add("retention_error","proportion difference","estimated_R_retention-true_R_retention","different supported R groups as specified",191,"NA if either retention is unavailable")
    return d


def display(value):
    if not math.isfinite(value):
        return "NA"
    if value==0:
        return "0"
    if abs(value)<0.00001:
        m,e=f"{value:.3e}".split("e")
        return rf"${m}\!\times\!10^{{{int(e)}}}$"
    return f"{value:.6f}"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=ROOT)
    out=ap.parse_args().output.resolve();result=out/"results/A3"
    assert sha(FORMAL/"performance_summary.csv")==SHA
    manifest=json.loads((FROZEN/"run_manifest.json").read_text(encoding="utf-8"))
    run=json.loads((FORMAL/"run_checks.json").read_text(encoding="utf-8"))
    truth=json.loads((FORMAL/"truth_checks.json").read_text(encoding="utf-8"))
    hashes=[]
    for p in sorted(FROZEN.rglob("*")):
        if not p.is_file():continue
        relative=p.relative_to(FROZEN).as_posix()
        hist=next((x for x in manifest["artifacts"] if x["path"]==relative),None)
        if relative=="source_code/reconstruction_simulation.py":
            expected=run["code_sha256"];source="formal/run_checks.json code_sha256"
        elif relative=="reconstruction_design.md":
            expected=run["design_sha256"];source="formal/run_checks.json design_sha256 + root manifest"
            assert hist["sha256"]==expected
        else:
            expected=hist["sha256"] if hist else None;source="historical run_manifest.json" if hist else "new capture only"
        actual=sha(p)
        assert expected is None or actual==expected,(relative,expected,actual)
        hashes.append(dict(path=relative,sha256=actual,expected_sha256=expected or "not listed",
                           historical_source=source,status="match" if expected else "captured; no self-hash"))
    write_rows(result/"historical_hash_verification.csv",hashes)
    summary=pd.read_csv(FORMAL/"performance_summary.csv",float_precision="round_trip")
    outer=pd.read_parquet(FORMAL/"outer_replicates.parquet")
    assert len(summary)==600 and summary.metric.nunique()==40
    assert not summary.duplicated(["scenario_id","metric"]).any()
    assert len(outer)==30000 and not outer.duplicated(["scenario_id","replicate"]).any()
    assert run["phase"]=="formal" and run["n_scenarios"]==15 and run["replicates_per_scenario"]==2000 and run["n_finite_panels"]==30000
    dictionary=metric_dictionary();assert set(dictionary)==set(summary.metric)
    records=[];scenario_rows=[];lookup={}
    for sid,g in summary.groupby("scenario_id",sort=True):
        assert len(g)==40
        process=PROCESSES[sid//3];tx,days=LAYOUTS[sid%3]
        assert g.process.eq(process).all() and g.n_tx_each_endpoint.eq(tx).all() and g.n_days_each_endpoint.eq(days).all()
        assert g.seed.eq(9132800+sid).all()
        saved=outer[outer.scenario_id.eq(sid)]
        assert sorted(saved.replicate.tolist())==list(range(2000))
        for key,value in [("process",process),("n_tx_each_endpoint",tx),("n_days_each_endpoint",days),("seed",9132800+sid)]:
            assert saved[key].eq(value).all()
        scenario_rows.append(dict(scenario_id=int(sid),process=process,n_tx_each_endpoint=tx,
            n_days_each_endpoint=days,seed=9132800+int(sid),n_replicates=2000,
            n_metrics=40,min_n_finite=int(g.n_finite.min()),max_n_nonfinite=int(g.n_nonfinite.max()),
            csv_first_data_line=int(g.index.min()+2),csv_last_data_line=int(g.index.max()+2)))
        for idx,r in g.iterrows():
            x=saved[r.metric].to_numpy(float);valid=np.isfinite(x);y=x[valid]
            assert len(x)==r.n_replicates and valid.sum()==r.n_finite and (~valid).sum()==r.n_nonfinite
            mean=float(y.mean()) if len(y) else None
            mcse=float(y.std(ddof=1)/np.sqrt(len(y))) if len(y)>1 else None
            assert mean is not None and mcse is not None # Frozen formal file has no unavailable metrics.
            delta_mean=abs(mean-r["mean"]);delta_mcse=abs(mcse-r.mcse)
            assert delta_mean<=1e-12 and delta_mcse<=1e-12
            entry=dict(scenario_id=int(sid),metric=r.metric,csv_line_including_header=int(idx+2),
                n_replicates=int(r.n_replicates),n_finite=int(r.n_finite),n_nonfinite=int(r.n_nonfinite),
                frozen_mean=float(r["mean"]),frozen_mcse=float(r.mcse),
                reaggregated_mean=mean,reaggregated_mcse=mcse,
                mean_absolute_difference=delta_mean,mcse_absolute_difference=delta_mcse)
            records.append(entry);lookup[int(sid),r.metric]=entry
    assert all(v==0 for v in run["nonfinite_counts"].values())
    assert summary.n_nonfinite.eq(0).all()
    assert np.isfinite(summary[["mean","mcse"]].to_numpy(float)).all()
    assert summary.mcse.ge(0).all()
    fiqr=float((outer.FIQR_combined_error-outer.FIQR_price_error_given_est_R-outer.FIQR_membership_change).abs().max())
    amount=float((outer.amount_combined_difference_from_full-outer.amount_reconstruction_error_on_A-outer.amount_support_composition_difference).abs().max())
    assert fiqr<=1e-12 and amount<=1e-12
    write_rows(result/"formal_result_index.csv",records)
    write_rows(result/"scenario_index.csv",scenario_rows)
    write_rows(result/"metric_dictionary.csv",[dictionary[m] for m in summary.metric.drop_duplicates()])
    dump(result/"metric_dictionary.json",[dictionary[m] for m in summary.metric.drop_duplicates()])
    panels=[
        ("Price errors",["E_mae","T_mae","D30_mae","residual_mae"],
         ["E MAE","T MAE",r"$D_{30}$ MAE","Residual MAE"]),
        ("Classification errors",["electricity_category_error","total_category_error","near_R_threshold_error","far_R_threshold_error"],
         ["E category", "T category",r"Near $R$ threshold",r"Far $R$ threshold"]),
        ("Support and population statistics",["A_coverage","FIQR_fixed_R_absolute_error","FIQR_combined_absolute_error",
         "amount_reconstruction_error_on_A","amount_support_composition_difference"],
         [r"$A$ coverage",r"$F$: fixed $R$",r"$F$: combined","Amount recon.","Amount comp."])
    ]
    selected=list(dict.fromkeys(m for _,mm,_ in panels for m in mm))
    compact=[]
    for r in scenario_rows:
        row=dict(r)
        for m in selected:
            v=lookup[r["scenario_id"],m];row[m]=v["frozen_mean"];row[m+"_mcse"]=v["frozen_mcse"]
        compact.append(row)
    write_rows(result/"scenario_compact.csv",compact)
    dump(result/"scenario_compact.json",compact)
    process_names=["IID low","IID high","Mixture","Shared day","Selective"]
    texts=[]
    for pi,(title,metrics,labels) in enumerate(panels):
        body=[]
        for r in scenario_rows:
            cells=[str(r["scenario_id"]),process_names[r["scenario_id"]//3],
                   str(r["n_tx_each_endpoint"])+"/"+str(r["n_days_each_endpoint"])]
            for m in metrics:
                v=lookup[r["scenario_id"],m]
                cells.append(r"\shortstack{"+display(v["frozen_mean"])+r"\\("+display(v["frozen_mcse"])+")}")
            body.append(" & ".join(cells)+r" \\")
        note=(r"Frozen formal experiment of 13 September 2026: 15 scenarios, 2,000 repetitions each. "
              r"Every displayed cell is the mean (Monte Carlo standard error); all 2,000 repetitions are finite "
              r"for every metric. Layout is transactions/tariff dates at each endpoint. "
              r"This package reaggregates saved outputs and does not rerun simulation. "
              r"All 40 metrics and exact source-row locators are supplied in the evidence package's "
              r"\texttt{results/A3/} directory; the original 600-row file is under "
              r"\texttt{frozen/reconstruction/formal/}. ")
        if pi==0:
            note+=r"E, T and residual errors are RMB/kWh; $D_{30}$ errors are reference RMB. Small nonzero MCSEs retain scientific notation."
        elif pi==1:
            note+=r"Errors are proportions, not percentages. The near group satisfies $|E^*_{\rm true}-0.01|\leq0.003$ RMB/kWh. Category thresholds are $\pm0.01$ for E and $\pm0.03$ for T."
        else:
            note+=(r"Coverage is a proportion. $F$ columns are mean absolute FIQR errors in RMB/kWh. "
                   r"Amount columns are signed differences in mean $|B_{30}|$, in reference RMB: "
                   r"reconstruction on $A$, and supported-minus-full true population, respectively. "
                   r"Zero composition differences in the first twelve scenarios are exact structural zeros. "
                   r"Support coverage here does not certify Beijing latent-price coverage.")
        texts.append(table("Fifteen-scenario reconstruction evidence: "+title+".",
            f"epkg:a3:panel{pi+1}","@{}ll"+"r"*(len(metrics)+1)+"@{}",
            "ID & Process & Layout & "+" & ".join(labels),"\n".join(body),note))
    (out/"tables").mkdir(parents=True,exist_ok=True)
    (out/"tables/A3_SI2_formal_scenarios.tex").write_text("\n".join(texts),encoding="utf-8")
    # SI2 outcome values are looked up from frozen numerical sources, never reconstructed from prose.
    si2=ROOT/"provenance/manuscript_context/supplement_si2/D_reliability.tex"
    lines=si2.read_text(encoding="utf-8").splitlines()
    mappings=[]
    for sid,metric,displayed,scale,digits in [
        (1,"near_R_threshold_error","33.55",100,2),
        (1,"far_R_threshold_error","0.0358",100,4),
        (13,"A_coverage","55.62",100,2),
        (0,"T_mae","0.004785",1,6),(1,"T_mae","0.002475",1,6),
        (9,"T_mae","0.041393",1,6),(10,"T_mae","0.040964",1,6),
        (11,"T_mae","0.022673",1,6),
        (13,"amount_support_composition_difference","-0.4117",1,4),
        (13,"amount_reconstruction_error_on_A","0.0017",1,4)]:
        row=lookup[sid,metric];value=row["frozen_mean"]
        assert f"{value*scale:.{digits}f}"==displayed,(sid,metric)
        line=next(i+1 for i,s in enumerate(lines) if displayed in s and 100<=i+1<=190)
        mappings.append(dict(kind="formal_performance",si2_line=line,displayed=displayed,
            scenario_id=sid,metric=metric,source="frozen/reconstruction/formal/performance_summary.csv",
            source_locator=f"line {row['csv_line_including_header']}; column mean",
            raw_value=value,scale_for_display=scale,decimal_places=digits,status="exact rounded match",
            clarification="selective-support 80/5 (scenario 13)" if sid==13 else ""))
    residual=truth["mixture_boundary_aggregation_residual_truth"]
    assert f"{residual:.6f}"=="-0.008416"
    mappings.append(dict(kind="known_truth",si2_line=next(i+1 for i,s in enumerate(lines) if "-0.008416" in s),
        displayed="-0.008416",scenario_id="multi_price (6,7,8)",metric="mixture_boundary_aggregation_residual_truth",
        source="frozen/reconstruction/formal/truth_checks.json",source_locator="mixture_boundary_aggregation_residual_truth",
        raw_value=residual,scale_for_display=1,decimal_places=6,status="exact rounded match",clarification="population truth, not estimated residual mean"))
    write_rows(result/"SI2_result_number_mapping.csv",mappings)
    # Grouped design-number locators cover the numerical design statements in the active V2b subsection.
    design=[
        ("180;30;six","fixed latent stations; stations per electricity type","source_code/reconstruction_simulation.py","20;67","111-112"),
        ("-0.020;0.003;0.009;0.011;0.050;0.200","electricity-change types","source_code/reconstruction_simulation.py","68","113"),
        ("0.60;0.75;0.45;0.50","before-component location ranges","source_code/reconstruction_simulation.py","70-71","114"),
        ("-0.070;-0.020;0;0.025;0.080","service-change cycle","source_code/reconstruction_simulation.py","69","115-116"),
        ("q=+1;two endpoints","one standardized boundary","reconstruction_design.md","V2b latent panel and known truth","118"),
        ("0;0.025;-0.020;0;-0.140;0.120","mixture class offsets","source_code/reconstruction_simulation.py","24-25","128-129"),
        ("0.55;0.30;0.15;0.35;0.45;0.20","before/after class probabilities","source_code/reconstruction_simulation.py","26","129-130"),
        ("0.015;0.020;0.060;0.080","low/high uniform halfwidths","source_code/reconstruction_simulation.py","102","131"),
        ("-1;1;0.040;0.060","day shock range and E/S loads","source_code/reconstruction_simulation.py","113-116","131-132"),
        ("five;20/5;80/5;80/20","process/layout cross","source_code/reconstruction_simulation.py","22-23","132-133"),
        ("0.55;0.90;0.01","selective support probabilities/true R threshold","source_code/reconstruction_simulation.py","126","134-135"),
        ("8;2;20/5","failed endpoint metadata and support rule","source_code/reconstruction_simulation.py","128-130","135-136"),
        ("+/-0.01;+/-0.03;30","classification tolerances and reference amount multiplier","source_code/reconstruction_simulation.py","137;152-160","142-145"),
        ("type 7","FIQR quartiles","source_code/reconstruction_simulation.py","91-96","152"),
        ("200","pilot repetitions","reconstruction/pilot/run_checks.json","replicates_per_scenario","165"),
        ("2000;30000","formal repetitions/scenario and panels","reconstruction/formal/run_checks.json","replicates_per_scenario;n_finite_panels","165-166"),
        ("0.003","near-threshold definition","source_code/reconstruction_simulation.py","156-159","169"),
    ]
    write_rows(result/"SI2_design_number_mapping.csv",[
        dict(numeric_tokens=t,meaning=desc,source="frozen/"+f,source_locator=loc,
             si2_line_hint=sl,status="checked against frozen code/design/checks; no simulation executed")
        for t,desc,f,loc,sl in design])
    check=dict(status="EXISTING_FORMAL_EVIDENCE_ARCHIVED_AND_REAGGREGATED",
        rows=600,scenarios=15,metrics_per_scenario=40,replicates_per_scenario=2000,
        stored_outer_rows=30000,nonfinite_metric_cells=0,unavailable_summary_cells=0,
        metrics_with_exact_zero_mcse=int(summary.mcse.eq(0).sum()),
        positive_mcse_below_1e_minus_12=int((summary.mcse.gt(0)&summary.mcse.lt(1e-12)).sum()),
        mean_reaggregation_max_abs_diff=max(r["mean_absolute_difference"] for r in records),
        mcse_reaggregation_max_abs_diff=max(r["mcse_absolute_difference"] for r in records),
        reaggregation_tolerance=1e-12,
        FIQR_decomposition_max_abs_residual=fiqr,amount_decomposition_max_abs_residual=amount,
        SI2_outcome_values_checked=len(mappings),SI2_outcome_rounding_matches=True,
        SI2_selective_support_clarification="55.62%, -0.4117 and 0.0017 refer to scenario 13 (80/5); specify layout in revised prose.",
        original_code_sha256_matches_run_checks=True,formal_summary_sha256=SHA,
        three_requested_formal_file_hashes_match_historical_manifest=True,
        historical_truth_check_status=truth["status"],historical_truth_checks_rerun=False,
        historical_exact_prefix_reproduction=run["exact_prefix_reproduction"],prefix_rerun_this_package=False,
        original_run_started_utc=run["started_utc"],original_run_finished_utc=run["finished_utc"],
        simulation_reexecuted=False,new_bootstrap=False,
        original_run_environment={k:run[k] for k in ["python","numpy","pandas","scipy"]},
        current_reaggregation_environment={"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__},
        unresolved_substantive_inconsistencies=[])
    dump(result/"A3_VERIFICATION.json",check)
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
