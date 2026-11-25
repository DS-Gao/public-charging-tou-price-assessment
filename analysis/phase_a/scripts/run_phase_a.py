"""Portable R4.27 Phase A analysis on captured frozen inputs.
Run: python scripts/run_phase_a.py [--output OUTPUT_DIR]
No raw-price extraction, calibration refit, manuscript edits or remote writes.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, platform, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT
KEY = ['month', 'boundary_hour', 'station_id']
LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
CONFIGS = ['E', 'E_plus_S', 'LOCAL']
PAIRS = [('E_to_E_plus_S', 'E', 'E_plus_S'), ('E_to_LOCAL', 'E', 'LOCAL'), ('E_plus_S_to_LOCAL', 'E_plus_S', 'LOCAL')]
CHECKS = []


def read(rel):
    return pd.read_csv(ROOT / rel, float_precision='round_trip')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(df, rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    compression = {'method': 'gzip', 'mtime': 0} if str(p).endswith('.gz') else None
    df.to_csv(p, index=False, encoding='utf-8-sig', float_format='%.17g', compression=compression)


def dump(obj, rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n', encoding='utf-8')


def check(name, actual, expected=True, tol=0):
    if isinstance(actual, (int, float, np.integer, np.floating)):
        passed = bool(abs(float(actual) - float(expected)) <= tol)
    else:
        passed = bool(actual == expected)
    CHECKS.append(dict(check=name, actual=actual, expected=expected, tolerance=tol, passed=passed))
    save(pd.DataFrame(CHECKS), 'validation/numerical_checks.csv')
    if not passed:
        raise AssertionError(f'{name}: {actual} vs {expected}, tolerance={tol}')


def maxdiff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise AssertionError('Different finite-value masks')
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0


spec = importlib.util.spec_from_file_location('frozen_stats', ROOT / 'sources/original_module_b_stats.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def weights(full, panel, kind):
    if kind == 'ST':
        return np.full(len(panel), 1 / len(panel))
    prefix = 'tx' if kind == 'TX' else 'kwh'
    before, after = full[prefix + '_before'].to_numpy(float), full[prefix + '_after'].to_numpy(float)
    base = pd.Series(.5 * before / before.sum() + .5 * after / after.sum(), index=full.index)
    selected = base.loc[panel.index].to_numpy(float)
    return selected / selected.sum()


def evaluate(frame, case='primary', group='all', member_mask=None, schemes=('ST', 'TX', 'KWH')):
    if member_mask is None:
        member_mask = pd.Series(True, index=frame.index)
    rows = []
    for layer in LAYERS:
        full = frame.loc[frame.layer_order.eq(layer)]
        panel = full.loc[full.V & member_mask.loc[full.index]]
        month, hour = str(full.month.iloc[0]), int(full.boundary_hour.iloc[0])
        for kind in schemes:
            w = weights(full, panel, kind) if len(panel) else np.array([])
            if len(w):
                check(f'weight sum {case}/{group}/{kind}/{layer}', float(w.sum()), 1, 1e-12)
            err = panel[CONFIGS].to_numpy(float) - panel.Y.to_numpy(float)[:, None]
            row = dict(case=case, group=group, weight=kind, scope='layer', month=month,
                       layer_order=layer, boundary_hour=hour, n=len(panel), n_layers=1,
                       status='ok' if len(panel) else 'empty')
            for j, config in enumerate(CONFIGS):
                row[config + '_mae'] = float(w @ np.abs(err[:, j])) if len(panel) else np.nan
                row[config + '_me'] = float(w @ err[:, j]) if len(panel) else np.nan
                row[config + '_p90'] = stats.weighted_quantile_tie(np.abs(err[:, j]), w, .9) if len(panel) else np.nan
            for label, a, b in PAIRS:
                row[label] = row[a + '_mae'] - row[b + '_mae']
            rows.append(row)
    layer_df = pd.DataFrame(rows)
    fields = [x for x in layer_df if x.endswith(('_mae', '_me', '_p90')) or x in [p[0] for p in PAIRS]]
    summaries = []
    for kind in schemes:
        for month in ['2025-01', '2025-07', 'ALL']:
            z = layer_df.loc[layer_df.weight.eq(kind) & (layer_df.month.eq(month) if month != 'ALL' else True)]
            expected = 10 if month == 'ALL' else (6 if month == '2025-01' else 4)
            ok = len(z) == expected and z.status.eq('ok').all()
            row = dict(case=case, group=group, weight=kind,
                       scope='overall_equal_10_layers' if month == 'ALL' else 'month_equal_layers',
                       month=month, layer_order='equal_layers', boundary_hour='', n=int(z.n.sum()),
                       n_layers=len(z), status='ok' if ok else 'incomplete')
            row.update({x: float(z[x].mean()) if ok else np.nan for x in fields})
            summaries.append(row)
    return pd.concat([layer_df, pd.DataFrame(summaries)], ignore_index=True)


def prepare():
    inventory = read('sources/source_inventory.csv')
    for _, r in inventory.iterrows():
        check('captured source hash ' + r.package_path, sha(ROOT / r.package_path), r.sha256)
    p = read('inputs/formation_panel.csv.gz')
    check('Af rows', len(p), 6883)
    check('V rows', int(p.V.sum()), 5846)
    check('Af stations', p.station_id.nunique(), 2342)
    check('V stations', p.loc[p.V, 'station_id'].nunique(), 2147)
    check('unique formation keys', not p.duplicated(KEY).any())
    check('layers', sorted(p.layer_order.unique().tolist()), LAYERS)
    check('V equals observable and input available', np.array_equal(p.V, p.O_v & p.input_all_available))
    p['E_plus_S'] = p.formation_E_star + p.formation_S_star
    for a, b in [('E', 'formation_E_star'), ('LOCAL', 'formation_T_star'), ('E_plus_S', 'formation_T_cons_star')]:
        check('formation identity ' + a, maxdiff(p[a], p[b]), 0, 1e-12)
    check('Z identity', maxdiff(p.Z, p.LOCAL - p.E), 0, 1e-12)
    check('Z grouping', np.array_equal(p.z_flat_group.eq('absZf_le_0.03'), p.Z.abs().le(.03)))
    check('Cf definition', np.array_equal(p.C_f, p.n_native_segments.eq(8) & p.all_formation_segments))
    check('evaluation prices finite', np.isfinite(p.loc[p.V, CONFIGS + ['Y']].to_numpy(float)).all())
    diag = read('inputs/fixed_V_aggregation.csv')
    join = p.loc[p.V].merge(diag, on=KEY, validate='one_to_one')
    check('diagnostic shared V', len(join), 5846)
    for a, b in [('E', 'E_median_f'), ('formation_S_star', 'S_median_f'), ('LOCAL', 'T_median_f'), ('Y', 'T_median_v')]:
        check('diagnostic price identity ' + a, maxdiff(join[a], join[b]), 0, 1e-12)
    cols = KEY + ['layer_order', 'expected_direction', 'E', 'formation_S_star', 'E_plus_S', 'LOCAL',
                  'Y', 'Z', 'V', 'O_v', 'C_f', 'z_flat_group', 'tx_before', 'tx_after', 'kwh_before', 'kwh_after']
    save(p[cols], 'task2/analysis_input_formation.csv.gz')
    return p, diag


def descriptive(p):
    frames = [POINT]
    coverage = []
    for name, mask in [('all_local', pd.Series(True, index=p.index)), ('complete', p.C_f), ('additional_local', ~p.C_f)]:
        frames.append(evaluate(p, case='support_groups', group=name, member_mask=mask, schemes=('ST',)))
        for layer in LAYERS + ['ALL']:
            z = p.loc[mask & (p.layer_order.eq(layer) if layer != 'ALL' else True)]
            coverage.append(dict(group=name, layer_order=layer, entry_n=len(z), observable_n=int(z.O_v.sum()),
                eval_n=int(z.V.sum()), missing_target_n=int((~z.O_v).sum()), availability=float(z.O_v.mean()),
                denominator='group formation entry', error_denominator='group intersection V'))
    frames.append(evaluate(p, case='station_influence', group='without_ST_003774', member_mask=p.station_id.ne('ST_003774')))
    check('removed station evaluation units', int((p.V & p.station_id.eq('ST_003774')).sum()), 7)
    for name in ['absZf_le_0.03', 'absZf_gt_0.03']:
        frames.append(evaluate(p, case='original_Z_groups', group=name, member_mask=p.z_flat_group.eq(name), schemes=('ST',)))
    cd = read('inputs/cd_formation_panel.csv.gz')
    check('CD Af rows', len(cd), 6883)
    check('CD support', int(cd.cd_support.sum()), 6390)
    check('CD shared V', int(cd.shared_V.sum()), 5572)
    match = p.merge(cd, on=KEY, suffixes=('_new', '_old'), validate='one_to_one')
    for c in ['E', 'LOCAL', 'Y', 'tx_before', 'tx_after', 'kwh_before', 'kwh_after']:
        check('CD original source identity ' + c, maxdiff(match[c + '_new'], match[c + '_old']), 0, 1e-12)
    check('CD finite component sum', np.isfinite(cd.loc[cd.shared_V, ['cd_E_star', 'cd_S_star', 'cd_T_star']].to_numpy(float)).all())
    for construction in ['original', 'CD']:
        z = cd.copy()
        z['V'] = z.shared_V
        z['E_plus_S'] = z.formation_E_star + z.formation_S_star if construction == 'original' else z.cd_E_star + z.cd_S_star
        if construction == 'CD':
            z['E'], z['LOCAL'] = z.cd_E_star, z.cd_T_star
        frames.append(evaluate(z, case='CD_' + construction))
    allpoints = pd.concat(frames, ignore_index=True)
    save(allpoints, 'task2/all_descriptive_metrics.csv')
    save(pd.DataFrame(coverage), 'task2/support_coverage.csv')
    for case in allpoints.case.unique():
        save(allpoints.loc[allpoints.case.eq(case)], f'task2/{case}_metrics.csv')
    influence = allpoints.loc[allpoints.case.eq('station_influence')].merge(
        POINT, on=['scope', 'month', 'layer_order', 'boundary_hour', 'weight'], suffixes=('_removed', '_primary'))
    for c in [x + '_mae' for x in CONFIGS] + [x[0] for x in PAIRS]:
        influence[c + '_change'] = influence[c + '_removed'] - influence[c + '_primary']
    save(influence, 'task2/station_influence_changes.csv')
    return allpoints


def compare_existing(allpoints, diag):
    rows = []
    def add(source, keys, metric, old, new, tol=1e-12):
        diff = float(new - old)
        rows.append(dict(source=source, keys=keys, metric=metric, old=float(old), new=float(new),
                         difference=diff, tolerance=tol, status='matched' if abs(diff) <= tol else 'unexplained',
                         difference_cause='rounding / floating precision only' if abs(diff) <= tol else 'requires investigation'))
    ref = read('references/A2/overall.csv')
    current = POINT.loc[POINT.scope.eq('overall_equal_10_layers') & POINT.weight.eq('ST')].iloc[0]
    for _, r in ref.loc[ref.configuration.isin(CONFIGS)].iterrows():
        for field, suffix in [('mae', '_mae'), ('me', '_me'), ('p90_abs_error', '_p90')]:
            add('R4.26 A2 overall', r.configuration, field, r[field], current[r.configuration + suffix])
    oldpoints = read('references/original_point_estimates.csv')
    for _, r in oldpoints.loc[oldpoints.group.eq('all') & oldpoints.configuration.isin(['E', 'LOCAL', 'E_to_LOCAL'])].iterrows():
        z = POINT.loc[POINT.scope.eq(r.scope) & POINT.month.eq(r.month) & POINT.layer_order.astype(str).eq(str(r.layer_order)) & POINT.weight.eq(r.weight)]
        if len(z) != 1:
            raise ValueError(f'Original point key match failed: {r.to_dict()}')
        col = r.configuration if r.configuration == 'E_to_LOCAL' else r.configuration + '_mae'
        add('R4.26 original points', f'{r.month}/{r.layer_order}/{r.weight}/{r.configuration}', 'mae', r.mae, z.iloc[0][col])
    for _, r in read('references/A1/overall.csv').iterrows():
        z = allpoints.loc[allpoints.case.eq('support_groups') & allpoints.group.eq(r.group) & allpoints.month.eq('ALL')].iloc[0]
        for c in ['E_mae', 'LOCAL_mae']:
            add('R4.26 A1 support', r.group, c, r[c], z[c])
    cdref = read('references/cd_point_estimates.csv')
    for _, r in cdref.loc[cdref.group.eq('all') & cdref.configuration.isin(['E', 'LOCAL', 'E_to_LOCAL'])].iterrows():
        z = allpoints.loc[allpoints.case.eq('CD_' + r.input_construction) & allpoints.scope.eq(r.scope) & allpoints.month.eq(r.month) & allpoints.layer_order.astype(str).eq(str(r.layer_order)) & allpoints.weight.eq(r.weight)].iloc[0]
        col = r.configuration if r.configuration == 'E_to_LOCAL' else r.configuration + '_mae'
        add('R4.26 CD', f'{r.input_construction}/{r.month}/{r.layer_order}/{r.weight}/{r.configuration}', 'mae', r.mae, z[col])
    oldloo = read('references/leave_one_station_cluster_out.csv')
    for _, r in oldloo.loc[oldloo.removed_station.eq('ST_003774') & oldloo.configuration.isin(['E', 'LOCAL'])].iterrows():
        z = allpoints.loc[allpoints.case.eq('station_influence') & allpoints.month.eq('ALL') & allpoints.weight.eq('ST')].iloc[0]
        add('R4.26 station leaveout', r.configuration, 'MAE', r.MAE, z[r.configuration + '_mae'])
    agg = []
    for method in ['median', 'mean', 'vwap']:
        x = diag.copy()
        x['E'], x['E_plus_S'] = x['E_' + method + '_f'], x['E_' + method + '_f'] + x['S_' + method + '_f']
        x['LOCAL'], x['Y'], x['layer_order'], x['V'] = x['T_' + method + '_f'], x['T_' + method + '_v'], x.layer_order_f, True
        z = evaluate(x, case='aggregation_' + method, schemes=('ST',))
        z['operator'] = method
        agg.append(z)
    agg = pd.concat(agg, ignore_index=True)
    save(agg, 'task5/aggregation_metrics.csv')
    prior = read('references/diagnostics/task1/cross_window_comparison.csv')
    for method in ['median', 'mean', 'vwap']:
        z = agg.loc[agg.operator.eq(method) & agg.month.eq('ALL')].iloc[0]
        for config in CONFIGS:
            name = 'E+S' if config == 'E_plus_S' else config
            old = prior.loc[prior.method.eq(method) & prior.month.eq('ALL') & prior.input.eq(name)].iloc[0]
            add('Nov16 operator check', method + '/' + config, 'mae', old.mae, z[config + '_mae'])
    comparison = pd.DataFrame(rows)
    save(comparison, 'task3/new_old_numeric_comparison.csv')
    check('unexplained numerical discrepancies', int(comparison.status.eq('unexplained').sum()), 0)
    return agg


def bootstrap(p, protocol):
    v = p.loc[p.V].copy()
    summaries, audits, main_draws = [], [], None
    metrics = ['E_mae', 'E_plus_S_mae', 'LOCAL_mae'] + [x[0] for x in PAIRS]
    for cfg in [protocol['main_frame'], protocol['sensitivity_frame']]:
        frame, B, seed = cfg['name'], cfg['draws'], cfg['seed']
        stations = np.array(sorted((p if frame == 'formation_2342' else v).station_id.astype(str).unique()))
        check(frame + ' station count', len(stations), cfg['stations'])
        N, lookup = len(stations), {s: i for i, s in enumerate(stations)}
        rng = np.random.default_rng(seed)
        initial_state = rng.bit_generator.state
        mult = np.stack([np.bincount(rng.choice(N, N, replace=True), minlength=N) for _ in range(B)]).astype(np.int16)
        check(frame + ' draw totals', np.all(mult.sum(axis=1) == N))
        check(frame + ' unselected multipliers zero', np.any(mult == 0))
        save(pd.DataFrame({'frame_index': np.arange(N), 'station_id': stations, 'in_V': np.isin(stations, v.station_id)}), f'task1/{frame}_station_frame.csv')
        multiplier_df = pd.DataFrame(mult, columns=stations)
        multiplier_df.insert(0, 'draw', np.arange(B))
        save(multiplier_df, f'task1/{frame}_multiplicities.csv.gz')
        layers = np.full((B, len(LAYERS), len(metrics)), np.nan)
        denominators = np.zeros((B, len(LAYERS)), dtype=np.int32)
        frames = []
        for j, layer in enumerate(LAYERS):
            z = v.loc[v.layer_order.eq(layer)]
            m = mult[:, [lookup[s] for s in z.station_id]].astype(float)
            denominator = m.sum(axis=1)
            denominators[:, j] = denominator.astype(np.int32)
            valid = denominator > 0
            w = np.divide(m, denominator[:, None], out=np.zeros_like(m), where=denominator[:, None] > 0)
            check(frame + f' normalized layer {layer}', float(np.max(np.abs(w[valid].sum(axis=1) - 1))), 0, 1e-12)
            loss = np.abs(z[CONFIGS].to_numpy(float) - z.Y.to_numpy(float)[:, None])
            mae = w @ loss
            values = np.column_stack([mae, mae[:, 0] - mae[:, 1], mae[:, 0] - mae[:, 2], mae[:, 1] - mae[:, 2]])
            values[~valid] = np.nan
            layers[:, j, :] = values
            check(frame + f' direct paired calculation {layer}', float(np.max(np.abs((w @ (loss[:, 0] - loss[:, 1]))[valid] - values[valid, 3]))), 0, 1e-12)
            df = pd.DataFrame(values, columns=metrics)
            frames.append(df.assign(frame=frame, seed=seed, draw=np.arange(B), scope='layer', month=str(z.month.iloc[0]),
                layer_order=str(layer), n=len(z), status=np.where(valid, 'ok', 'empty_layer'), denominator=denominator))
        for month, idx in [('2025-01', list(range(6))), ('2025-07', list(range(6, 10))), ('ALL', list(range(10)))]:
            valid = (denominators[:, idx] > 0).all(axis=1)
            values = layers[:, idx, :].mean(axis=1)
            values[~valid] = np.nan
            df = pd.DataFrame(values, columns=metrics)
            frames.append(df.assign(frame=frame, seed=seed, draw=np.arange(B),
                scope='overall_equal_10_layers' if month == 'ALL' else 'month_equal_layers', month=month,
                layer_order='equal_layers', n=int(v.layer_order.isin([LAYERS[k] for k in idx]).sum()),
                status=np.where(valid, 'ok', 'invalid_empty_layer'), denominator=np.nan))
        draws = pd.concat(frames, ignore_index=True)
        if frame == 'formation_2342':
            main_draws = draws
        status = pd.DataFrame({'frame': frame, 'draw': np.arange(B), 'valid_overall': (denominators > 0).all(axis=1),
            'empty_layers': [';'.join(str(LAYERS[i]) for i in np.flatnonzero(x == 0)) for x in denominators]})
        save(status, f'task1/{frame}_draw_status.csv')
        save(draws, f'task1/{frame}_draw_metrics.csv.gz')
        for keys, z in draws.groupby(['scope', 'month', 'layer_order'], sort=False):
            valid = z.status.eq('ok')
            point = POINT.loc[POINT.weight.eq('ST') & POINT.scope.eq(keys[0]) & POINT.month.eq(keys[1]) & POINT.layer_order.astype(str).eq(keys[2])].iloc[0]
            for metric in metrics:
                interval = np.quantile(z.loc[valid, metric], [.025, .975], method='linear') if valid.any() else [np.nan, np.nan]
                summaries.append(dict(frame=frame, seed=seed, scope=keys[0], month=keys[1], layer_order=keys[2],
                    metric=metric, point=float(point[metric]), ci_low=interval[0], ci_high=interval[1], planned=B,
                    valid=int(valid.sum()), invalid=int((~valid).sum()), n=int(z.n.iloc[0]), unit='RMB/kWh',
                    inference_role='primary' if keys[1] == 'ALL' and metric == 'E_to_E_plus_S' and frame == 'formation_2342' else 'conditional supporting / sensitivity'))
        audits.append(dict(frame=frame, N=N, B=B, seed=seed, initial_rng_state=initial_state,
            final_rng_state=rng.bit_generator.state, multiplier_sha256=hashlib.sha256(mult.tobytes()).hexdigest(),
            valid_overall=int((denominators > 0).all(axis=1).sum()), invalid_overall=int((denominators == 0).any(axis=1).sum()),
            invalid_reason='any zero V-layer multiplier denominator', sample_order='lexicographically sorted station strings',
            multiplier_dtype=str(mult.dtype), multiplier_shape=list(mult.shape)))
        print('Bootstrap complete', frame, audits[-1]['valid_overall'], flush=True)
    summary = pd.DataFrame(summaries)
    save(summary, 'task1/bootstrap_summary.csv')
    dump(audits, 'task1/random_stream_and_sampling_record.json')
    original = read('references/original_bootstrap_draws.csv.gz')
    original = original.loc[original.weight.eq('ST') & original.group.eq('all') & original.configuration.isin(['E', 'LOCAL', 'E_to_LOCAL'])].copy()
    comparison = []
    for config, metric in [('E', 'E_mae'), ('LOCAL', 'LOCAL_mae'), ('E_to_LOCAL', 'E_to_LOCAL')]:
        z = original.loc[original.configuration.eq(config)].copy()
        z['layer_order'] = z.layer_order.astype(str)
        matched = z.merge(main_draws, on=['scope', 'month', 'layer_order', 'draw'], suffixes=('_old', '_new'), validate='one_to_one')
        difference = float(np.max(np.abs(matched.mae - matched[metric])))
        check('saved original draw replay ' + config, difference, 0, 1e-10)
        check('saved original draw matched rows ' + config, len(matched), 26000)
        comparison.append(dict(configuration=config, matched_draw_rows=len(matched), max_abs_difference=difference,
                               tolerance=1e-10, passed=difference <= 1e-10))
    save(pd.DataFrame(comparison), 'validation/original_draw_replay.csv')
    return summary


def table_data(summary, agg):
    overall = POINT.loc[POINT.month.eq('ALL') & POINT.weight.eq('ST')].iloc[0]
    s = summary.loc[summary.frame.eq('formation_2342') & summary.month.eq('ALL')]
    A, B = [], []
    for config in CONFIGS:
        ci = s.loc[s.metric.eq(config + '_mae')].iloc[0]
        A.append(dict(input=config, n=5846, mae=overall[config + '_mae'], me=overall[config + '_me'],
            mean_layer_p90=overall[config + '_p90'], mae_ci_low=ci.ci_low, mae_ci_high=ci.ci_high,
            frame=ci.frame, valid=ci.valid, unit='RMB/kWh'))
    for label, a, b in PAIRS:
        ci = s.loc[s.metric.eq(label)].iloc[0]
        B.append(dict(comparison=label, n=5846, paired_mae_improvement=overall[label], ci_low=ci.ci_low,
            ci_high=ci.ci_high, valid=ci.valid, frame=ci.frame, unit='RMB/kWh',
            role='primary' if label == 'E_to_E_plus_S' else 'supporting'))
    save(pd.DataFrame(A), 'task5/Table8_panelA.csv')
    save(pd.DataFrame(B), 'task5/Table8_panelB.csv')
    save(agg.loc[agg.month.eq('ALL'), ['operator', 'n', 'n_layers', 'E_mae', 'E_plus_S_mae', 'LOCAL_mae', 'E_to_E_plus_S']], 'task5/Table9_panelA.csv')
    transition = read('references/diagnostics/task1/direction_amount_transition.csv')
    counts = []
    for method in ['mean', 'vwap']:
        z = transition.loc[transition.layer.astype(str).eq('ALL') & transition.population.eq('fixed_H_median') & transition.target_method.eq(method) & transition.classification.eq('amount')]
        check('transition denominator ' + method, int(z.n.sum()), 7479)
        changed = int(z.loc[z.from_median.ne(z.to_target), 'n'].sum())
        check('transition changed ' + method, changed, 846 if method == 'mean' else 894)
        counts.append(dict(operator=method, fixed_original_H=7479, changed=changed, unchanged=7479-changed,
                           changed_fraction=changed/7479, requalified_H=7571 if method == 'mean' else 7575))
    save(pd.DataFrame(counts), 'task5/Table9_panelB.csv')
    save(pd.DataFrame(A), 'task5/Figure7_panelA.csv')
    fig = summary.loc[summary.frame.eq('formation_2342') & summary.metric.eq('E_to_E_plus_S') & (summary.scope.eq('layer') | summary.month.eq('ALL'))].copy()
    fig['display_interval'] = fig.month.eq('ALL')
    fig['display_ci_low'] = fig.ci_low.where(fig.display_interval)
    fig['display_ci_high'] = fig.ci_high.where(fig.display_interval)
    save(fig, 'task5/Figure7_panelB.csv')
    comp = summary.loc[summary.month.eq('ALL')].pivot(index='metric', columns='frame', values=['point', 'ci_low', 'ci_high']).reset_index()
    comp.columns = ['_'.join(x).strip('_') for x in comp.columns.to_flat_index()]
    save(comp, 'task1/frame_comparison.csv')


def main():
    global OUT, POINT
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT)
    args = ap.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / 'protocol/analysis_protocol.json').read_text(encoding='utf-8'))
    preflight = json.loads((ROOT / 'validation/input_preflight.json').read_text(encoding='utf-8'))
    check('preflight passed', preflight['all_passed'])
    p, diag = prepare()
    POINT = evaluate(p)
    save(POINT, 'task2/primary_metrics.csv')
    estimate = POINT.loc[POINT.month.eq('ALL') & POINT.weight.eq('ST'), 'E_to_E_plus_S'].iloc[0]
    check('primary paired point', estimate, protocol['expected_point'], 1e-12)
    print('Point estimate reproduced', repr(estimate), flush=True)
    desc = descriptive(p)
    agg = compare_existing(desc, diag)
    summary = bootstrap(p, protocol)
    table_data(summary, agg)
    dump(dict(python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__,
        platform=platform.platform(), protocol_sha256=sha(ROOT / 'protocol/analysis_protocol.json'),
        analysis_script_sha256=sha(Path(__file__)), raw_reconstruction_runs=0, bootstrap_frames=2,
        draws_per_frame=2000, regressions=0, phase_B_started=False), 'validation/run_environment.json')
    print(summary.loc[summary.month.eq('ALL') & summary.metric.eq('E_to_E_plus_S')].to_string(index=False), flush=True)
    print('All checks passed:', len(CHECKS), flush=True)


if __name__ == '__main__':
    main()

