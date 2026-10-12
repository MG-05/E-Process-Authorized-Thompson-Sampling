"""Paper estimands, ranks, and the original declared resampling designs."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import numpy as np
from eats.io import load_rows, load_json, output_directory
from eats.environments.registered import MECHANISMS
from eats.experiments.calibration_math import _matched_threshold

K_VALUES = (4,8,16,32,64)
B_VALUES = (250,625,1250,2500)
NONSTATIONARY = tuple(m for m in MECHANISMS if m!='stationary')
GATES = ('ots','open-ramp','e-ats','certified-glr','practical-glr')
LABELS = {'ots':'OTS','open-ramp':'Open ramp','e-ats':'e-ATS','certified-glr':'Certified GLR','practical-glr':'Practical GLR','calibrated-glr':'Calibrated GLR'}

def mean(rows, policy):
    return float(np.mean([r['outcomes'][policy]['regret'] for r in rows]))

def difference(row, left, right='e-ats'):
    return row['outcomes'][left]['regret']-row['outcomes'][right]['regret']

def seed(label, modulo=True):
    value=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')
    return value % (2**63) if modulo else value

def interval(samples, *, interpolated=False):
    return [float(np.quantile(samples,.025,method='linear' if interpolated else 'lower')),
            float(np.quantile(samples,.975,method='linear' if interpolated else 'higher'))]

def registered_bootstrap(rows, pairs, seed_value, replicates=50_000):
    groups=defaultdict(list)
    for r in rows:
        t=r['task'];groups[(t['K'],t['B'],t['mechanism'])].append(r)
    blocks=[]
    for k in K_VALUES:
        for b in B_VALUES:
            mechanisms=[('isolated_incumbent_down','isolated_challenger_up')]+[(m,) for m in NONSTATIONARY if m not in ('isolated_incumbent_down','isolated_challenger_up')]
            for block in mechanisms:
                values=[]
                for m in block:
                    members=sorted(groups[k,b,m],key=lambda r:r['task']['source_index'])
                    if [r['task']['source_index'] for r in members]!=list(range(5,96)):
                        raise ValueError('Registered resampling census mismatch')
                    values.append([[difference(r,*pair) for pair in pairs] for r in members])
                blocks.append(np.asarray(values))
    rng=np.random.default_rng(seed_value)
    samples=np.empty((replicates,len(pairs)))
    for start in range(0,replicates,250):
        stop=min(start+250,replicates);totals=np.zeros((stop-start,len(pairs)))
        for block in blocks:
            indices=rng.integers(0,91,size=(stop-start,91))
            totals+=np.sum(np.take(block,indices,axis=1),axis=(0,2))
        samples[start:stop]=totals/len(rows)
    return samples

def replay_bootstrap(rows, *, corrected, replicates=50_000):
    groups=defaultdict(list)
    for r in rows: groups[(r['task']['environment_family'],r['task']['K'])].append(r)
    pairs=[('e-ats','open-ramp')] if corrected else [('practical-glr','e-ats'),('certified-glr','e-ats'),('e-ats','open-ramp')]
    cells=[]
    # Corrected audit iterated first-seen cells: archived rows put K=32 first.
    keys=list(groups) if corrected else sorted(groups)
    for family,k in keys:
        members=groups[family,k]
        if corrected and family in ('abrupt','fast','slow'):
            units=np.asarray([[difference(r,*p) for p in pairs] for r in members])
        else:
            clusters=defaultdict(list)
            for r in members: clusters[r['environment_index']].append(r)
            indices=list(clusters) if corrected else sorted(clusters)
            if len(clusters)!=50 or any(len(v)!=5 for v in clusters.values()): raise ValueError('Replay cluster census')
            units=np.asarray([[np.mean([difference(r,*p) for r in clusters[i]]) for p in pairs] for i in indices])
        cells.append(units)
    label=('e_ats_table5_gate_policy_rng_sensitivity_v1/original-policy-seed-corrected-sampling-units' if corrected else
           'e_ats_table5_exact_suite_gate_comparison_v1/mean-array-cluster-bootstrap')
    rng=np.random.default_rng(seed(label,modulo=not corrected))
    samples=np.zeros((replicates,len(pairs)))
    if corrected:
        for units in cells:
            for start in range(0,replicates,500):
                stop=min(start+500,replicates);n=len(units)
                indices=rng.integers(0,n,size=(stop-start,n))
                samples[start:stop]+=np.mean(units[indices],axis=1)/18
    else:
        for start in range(0,replicates,500):
            stop=min(start+500,replicates)
            for units in cells:
                indices=rng.integers(0,50,size=(stop-start,50))
                samples[start:stop]+=np.sum(np.take(units,indices,axis=0),axis=1)
            samples[start:stop]/=900
    return [interval(samples[:,i],interpolated=corrected) for i in range(len(pairs))]

def calibrated_bootstrap(rows, replicates=50_000):
    groups=defaultdict(list)
    for r in rows:
        t=r['task'];groups[t['K'],t['B'],t['mechanism']].append(r)
    values=np.asarray([[difference(r,'calibrated-glr') for r in sorted(groups[key],key=lambda r:r['task']['source_index'])] for key in sorted(groups)])
    rng=np.random.default_rng(seed('e_ats_glr_full_grid_synthesis/registered'))
    samples=np.empty(replicates);strata=np.arange(180)[None,:,None]
    for start in range(0,replicates,250):
        stop=min(start+250,replicates);indices=rng.integers(0,91,size=(stop-start,180,91))
        samples[start:stop]=np.mean(values[strata,indices],axis=(1,2))
    return interval(samples)

def census(registered,replay):
    expected=Counter({(m,k,b,kind):91 if kind=='random_environment' else 27 for m in MECHANISMS for k in K_VALUES for b in B_VALUES for kind in ('random_environment','sentinel')})
    observed=Counter((r['task']['mechanism'],r['task']['K'],r['task']['B'],r['task']['source_kind']) for r in registered)
    if observed!=expected: raise ValueError('Registered suite census mismatch')
    if len({r['group_id'] for r in registered})!=23600: raise ValueError('Duplicate registered group')
    cells=Counter((r['task']['environment_family'],r['task']['K']) for r in replay)
    if len(cells)!=18 or set(cells.values())!={250} or len({r['group_id'] for r in replay})!=4500:
        raise ValueError('Replay suite census mismatch')
    if len({r['mean_sha256'] for r in replay})!=606: raise ValueError('Replay unique-mean census mismatch')

def compute(*,bootstrap=False):
    registered,replay=load_rows('registered'),load_rows('replay')
    census(registered,replay)
    random=[r for r in registered if r['task']['source_kind']=='random_environment']
    nonstat=[r for r in random if r['task']['mechanism']!='stationary']
    stationary=[r for r in random if r['task']['mechanism']=='stationary']
    stress=[r for r in registered if r['task']['source_kind']=='sentinel' and r['task']['mechanism']!='stationary']
    table=[{'policy':LABELS[p],'id':p,'registered':mean(nonstat,p),'replay':mean(replay,p),
            'registered_difference':mean(nonstat,p)-mean(nonstat,'e-ats'),'replay_difference':mean(replay,p)-mean(replay,'e-ats')} for p in GATES]
    by_cell=defaultdict(list)
    for r in replay: by_cell[r['task']['environment_family'],r['task']['K']].append(r)
    baseline_ids=[r['row_id'] for r in load_json('baseline_configurations.json')['4']]+['e-ats']
    baseline_labels={r['row_id']:r['label'] for r in load_json('baseline_configurations.json')['4']};baseline_labels['e-ats']='e-ATS'
    cell_rows=[]
    for (family,k),members in sorted(by_cell.items()):
        means={p:mean(members,p) for p in baseline_ids}
        for p,value in means.items():
            cell_rows.append({'family':family,'K':k,'policy':p,'label':baseline_labels[p], 'regret':value,'rank':1+sum(v<value for v in means.values())})
    ranks=[]
    for p in baseline_ids:
        row={'policy':p,'label':baseline_labels[p]}
        for k in [4,32,None]:
            members=[r for r in cell_rows if r['policy']==p and (k is None or r['K']==k)]
            suffix=str(k) if k else 'overall'
            row['regret_'+suffix]=float(np.mean([r['regret'] for r in members]))
            row['rank_'+suffix]=float(np.mean([r['rank'] for r in members]))
        ranks.append(row)
    ranks.sort(key=lambda r:r['rank_overall'])
    null=load_rows('null');null_counts=[]
    for g in range(3):
        members=[r for r in null if r['geometry_index']==g]
        if len(members)!=4000: raise ValueError('Null calibration census')
        k,t=members[0]['K'],members[0]['T']
        null_counts.append({'K':k,'T':t,'runs':len(members),'crossings':sum(r['maximum_log_e']>=math.log(k/.05) for r in members)})
    calibration=load_rows('calibration');groups=defaultdict(list)
    for r in calibration: groups[r['task']['K'],r['task']['B']].append(r['calibration'])
    for (k,b),members in groups.items():
        threshold=_matched_threshold(np.array([r['maximum_glr'] for r in members]),sum(r['e_process_crossed'] for r in members))
        if threshold!=load_json('calibrated_thresholds.json')[f'k{k}_b{b}']['glr_threshold']: raise ValueError('Fitted threshold mismatch')
    report={'gate_table':table,'replay_leaderboard':ranks,'replay_cells':cell_rows,
            'counts':{'registered_random_nonstationary':len(nonstat),'registered_random_stationary':len(stationary),'registered_stress':len(registered)-len(random),'replay':len(replay),'replay_unique_means':606,'calibration':len(calibration)},
            'stationary':{p:mean(stationary,p) for p in GATES},
            'stationary_authorized_arms':sum(r['outcomes']['e-ats']['authorized_arms'] for r in stationary),
            'calibrated_stationary_crossings':sum(r['outcomes']['calibrated-glr']['authorized_arms']>0 for r in stationary),
            'calibrated_regret':mean(nonstat,'calibrated-glr'),
            'calibrated_difference':float(np.mean([difference(r,'calibrated-glr') for r in nonstat])),
            'calibration_zero_crossing_geometries':sum(not any(r['e_process_crossed'] for r in members) for members in groups.values()),
            'null_crossings':null_counts,'stress_e_minus_open':float(np.mean([difference(r,'e-ats','open-ramp') for r in stress])),
            'registered_by_k':{k:float(np.mean([difference(r,'open-ramp') for r in nonstat if r['task']['K']==k])) for k in K_VALUES},
            'registered_by_b':{b:float(np.mean([difference(r,'open-ramp') for r in nonstat if r['task']['B']==b])) for b in B_VALUES},
            'registered_by_mechanism':{m:float(np.mean([difference(r,'open-ramp') for r in nonstat if r['task']['mechanism']==m])) for m in NONSTATIONARY},
            'replay_by_k':{k:float(np.mean([difference(r,'open-ramp') for r in replay if r['task']['K']==k])) for k in (4,32)},
            'replay_open_ramp_groups_better':sum(difference(r,'open-ramp')<0 for r in replay),
            'replay_open_ramp_cells_better':sum(mean(v,'open-ramp')<mean(v,'e-ats') for v in by_cell.values()),
            'replay_practical_glr_cells_better':sum(mean(v,'practical-glr')<mean(v,'e-ats') for v in by_cell.values())}
    for label,rows in [('registered',nonstat),('replay',replay)]:
        e,opened=mean(rows,'e-ats'),mean(rows,'open-ramp')
        report[label+'_open_minus_e_percent_of_e']=100*(opened-e)/e
        report[label+'_e_minus_open_percent_of_open']=100*(e-opened)/opened
        report[label+'_authorization_fraction']=float(np.mean([r['outcomes']['e-ats']['authorized_arms']/r['task']['K'] for r in rows]))
        field='mechanism' if label=='registered' else 'environment_family'
        report[label+'_authorization_by_family']={f:float(np.mean([r['outcomes']['e-ats']['authorized_arms']/r['task']['K'] for r in rows if r['task'][field]==f])) for f in sorted({r['task'][field] for r in rows})}
    if bootstrap:
        print('Recomputing 50,000-resample intervals for the paper...',flush=True)
        samples=registered_bootstrap(nonstat,[('e-ats','open-ramp')],seed('e_ats_full_budget_20_geometry_regret_v1/nonstationary_bootstrap'))[:,0]
        cis={'registered_e_minus_open':interval(samples),'registered_primary_upper_one_sided':float(np.quantile(samples,.95,method='higher'))}
        print('Primary registered interval verified; recomputing GLR intervals...',flush=True)
        samples=registered_bootstrap(nonstat,[('practical-glr','e-ats'),('certified-glr','e-ats')],seed('e_ats_time_uniform_glr_gate_full_grid_v1/bootstrap/matched_directions'))
        cis['registered_practical_minus_e']=interval(samples[:,0]);cis['registered_certified_minus_e']=interval(samples[:,1])
        cis['registered_calibrated_minus_e']=calibrated_bootstrap(nonstat)
        print('Recomputing replay intervals with the paper’s two sampling conventions...',flush=True)
        cis['replay_e_minus_open_corrected']=replay_bootstrap(replay,corrected=True)[0]
        cis['replay_practical_minus_e'],cis['replay_certified_minus_e'],cis['replay_e_minus_open_original']=replay_bootstrap(replay,corrected=False)
        report['intervals']=cis
    return report

def verify_paper(report):
    expected=[('ots',.0494,.2157),('open-ramp',.0540,.1414),('e-ats',.0390,.1530),('certified-glr',.0418,.1750),('practical-glr',.0392,.1529)]
    for p,a,b in expected:
        row=next(r for r in report['gate_table'] if r['id']==p)
        if round(row['registered'],4)!=a or round(row['replay'],4)!=b: raise ValueError(f'Paper table mismatch: {p}')
    target=next(r for r in report['replay_leaderboard'] if r['policy']=='e-ats')
    if round(target['rank_overall'],2)!=9.72: raise ValueError('Paper rank mismatch')
    for key,expected in [('stationary_authorized_arms',1),('calibrated_stationary_crossings',7),('calibration_zero_crossing_geometries',17),('replay_open_ramp_groups_better',3204),('replay_open_ramp_cells_better',15),('replay_practical_glr_cells_better',13)]:
        if report[key]!=expected: raise ValueError(f'Paper claim mismatch: {key}')
    if [r['crossings'] for r in report['null_crossings']]!=[4,1,4]: raise ValueError('Null crossing count mismatch')
    for key,expected,digits in [
        ('registered_open_minus_e_percent_of_e',38.4,1),('replay_open_minus_e_percent_of_e',-7.5,1),
        ('registered_e_minus_open_percent_of_open',-27.7,1),('replay_e_minus_open_percent_of_open',8.2,1),
        ('registered_authorization_fraction',.0997,4),('replay_authorization_fraction',.3827,4),
        ('calibrated_regret',.0387,4)]:
        if round(report[key],digits)!=expected: raise ValueError(f'Paper claim mismatch: {key}')
    if 'intervals' in report:
        expected=load_json('paper_intervals.json')
        for name,value in report['intervals'].items():
            if not np.allclose(value,expected[name],rtol=0,atol=2e-13):
                raise ValueError(f'Paper interval mismatch: {name}: {value} != {expected[name]}')

def write_report(report, destination):
    out=output_directory(destination)
    (out/'paper_results.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in [('gate_comparison',report['gate_table']),('replay_leaderboard',report['replay_leaderboard']),('replay_cells',report['replay_cells'])]:
        with (out/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Recomputed paper results','', 'Normalized dynamic pseudo-regret; lower is better. Suites are separate populations.','',
           '| Policy | Registered | Replay |','|---|---:|---:|']
    for r in report['gate_table']: lines.append(f'| {r["policy"]} | {r["registered"]:.7f} | {r["replay"]:.7f} |')
    lines+=['','All counts, per-family results, authorization fractions and 17-method ranks are in `paper_results.json`.','',
            'Intervals recomputed with 50,000 resamples.' if 'intervals' in report else 'Intervals were not recomputed in this invocation; use `--bootstrap`.','']
    (out/'paper_results.md').write_text('\n'.join(lines))
    return out
