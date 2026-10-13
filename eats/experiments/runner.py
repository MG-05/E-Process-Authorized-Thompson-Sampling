"""Rerun the frozen paper groups; never substitute archived policy outcomes."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing as mp
import platform
import time
import numpy as np

from eats.io import ROOT, DATA, array_hash, load_json, load_rows, output_directory, rng
from eats.algorithms import baselines, parameters
from eats.algorithms.e_ats import RestartEProcessState, run_e_ats_full_budget_ramp, run_e_ats_full_budget_ramp_no_ce
from eats.algorithms.glr import TimeUniformGLRAuthorizationConfig, run_e_ats_time_uniform_glr_full_budget_ramp
from eats.algorithms.calibrated_glr import GLRAuthorizationConfig, run_e_ats_glr_full_budget_ramp
from eats.environments.registered import EnvironmentTask, bounded_generate
from eats.environments.replay import build_scale_environment
from eats.experiments.calibration_math import maximum_glr_on_trace
from eats.experiments.null_stream import _prior_predictive_table, _optimistic_ts, _rng
from eats.experiments.null_seeds import CALIBRATION_NAMESPACE_ID

GATES = ('e-ats', 'open-ramp', 'ots', 'certified-glr', 'practical-glr')
BASELINES = dict(zip(('dts','dots','aff-ts','rexp3','dynamic-ts','sw-ucb','d-ucb','m-ucb','cusum-ucb',
                     'glr-klucb','beta-swts','gamma-swgts','global-cts','d-linucb','ots','ts'),
                    (f'historical_{i:02d}' for i in range(4,20)),strict=True))

def runtime_platform():
    """Separate checkpoints by OS, architecture, and Python build."""
    return {'system': platform.system(), 'architecture': platform.machine(),
            'platform': platform.platform(), 'python_build': list(platform.python_build())}

def mean_table_drift(row, means):
    task = row['task']
    return ValueError(
        f'Mean-table drift: {row["group_id"]}\n'
        f'Expected shape {(task["T"], task["K"])}; observed {means.shape}.\n'
        f'Expected SHA-256: {row["mean_sha256"]}\n'
        f'Observed SHA-256: {array_hash(means, "<f8")}\n'
        f'Runtime: Python {platform.python_version()}, NumPy {np.__version__}, '
        f'{platform.platform()} ({platform.machine()}).\n'
        'Reference validation platform: macOS ARM64 with the pinned runtime. '
        'Version pins alone do not guarantee bitwise equality across platforms; '
        'this mismatch has not been accepted as rounding error. '
        'For saved-result tables and intervals, use '
        '`python -m eats reproduce --bootstrap --no-figures`. '
        'See docs/REPRODUCIBILITY.md for the exact-regeneration contract.')

def environment(row):
    task = row['task']
    if task['bank'] == 'fresh_atlas':
        means, _ = bounded_generate(EnvironmentTask(
            mechanism=task['mechanism'], n_arms=task['K'], budget=task['B'],
            source_kind=task['source_kind'], source_index=task['source_index'],
            environment_seed=task['environment_seed']))
    else:
        means = build_scale_environment(family=task['environment_family'], n_arms=task['K'],
            seed=task['environment_seed'], horizon=task['T']).means_tk
        if array_hash(means, '<f8') != row['mean_sha256']:
            records = load_json('replay_runtime_bridge.json')['records']
            match = next((r for r in records if r['task_id'] == task['task_id']), None)
            if match is None:
                raise mean_table_drift(row, means)
            path = DATA/'fixtures'/match['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != match['file_sha256']:
                raise ValueError('Runtime bridge file hash mismatch')
            means = np.load(path, allow_pickle=False)
    if means.shape != (task['T'], task['K']) or array_hash(means, '<f8') != row['mean_sha256']:
        raise mean_table_drift(row, means)
    mu = means.T
    uniforms = rng(row['potential_seed']).random(mu.shape)
    returns = np.asarray(uniforms < mu, dtype=np.uint8)
    if array_hash(uniforms, '<f8') != row['uniform_sha256'] or array_hash(returns, 'u1') != row['reward_sha256']:
        raise ValueError(f'Reward-tape drift: {row["group_id"]}')
    return mu, returns

@lru_cache
def baseline_config(k, policy):
    return next(r['candidate_signature'] for r in load_json('baseline_configurations.json')[str(k)] if r['row_id'] == policy)

def run_policy(policy, mu, returns, seed, budget):
    diagnostics = {'_record_played_arms': True}
    generator = rng(seed)
    kwargs = {'returns': returns, 'diagnostics': diagnostics}
    if policy == 'e-ats':
        rewards = run_e_ats_full_budget_ramp(mu, generator, **kwargs)
    elif policy == 'open-ramp':
        rewards = run_e_ats_full_budget_ramp_no_ce(mu, generator, **kwargs)
    elif policy == 'ots':
        rewards = baselines.run_dOTS(mu, parameters.dOTSParams(gamma=1.0), generator, **kwargs)
    elif policy in ('certified-glr', 'practical-glr'):
        kind = 'besson_lemma2' if policy == 'certified-glr' else 'besson_practical'
        rewards = run_e_ats_time_uniform_glr_full_budget_ramp(mu, generator,
            authorization=TimeUniformGLRAuthorizationConfig(boundary_kind=kind), **kwargs)
    elif policy == 'calibrated-glr':
        threshold = load_json('calibrated_thresholds.json')[f'k{mu.shape[0]}_b{budget}']['glr_threshold']
        rewards = run_e_ats_glr_full_budget_ramp(mu, generator,
            authorization=GLRAuthorizationConfig(threshold=threshold, calibration_id='paper-frozen-count-matching'), **kwargs)
    else:
        config = baseline_config(mu.shape[0], policy)
        runner = getattr(baselines, config['runner'].split('.')[-1])
        params = getattr(parameters, config['params_type'].split('.')[-1])(**config['params'])
        rewards = runner(mu, params, generator, **kwargs)
    actions = np.asarray(diagnostics['played_arms'], dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.uint8)
    if not np.array_equal(rewards, returns[actions, np.arange(mu.shape[1])]):
        raise ValueError('Policy rewards disagree with played actions')
    value = {'regret': float(np.mean(np.max(mu, axis=0)-mu[actions, np.arange(mu.shape[1])])),
             'actions_sha256': array_hash(actions, '<i8'), 'rewards_sha256': array_hash(rewards, 'u1')}
    if 'authorized_arm_count' in diagnostics:
        value['authorized_arms'] = int(diagnostics['authorized_arm_count'])
    return value, actions, rewards

def check_outcome(actual, expected, group, policy):
    for field, value in expected.items():
        equal = math.isclose(actual[field], value, abs_tol=2e-13, rel_tol=0) if field == 'regret' else actual.get(field) == value
        if not equal:
            raise ValueError(f'Frozen result mismatch: {group}, {policy}, {field}: {actual.get(field)} != {value}')

def monitor(actions, rewards, k):
    states = [RestartEProcessState(0.25) for _ in range(k)]
    maximum = 0.0
    for arm, reward in zip(actions, rewards, strict=True):
        maximum = max(maximum, states[int(arm)].update(int(reward)))
    return maximum

def run_group(job):
    suite, row, policies = job
    if suite == 'null':
        g, i = row['geometry_index'], row['trajectory_index']
        _, returns = _prior_predictive_table(g, i, namespace=CALIBRATION_NAMESPACE_ID)
        actions, rewards = _optimistic_ts(returns, _rng(CALIBRATION_NAMESPACE_ID, g, i, 1))
        for field, value in [('actions_sha256', array_hash(actions, '<i8')),
                             ('rewards_sha256', array_hash(rewards, 'i1')),
                             ('reward_table_sha256', array_hash(returns, 'i1'))]:
            if value != row[field]: raise ValueError(f'Null trace drift: {g}, {i}, {field}')
        maximum = monitor(actions, rewards, row['K'])
        if not math.isclose(maximum, row['maximum_log_e'], abs_tol=5e-10, rel_tol=0):
            raise ValueError('Null evidence drift')
        return {'group_id':f'null:{g}:{i}', 'maximum_log_e':maximum,
                'crossed':maximum >= math.log(row['K']/0.05), 'matches_frozen':True}
    mu, returns = environment(row)
    if suite == 'calibration':
        value, actions, rewards = run_policy('ots', mu, returns, row['policy_seed'], row['task']['B'])
        expected = row['calibration']
        if value['actions_sha256'] != expected['ots_action_sha256_int64_le'] or value['rewards_sha256'] != expected['ots_reward_sha256_uint8']:
            raise ValueError('Calibration OTS trace mismatch')
        maximum = maximum_glr_on_trace(actions, rewards, mu.shape[0])
        crossed = monitor(actions, rewards, mu.shape[0]) >= math.log(mu.shape[0]/0.05)
        if maximum != expected['maximum_glr'] or crossed != expected['e_process_crossed']:
            raise ValueError('Calibration crossing/statistic mismatch')
        return {'group_id':row['group_id'], 'maximum_glr':maximum, 'e_process_crossed':crossed, 'matches_frozen':True}
    output = {}
    for policy in policies:
        if policy not in row['outcomes']: continue
        value, _, _ = run_policy(policy, mu, returns, row['policy_seed'], row['task'].get('B',625))
        check_outcome(value, row['outcomes'][policy], row['group_id'], policy)
        output[policy] = value
    return {'group_id':row['group_id'], 'outcomes':output, 'matches_frozen':True}

def select_rows(suite, *, smoke, k=None, family=None, limit=None):
    rows = load_rows(suite)
    if k is not None: rows = [r for r in rows if r.get('task',r)['K'] == k]
    if family is not None: rows = [r for r in rows if r.get('task',{}).get('mechanism',r.get('task',{}).get('environment_family')) == family]
    if smoke:
        if suite == 'null': rows = [r for r in rows if r['trajectory_index'] == 0]
        elif suite == 'calibration': rows = [r for r in rows if r['task']['K']==(k or 4) and r['task']['B']==250 and r['task']['source_index']==5 and r['tape_index']==0]
        else:
            rows = [r for r in rows if r['task']['K'] == (k or 4)]
            if suite == 'registered':
                rows = [r for r in rows if r['task']['B']==250 and r['task']['source_kind']=='random_environment' and r['task']['source_index']==5]
            else: rows = [r for r in rows if r['environment_index']==0 and r['tape_index']==0]
    if limit is not None: rows = rows[:limit]
    if not rows: raise ValueError('No groups match the requested selection')
    return rows

def run(args):
    rows = select_rows(args.suite, smoke=args.smoke, k=args.k, family=args.family, limit=args.limit)
    if args.policies == ['all']:
        policies = list(GATES)
        if args.suite == 'replay': policies += [r['row_id'] for r in load_json('baseline_configurations.json')['4'] if r['row_id']!='historical_18']
    else: policies = [p if p in GATES else BASELINES.get(p,p) for p in args.policies]
    if args.suite in ('calibration','null'):
        if args.policies!=['all']:
            raise ValueError('Calibration and null experiments have a fixed OTS monitoring policy; omit --policies')
        policies=[]
    if len(set(policies))!=len(policies): raise ValueError('Choose each policy only once')
    allowed = set(GATES)|{'calibrated-glr'}|{r['row_id'] for r in load_json('baseline_configurations.json')['4']}
    if not set(policies) <= allowed: raise ValueError(f'Unknown policies: {set(policies)-allowed}')
    if args.suite in ('registered','replay'):
        if any(not any(p in r['outcomes'] for r in rows) for p in policies):
            raise ValueError('A selected policy has no outcomes in this suite')
        rows=[r for r in rows if any(p in r['outcomes'] for p in policies)]
    destination = output_directory(args.output or f'results/{args.suite}')
    config = {'suite':args.suite,'groups':[r.get('group_id',f'null:{r.get("geometry_index")}:{r.get("trajectory_index")}') for r in rows],
              'policies':policies,'python':platform.python_version(),'numpy':np.__version__,
              'runtime_platform':runtime_platform(),
              'release_manifest_sha256':hashlib.sha256((ROOT/'provenance/release.json').read_bytes()).hexdigest()}
    identity = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    path = destination/f'{identity[:16]}.jsonl'
    metadata = destination/f'{identity[:16]}.json'
    metadata.write_text(json.dumps({'identity':identity,**config},indent=2)+'\n')
    completed = {}
    if path.exists():
        raw=path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            # Only a final incomplete append can be discarded; complete rows are checked below.
            end=raw.rfind(b'\n')+1
            with path.open('r+b') as stream: stream.truncate(end)
        expected_rows=dict(zip(config['groups'],rows,strict=True))
        for line in path.read_text().splitlines():
            value=json.loads(line)
            if value.get('matches_frozen') is not True: raise ValueError('Unverified checkpoint row')
            if value['group_id'] in completed: raise ValueError('Duplicate checkpoint row')
            expected=expected_rows[value['group_id']]
            if args.suite in ('registered','replay'):
                wanted={p:expected['outcomes'][p] for p in policies if p in expected['outcomes']}
                if set(value['outcomes'])!=set(wanted): raise ValueError('Checkpoint policy census mismatch')
                for p,outcome in wanted.items(): check_outcome(value['outcomes'][p],outcome,value['group_id'],p)
            elif args.suite=='calibration':
                for field in ('maximum_glr','e_process_crossed'):
                    if value[field]!=expected['calibration'][field]: raise ValueError('Calibration checkpoint mismatch')
            elif not math.isclose(value['maximum_log_e'],expected['maximum_log_e'],abs_tol=5e-10,rel_tol=0) or value['crossed']!=(expected['maximum_log_e']>=math.log(expected['K']/.05)):
                raise ValueError('Null checkpoint mismatch')
            completed[value['group_id']]=value
    pending = [r for r,gid in zip(rows,config['groups'],strict=True) if gid not in completed]
    print(f'{args.suite}: {len(rows)} groups, {len(pending)} pending; Python {platform.python_version()}, NumPy {np.__version__}',flush=True)
    start=time.monotonic()
    jobs=((args.suite,row,policies) for row in pending)
    executor = ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) if args.workers>1 else None
    try:
        results=executor.map(run_group,jobs,chunksize=1) if executor else map(run_group,jobs)
        with path.open('a') as stream:
            for count,result in enumerate(results,1):
                stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush()
                if count==1 or count%10==0 or count==len(pending):
                    print(f'{len(completed)+count}/{len(rows)} verified ({time.monotonic()-start:.1f}s)',flush=True)
    finally:
        if executor: executor.shutdown(cancel_futures=True)
    summary={'suite':args.suite,'groups_verified':len(rows),
             'policy_names':policies or ['OTS with e-process/GLR monitoring' if args.suite=='calibration' else 'OTS with e-process monitoring'],
             'complete_suite':not(args.smoke or args.k or args.family or args.limit),
             'checkpoint':path.name,'seconds_this_invocation':time.monotonic()-start}
    (destination/f'{identity[:16]}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
