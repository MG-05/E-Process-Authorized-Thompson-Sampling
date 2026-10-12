import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import numpy as np
import pytest

from eats import run_e_ats
from eats.algorithms.e_ats import RestartEProcessState
from eats.algorithms.baselines import run_dOTS
from eats.algorithms.parameters import dOTSParams
from eats.analysis import compute, verify_paper
from eats.experiments.runner import environment, run_group, select_rows, GATES, check_outcome
from eats.io import ROOT, load_rows, load_json, verify_integrity, output_directory

def test_release_integrity():
    assert verify_integrity()>40

def test_extracted_function_bodies_match_original_symbol_hashes():
    available={}
    for row in json.loads((ROOT/'provenance/code.json').read_text()):
        path=row['destination']
        if path not in available:
            text=(ROOT/path).read_text();lines=text.splitlines();hashes=set()
            for node in ast.parse(text).body:
                start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
                segment='\n'.join(lines[start-1:node.end_lineno])
                hashes.add(hashlib.sha256(segment.encode()).hexdigest())
            available[path]=hashes
        assert row['sha256'] in available[path],row

def test_e_process_against_exact_beta_bernoulli_marginals():
    def marginal(bits):
        successes=sum(bits);failures=len(bits)-successes
        return math.factorial(successes)*math.factorial(failures)/math.factorial(len(bits)+1)
    for bits in itertools.product((0,1),repeat=8):
        state=RestartEProcessState(.25)
        for n,reward in enumerate(bits,1):
            log_e=state.update(reward)
            starts=[];s=2
            while s<=n:
                starts.append(s);s=max(s+1,math.ceil(1.25*s))
            exact=1/(len(starts)+1)
            for j,s in enumerate(starts,1):
                exact+=marginal(bits[:s-1])*marginal(bits[s-1:n])/marginal(bits[:n])/(j*(j+1))
            assert math.exp(log_e)==pytest.approx(exact,rel=5e-14)

def test_no_authorization_preserves_anchor_actions_rewards_and_rng_state():
    mu=np.full((4,500),.5);returns=np.zeros_like(mu,dtype=np.uint8)
    r1,r2=np.random.default_rng(771),np.random.default_rng(771)
    e,ots={}, {'_record_played_arms':True}
    a=run_e_ats(mu,r1,returns,e)
    b=run_dOTS(mu,dOTSParams(gamma=1.),r2,returns=returns,diagnostics=ots)
    assert e['authorized_arm_count']==0
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(e['played_arms'],ots['played_arms'])
    assert r1.bit_generator.state==r2.bit_generator.state

@pytest.mark.parametrize('suite',['registered','replay','calibration','null'])
def test_frozen_group_reexecution(suite):
    row=select_rows(suite,smoke=True)[0]
    assert run_group((suite,row,list(GATES)))['matches_frozen']

def test_all_seventeen_replay_methods_match_archived_actions():
    row=select_rows('replay',smoke=True)[0]
    policies=[r['row_id'] for r in load_json('baseline_configurations.json')['4']]+['e-ats']
    assert len(run_group(('replay',row,policies))['outcomes'])==17

def test_corrupted_input_and_output_hashes_fail_closed():
    row=select_rows('registered',smoke=True)[0]
    with pytest.raises(ValueError,match='Reward-tape'):
        environment({**row,'reward_sha256':'0'*64})
    with pytest.raises(ValueError,match='Frozen result mismatch'):
        check_outcome({'actions_sha256':'x'},{'actions_sha256':'y'},'group','policy')

def test_output_cannot_overwrite_inputs():
    with pytest.raises(ValueError): output_directory(ROOT/'data'/'new')

def test_all_paper_leaderboard_numbers_and_claims():
    report=compute();verify_paper(report)
    manuscript=(ROOT/'paper/appendix.tex').read_text()
    table=manuscript.split(r'\label{tab:literature-ranks}')[1].split(r'\end{tabular}')[0]
    values=[]
    for line in table.splitlines():
        cells=line.split('&')
        if len(cells)==7:
            numbers=[re.search(r'\d+\.\d+',c) for c in cells[1:]]
            if all(numbers): values.append([m.group() for m in numbers])
    assert len(values)==17
    for printed,row in zip(values,report['replay_leaderboard'],strict=True):
        for expected,field,decimals in zip(printed,['regret_4','rank_4','regret_32','rank_32','regret_overall','rank_overall'],[4,2,4,2,4,2],strict=True):
            assert f'{row[field]:.{decimals}f}'==expected,(row['label'],field)

def test_only_manuscript_figures_are_shipped():
    tex='\n'.join(p.read_text() for p in (ROOT/'paper').glob('*.tex'))
    used=set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex))
    shipped={str(p.relative_to(ROOT/'paper')) for p in (ROOT/'paper/figures').glob('*')}
    assert used==shipped
