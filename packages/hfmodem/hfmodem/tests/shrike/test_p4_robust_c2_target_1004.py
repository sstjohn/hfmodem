"""Approved placed-domain orchestration; no recorded target recovery before freeze."""
from pathlib import Path
import json
import hashlib
import tomllib
import numpy as np
import pytest
from hfmodem.shrike import p4c2, p4robustc2 as target

FIXTURES = Path(__file__).parent/'fixtures/robust-c2-oracle-1004'


def _fixed(sl, seed=100300):
    if not (FIXTURES/'FIXED-PANEL.json').is_file():
        pytest.skip("private independent robust fixture panel not present")
    panel = json.loads((FIXTURES/'FIXED-PANEL.json').read_bytes())
    row = next(r for r in panel['cases'] if r['speedlevel']==sl and r['seed']==seed and r['es_n0_db']==24)
    path = FIXTURES/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['file_sha256']
    with np.load(path, allow_pickle=False) as data:
        return {name:data[name].copy() for name in ('clean_symbols','systematic','wire','serialized')}


@pytest.mark.parametrize('sl', [2, 4])
def test_canonical_resources_match_original_provenance_and_are_immutable(sl):
    order = target._resources()[sl]
    g = target._geometry(sl)
    assert len(order)==g.systematic
    assert np.array_equal(np.sort(order),np.arange(g.systematic))
    with pytest.raises(ValueError): order[0]=0
    if not (FIXTURES/f'c2-sl{sl}.u16le').is_file():
        pytest.skip("private original robust C2 reference not present")
    oracle_raw = (FIXTURES/f'c2-sl{sl}.u16le').read_bytes()
    assert order.astype('<u2').tobytes()==oracle_raw


@pytest.mark.parametrize('which', ['missing', 'short', 'swapped', 'corrupt'])
def test_generated_table_failure_precedes_zero_input_refusal(monkeypatch, which):
    original = p4c2.permutation
    def broken(size):
        if size != 1160:
            return original(size)
        if which == 'missing':
            raise OSError('missing swap resource')
        order = original(size).copy()
        if which == 'short': return order[:-1]
        if which == 'swapped': return np.argsort(order)
        order[0] ^= 1
        return order
    monkeypatch.setattr(p4c2, 'permutation', broken)
    with pytest.raises((ValueError, OSError)):
        target.decode_placed(np.zeros(349), speedlevel=2, component_budget=0)


def test_only_replay_swaps_and_proof_sources_declared_in_package_data():
    root = Path(__file__).resolve().parents[3]
    config = tomllib.loads((root/'pyproject.toml').read_text())
    data = config['tool']['setuptools']['package-data']
    assert data['hfmodem.shrike'] == ['assets/c2-*.swaps.z']
    assert set(data['hfmodem.shrike.c2proof']) == {
        'cml_generate.c', 'upstream/srandom.h', 'NOTICE.md', 'COPYING.LESSER'}


@pytest.mark.parametrize('sl',[True,False,3,1,5,2.0,'2'])
def test_invalid_speedlevel(sl):
    with pytest.raises(ValueError):target.decode_placed(np.zeros(349),speedlevel=sl)


@pytest.mark.parametrize('iterations',[True,-1,9,1.5,'8'])
def test_invalid_iteration_domain(iterations):
    with pytest.raises(ValueError):target.decode_placed(np.zeros(349),speedlevel=2,iterations=iterations)


@pytest.mark.parametrize('budget',[True,-1,1.5,'8'])
def test_invalid_component_budget(budget):
    with pytest.raises(ValueError):target.decode_placed(np.zeros(349),speedlevel=2,component_budget=budget)


@pytest.mark.parametrize('sl',[2,4])
@pytest.mark.parametrize('kind',['short','extra','rank','nan','infinity'])
def test_input_shape_and_finiteness_before_decoder(sl,kind):
    size=target._geometry(sl).symbols+1
    raw=np.zeros(size,complex)
    if kind=='short':raw=raw[:-1]
    if kind=='extra':raw=np.append(raw,0)
    if kind=='rank':raw=raw.reshape(1,-1)
    if kind=='nan':raw[0]=np.nan
    if kind=='infinity':raw[0]=np.inf
    with pytest.raises(ValueError):target.decode_placed(raw,speedlevel=sl)


@pytest.mark.parametrize('sl',[2,4])
def test_all_zero_precedence_without_kernel_or_budget(monkeypatch,sl):
    def forbidden(*args,**kwargs):raise AssertionError('decoder work for zero input')
    monkeypatch.setattr(target,'_eligible_members',forbidden)
    monkeypatch.setattr(target,'_decode_member',forbidden)
    result=target.decode_placed(np.zeros(target._geometry(sl).symbols+1),speedlevel=sl,component_budget=0)
    assert result.complete and result.reason=='no-hit' and result.frame is None
    assert result.eligible_members==result.completed_members==result.component_calls==result.crc_checks==0


def test_nonzero_wrong_layout_precedes_budget(monkeypatch):
    monkeypatch.setattr(target.p4robustrx,'syndrome',lambda fields,sl:1.0)
    result=target.decode_placed(np.ones(349),speedlevel=2,component_budget=0)
    assert result.complete and result.reason=='layout-ineligible' and result.frame is None
    assert result.eligible_members==result.component_calls==result.crc_checks==0


@pytest.mark.parametrize('layout',[2,3])
def test_exact_layout_ranking_keeps_all_equal_best_turns(monkeypatch,layout):
    monkeypatch.setattr(target.p4robustrx,'syndrome',lambda fields,sl:1.0 if sl==layout else 0.9)
    members=target._eligible_members(np.ones(349, dtype=complex),target._geometry(2))
    assert [m.quarter_turn for m in members] == (list(range(4)) if layout==2 else [])


def test_sf8_declared_order_is_all_unit_then_all_amplitude():
    members=target._eligible_members(_fixed(4)['clean_symbols'],target._geometry(4))
    assert [(m.weighting,m.quarter_turn) for m in members]==[
        (weighting,turn) for weighting in ('unit','amplitude') for turn in range(4)]
    assert all(len(m.fields)==1392 for m in members)


@pytest.mark.parametrize('budget',[0,1,2,None])
def test_sticky_ambiguity_and_incomplete_precedence_with_controlled_candidates(monkeypatch,budget):
    a,b=_fixed(4),_fixed(4,100301)
    raw=a['clean_symbols'];unchanged=raw.copy()
    def members(raw,g):return (target._Member('unit',0,np.ones(1392)),)
    def kernel(member,g,order,iterations,work):
        yield target._Candidate('systematic',0,a['systematic'])
        work.call('c1-only',g.systematic,lambda:None)
        yield target._Candidate('c1-only',0,b['systematic'])
        work.call('iteration-1-c1',g.systematic,lambda:None)
        yield target._Candidate('turbo',1,a['systematic'])
    monkeypatch.setattr(target,'_eligible_members',members)
    monkeypatch.setattr(target,'_decode_member',kernel)
    result=target.decode_placed(raw,speedlevel=4,iterations=1,component_budget=budget)
    assert np.array_equal(raw,unchanged)
    assert result.frame is None
    assert result.identity_count==(1 if budget==0 else 2)
    assert result.ambiguous==(budget!=0)
    assert result.complete==(budget is None or budget>=2)
    assert result.reason==('ambiguous' if result.complete else 'incomplete-work')
    assert result.completed_members==int(result.complete)
    assert result.crc_checks==(1 if budget==0 else 2 if budget==1 else 3)
    assert len(result.identities)==result.identity_count
    assert len(result.provenance)==result.crc_checks


def test_constituent_exchange_and_transmitted_suffix_are_exact(monkeypatch):
    from types import SimpleNamespace
    g=target._geometry(4)
    order=target._resources()[4]
    fields=np.linspace(-2.0,3.0,2*g.symbols)
    member=target._Member('unit',0,fields)
    calls=[]
    def component(systematic,parity,apriori=None,*,initial_state=None,end_state=None):
        calls.append((systematic.copy(),parity.copy(),None if apriori is None else apriori.copy(),initial_state,end_state))
        index=len(calls)
        return SimpleNamespace(posterior=np.full(g.systematic,index,dtype=float),
                               extrinsic=np.arange(g.systematic,dtype=float)+index)
    monkeypatch.setattr(target.p4decode,'component_decode',component)
    work=target._Work(None)
    outcomes=list(target._decode_member(member,g,order,1,work))
    assert [(o.method,o.iteration) for o in outcomes]==[('systematic',0),('c1-only',0),('turbo',1)]
    assert work.calls==3 and work.positions==3*g.systematic
    assert [c[3:] for c in calls]==[(0,0),(0,0),(0,None)]
    assert calls[0][2] is None and np.array_equal(calls[1][2],np.zeros(g.systematic))
    expected1=np.zeros(g.systematic);expected1[::10]=fields[g.systematic:g.systematic+g.parity]
    expected2=np.zeros(g.systematic);expected2[4::10]=fields[g.systematic+g.parity:]
    assert np.array_equal(calls[0][1],expected1) and np.array_equal(calls[1][1],expected1)
    assert np.array_equal(calls[2][1],expected2)
    assert np.array_equal(calls[2][0],fields[:g.systematic][order])
    assert np.array_equal(calls[2][2],(np.arange(g.systematic)+2)[order])
    assert calls[0][1][1150]!=0 and calls[2][1][1154]!=0
    assert len(calls[2][0])==1160


@pytest.mark.parametrize('sl',[2,4])
def test_clean_independent_fixed_member_completes_declared_domain(sl):
    fixture=_fixed(sl)
    original=fixture['clean_symbols'].copy()
    result=target.decode_placed(original,speedlevel=sl)
    assert result.complete and result.reason=='unique' and result.frame is not None
    assert np.array_equal(original,fixture['clean_symbols'])
    assert result.iterations==8 and result.completed_members==result.eligible_members
    assert result.component_calls==17*result.eligible_members
    assert result.crc_checks==10*result.eligible_members
    assert result.constituent_positions==target._geometry(sl).systematic*result.component_calls
    assert result.component_calls<=(68 if sl==2 else 136)
    if sl==4:assert result.eligible_members==8
    assert result.identity_count==1 and not result.ambiguous
    assert result.selected_provenance is not None
    assert len(result.provenance)==result.crc_checks
    expected=fixture['wire'].tobytes()
    actual=result.frame.data+bytes((result.frame.status.byte,))+int(result.frame.crc).to_bytes(2,'little')
    assert actual==expected


@pytest.mark.parametrize('budget',[0,1,2,3,17,135,136])
def test_actual_sf8_component_budget_never_publishes_partial_domain(budget):
    fixture=_fixed(4)
    result=target.decode_placed(fixture['clean_symbols'],speedlevel=4,component_budget=budget)
    assert result.component_calls==budget
    assert result.complete==(budget==136)
    if budget<136:
        assert result.reason=='incomplete-work' and result.frame is None
    else:
        assert result.reason=='unique' and result.frame is not None
    assert result.component_calls*1160==result.constituent_positions


def test_systematic_only_is_explicit_zero_iteration_domain():
    result=target.decode_placed(_fixed(4)['clean_symbols'],speedlevel=4,iterations=0,component_budget=0)
    assert result.complete and result.reason=='unique'
    assert result.iterations==0 and result.component_calls==0
    assert result.eligible_members==result.completed_members==result.crc_checks==8
    assert all(p.method=='systematic' for p in result.provenance)


@pytest.mark.parametrize('sl',[2,4])
@pytest.mark.parametrize('turn',range(4))
def test_independent_clean_field_preserved_under_all_carrier_quarter_turns(sl,turn):
    fixture=_fixed(sl)
    # Incremental carrier phase rotates every differential by a quarter-turn;
    # this exercises the declared ambiguity domain, not just common phase.
    symbols=fixture['clean_symbols']*(1j**turn)**np.arange(len(fixture['clean_symbols']))
    result=target.decode_placed(symbols,speedlevel=sl)
    assert result.complete and result.frame is not None
    expected=fixture['wire'].tobytes()
    actual=result.frame.data+bytes((result.frame.status.byte,))+int(result.frame.crc).to_bytes(2,'little')
    assert actual==expected


@pytest.mark.parametrize('sl',[2,4])
def test_native_symbol_scatter_not_reversed_and_old_sl2_guess_not_c2(monkeypatch,sl):
    fixture=_fixed(sl)
    g=target._geometry(sl)
    members=target._eligible_members(fixture['clean_symbols'],g)
    assert any(np.array_equal(m.fields<0,fixture['serialized']) for m in members)
    if sl==2:
        native=target._resources()[2]
        guessed=target.p4robustrx._walk(348,18)
        assert np.count_nonzero(native==guessed)==331
        monkeypatch.setattr(target.p4robustrx,'_SYMBOL_ORDER',target.p4robustrx._SYMBOL_ORDER[::-1])
    else:
        monkeypatch.setattr(target.p4sf8rx,'ORDER',target.p4sf8rx.ORDER[::-1])
    wrong=target._eligible_members(fixture['clean_symbols'],g)
    assert not any(np.array_equal(m.fields<0,fixture['serialized']) for m in wrong)
