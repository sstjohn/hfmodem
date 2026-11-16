# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Placed-domain compatibility and original C2 resource provenance."""
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
        'cml/cml_generate.c', 'cml/COPYING.LESSER', 'NOTICE.md'}


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


@pytest.mark.parametrize('sl',[2,4])
def test_clean_independent_fixed_member_completes_declared_domain(sl):
    fixture=_fixed(sl)
    original=fixture['clean_symbols'].copy()
    result=target.decode_placed(original,speedlevel=sl)
    assert result.complete and result.reason=='unique' and result.frame is not None
    assert np.array_equal(original,fixture['clean_symbols'])
    assert result.iterations==8 and result.completed_members==result.eligible_members
    assert result.component_calls==16*result.eligible_members
    assert result.crc_checks==10*result.eligible_members
    assert result.constituent_positions==target._geometry(sl).systematic*result.component_calls
    assert result.component_calls<=(64 if sl==2 else 128)
    if sl==4:assert result.eligible_members==8
    assert result.identity_count==1 and not result.ambiguous
    assert result.selected_provenance is not None
    assert len(result.provenance)==result.crc_checks
    expected=fixture['wire'].tobytes()
    actual=result.frame.data+bytes((result.frame.status.byte,))+int(result.frame.crc).to_bytes(2,'little')
    assert actual==expected


@pytest.mark.parametrize('budget',[0,1,2,3,8,16,17,127,128])
def test_actual_sf8_component_budget_never_publishes_partial_domain(budget):
    fixture=_fixed(4)
    result=target.decode_placed(fixture['clean_symbols'],speedlevel=4,component_budget=budget)
    # An unfinished tier is resumable,
    # while publication still requires the complete declared member domain.
    # C1 and the saved-C1 half of turbo iteration one each cost one call;
    # subsequent member iterations cost two calls and cannot be split.
    expected = budget if budget <= 16 else 16 + 2*((budget-16)//2)
    assert result.component_calls == expected
    checks = 8 + min(expected, 16) + max(0, expected-16)//2
    assert result.crc_checks == len(result.provenance) == checks
    declared = [(method, iteration, member)
                for method, iteration in [('systematic', 0), ('c1-only', 0)]
                    + [('turbo', iteration) for iteration in range(1, 9)]
                for member in range(8)]
    assert [(p.method, p.iteration, p.member_index) for p in result.provenance] == declared[:checks]
    assert result.eligible_members == 8
    assert result.identity_count == 1 and not result.ambiguous
    assert result.complete == (budget == 128)
    if budget < 128:
        assert result.reason=='incomplete-work' and result.frame is None
        assert result.completed_members == 0
    else:
        assert result.reason=='unique' and result.frame is not None
        assert result.completed_members == result.eligible_members == 8
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
