# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Standalone normal-LONG gates; no live receiver or transmitter imports."""
import importlib.util
import math
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.shrike import p4decode, p4longrates as rates

ROOT = Path(__file__).resolve().parents[5]
ORACLE = Path(os.environ.get("P4_NORMAL_LONG_ORACLE", str(ROOT / "working/p4-plan-0930/normal-long-rates-oracle-1003/oracle.py")))


MEASUREMENTS = []


def _cost(result, stage, sl):
    MEASUREMENTS.append(dict(stage=stage, speedlevel=sl,
                             seconds=result.elapsed_seconds,
                             iterations=result.iterations_done,
                             complete=result.complete,
                             identities=result.identity_count))


@pytest.fixture(scope="module", autouse=True)
def _preserve_costs():
    yield
    target = os.environ.get("NORMAL_LONG_COSTS_PATH")
    if target:
        with Path(target).open("x") as stream:
            json.dump(MEASUREMENTS,stream,indent=2)
            stream.write("\n")


@pytest.fixture(scope="module")
def oracle():
    if not ORACLE.is_file():
        pytest.skip("set P4_NORMAL_LONG_ORACLE to the private independent scalar oracle")
    spec = importlib.util.spec_from_file_location("normal_long_independent_oracle", ORACLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sl,d,n,p", [(6,4140,517,1),(7,8280,1034,5),(8,12420,1552,1),(9,16560,2069,5),(10,20700,2587,1)])
def test_geometry_and_immutable_native_bijections(sl, d, n, p):
    g = rates.geometry(sl)
    assert (g.systematic, g.net_bytes, g.padding, g.parity_each) == (d,n,p,d//10)
    order = rates.component_permutation(sl)
    assert np.array_equal(np.sort(order), np.arange(d))
    assert not order.flags.writeable
    with pytest.raises(ValueError):
        order.setflags(write=True)
    assert rates.constellation(sl).shape == (1 << g.width,)
    assert rates.symbol_permutation().shape == (4968,)


@pytest.mark.parametrize("sl,energy", [(6,1.),(7,1.),(8,.9998931884765625),(9,.7834351360797882),(10,.6502655297517776)])
def test_native_constellation_mean_energy(sl, energy):
    assert float(np.mean(abs(rates.constellation(sl))**2)) == energy


@pytest.mark.parametrize("sl", rates.LEVELS)
def test_demap_exact_logsum_marginals(sl):
    points = rates.constellation(sl)
    z = np.resize(np.array([.3+.7j,-.9+.15j,0j]), rates.SYMBOLS)
    n0 = .41
    result = rates.demap(z, n0, sl).reshape(rates.SYMBOLS, sl-5)
    for logical in [0,1,101,4967]:
        observed = z[rates.symbol_permutation()[logical]]
        likelihood = [math.exp(-abs(observed-x)**2/n0) for x in points]
        for column in range(sl-5):
            bit = sl-6-column
            expected = math.log(math.fsum(v for i,v in enumerate(likelihood) if not (i>>bit)&1)) - math.log(math.fsum(v for i,v in enumerate(likelihood) if (i>>bit)&1))
            assert result[logical,column] == pytest.approx(expected, rel=2e-14, abs=2e-14)


def test_bpsk_complex_noise_factor_and_body_energy_proxy():
    z = np.linspace(-2,2,rates.SYMBOLS) + .3j
    assert np.allclose(rates.demap(z,.7,6), (4*z.real/.7)[rates.symbol_permutation()], rtol=2e-14, atol=2e-14)
    ob = rates.BodyObservation(z, .7+.2j, .13, 2.3, .07)
    for sl in rates.LEVELS:
        es = float(np.mean(abs(rates.constellation(sl))**2))
        assert ob.noise(sl) == (.13*2.3+es*.07)/abs(.7+.2j)**2


@pytest.mark.parametrize("level", [True,5,11,6.5,"6"])
def test_invalid_level(level):
    with pytest.raises(ValueError): rates.geometry(level)


@pytest.mark.parametrize("iterations", [True,0,9,1.5])
def test_invalid_iterations(iterations):
    with pytest.raises(ValueError): rates.decode_fields(np.zeros(4968),6,iterations)


@pytest.mark.parametrize("noise", [0,-1,float('nan'),float('inf'),1j,True])
def test_invalid_complex_noise(noise):
    with pytest.raises(ValueError): rates.demap(np.ones(4968),noise,6)


def test_invalid_arrays_and_declared_domains():
    for values in [np.ones(4967),np.ones((1,4968)),np.full(4968,np.nan),np.ones(4968,dtype=complex)]:
        with pytest.raises(ValueError): rates.decode_fields(values,6)
    for modes in [(),(6,6),(6,11),(True,),([6],)]:
        with pytest.raises(ValueError): rates.decode_equalized(np.ones(4968),1,speedlevels=modes)
    for budget in [True,-1,.5]:
        with pytest.raises(ValueError): rates.decode_fields(np.ones(4968),6,max_work=budget)


def test_zero_and_incomplete_work_never_publish():
    zero = rates.decode_equalized(np.zeros(4968),1,max_work=0)
    assert zero.complete and zero.reason == 'no-hit' and zero.frame is None
    assert zero.iterations_done == zero.crc_checks == zero.identity_count == 0
    for sl in rates.LEVELS:
        fields = rates.decode_fields(np.zeros(4968*(sl-5)),sl,max_work=0)
        assert not fields.complete and fields.frame is None and fields.crc_checks == 0
        assert fields.reason == "incomplete" and fields.modes_completed == ()
        assert fields.declared_modes == (sl,) and fields.iterations_done == 0
    incomplete = rates.decode_equalized(np.ones(4968),1,max_work=0)
    assert not incomplete.complete and incomplete.reason == 'incomplete'
    assert incomplete.frame is None and incomplete.modes_completed == ()


def _payload(sl, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0,256,rates.geometry(sl).data_bytes,dtype=np.uint8).tobytes()


@pytest.mark.parametrize("sl", rates.LEVELS)
def test_independent_clean_field_and_symbol_roundtrip(sl, oracle):
    data = _payload(sl, 100300+sl)
    status = (0,35,68,131,255)[sl-6]
    fixture = oracle.generate(sl,data,status)
    fields = (1-2*np.asarray(fixture.fields,dtype=float))*8
    result = rates.decode_fields(fields,sl)
    _cost(result,"clean-fields",sl)
    assert result.complete and result.identity_count == 1 and result.iterations_done == 8
    assert result.frame is not None and result.frame.data == data and result.frame.status.byte == status
    from_symbols = rates.decode_equalized(fixture.symbols,.025,speedlevels=(sl,))
    _cost(from_symbols,"clean-symbols",sl)
    assert from_symbols.complete and from_symbols.identity_count == 1
    assert from_symbols.frame is not None and from_symbols.frame.data == data
    assert from_symbols.frame.status.byte == status
    for decoded in (result, from_symbols):
        assert decoded.declared_modes == (sl,)
        assert decoded.declared_iterations == 8
        assert decoded.identities == (decoded.frame,)
        assert len(decoded.outcomes) == 8
        assert all(o.crc_valid and o.padding_valid and o.flush_valid for o in decoded.outcomes)
        assert all(o.identity_index == 0 for o in decoded.outcomes)
        assert all(o.reencoding.full_hard_disagreements == 0 for o in decoded.outcomes)
        costs = dict(decoded.stage_seconds)
        assert set(costs) == {"observe", "demap", "depuncture", "constituent-c1", "constituent-c2", "crc", "reencode"}
        assert all(value >= 0 and np.isfinite(value) for value in costs.values())
        assert costs["depuncture"] > 0 and costs["constituent-c1"] > 0
        assert costs["constituent-c2"] > 0 and costs["crc"] > 0 and costs["reencode"] > 0
        assert sum(costs.values()) <= decoded.elapsed_seconds
    assert result.coordinates.input_stage == "serialized-llrs"
    assert all(o.reencoding.normalized_complex_distance is None for o in result.outcomes)
    assert from_symbols.coordinates.input_stage == "equalized-symbols"
    assert all(o.reencoding.normalized_complex_distance == 0 for o in from_symbols.outcomes)
    assert from_symbols.noise_diagnostics[0].provenance == "caller-supplied"
    assert not from_symbols.noise_diagnostics[0].calibrated_recording_snr
    g = rates.geometry(sl)
    regions = from_symbols.outcomes[0].reencoding.regions
    assert [(r.region, r.serialized_start, r.serialized_stop) for r in regions] == [
        ("systematic", 0, g.systematic),
        ("retained-c1", g.systematic, g.systematic + g.parity_each),
        ("retained-c2", g.systematic + g.parity_each, 4968*g.width),
        ("alignment-padding", 8*g.net_bytes, g.systematic-3),
        ("c1-flush", g.systematic-3, g.systematic)]


def test_later_iteration_ambiguity_is_sticky(monkeypatch, oracle):
    a=oracle.generate(6,_payload(6,11001),1)
    b=oracle.generate(6,_payload(6,11002),2)
    monkeypatch.setattr(rates,'_iterations_for',lambda fields,g,n:iter([a.systematic,b.systematic]+[a.systematic]*6))
    result=rates.decode_fields(np.ones(4968),6)
    assert result.complete and result.reason=='ambiguous' and result.frame is None and result.identity_count==2
    cut=rates.decode_fields(np.ones(4968),6,max_work=1)
    assert not cut.complete and cut.reason=='incomplete' and cut.identity_count==1 and cut.frame is None
    cut2=rates.decode_fields(np.ones(4968),6,max_work=2)
    assert not cut2.complete and cut2.reason=='incomplete' and cut2.identity_count==2 and cut2.frame is None


def test_later_mode_ambiguity_and_duplicate_identity(monkeypatch, oracle):
    fixtures={sl:oracle.generate(sl,_payload(sl,12000+sl),1) for sl in (6,7)}
    monkeypatch.setattr(rates,'_iterations_for',lambda fields,g,n:iter([fixtures[g.speedlevel].systematic]*n))
    result=rates.decode_equalized(np.ones(4968),1,speedlevels=(6,7))
    assert result.complete and result.reason=='ambiguous' and result.identity_count==2 and result.frame is None
    first=rates.decode_equalized(np.ones(4968),1,speedlevels=(6,))
    assert first.complete and first.identity_count==1 and first.frame is not None and first.iterations_done==8
    cut=rates.decode_equalized(np.ones(4968),1,speedlevels=(6,7),max_work=8)
    assert not cut.complete and cut.identity_count==1 and cut.frame is None and cut.modes_completed==(6,)
    assert len(result.identities) == 2
    assert result.identities[0].data != result.identities[1].data
    assert result.declared_modes == (6, 7)
    assert len(result.outcomes) == 16
    assert {o.identity_index for o in result.outcomes} == {0, 1}
    assert cut.declared_modes == (6, 7) and len(cut.identities) == 1
    assert cut.identities[0] == first.frame
    assert len(cut.outcomes) == cut.iterations_done


@pytest.mark.parametrize("bad", [True,-1,8,.5])
def test_invalid_component_endpoint(bad):
    with pytest.raises(ValueError): p4decode.component_decode(np.zeros(4),np.zeros(4),end_state=bad)


@pytest.mark.parametrize("sl", rates.LEVELS)
def test_heldout_equalized_noise_and_wrong_mode(sl, oracle):
    data = _payload(sl, 210040+sl)
    fixture = oracle.generate(sl,data,2)
    rng = np.random.default_rng(310040+sl)
    n0 = .001
    z = np.asarray(fixture.symbols) + math.sqrt(n0/2)*(rng.normal(size=4968)+1j*rng.normal(size=4968))
    result = rates.decode_equalized(z,n0,speedlevels=(sl,))
    _cost(result,"heldout-noise",sl)
    assert result.complete and result.frame is not None and result.frame.data == data
    wrong = 6 if sl == 10 else sl+1
    rejected = rates.decode_equalized(z,n0,speedlevels=(wrong,))
    _cost(rejected,"wrong-mode",wrong)
    assert rejected.complete and rejected.frame is None and rejected.identity_count == 0


def _body(symbols, *, seed, noise=.0005, echo=.12, cfo=2.7):
    from hfmodem.shrike import p4normal, p4rx
    # Observation test scaffold: established pilot/pulse implementation, with
    # independently encoded body symbols. No independent pilot proof is claimed.
    chips = np.zeros(25*32+24*207,complex)
    for i in range(25):
        at=i*(32+207)
        chips[at:at+32]=p4normal.training_sequence(i+1)
        if i<24: chips[at+32:at+239]=symbols[i*207:(i+1)*207]
    impulses=np.zeros(len(chips)*16,complex)
    impulses[::16]=chips
    z=np.pad(np.convolve(impulses,p4rx.RRC_TAPS),(93,200))
    z[16:]+=echo*np.exp(.8j)*z[:-16].copy()
    z*=np.exp(1j*(.7+2*np.pi*cfo*np.arange(len(z))/28800))
    rng=np.random.default_rng(seed)
    z+=noise*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
    return z


@pytest.mark.parametrize("sl", rates.LEVELS)
def test_placed_body_noise_cfo_and_channel(sl,oracle):
    data=_payload(sl,410040+sl)
    fixture=oracle.generate(sl,data,3)
    z=_body(fixture.symbols,seed=510040+sl)
    result=rates.decode_body(z,first_training_start=93,frequency_offset=2.7,speedlevels=(sl,))
    _cost(result,"placed-body",sl)
    assert result.complete and result.frame is not None
    assert result.frame.data==data and result.frame.status.byte==3
    assert result.observation is not None
    assert result.observation.coordinates == result.coordinates
    assert result.coordinates.input_samples == len(z)
    assert result.coordinates.first_training_start == 93
    assert result.coordinates.frequency_offset == 2.7
    diagnostic = result.noise_diagnostics[0]
    assert diagnostic.speedlevel == sl
    assert diagnostic.provenance == "pilot-residual-isi-proxy"
    assert diagnostic.convention == "total-complex-variance"
    assert not diagnostic.calibrated_recording_snr
    assert diagnostic.mean_symbol_energy == pytest.approx(np.mean(np.abs(oracle.constants(sl).points)**2))
    assert diagnostic.complex_variance == result.observation.noise(sl)
    assert dict(result.stage_seconds)["observe"] > 0
    assert dict(result.stage_seconds)["demap"] > 0
    assert all(o.reencoding.normalized_complex_distance is not None
               for o in result.outcomes if o.crc_valid)


def test_complex_observer_bpsk_equivalence_and_incomplete_body():
    from hfmodem.shrike import p4long
    rng=np.random.default_rng(610043)
    z=_body(1-2*rng.integers(0,2,4968),seed=710043)
    observed=rates.observe_body(z,first_training_start=93,frequency_offset=2.7)
    assert observed is not None
    fields=rates.demap(observed.symbols,observed.noise(6),6)
    original=p4long.body_scores(z,first_training_start=93,frequency_offset=2.7)
    assert np.allclose(fields,original,rtol=3e-13,atol=3e-10)
    incomplete=rates.decode_body(z[:-1000],first_training_start=93,frequency_offset=2.7)
    assert not incomplete.complete and incomplete.reason=='incomplete-input' and incomplete.frame is None
    zero=rates.decode_body(np.zeros(len(z)),first_training_start=93)
    assert zero.complete and zero.frame is None


def test_complete_all_mode_union_on_independent_sl10(oracle):
    data=_payload(10,810043)
    fixture=oracle.generate(10,data,2)
    result=rates.decode_equalized(fixture.symbols,.025)
    _cost(result,"all-five-modes",10)
    assert result.complete and result.modes_completed==rates.LEVELS
    assert result.iterations_done==40 and result.identity_count==1
    assert result.frame is not None and result.frame.data==data


def test_noise_wrong_order_and_conjugation_are_not_fields(oracle):
    fixture=oracle.generate(10,_payload(10,910043),1)
    rng=np.random.default_rng(1010043)
    for z in [rng.normal(size=4968)+1j*rng.normal(size=4968),
              np.asarray(fixture.symbols)[::-1],np.conj(fixture.symbols)]:
        result=rates.decode_equalized(z,.1,speedlevels=(10,))
        assert result.complete and result.frame is None and result.identity_count==0


def test_c2_causally_repairs_fixed_systematic_erasures(oracle,monkeypatch):
    fixture=oracle.generate(6,_payload(6,1110043),1)
    fields=(1-2*np.asarray(fixture.fields,dtype=float))*12
    rng=np.random.default_rng(1210043)
    erased=rng.choice(8*rates.geometry(6).data_bytes,16,replace=False)
    fields[erased]=0
    fields[4140:4554]=0
    good=rates.decode_fields(fields,6)
    assert good.complete and good.frame is not None and good.frame.data==_payload(6,1110043)
    wrong=np.roll(rates.component_permutation(6),1)
    monkeypatch.setattr(rates,'component_permutation',lambda sl:wrong)
    bad=rates.decode_fields(fields,6)
    assert bad.complete and bad.frame is None


@pytest.mark.parametrize("bad", [True,-1,8,.5])
def test_invalid_component_initial_state(bad):
    with pytest.raises(ValueError):
        p4decode.component_decode(np.zeros(7),np.zeros(7),initial_state=bad)


@pytest.mark.parametrize("which", ['padding','flush'])
def test_crc_without_declared_padding_or_flush_is_not_accepted(which,oracle,monkeypatch):
    fixture=oracle.generate(6,_payload(6,1310043),1)
    wrong=np.asarray(fixture.systematic).copy()
    wrong[8*rates.geometry(6).net_bytes if which=='padding' else -1] ^= 1
    monkeypatch.setattr(rates,'_iterations_for',lambda fields,g,n:iter([wrong]*n))
    result=rates.decode_fields(np.ones(4968),6)
    assert result.complete and result.identity_count==0 and result.frame is None


@pytest.mark.parametrize("sl", rates.LEVELS)
def test_target_constants_match_independent_native_extraction(sl,oracle):
    native=oracle.constants(sl)
    assert np.array_equal(rates.constellation(sl),native.points)
    assert np.array_equal(rates.component_permutation(sl),native.c2_order)
    assert np.array_equal(rates.symbol_permutation(),native.symbol_order)


@pytest.mark.parametrize("initial", [None,0])
@pytest.mark.parametrize("final", [None,0])
def test_component_boundaries_against_independent_all_path_likelihood(initial,final,oracle):
    rng=np.random.default_rng(1410043)
    systematic=rng.normal(size=7)
    parity=rng.normal(size=7)
    prior=rng.normal(size=7)*.3
    exact=oracle.trellis_logpost(systematic,parity,prior,initial_state=initial,final_state=final)
    got=p4decode.component_decode(systematic,parity,prior,initial_state=initial,end_state=final)
    assert exact.admitted_paths>0
    assert np.allclose(got.posterior,exact.posterior,rtol=2e-12,atol=2e-12)
    assert np.allclose(got.extrinsic,exact.extrinsic,rtol=2e-12,atol=2e-12)


def test_oracle_does_not_import_target_implementation(oracle):
    import ast
    tree=ast.parse(ORACLE.read_bytes())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom): imports.append(node.module or '')
    assert not any(name.startswith('hfmodem') for name in imports)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_fixed_boundary_certainty_and_empty_path_against_oracle(length, oracle):
    systematic = np.linspace(-0.7, 0.9, length)
    parity = np.linspace(0.4, -0.8, length)
    prior = np.linspace(-0.2, 0.3, length)
    feasible = impossible = certain = 0
    for initial in range(8):
        for final in range(8):
            try:
                expected = oracle.trellis_logpost(
                    systematic, parity, prior,
                    initial_state=initial, final_state=final)
            except ValueError:
                impossible += 1
                with pytest.raises(ValueError):
                    p4decode.component_decode(systematic, parity, prior,
                        initial_state=initial, end_state=final)
                continue
            actual = p4decode.component_decode(systematic, parity, prior,
                initial_state=initial, end_state=final, parity_output=True)
            np.testing.assert_allclose(actual.posterior, expected.posterior, atol=1e-12)
            np.testing.assert_allclose(actual.extrinsic, expected.extrinsic, atol=1e-12)
            assert not np.any(np.isnan(actual.parity_extrinsic))
            feasible += 1
            certain += int(np.any(np.isinf(actual.posterior)))
    assert feasible > 0 and certain > 0
    assert (impossible > 0) == (length < 3)


def test_zero_member_consumes_full_declared_work():
    fields = np.zeros(4968)
    result = rates.decode_fields(fields, 6)
    assert result.complete and result.frame is None
    assert result.iterations_done == result.crc_checks == 8
    assert result.modes_completed == (6,)
    refused = rates.decode_fields(fields, 6, max_work=7)
    assert not refused.complete and refused.frame is None
    assert refused.iterations_done == refused.crc_checks == 7
    assert refused.modes_completed == ()


@pytest.mark.parametrize("budget", [0, 39, 40])
def test_zero_llr_member_cannot_disappear_from_all_mode_domain(monkeypatch, budget):
    visited = []
    def candidates(fields, g, iterations):
        assert not np.any(fields)
        for iteration in range(iterations):
            visited.append((g.speedlevel, iteration))
            yield np.zeros(g.systematic, np.uint8)
    monkeypatch.setattr(rates, "demap", lambda z, noise, sl: np.zeros(4968*rates.geometry(sl).width))
    monkeypatch.setattr(rates, "_iterations_for", candidates)
    result = rates.decode_equalized(np.ones(4968), 1, max_work=budget)
    assert result.declared_modes == (6, 7, 8, 9, 10)
    assert result.iterations_done == result.crc_checks == len(visited) == budget
    assert result.complete == (budget == 40)
    assert result.frame is None and result.identities == ()
    assert result.modes_completed == tuple(range(6, 6+budget//8))
    assert len(result.outcomes) == budget


def test_reencoding_regions_and_distance_use_independent_full_codeword(oracle):
    sl = 10
    fixture = oracle.generate(sl, _payload(sl, 40101), 0)
    g = rates.geometry(sl)
    fields = 1 - 2*np.asarray(fixture.fields, dtype=float)
    # Flip observations in three disjoint coding regions, including the true
    # final systematic flush bit. Candidate bits and all received samples stay.
    positions = (g.systematic-1, g.systematic, g.systematic+g.parity_each)
    fields[list(positions)] *= -1
    symbols = np.asarray(fixture.symbols).copy() + (0.02+0.03j)
    diagnostic = rates._reencoding(fixture.systematic, g, fields, symbols)
    assert diagnostic.full_hard_disagreements == 3
    regions = {r.region: r for r in diagnostic.regions}
    assert regions["systematic"].hard_disagreements == 1
    assert regions["retained-c1"].hard_disagreements == 1
    assert regions["retained-c2"].hard_disagreements == 1
    assert regions["alignment-padding"].hard_disagreements == 0
    assert regions["c1-flush"].hard_disagreements == 1
    energy = float(np.mean(np.abs(fixture.symbols)**2))
    expected = abs(0.02+0.03j)**2 / energy
    assert diagnostic.normalized_complex_distance == pytest.approx(expected)
    assert all(r.normalized_complex_distance == pytest.approx(expected) for r in diagnostic.regions)


@pytest.mark.parametrize("budget", [0, 40])
def test_raw_zero_observation_ineligible_before_candidate_work(monkeypatch, budget):
    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible observation reached demapper")
    monkeypatch.setattr(rates, "demap", forbidden)
    result = rates.decode_equalized(np.zeros(4968), 1, max_work=budget)
    assert result.complete and result.reason == "no-hit"
    assert not result.observation_eligible
    assert result.declared_modes == (6, 7, 8, 9, 10)
    assert result.iterations_done == result.crc_checks == 0
    assert result.outcomes == result.modes_completed == result.identities == ()
    assert result.frame is None


def test_zero_later_member_retains_earlier_identity_but_requires_completion(monkeypatch, oracle):
    fixture = oracle.generate(6, _payload(6, 40102), 0)
    original = rates.demap
    def demap(z, noise, sl):
        return original(z, noise, sl) if sl == 6 else np.zeros(4968*2)
    monkeypatch.setattr(rates, "demap", demap)
    cut = rates.decode_equalized(fixture.symbols, 0.03, speedlevels=(6, 7), max_work=8)
    assert not cut.complete and cut.frame is None
    assert len(cut.identities) == 1 and cut.identities[0].data == _payload(6, 40102)
    assert cut.modes_completed == (6,) and cut.iterations_done == 8
    full = rates.decode_equalized(fixture.symbols, 0.03, speedlevels=(6, 7), max_work=16)
    assert full.complete and full.frame is not None
    assert full.frame == cut.identities[0]
    assert full.modes_completed == (6, 7)
    assert full.iterations_done == full.crc_checks == 16
    assert all(not o.crc_valid for o in full.outcomes[8:])
