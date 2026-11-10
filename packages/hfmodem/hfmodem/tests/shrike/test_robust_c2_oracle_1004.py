# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent scalar generator checks. No target encoder/reader imported."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from . import robust_c2_oracle as oracle

require_input(oracle.DATA, allow_module_level=True)


def native(name, dtype):
    return np.frombuffer((oracle.DATA / name).read_bytes(), dtype=dtype)


def table_encode(inputs, state=0):
    """Separate original-table lookup, rather than scalar polynomial model."""
    transitions = native('native-transition.u8', np.uint8)
    parities = native('native-parity.u8', np.uint8)
    output = []
    for bit in inputs:
        index = 2 * state + int(bit)
        output.append(int(parities[index]))
        state = int(transitions[index])
    return np.asarray(output, dtype=np.uint8), state


def bytewise_whitening(data):
    """Separate whole-byte recurrence checked against bit-at-a-time model."""
    state, output = 12021965, []
    for byte in data:
        output.append(byte ^ (state & 255))
        for _ in range(8):
            carry = state & 1
            state //= 2
            state ^= 1207959552 if carry else 0
    return bytes(output)


def test_no_target_or_modem_generator_imports():
    tree = ast.parse(Path(oracle.__file__).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in imports for a in n.names]
    assert set(names) <= {'annotations', 'dataclass', 'field', 'hashlib', 'itertools', 'json', 'math', 'Path', 'numbers', 'numpy'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('eval', 'exec', '__import__') for n in ast.walk(tree))


def test_all_scalar_transitions_and_flush_states_match_native():
    parities = native('native-parity.u8', np.uint8)
    transitions = native('native-transition.u8', np.uint8)
    flushes = native('native-flush.u8', np.uint8)
    for state in range(8):
        for bit in range(2):
            nxt, parity = oracle.rsc_step(state, bit)
            assert nxt == transitions[2 * state + bit]
            assert parity == parities[2 * state + bit]
        current = state
        for bit in oracle.terminate(state):
            assert bit == flushes[current]
            current = int(transitions[2 * current + int(bit)])
        assert current == 0


def test_crc_native_table_published_check_and_residue():
    table = native('native-crc.u16le', '<u2')
    for byte in range(256):
        state = byte
        for _ in range(8):
            state = (state >> 1) ^ (33800 if state & 1 else 0)
        assert state == table[byte]
    assert oracle.crc_x25(b'123456789') == 36974
    for count in (0, 1, 40, 141, 256):
        data = bytes((17 * i + 3) % 256 for i in range(count))
        state = 65535
        for byte in data:
            state = (state >> 8) ^ int(table[(state ^ byte) & 255])
        assert oracle.crc_register(data) == state
        crc = state ^ 65535
        assert oracle.crc_register(data + bytes([crc & 255, crc >> 8])) == 61624


@pytest.mark.parametrize('sl', (2, 4))
def test_all_status_bytes_exact_full_field_crc_and_whitening(sl):
    length, d, _, padding, period, phase = oracle.geometry(sl)
    data = bytes((13 * i + 7) % 256 for i in range(length))
    for status in range(256):
        fixture = oracle.generate(sl, data, status)
        assert len(fixture.wire) == length + 3
        assert fixture.wire[:-2] == data + bytes([status])
        assert oracle.crc_register(fixture.wire) == 61624
        assert fixture.whitened == bytewise_whitening(fixture.wire)
        assert oracle.whiten(fixture.whitened) == fixture.wire
        wire_bits = np.asarray([(b >> i) & 1 for b in fixture.whitened for i in range(8)])
        assert np.array_equal(fixture.systematic[:len(wire_bits)], wire_bits)
        assert np.array_equal(fixture.systematic[len(wire_bits):len(wire_bits)+padding], np.zeros(padding))
        c1, final = table_encode(fixture.systematic)
        assert final == 0 and np.array_equal(c1, fixture.c1_full)
        c2, final = table_encode(fixture.c2_inputs)
        assert final == fixture.c2_final and np.array_equal(c2, fixture.c2_full)
        assert len(fixture.c2_inputs) == d
        assert np.array_equal(fixture.serialized, np.concatenate((fixture.systematic, c1[0::period], c2[phase::period])))
        assert (d - period, d - period + phase) == ((346,347) if sl == 2 else (1150,1154))


@pytest.mark.parametrize('sl', (2, 4))
@pytest.mark.parametrize('pattern', ('zero', 'one', 'alternating', 'seeded'))
def test_full_codeword_fields_and_reference_integration(sl, pattern):
    length, d, m, padding, _, _ = oracle.geometry(sl)
    data = {'zero': bytes(length), 'one': bytes([255]) * length,
            'alternating': bytes([85,170]) * (length//2) + (bytes([85]) if length % 2 else b''),
            'seeded': oracle.seeded_data(sl, 100300)}[pattern]
    fixture = oracle.generate(sl, data, 37)
    assert len(fixture.systematic) == d and len(fixture.symbols) == m + 1
    assert len(fixture.systematic) == 8 * (length + 3) + padding + 3
    assert len(fixture.serialized) == 2 * m
    # Independent cumulative loop driven by original scatter table.
    scatter = oracle.symbol_order(m)
    expected_physical = [None] * m
    for i, differential in enumerate(fixture.logical_differentials):
        expected_physical[int(scatter[i])] = differential
    current, expected = 1+0j, [1+0j]
    for differential in expected_physical:
        current *= differential
        expected.append(current)
    assert np.array_equal(fixture.symbols, np.asarray(expected))
    for turn in range(4):
        rotated = oracle.generate(sl, data, 37, reference=1j**turn)
        assert np.array_equal(rotated.symbols, fixture.symbols * (1j**turn))
        assert rotated.wire == fixture.wire
    for array in (fixture.systematic, fixture.c1_full, fixture.c2_inputs, fixture.c2_full, fixture.serialized, fixture.logical_differentials, fixture.symbols):
        assert not array.flags.writeable


@pytest.mark.parametrize('sl', (2, 4))
def test_direct_gather_all_basis_positions_and_direction_mutations(sl):
    _, d, _, _, period, phase = oracle.geometry(sl)
    order = oracle.c2_order(sl)
    inverse = np.argsort(order)
    for at in range(d):
        basis = np.zeros(d, dtype=np.uint8)
        basis[at] = 1
        # A source basis at j appears at the sole consumer i with order[i]=j.
        gathered = [int(basis[int(source)]) for source in order]
        assert sum(gathered) == 1
        assert gathered[int(inverse[at])] == 1
    fixture = oracle.generate(sl, oracle.seeded_data(sl, 100300), 37)
    assert np.array_equal(fixture.c2_inputs, fixture.systematic[order])
    assert not np.array_equal(fixture.c2_inputs, fixture.systematic[inverse])
    expected, final = table_encode(fixture.systematic[order])
    assert np.array_equal(expected, fixture.c2_full) and final == fixture.c2_final
    # Mutations directly contradict algebra, irrespective of a CRC result.
    wrong, _ = table_encode(fixture.systematic[inverse])
    assert not np.array_equal(wrong[phase::period], fixture.c2_full[phase::period])
    wrong, _ = table_encode(fixture.c2_inputs[:-3])
    assert len(wrong) != len(fixture.c2_full)
    assert not np.array_equal(fixture.c2_full[phase::period], fixture.c2_full[0::period])
    if sl == 2:
        guessed = oracle.walk(d, 18)
        assert np.count_nonzero(guessed == order) == 331
        wrong, _ = table_encode(fixture.systematic[guessed])
        assert not np.array_equal(wrong[phase::period], fixture.c2_full[phase::period])


def test_sl3_full_native_order_equality_control():
    order = oracle.c2_order(3)
    assert np.array_equal(order, oracle.walk(580, 24))
    fixture = oracle.generate(3, oracle.seeded_data(3, 100300), 37)
    assert len(fixture.systematic) == 580 and len(fixture.serialized) == 696
    expected, _ = table_encode(fixture.systematic[order])
    assert np.array_equal(expected, fixture.c2_full)


@pytest.mark.parametrize('m', (348,696))
def test_native_scatter_matches_scalar_walk_and_not_reverse(m):
    order = oracle.symbol_order(m)
    assert np.array_equal(order, oracle.walk(m, 29))
    assert not np.array_equal(order, np.argsort(order))
    labels = oracle.dqpsk_labels(np.asarray(([0,0,0,1,1,0,1,1] * ((2*m+7)//8))[:2*m]))
    assert not np.array_equal(oracle.integrate(labels,order), oracle.integrate(labels,np.argsort(order)))


def test_pair_labels_all_four_signs():
    inputs = np.asarray([0,0,0,1,1,0,1,1], dtype=np.uint8)
    labels = oracle.dqpsk_labels(inputs)
    assert np.array_equal(labels, np.asarray([1,1j,-1j,-1], dtype=complex))
    assert np.array_equal((labels.real+labels.imag) < 0, inputs[0::2])
    assert np.array_equal((labels.real-labels.imag) < 0, inputs[1::2])


@pytest.mark.parametrize('n', (0,1,2,3,4,6,8))
@pytest.mark.parametrize('boundary', ((0,0),(0,None),(None,None)))
def test_short_paths_exact_admitted_counts_and_finite_or_certain(n,boundary):
    initial, final = boundary
    result = oracle.enumerate_trellis(np.linspace(-1.5,2.5,n), np.linspace(.3,-.7,n), initial=initial,final=final)
    expected = (8 * 2**n if initial is None else 2**n if final is None else max(1,2**max(n-3,0)))
    assert result.admitted_paths == expected
    assert not np.any(np.isnan(result.posterior))
    assert not np.any(np.isnan(result.extrinsic))
    finite = np.isfinite(result.posterior)
    assert np.allclose((result.posterior-result.extrinsic)[finite], np.linspace(-1.5,2.5,n)[finite], rtol=0, atol=1e-12) if np.all(finite) else np.array_equal(np.sign(result.posterior[~finite]),np.sign(result.extrinsic[~finite]))


def test_degenerate_one_path_certainty_and_impossible_path():
    assert oracle.enumerate_trellis([.3],[.7],initial=0,final=0).posterior[0] == math.inf
    assert oracle.enumerate_trellis([.3],[.7],initial=0,final=4).posterior[0] == -math.inf
    with pytest.raises(ValueError, match='no admitted'):
        oracle.enumerate_trellis([.3],[.7],initial=0,final=1)
    with pytest.raises(ValueError, match='no admitted'):
        oracle.enumerate_trellis([],[],initial=0,final=1)


@pytest.mark.parametrize('n', (1,2,3,4,6,8))
def test_free_free_matches_untouched_component_numerical_control(n):
    # This import is solely a comparison to the unchanged old free/free core.
    # Oracle generation and path enumeration have no dependency on it.
    from hfmodem.shrike import p4decode
    systematic=np.linspace(-2.1,3.2,n)
    parity=np.linspace(.6,-1.3,n)
    prior=np.linspace(.1,.7,n)
    expected=oracle.enumerate_trellis(systematic+prior,parity,initial=None,final=None)
    actual=p4decode.component_decode(systematic,parity,prior)
    assert np.allclose(actual.posterior,expected.posterior,rtol=0,atol=1e-12)
    assert np.allclose(actual.extrinsic,expected.extrinsic,rtol=0,atol=1e-12)


@pytest.mark.parametrize('sl',(2,4))
def test_fixed_panel_all_96_cases_stage_hashes_noise_and_reference(sl):
    import json
    manifest=json.loads((oracle.DATA/'FIXED-PANEL.json').read_text())
    rows=[r for r in manifest['cases'] if r['speedlevel']==sl]
    assert len(rows)==96
    assert [(r['seed'],r['es_n0_db']) for r in rows] == [(seed,snr) for seed in range(100300,100316) for snr in (24,21,18,15,12,9)]
    for row in rows:
        path=oracle.DATA/row['path']
        assert oracle.digest(path.read_bytes())==row['file_sha256']
        fixture,noisy=oracle.noisy_case(sl,row['seed'],row['es_n0_db'])
        assert fixture.stage_hashes==row['stage_sha256']
        assert oracle.digest(noisy.astype('<c16').tobytes())==row['noisy_symbols_sha256']
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(saved['noisy_symbols'],noisy)
            assert np.array_equal(saved['clean_symbols'],fixture.symbols)
            assert np.array_equal(saved['wire'],np.frombuffer(fixture.wire,dtype=np.uint8))
        clean=fixture.symbols/np.sqrt(np.mean(abs(fixture.symbols)**2))
        draws=np.random.Generator(np.random.PCG64(row['seed'])).standard_normal((len(clean),2))
        noise=np.sqrt(10.**(-row['es_n0_db']/10)/2)*(draws[:,0]+1j*draws[:,1])
        assert np.array_equal(noisy,clean+noise)
        assert noise[0]!=0  # Reference is also corrupted by the declared model.


def test_missing_corrupt_or_nonbijective_orders_raise(tmp_path,monkeypatch):
    monkeypatch.setattr(oracle,'DATA',tmp_path)
    with pytest.raises(FileNotFoundError):oracle.c2_order(2)
    (tmp_path/'c2-sl2.u16le').write_bytes(bytes(696))
    with pytest.raises(ValueError):oracle.c2_order(2)
    for invalid in ([0,0],[-1,1],[.0,1.0],[[0,1]]):
        with pytest.raises(ValueError):oracle.permutation(invalid,2)


def test_generator_and_enumerator_reject_malformed_arguments():
    for sl in (True,1,5,2.0):
        with pytest.raises(ValueError):oracle.geometry(sl)
    for status in (True,-1,256,.5):
        with pytest.raises(ValueError):oracle.generate(2,bytes(40),status)
    for data in (bytes(39),bytes(41),[0]*40):
        with pytest.raises(ValueError):oracle.generate(2,data,0)
    for initial,final in ((True,0),(0,False),(-1,0),(0,8)):
        with pytest.raises(ValueError):oracle.enumerate_trellis([0],[0],initial=initial,final=final)
    for sys,par in (([np.nan],[0]),([0],[np.inf]),([[0]],[[0]]),([0],[]),([0]*11,[0]*11),([1e308],[1e308])):
        with pytest.raises(ValueError):oracle.enumerate_trellis(sys,par)
    for reference in (0,complex(np.nan,0),complex(0,np.inf)):
        with pytest.raises(ValueError):oracle.integrate([1],[0],reference)


def test_constituent_endpoint_mutations_change_admitted_paths():
    sys, par = [.3,-.7,.6,1.2], [.8,.1,-.4,.7]
    fixed = oracle.enumerate_trellis(sys,par,initial=0,final=0)
    free_end = oracle.enumerate_trellis(sys,par,initial=0,final=None)
    free_both = oracle.enumerate_trellis(sys,par,initial=None,final=None)
    assert (fixed.admitted_paths, free_end.admitted_paths, free_both.admitted_paths) == (2,16,128)
    assert not np.allclose(fixed.posterior, free_end.posterior)
    assert not np.allclose(fixed.posterior, free_both.posterior)
    # C1 freeing and C2 terminating are opposite mutations of these boundaries.
    assert oracle.enumerate_trellis([.3],[.8],initial=0,final=None).admitted_paths == 2
    assert oracle.enumerate_trellis([.3],[.8],initial=0,final=0).admitted_paths == 1


def test_scalar_contract_failure_precedes_codeword_construction(monkeypatch):
    def invalid(_):
        return 0
    monkeypatch.setattr(oracle,"crc_x25",invalid)
    with pytest.raises(ValueError,match="scalar CRC check mismatch"):
        oracle.generate(2,bytes(40),0)


def test_separate_whitening_golden_failure_precedes_codeword_construction(monkeypatch):
    monkeypatch.setattr(oracle,"whiten",lambda data:data)
    with pytest.raises(ValueError,match="whitening contradicts"):
        oracle.generate(4,bytes(141),0)
