# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Observation-domain adapters preserve the shared LONG engine contract."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import p4long, p4longdecode, p4robustrx, p4sf8rx


def symbols(count):
    rng = np.random.default_rng(1111)
    return rng.uniform(.3, 2, count) * np.exp(1j * rng.uniform(-np.pi, np.pi, count))


def assert_ranked_pairs(members):
    scores = []
    for unit, amplitude in zip(members[::2], members[1::2]):
        assert (unit.weighting, amplitude.weighting) == ('unit', 'amplitude')
        assert unit.geometry == amplitude.geometry
        assert unit.label == amplitude.label
        assert unit.quarter_turn == amplitude.quarter_turn
        np.testing.assert_array_equal(unit.fields < 0, amplitude.fields < 0)
        assert not unit.fields.flags.writeable
        scores.append(p4robustrx.syndrome(unit.fields, unit.geometry.speedlevel))
    assert scores == sorted(scores, reverse=True)


def test_sf16_complete_domain_and_locked_369_symbol_support():
    raw = symbols(373)
    members = p4robustrx.build_members(raw)
    assert isinstance(members, tuple) and len(members) == 7 * 4 * 2 * 2
    assert {m.label for m in members} == {str(k) for k in range(18, 25)}
    assert {m.geometry.speedlevel for m in members} == {2, 3}
    assert_ranked_pairs(members)
    assert min(p4robustrx.syndrome(m.fields, m.geometry.speedlevel) for m in members) < .8
    locked = p4robustrx.build_members(raw[:369], starts=(20,))
    assert len(locked) == 16 and {m.label for m in locked} == {'20'}
    assert p4robustrx.build_members(raw[:368], starts=(20,)) == ()


def test_sf8_complete_domain_and_compatibility_filters():
    raw = symbols(697)
    members = p4sf8rx.build_members(raw)
    assert isinstance(members, tuple) and len(members) == 8
    assert {m.geometry.speedlevel for m in members} == {4}
    assert_ranked_pairs(members)
    restricted = p4sf8rx.build_members(raw, amplitude=False, turns=(2,), stride=26)
    assert len(restricted) == 1 and restricted[0].quarter_turn == 2
    assert restricted[0].weighting == 'unit'
    assert not np.array_equal(restricted[0].fields,
                              next(m.fields for m in members if m.quarter_turn == 2))


def test_sl5_members_preserve_observed_parity_tails():
    fields = np.arange(4968, dtype=float) + 1
    members = p4long.build_members(fields)
    assert tuple(m.polarity for m in members) == (1, -1)
    for member in members:
        np.testing.assert_array_equal(member.fields, fields * member.polarity)
        assert np.all(member.fields[3304:3312]) and np.all(member.fields[-8:])


def test_wrappers_submit_one_complete_engine_domain(monkeypatch):
    calls = []

    class Search:
        def __init__(self, members, **kwargs):
            calls.append((tuple(members), kwargs))

        def advance(self):
            return SimpleNamespace(frame=None, selected_provenance=None)

    monkeypatch.setattr(p4longdecode, 'LongSearch', Search)
    assert p4robustrx.decode_symbols(symbols(373), starts=(20,), iterations=2) is None
    assert len(calls[-1][0]) == 16 and calls[-1][1]['iterations'] == 2
    assert p4sf8rx.decode_symbols(symbols(697), c1=False, turns=(1,)) is None
    assert len(calls[-1][0]) == 2 and calls[-1][1]['iterations'] == 0
    fields = np.arange(4968, dtype=float)
    assert p4long.decode_scores(fields, polarity=-1, iterations=3) is None
    np.testing.assert_array_equal(calls[-1][0][0].fields, -fields)
    assert calls[-1][1]['iterations'] == 3
    monkeypatch.setattr(p4long, 'body_scores', lambda *a, **kw: fields)
    assert p4long.decode_body([], first_training_start=0) is None
    assert tuple(m.polarity for m in calls[-1][0]) == (1, -1)
    assert len(calls) == 4


@pytest.mark.parametrize('builder,raw,kwargs', [
    (p4robustrx.build_members, symbols(373), {'starts': (20, 20)}),
    (p4sf8rx.build_members, symbols(697), {'turns': (4,)}),
    (p4long.build_members, np.ones(4968), {'polarities': (1, 1)}),
])
def test_invalid_domains_rejected(builder, raw, kwargs):
    with pytest.raises(ValueError):
        builder(raw, **kwargs)


def test_receptions_use_engine_selected_member_provenance(monkeypatch):
    frame = object()

    class Search:
        def __init__(self, members, **kwargs):
            self.members = tuple(members)

        def advance(self):
            chosen = SimpleNamespace(member_index=1, method='c1-only',
                                     agreements=(SimpleNamespace(hard_disagreements=7),))
            return SimpleNamespace(frame=frame, selected_provenance=chosen)

    monkeypatch.setattr(p4longdecode, 'LongSearch', Search)
    raw = symbols(373)
    members = p4robustrx.build_members(raw, starts=(21,))
    reception = p4robustrx.decode_symbols(raw, starts=(21,))
    assert reception.frame is frame and reception.field_start == 21
    assert reception.speedlevel == members[1].geometry.speedlevel
    assert reception.corrected_bits == 7
    np.testing.assert_array_equal(reception.fields, members[1].fields)
    raw = symbols(697)
    members = p4sf8rx.build_members(raw)
    reception = p4sf8rx.decode_symbols(raw)
    assert reception.frame is frame and reception.quarter_turn == members[1].quarter_turn
    assert reception.correction == 'amplitude-c1'
    np.testing.assert_array_equal(reception.fields, members[1].fields)


def test_normal_rate_diagnostics_delegate_all_public_paths(monkeypatch):
    from hfmodem.shrike import p4longrates as rates

    calls = []

    class Search:
        def __init__(self, members, **kwargs):
            calls.append((tuple(members), kwargs))

        def advance(self):
            return SimpleNamespace(state='unread', complete=True, frame=None,
                                   identities=(), provenance=(), stage_seconds=())

    monkeypatch.setattr(p4longdecode, 'LongSearch', Search)
    fields = np.arange(4968, dtype=float)
    result = rates.decode_fields(fields, 6, iterations=3, max_work=2)
    assert result.complete and result.reason == 'no-hit' and result.declared_modes == (6,)
    np.testing.assert_array_equal(calls[-1][0][0].fields, fields)
    assert calls[-1][1] == {'policy': 'exhaustive', 'iterations': 3, 'component_budget': 4}
    monkeypatch.setattr(rates, 'demap', lambda z, n0, sl: np.ones(4968 * (sl - 5)))
    result = rates.decode_equalized(np.ones(4968, dtype=complex), 1,
                                   speedlevels=(6, 7), iterations=1)
    assert result.declared_modes == (6, 7)
    assert tuple(m.geometry.speedlevel for m in calls[-1][0]) == (6, 7)
    observed = SimpleNamespace(symbols=np.ones(4968, dtype=complex), noise=lambda sl: 1)
    monkeypatch.setattr(rates, 'observe_body', lambda *a, **kw: observed)
    result = rates.decode_body([], first_training_start=0, speedlevels=(8,), iterations=2)
    assert result.observation is observed and result.declared_modes == (8,)
    assert calls[-1][0][0].geometry.speedlevel == 8
    assert len(calls) == 3


def test_robust_diagnostic_domain_preserves_bench_contract_and_uses_common_engine(monkeypatch):
    from hfmodem.shrike import p4robustc2

    # The diagnostic contract remains narrower;
    # the live builder's broader ambiguity domain does not replace it.
    monkeypatch.setattr(p4robustrx, 'syndrome',
                        lambda fields, level: 1. if level == 2 else .5)
    calls = []
    class Search:
        def __init__(self, members, **kwargs):
            calls.append((tuple(members), kwargs))

        def advance(self):
            return SimpleNamespace(state='unread', complete=True, frame=None,
                identities=(), provenance=(), component_calls=0,
                constituent_positions=0, stage_seconds=())

    monkeypatch.setattr(p4longdecode, 'LongSearch', Search)
    for level, count in ((2, 349), (4, 697)):
        raw = symbols(count)
        members = p4robustc2._eligible_members(raw, p4longdecode.geometry(level))
        assert [(m.weighting, m.quarter_turn) for m in members] == [
            (weighting, turn)
            for weighting in (('unit',) if level == 2 else ('unit', 'amplitude'))
            for turn in range(4)]
        if level == 4:
            expected = p4sf8rx.build_members(raw)
        else:
            expected = tuple(m for m in p4robustrx.build_members(
                np.r_[np.zeros(20), raw], starts=(20,))
                if m.geometry.speedlevel == 2 and m.weighting == 'unit')
        expected = {(m.weighting, m.quarter_turn): m for m in expected}
        for actual in members:
            canonical = expected[(actual.weighting, actual.quarter_turn)]
            np.testing.assert_array_equal(actual.fields < 0, canonical.fields < 0)
            # Bench normalization precedes its declared field-only support;
            # the live builder normalizes the padded packet support. Pin the
            # bench's exact arithmetic independently rather than allow an ulp
            # tolerance between those two intentionally different supports.
            differential = raw[1:] * raw[:-1].conjugate()
            unit = differential / np.maximum(abs(differential), 1e-30)
            unit *= np.exp(-1j*np.angle(np.mean(unit**4))/4)
            if level == 2:
                order = np.concatenate([np.arange(k, 348, 29) for k in range(29)])
                rotated = unit[order] * (1j**actual.quarter_turn)
                fields = 4*np.stack((rotated.real+rotated.imag,
                                     rotated.real-rotated.imag), axis=-1).ravel()
                np.testing.assert_array_equal(actual.fields, fields)
            else:
                np.testing.assert_array_equal(actual.fields, canonical.fields)
        result = p4robustc2.decode_placed(raw, speedlevel=level,
                                        iterations=3, component_budget=5)
        submitted, options = calls[-1]
        assert options == dict(policy='exhaustive', iterations=3, component_budget=5)
        assert result.complete and result.reason == 'no-hit'
        assert result.eligible_members == result.completed_members == len(members)
        assert len(submitted) == len(members)
        for actual, original in zip(submitted, members):
            assert actual.geometry == p4longdecode.geometry(level)
            assert (actual.weighting, actual.quarter_turn) == (original.weighting, original.quarter_turn)
            assert not actual.fields.flags.writeable
            np.testing.assert_array_equal(actual.fields, original.fields)


def test_live_sl2_domain_retains_all_eight_members_below_bench_syndrome_cutoff():
    raw = np.r_[np.zeros(20), symbols(349)]
    members = tuple(m for m in p4robustrx.build_members(raw, starts=(20,))
                    if m.geometry.speedlevel == 2)
    assert len(members) == 8
    assert [m.weighting for m in members] == ['unit', 'amplitude'] * 4
    assert min(p4robustrx.syndrome(m.fields, 2) for m in members) < .8
    assert {m.quarter_turn for m in members} == set(range(4))
    assert_ranked_pairs(members)
    # Reconstruct the serialized observations independently, without replacing
    # syndrome or the builder: weak SL2 models must survive live admission.
    differential = raw[1:] * raw[:-1].conjugate()
    unit = differential / np.maximum(abs(differential), 1e-30)
    phase = np.exp(-1j*np.angle(np.mean(unit**4))/4)
    order = np.concatenate([np.arange(k, 348, 29) for k in range(29)])
    data = unit[20:368][order] * phase
    weights = abs(differential[20:368][order]) / np.mean(abs(differential[20:368]))
    for member in members:
        rotated = data * (1j**member.quarter_turn)
        fields = 4*np.stack((rotated.real+rotated.imag, rotated.real-rotated.imag), axis=-1).ravel()
        if member.weighting == 'amplitude':
            fields = fields * np.repeat(weights, 2) * 2
        np.testing.assert_array_equal(member.fields, fields)
