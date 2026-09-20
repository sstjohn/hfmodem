# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Explicit control A/B profiles: real peer body through the duplex TX seam."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, placement
from hfmodem.tests.shrike.test_entry_candidate_body import META, META_PATH, scene
from hfmodem.tests.shrike.test_grid import _Rig

FS = onair.FS
GOLDEN_PATH = Path(__file__).with_name("fixtures") / "historical-control-cs1.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else None


@pytest.mark.skipif(GOLDEN is None,
                    reason=f"the frozen control vector {GOLDEN_PATH.name} is "
                           "not installed")
def test_historical_cs1_matches_frozen_7129_and_is_read_only():
    # This vector was generated with imports and placement.py bytes asserted
    # against the frozen 7129bb83 snapshot, before reading the new helper.
    # Compare numeric samples rather than a platform-sensitive FFT byte hash.
    audio = placement.historical_control_signal(arq.CS_ACK)
    assert len(audio) == GOLDEN["samples"] == 11939
    np.testing.assert_allclose(audio[GOLDEN["indices"]], GOLDEN["values"],
                               rtol=0, atol=2e-12)
    assert placement.historical_control_signal(arq.CS_ACK) is audio
    assert not audio.flags.writeable
    with pytest.raises(ValueError):
        audio[0] = 1
    # The new control includes a different physical ending and a stagger;
    # aliasing the two profiles must not silently pass this experiment.
    assert len(placement.control_signal(arq.CS_ACK)) != len(audio)


def _received_reply(tmp_path, monkeypatch, waveform, reference,
                    follow="none", tail=None, stagger=None, p1_changeover=False):
    if META is None or not META_PATH.with_suffix(".wav").exists():
        pytest.skip("recorded WS8EOC first-changeover fixture is not installed")
    session, tx, live, grid, seg, first = scene(tmp_path)
    session.host.p3_changeover_p1_cs = p1_changeover
    # The recorded peer sits at -15 Hz, so the shipping carrier follow would
    # move every sample these profiles are compared on. It is its own subject
    # (`test_control_offset_follow`), and nominal is what these A/B arms keyed.
    tx.p3_follow_offset = follow
    if waveform is not None:
        tx.p3_control_waveform = waveform
    if reference is not None:
        tx.p3_control_placement = reference
    if tail is not None:
        tx.p3_control_tail = tail
    if stagger is not None:
        tx.p3_control_stagger = stagger
    tx.live, tx.sessrx = live, session.rx
    tx.transmit, tx.rig = True, _Rig()  # No device or serial connection.
    emitted = []
    transmit = live.transmit

    def record(audio, **kwargs):
        emitted.append(audio.copy())
        return transmit(audio, **kwargs)

    monkeypatch.setattr(live, "transmit", record)
    slot, _, _ = onair._regrid(live, grid, tx, session.host, session.rx,
                              17, seg, first, 1920)
    assert [ev.packet[2] for ev in session.packets] == [b"RMS"]
    assert session.host.arq.role == arq.IRS
    assert not session.host.arq.entry_pending
    assert tx._pending_p3_cs == arq.CS_ACK
    assert not emitted and not live.emissions
    # Fixed-clock FSK polarity must not determine the P3 control's order.
    monkeypatch.setattr(grid, "shift", lambda slot: False)
    session.host.tick()
    assert tx._pending_p3_cs == arq.CS_ACK and not emitted
    return session, tx, live, grid, slot, emitted


@pytest.mark.parametrize("waveform,reference", [
    (None, None),  # Unconfigured waveform and placement take the defaults below.
    ("current", "pulse-center"),
    ("current", "audio-start"),
    ("historical", "pulse-center"),
    ("historical", "audio-start"),
])
def test_recorded_cs3_emits_plain_cs1_with_selected_profile(
        tmp_path, monkeypatch, waveform, reference):
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, waveform, reference)
    historical = (waveform or "historical") == "historical"
    audio_start = (reference or "audio-start") == "audio-start"
    if historical:
        raw = placement.historical_control_signal(arq.CS_ACK)
        lead = placement.historical_control_pulse_lead(arq.CS_ACK)
        reserve = max(placement.historical_control_pulse_lead(ci)
                      for ci in range(6))
    else:
        # Real fixture is peer swapFalse, this side is caller; projected reply
        # is swapTrue. This is independent of the grid.shift override above.
        raw = placement.control_signal(arq.CS_ACK, swapped=True)
        lead = placement.control_pulse_lead(arq.CS_ACK, swapped=True)
        reserve = onair.P3_CS_KEY_LEAD_N
    # The comb arrives at the reversal half a millisecond past the answer slot,
    # carried from the PACTOR-1 phase. `_regrid` spends that residue where the
    # cycle's window can still be measured against the result, so the comb is
    # already on the peer's packet here and the emit has nothing left to move.
    target = grid._p3_peer[0] + NOMINAL_REPLY_N
    assert grid.boundary(slot) == target
    assert grid.p3_reply_shift(slot) is None
    assert tx.key_instant(grid, slot) == grid.boundary(slot) - (
        0 if audio_start else reserve)
    previous_slots = list(tx.slots_used)
    tx.emit_pending_cs()
    assert grid.boundary(slot) == target
    assert len(emitted) == len(live.emissions) == tx.n == 1
    assert not tx.refused and tx._pending_p3_cs is None
    assert tx.slot == slot
    expected = onair._trim_silence(raw)
    # Duplex audio is float32; compare at the actual DAC transport precision.
    expected = (expected * (tx.drive / max(abs(expected)))).astype(np.float32)
    np.testing.assert_allclose(emitted[0], expected, rtol=0, atol=2e-12)
    assert tx.tx_audio_start == target - (0 if audio_start else lead)
    offsets = (lead, lead) if historical else (lead + FS//200, lead)
    assert tx.tx_pulse_offsets == offsets
    assert tx.tx_audio_start + min(offsets) == target + (lead if audio_start else 0)
    # RECORDED AS THE BURST WAS AIMED, in both profiles. The reply-timing chain
    # compares this against `boundary`; recording the pulse center under
    # audio-start placement put the read instant a lead past our own comb and
    # `_clamp_refusal` then refused every changeover by about 18 ms.
    assert grid._p3_controls[-1] == (target, grid.cycle_n)
    assert tx.tx_end - tx.tx_audio_start == len(emitted[0])
    assert live.emissions == [(tx.tx_audio_start, tx.tx_end)]
    assert tx.slots_used == previous_slots + [slot]
    assert tx.keyed == [(slot, len(emitted[0])/FS)]
    assert tx.keyings == [(tx.tx_key_up, tx.tx_end)]
    assert live.pos == live.floor == tx.tx_end
    assert tx.rig.edges == [True, False]
    sidecar = json.loads((tmp_path / "tx_01.json").read_text())
    assert sidecar["audio_start"] == tx.tx_audio_start
    assert sidecar["audio_end"] == tx.tx_end
    assert sidecar["pulse_offsets"] == list(offsets)
    assert not session.host.arq.entry_pending


NOMINAL_REPLY_N = round(.890 * FS)


@pytest.mark.parametrize("reference,lead", [(None, 0), ("pulse-center", 852)])
def test_default_profile_answers_a_changeover_at_the_nominal_reply_instant(
        tmp_path, monkeypatch, reference, lead):
    """`--p3-control-placement` measured against the peer packet, not the slot."""
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, None, reference)
    at = grid._p3_peer[0]
    # The transmit anchor is carried through the reversal from the PACTOR-1
    # phase and arrives half a millisecond past the answer slot. Nothing here
    # corrects it by hand: `_p3_place_reply` holds the comb on the peer's own
    # packet, which is what the answer slot is measured from, and it runs
    # inside `_regrid` rather than at the emit.
    assert grid.boundary(slot) == at + NOMINAL_REPLY_N
    tx.emit_pending_cs()
    assert len(emitted) == 1 and not tx.refused
    assert tx.tx_pulse_offsets == (852, 852)
    assert tx.tx_audio_start - at == NOMINAL_REPLY_N - lead
    assert grid._p3_controls[-1][0] - at == NOMINAL_REPLY_N


@pytest.mark.parametrize("waveform", ["current", "historical"])
@pytest.mark.parametrize("reference", ["pulse-center", "audio-start"])
def test_profile_cannot_bypass_invalid_peer_clock(
        tmp_path, monkeypatch, waveform, reference):
    session, tx, live, grid, _, emitted = _received_reply(
        tmp_path, monkeypatch, waveform, reference)
    # Exercise actual clock-admission policy, not a stubbed _tx or fake refusal.
    grid._p3_reply_phase_invalid = True
    slots = list(tx.slots_used)
    controls = list(grid._p3_controls)
    tx.emit_pending_cs()
    assert tx.refused
    assert not emitted and not live.emissions and not tx.rig.edges
    assert tx.n == 0 and tx.tx_audio_start is None and tx.tx_end is None
    assert tx.slots_used == slots and grid._p3_controls == controls
    assert not tx.keyed and not tx.keyings
    assert not (tmp_path / "tx_01.json").exists()
    assert session.host.arq.role == arq.IRS
    assert [ev.packet[2] for ev in session.packets] == [b"RMS"]


@pytest.mark.parametrize("tail,stagger", [
    ("repeat", "off"), ("off", "lead-5"), ("off", "lead-12"),
    ("repeat", "lead-5"), ("repeat", "lead-12")])
def test_control_flags_key_the_measured_shape_and_name_the_foot(
        tmp_path, monkeypatch, tail, stagger):
    """`--p3-control-tail` and `--p3-control-stagger`, on the default waveform."""
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, None, None, tail=tail, stagger=stagger)
    tx.emit_pending_cs()
    assert len(emitted) == 1 and not tx.refused
    # `grid.shift` is pinned False above, so this is the session's first foot
    # and the flag names it outright.
    swapped = None if stagger == "off" else stagger == "lead-12"
    expected = onair._trim_silence(
        placement.control_burst(arq.CS_ACK, tail=tail == "repeat",
                                swapped=swapped))
    expected = (expected * (tx.drive / max(abs(expected)))).astype(np.float32)
    np.testing.assert_allclose(emitted[0], expected, rtol=0, atol=2e-12)
    lead = placement.control_burst_pulse_lead(
        arq.CS_ACK, tail=tail == "repeat", swapped=swapped)
    half = FS // 200
    assert tx.tx_pulse_offsets == (
        (lead, lead) if swapped is None else
        (lead + half, lead) if swapped else (lead, lead + half))
    # The foot has to reach the tape: the alternation is ours and a capture has
    # nothing else to score it against.
    shape = (([] if stagger == "off" else [stagger])
             + (["tail"] if tail == "repeat" else []))
    sidecar = json.loads((tmp_path / "tx_01.json").read_text())
    assert sidecar["control"] == f"CS1 ACK seq=0 [{', '.join(shape)}]"
    assert not session.host.arq.entry_pending


@pytest.mark.parametrize("stagger,first", [("lead-5", False), ("lead-12", True)])
def test_stagger_alternates_every_arq_cycle_from_the_named_foot(
        tmp_path, monkeypatch, stagger, first):
    _, tx, _, _, _, _ = _received_reply(tmp_path, monkeypatch, None, None,
                                        stagger=stagger)
    # Slot parity IS the ARQ cycle count, so the arrangement follows the cycle
    # the peer is counting and not the bursts we happen to key. Whichever parity
    # a session opens on, the first control goes out on the named foot.
    for anchor in (False, True):
        tx._p3_stagger_anchor = None
        cycles = [anchor ^ bool(n & 1) for n in range(8)]
        assert tx._p3_stagger_foot(cycles[0]) == first
        tx._p3_stagger_anchor = anchor
        assert ([tx._p3_stagger_foot(c) for c in cycles]
                == [first ^ bool(n & 1) for n in range(8)])
    # A burst the guards decline must not spend the foot: the next cycle is the
    # other parity, and anchoring on a refused render would key the arm on the
    # foot the flag did not name.
    tx._p3_stagger_anchor = None
    tx._p3_stagger_foot(True)
    assert tx._p3_stagger_anchor is None


def test_our_receiver_reads_every_flag_combination():
    """Loopback and bench decodes have to survive both flags, in every mixture.

    Our own acquisition cannot GRADE any of these -- it read the synchronous
    burst and both staggered orientations at quality 1.000 all through the
    session that found the difference -- but it still has to read what we key.
    """
    for cs in range(6):
        for tail in (False, True):
            for swapped in (None, False, True):
                audio = placement.control_burst(cs, tail=tail, swapped=swapped)
                got = p3acquire.control_signal(np.pad(audio, (4800, 4800)))
                assert got is not None, (cs, tail, swapped)
                assert got.event.cs == cs and got.quality > .99
