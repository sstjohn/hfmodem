# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recorded RMS deadlines and false CS3 evidence, without hardware I/O."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session

CAP = Path(__file__).resolve().parents[5] / "captures/onair-0914-1731"


@pytest.fixture(scope="module")
def recording():
    if not (CAP / "stream.wav").exists():
        pytest.skip("WS8EOC 1731 recording is not installed")
    rate, pcm = wavfile.read(CAP / "stream.wav")
    assert rate == onair.FS
    return pcm.astype(float) / 32768


def scene(tmp_path, recording, hold=5, delivery_phase=0):
    meta = json.loads((CAP / f"hold_{hold:02d}.json").read_text())
    end = meta["end_stream_sample"]
    origin = end - meta["samples"]
    phase = 1685546 + (hold - 3) * 60000
    s = _Session(role=arq.IRS)
    s.rx.p3_wideband_prekey = True
    s.rx.p3_receive_offset_hz = -13.8
    s.rx._seed_p3_changeover_clock(phase - 60000)
    s.rx._p3_clock_role = arq.IRS
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.defer_p3_cs = True
    tx.p3_control_waveform = "historical"
    tx.p3_control_placement = "pulse-center"
    g = onair._MasterGrid(phase + 42720 - 60000, 60000, 8880,
                         packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    tx.aim(g, 1)
    # Deliver the unchanged recorded samples at each possible callback phase.
    # Extra delivered samples remain unread, rather than inventing new audio.
    delivered = ((end - delivery_phase + 383) // 384) * 384 + delivery_phase
    clock = SimpleNamespace(samples=delivered, key_notice=2612)
    clock.transmit = lambda *a, **kw: pytest.fail("hardware transmission")
    clock.clamp_late = lambda at: max(0, clock.samples + clock.key_notice - at)
    tx.live = clock
    return s, tx, g, clock, recording[origin:end], origin


@pytest.mark.parametrize("hold", [5, 6, 7])
def test_saved_rms_windows_reach_prekey_crc(tmp_path, recording, hold):
    s, tx, g, clock, audio, origin = scene(tmp_path, recording, hold)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert [ev.packet for ev in s.packets] == [(1, 0, b"RMS", True)]
    assert onair._p3_current_packet_crc(s.rx, g, 1)
    assert not clock.clamp_late(tx.key_instant(g, 1))


@pytest.mark.parametrize("hold", [5, 6, 7])
@pytest.mark.parametrize("phase", [0, 96, 192, 288])
def test_callback_phases_obey_body_processing_reserve(tmp_path, recording, hold, phase):
    s, tx, g, clock, audio, origin = scene(tmp_path, recording, hold, phase)
    fits = s.rx._p3_acquisition_fits(s.rx.CHANGEOVER_BODY_RESERVE_S)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert bool(s.packets) == fits
    assert onair._p3_current_packet_crc(s.rx, g, 1) == fits
    assert not tx.keyed


@pytest.mark.parametrize("change", ["slot", "row", "role", "state", "entry", "new-cycle", "geometry"])
def test_crc_proof_cannot_escape_its_opportunity(tmp_path, recording, change):
    s, tx, g, clock, audio, origin = scene(tmp_path, recording)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert onair._p3_current_packet_crc(s.rx, g, 1)
    slot = 1
    if change == "slot":
        slot = 2
    elif change == "row":
        s.rx._p3_delivered_at += 60000
    elif change == "role":
        s.host.arq.role = arq.ISS
    elif change == "state":
        s.host.arq.state = arq.State.DISCONNECTED
    elif change == "entry":
        s.host.arq.entry_pending = True
    elif change == "new-cycle":
        s.rx.new_cycle()
    else:
        at, width, cycle, seen = g._p3_peer
        g._p3_peer = (at - 60000, width, cycle, seen)
    assert not onair._p3_current_packet_crc(s.rx, g, slot)


def test_previous_window_delivery_is_not_prekey_proof(tmp_path, recording):
    s, tx, g, clock, audio, origin = scene(tmp_path, recording)
    onair._scan_frame(s.rx, audio, origin)
    assert s.packets
    assert not onair._p3_current_packet_crc(s.rx, g, 1)


def test_incomplete_cs3_body_does_not_deliver_data(tmp_path, recording):
    s, tx, g, clock, audio, origin = scene(tmp_path, recording)
    onair._scan_frame(s.rx, audio[:24000], origin, tracked_only=True)
    assert not s.packets
    assert not onair._p3_current_packet_crc(s.rx, g, 1)


def test_recorded_false_breakin_fails_coherent_head_gate(recording):
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -13.8
    phase = 1685546 + 11 * 60000
    origin = phase - 7200
    audio = recording[origin:phase + 48000]
    ev = rxfront.SyncedRx().control_signal_at(
        p3acquire.compensate(audio, -13.8), 8400, details=True)
    assert ev is not None and ev.cs == arq.CS_BREAKIN and ev.packet is None
    assert not s.rx._p3_bodyless_head_supported(audio, ev)
    assert s.rx._p3_cs(audio, 8400, seg_start=origin) is None
    assert s.rx._p3_answer_at is None
    assert not s.rx.cs_log


@pytest.mark.parametrize("epoch", range(5))
def test_captured_scs_changeover_head_survives_stronger_check(recording, epoch):
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -13.8
    phase = 1685546 + epoch * 60000
    audio = recording[phase - 7200:phase + 48000]
    ev = rxfront.SyncedRx().control_signal_at(
        p3acquire.compensate(audio, -13.8), 7200, details=True)
    assert ev is not None and ev.packet == (1, 0, b"RMS", True)
    # Withhold the already-proved body to exercise head-only acceptance.
    head = replace(ev, kind="cs", packet=None, breakin=False)
    assert s.rx._p3_bodyless_head_supported(audio, head)


def test_crc_packet_and_iss_control_are_not_subject_to_extra_gate(monkeypatch):
    s = _Session(role=arq.IRS)
    ev = SimpleNamespace(cs=arq.CS_BREAKIN, packet=(1, 0, b"RMS", True))
    monkeypatch.setattr(p3acquire, "changeover", lambda *a, **kw: pytest.fail("extra read"))
    assert s.rx._p3_bodyless_head_supported(np.zeros(1), ev)
    s.host.arq.role = arq.ISS
    ev.packet = None
    assert s.rx._p3_bodyless_head_supported(np.zeros(1), ev)
