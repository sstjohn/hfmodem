# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Which read reports the peer's phase, and which only confirms it.

`p3_reply_shift` carries the receiver's coordinate into the transmit comb, so
the two PACTOR-3 changeover readers have to agree about where a packet is. They
did not: over `ws8eoc-0910/first-rms.wav` the acquisition reports the head the
crop's own manifest records, and the tracked reader reports it 300 samples
earlier and stays there. The fixture PCM decides between them here.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p3acquire, placement, rx, rxfront, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_p3_rx_recovery import scene

FS = onair.FS
SPS = rxfront.SPS
CYCLE = round(spec.CYCLE_SHORT_S * FS)
METADATA = Path(__file__).with_name("fixtures") / "ws8eoc-0910" / "metadata.json"
pytestmark = pytest.mark.skipif(
    not METADATA.exists(),
    reason=f"WS8EOC morning recordings absent: {METADATA}")


def crop():
    """The recorded changeover, the head its manifest records, and its CFO."""
    row = next(r for r in json.loads(METADATA.read_text())["fixtures"]
               if r["file"] == "first-rms.wav")
    pcm = recorded_pcm({"file": "ws8eoc-0910/first-rms.wav",
                        "sha256": row["pcm_sha256"]})
    return pcm, row["onset_relative"], float(row["correction_hz"])


def placed(pcm: np.ndarray, head: int, at: int, lead: int = FS):
    """The crop dropped into quiet so its head lands on `at` of a long buffer."""
    buf = np.zeros(lead + len(pcm) + FS, np.float32)
    buf[lead:lead + len(pcm)] = pcm
    return buf, at - (lead + head)


def coherent_quality(audio: np.ndarray, hz: float, at: int, half=4 * SPS):
    """`p3acquire`'s own CS3 metric at full rate, one alignment per sixteenth.

    The acquisition maximises this over the window; `cycle_scan.cs3_profile` is
    the same arithmetic with the accept gates taken off. It is the only one of
    the two estimators that can place a codeword inside its own capture range.
    """
    shifted = p3acquire.compensate(audio, hz)
    pulse = rx._pulse(SPS)
    delay = (len(pulse) - 1) // 2
    Z = {cn: rx._baseband(shifted, cn, FS, pulse) for cn in spec.VH_CHANNELS}
    signs = 1.0 - 2.0 * rx._CS_TABLE[placement.BREAKIN_CS]
    out = []
    for st in range(at - half, at + half + 1, SPS // 16):
        idx = st + np.arange(21) * SPS + delay
        y = np.asarray([Z[cn][idx] for cn in spec.VH_CHANNELS])
        d = y[..., 1:] * np.conj(y[..., :-1])
        out.append((float((d.real.sum(axis=0) * signs).sum()
                          / max(np.abs(d).sum(), 1e-30)), st))
    return out


# WHERE THIS CROP'S CARRIERS ARE, as against the `correction_hz` its manifest
# records, which is the coarse hypothesis that read the head and not a
# measurement of it. The head's own twenty symbol pairs put them at -47.5 Hz
# (`p3acquire._refined`); an independent template fit against nominal 1080/1920
# says -47.56 and a 0.5 Hz quality sweep peaks at -48.5.
TRUE_HZ = -47.5


def test_the_acquisition_reports_the_head_the_recording_holds():
    """The coherent argmax, the manifest's onset and `p3acquire` are one sample."""
    pcm, head, hz = crop()
    got = p3acquire.changeover(pcm)
    assert got is not None and got.event.packet is not None
    assert got.coarse_hz == hz and got.offset_hz == TRUE_HZ
    assert got.event.start == head
    assert max(coherent_quality(pcm, hz, head))[1] == head


def test_the_tracked_reader_lands_on_the_head_the_recording_holds():
    """...and it now does, on a soft metric the hard decision cannot supply.

    Every alignment from 300 samples early to 120 late decodes CS3 at zero bit
    errors -- twenty bits over six words at mutual distance twelve -- so the
    band below is what the codeword alone can say, and its first member is where
    `_best_cs` used to stop. Choosing inside that band by `_cs_coherence` puts
    the reading on the same sample the acquisition and the manifest hold.
    """
    pcm, head, hz = crop()
    audio, origin = placed(pcm, head, 0)
    at = -origin
    shifted = p3acquire.compensate(audio, hz)
    pulse = rx._pulse(SPS)
    delay = (len(pulse) - 1) // 2
    Z = {cn: rx._baseband(shifted, cn, FS, pulse) for cn in rxfront.HDR_TONES}
    zero = [st - at for st in range(at - 8 * SPS, at + 8 * SPS + 1, SPS // 16)
            if rx.nearest_control_signal(rx.cs_bits(Z, st, delay))
            == (placement.BREAKIN_CS, 0)]
    assert zero == list(range(-300, 121, SPS // 16))
    ev = rxfront.SyncedRx().control_signal_at(shifted, at)
    assert ev is not None and ev.packet is not None
    assert ev.start - at == 0
    assert zero[0] == -300          # what the band alone says, and it is not it


def test_a_tracked_changeover_is_delivered_on_the_raster_it_confirms(tmp_path):
    """So the receiver carries the instant it aimed at, not the plateau's edge.

    One repeat read through the production tracked seam: the delivery, the
    clock it leaves behind and the grid's copy of the peer's phase are all the
    projected head, and the comb answers exactly `P3_REPLY_S` past it.
    """
    pcm, head, hz = crop()
    seeded = 3 * CYCLE
    session, _, grid = scene(tmp_path, seeded_at=seeded, slot=62)
    grid.sending = False          # An IRS holding the link, which is whose comb this is.
    session.rx.p3_receive_offset_hz = hz
    audio, origin = placed(pcm, head, seeded + CYCLE)
    session.rx.new_cycle()
    onair._scan_frame(session.rx, audio, origin, tracked_only=True)

    assert [ev.packet[:3] for ev in session.packets] == [(1, 0, b"RMS")]
    assert session.rx._p3_delivered_at == seeded + CYCLE
    assert grid._p3_peer[0] == seeded + CYCLE
    assert grid.p3_reply_shift(62) is not None
    assert ((grid.boundary(62) - grid._p3_peer[0]) % grid.slot_n
            == round(onair.P3_REPLY_S * FS))


def test_a_peer_that_moved_is_measured_again_rather_than_projected(tmp_path):
    """The clock still closes on the audio, which is what bounds the projection.

    A repeat five symbols off the raster is past the tracked reader's four, so
    it misses, the acquisition runs, and the delivery is that read's own
    measurement -- not the instant the projection wanted.
    """
    pcm, head, hz = crop()
    seeded = 3 * CYCLE
    moved = 5 * SPS
    session, _, grid = scene(tmp_path, seeded_at=seeded, slot=62)
    session.rx.p3_receive_offset_hz = hz
    audio, origin = placed(pcm, head, seeded + CYCLE + moved)
    session.rx.new_cycle()
    onair._scan_frame(session.rx, audio, origin)

    assert [ev.packet[:3] for ev in session.packets] == [(1, 0, b"RMS")]
    assert session.rx._p3_delivered_at == seeded + CYCLE + moved
    assert grid._p3_peer[0] == seeded + CYCLE + moved


def test_the_recorded_repeats_walk_the_raster_by_their_own_cycle(tmp_path):
    """Four physical repeats of one stint, each read where the last one put it.

    The crops sit 179940, 180000 and 120012 samples apart -- the peer's own
    clock against our nominal 60000 -- so this is what the projection costs
    when nothing re-measures it: a couple of milliseconds over the stint,
    against the 300 samples the plateau's edge takes off every reading.
    """
    rows = {r["file"]: r for r in json.loads(METADATA.read_text())["fixtures"]}
    names = ["first-rms.wav", "repeat-rms.wav", "guard-rms.wav", "next-rms.wav"]
    onsets = [rows[n]["start_sample"] + rows[n]["onset_relative"] for n in names]
    session, _, grid = scene(tmp_path, seeded_at=onsets[0], slot=62)
    session.rx.p3_receive_offset_hz = float(rows[names[0]]["correction_hz"])
    walk = []
    for name, onset in zip(names[1:], onsets[1:]):
        row = rows[name]
        pcm = recorded_pcm({"file": "ws8eoc-0910/" + name,
                            "sha256": row["pcm_sha256"]})
        audio, origin = placed(pcm, row["onset_relative"], onset)
        session.rx.new_cycle()
        onair._scan_frame(session.rx, audio, origin, tracked_only=True)
        assert session.packets[-1].packet[:3] == (1, 0, b"RMS")
        walk.append(session.rx._p3_delivered_at - onset)
    assert walk == [60, 60, 48]          # 1.25 ms at worst, against the plateau edge's 300.
    # ...and every one of them still lands on one raster, which is what the
    # comb is placed from.
    assert grid._p3_raster_run == 3
    assert grid._peer_raster_position(grid._p3_peer[0], CYCLE) is not None
