# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The first v24 SL3 counter must reach its own CS choice, without hardware."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3frame, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
from hfmodem.tests.shrike.test_slot_deadline import charging

FIXTURE = Path(__file__).with_name("fixtures") / "wideband-prekey-0914"
SEED = 2388632
ROW0 = SEED + p3frame.DATA_OFFSET * rxfront.SPS + 31 * 60000
PAYLOAD = b" Trimode 1.4.3.0\r\nW9SSJ has 85 daily minutes remaining with"


@pytest.fixture(scope="module")
def recording():
    if not (FIXTURE / "metadata.json").exists():
        pytest.skip(f"no {FIXTURE.name} cut under {FIXTURE.parent}")
    meta = json.loads((FIXTURE / "metadata.json").read_text())
    row = meta["rows"][0]
    path = FIXTURE / row["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    fs, pcm = wavfile.read(path)
    assert fs == onair.FS and len(pcm) == row["end"] - row["start"]
    return row, pcm.astype(float) / 32768


def receiver(enabled=True):
    session = _Session(role=arq.IRS)
    session.rx.p3_wideband_prekey = enabled
    session.rx._seed_p3_changeover_clock(SEED)
    session.rx._p3_clock_role = arq.IRS
    session.rx.sync.packet_level = 1
    session.rx.p3_receive_offset_hz = -19.9
    session.host.arq._rx_seen = True
    session.host.arq._expected_seq = 1
    session.host.arq.cfg.long_cycle = False
    return session


def test_first_wideband_frame_arrives_without_a_later_decode(recording):
    row, pcm = recording
    s = receiver()
    onair._scan_frame(s.rx, pcm, row["start"], tracked_only=True)
    assert [ev.packet[:3] for ev in s.packets] == [(3, 0x21, PAYLOAD)]
    assert abs(s.rx._p3_row0 - ROW0) <= rxfront.SPS // 4
    assert s.rx.sync.packet_level == 3
    assert s.host.arq.rx_seq == 1
    assert s.host.peer.sent[-1] == ("cs", arq.CS_REQUEST)


def test_existing_path_misses_that_same_partial_window(recording, monkeypatch):
    row, pcm = recording
    s = receiver(False)
    # No broad acquisition fits here in the real arm.
    monkeypatch.setattr(s.rx, "_p3_acquisition_fits", lambda _: False)
    onair._scan_frame(s.rx, pcm, row["start"], tracked_only=True)
    assert s.packets == []


def test_one_packet_cannot_be_delivered_twice(recording):
    row, pcm = recording
    s = receiver()
    for _ in range(2):
        s.rx.new_cycle()
        onair._scan_frame(s.rx, pcm, row["start"], tracked_only=True)
    assert len(s.packets) == 1


def test_lost_preferred_level_never_opens_experimental_ladder(recording, monkeypatch):
    row, pcm = recording
    s = receiver()
    s.rx.sync.packet_level = None
    s.rx._tracked_only, s.rx._scan_origin = True, row["start"]
    monkeypatch.setattr(s.rx.sync, "wideband_packet_at", lambda *a, **k: (None, False))
    monkeypatch.setattr(s.rx, "_read_p3_packet", lambda *a, **k: pytest.fail("unbounded level trial"))
    assert s.rx._p3_packet(pcm) is None


def test_missing_too_much_tail_does_not_reach_the_crc(recording, monkeypatch):
    row, pcm = recording
    at = ROW0 - row["start"]
    short = pcm[:at + 34560 - rxfront.SyncedRx.WIDEBAND_EARLY_N - 1]
    monkeypatch.setattr(rxfront.p3rx, "decode_at", lambda *a, **k: pytest.fail("short field"))
    assert rxfront.SyncedRx().wideband_packet_at(short, at) == (None, False)


@pytest.mark.parametrize("state", ["disabled", "iss", "entry", "long", "command", "higher-level", "no-clock"])
def test_feature_is_scoped_to_short_irs_cycles(state):
    s = receiver()
    if state == "disabled":
        s.rx.p3_wideband_prekey = False
    elif state == "iss":
        s.host.arq.role = arq.ISS
    elif state == "entry":
        s.host.arq.entry_pending = True
    elif state == "long":
        s.host.arq.cycle_long = True
    elif state == "command":
        s.host.arq._cycle_command_emitted = True
    elif state == "higher-level":
        s.rx.sync.packet_level = 5
    else:
        s.rx._p3_row0 = None
    assert not s.rx.wideband_prekey_active()


def test_noise_and_bare_controls_do_not_become_wideband_packets():
    sync = rxfront.SyncedRx()
    rng = np.random.default_rng(91438)
    for _ in range(200):
        assert sync.wideband_packet_at(rng.normal(0, .1, 43000), 6720)[0] is None
    for cs in range(6):
        wave = np.pad(placement.control_signal(cs), (2400, 48000))
        assert sync.wideband_packet_at(wave, 6720)[0] is None


def test_recorded_deadline_selects_current_partial_packet(recording):
    row, _ = recording
    s = receiver()
    deadline = onair._p3_decode_deadline(SimpleNamespace(key_notice=2612), row["key"], 1920)
    ready = onair._p3_frame_ready(s.rx, deadline, 660)
    delivered = (ready - 532) // 128 * 128
    assert ready <= deadline
    assert delivered >= row["end"]
    assert onair._p3_wideband_target(s.rx, delivered) == ROW0


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("early_ms", [0, 12])
def test_header_selects_short_sl3(swapped, early_ms):
    sl = 3
    audio = np.pad(placement.link_packet(sl, b"wideband", 0x21, swapped=swapped),
                   (4800, 4800))
    at = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + p3frame.DATA_OFFSET * rxfront.SPS
    span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[sl]))
    ev, owned = rxfront.SyncedRx().wideband_packet_at(audio[:at + span - early_ms*48], at)
    assert owned and ev is not None
    assert ev.packet == (sl, 0x21, b"wideband", True)
    assert ev.carrier_swapped == swapped
    assert ev.cycle_long is False


@pytest.mark.parametrize("sl", [4, 5, 6])
@pytest.mark.parametrize("swapped", [False, True])
def test_other_wideband_levels_are_not_given_an_early_crc_trial(sl, swapped, monkeypatch):
    audio = np.pad(placement.link_packet(sl, b"other level", 0x21, swapped=swapped),
                   (4800, 4800))
    at = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + p3frame.DATA_OFFSET * rxfront.SPS
    monkeypatch.setattr(rxfront.p3rx, "decode_at", lambda *a, **k: pytest.fail("non-SL3 trial"))
    assert rxfront.SyncedRx().wideband_packet_at(audio, at) == (None, False)


@pytest.mark.parametrize("long_reply", [False, True])
def test_enabled_flag_preserves_existing_cs6_acquisition(tmp_path, long_reply):
    from hfmodem.tests.shrike.test_cs6_driver import _arm, _window
    from hfmodem.tests.shrike.test_cs6_rx import FIXTURES, ROWS, recorded
    if not ROWS or not (FIXTURES / "reference-long.wav").is_file():
        pytest.skip(f"no CS6 recorded PCM under {FIXTURES}")
    if long_reply:
        fs, raw = wavfile.read(FIXTURES / "reference-long.wav")
        assert fs == onair.FS
        pcm = raw.astype(float) / 32768
    else:
        pcm = recorded(ROWS[0])
    s, tx, grid, clock = _arm(tmp_path, pcm, sl=3 if long_reply else 1,
                            expected_seq=0 if long_reply else 1)
    s.rx.p3_wideband_prekey = True
    slot, _, _ = _window(s, tx, grid, clock)
    assert slot == (3 if long_reply else 1)
    assert len(s.packets) == 1
    assert s.packets[0].cycle_long == long_reply
    assert not s.host.arq.cycle_command_emitted


def test_long_header_is_left_to_existing_reader():
    audio = np.pad(placement.link_packet(3, b"long", 0x21, long_cycle=True), (4800, 4800))
    at = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + p3frame.DATA_OFFSET * rxfront.SPS
    assert rxfront.SyncedRx().wideband_packet_at(audio, at) == (None, False)


def test_no_budget_does_not_start_either_decoder(recording, monkeypatch):
    row, pcm = recording
    s = receiver()
    monkeypatch.setattr(s.rx, "_p3_acquisition_fits", lambda _: False)
    monkeypatch.setattr(s.rx.sync, "wideband_packet_at", lambda *a: pytest.fail("late wideband trial"))
    monkeypatch.setattr(s.rx, "_read_p3_packet", lambda *a, **k: pytest.fail("late fallback"))
    s.rx._tracked_only, s.rx._scan_origin = True, row["start"]
    assert s.rx._p3_packet(pcm) is None


def test_owned_crc_miss_does_not_start_a_level_ladder(recording, monkeypatch):
    row, pcm = recording
    s = receiver()
    monkeypatch.setattr(s.rx.sync, "wideband_packet_at", lambda *a, **k: (None, True))
    monkeypatch.setattr(s.rx, "_read_p3_packet", lambda *a, **k: pytest.fail("CRC miss fallback"))
    s.rx._tracked_only, s.rx._scan_origin = True, row["start"]
    assert s.rx._p3_packet(pcm) is None


def test_full_receive_tick_render_keeps_first_packet_reply(recording, tmp_path, capsys):
    row, pcm = recording
    s = receiver()
    g = irs_grid(0)
    g.note_p3_packet(SEED + 30*60000, 38880, 60000)
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.sessrx = s.rx
    tx.p3_control_placement = "pulse-center"
    tx.defer_p3_cs = True
    bench = _Bench(seconds=95, blk=128, holdback=660, lat_in=532)
    bench._lat, bench.tx_latency_n = 1268, 960
    bench.audio[row["start"]:row["end"]] = pcm
    bench.now = row["end"] + bench.lat_in
    bench.pos = row["end"]
    tx.live = bench
    slot = 71
    tx.aim(g, slot)
    onair._p3_place_reply(g, tx, slot)
    key = tx.key_instant(g, slot)
    cost = {}
    with charging(bench, cost):
        onair._scan_frame(s.rx, pcm, row["start"], tracked_only=True)
        assert s.host.arq.rx_seq == 1
        assert tx._pending_p3_cs == arq.CS_REQUEST
        slot, _, _ = onair._regrid(bench, g, tx, s.host, s.rx, slot, pcm, row["start"], 1920)
        # Charge the tick and control render as well as the receive stages.
        tick = time.perf_counter()
        s.host.tick()
        bench.spend(round((time.perf_counter()-tick)*onair.FS))
        render = tx._tx

        def charged_render(*args, **kwargs):
            # Rendering before _tx is charged by the interval from emit start.
            bench.spend(round((time.perf_counter()-started)*onair.FS))
            return render(*args, **kwargs)

        tx._tx = charged_render
        started = time.perf_counter()
        tx.emit_pending_cs()
    assert slot == 71
    assert tx.n == 1 and not tx.refused
    assert len(bench.emissions) == 1
    assert abs(tx.tx_audio_start - row["key"]) < round(.005 * onair.FS)
    assert not tx._pending_p3_cs
    said = capsys.readouterr().out
    assert "CS2 ACK seq=1" in said
    assert "IS GONE" not in said and "LATE TO THE KEY" not in said
    print(f"first SL3 scan {cost['scan'][0]:.2f} ms; original key {key}; emitted {tx.tx_audio_start}")
