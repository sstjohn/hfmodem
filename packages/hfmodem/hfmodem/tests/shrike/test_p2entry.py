# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The `p2sl1` rung: a PACTOR-2 SL1 short frame where the PACTOR-3 entry goes.

The offline checks for the August 31 entry experiment cover tones, rate, extent and raster fit; a decode through the P2 receive
path; negative controls on the P1 and P3 decoders; and the `template` control
untouched. The arm itself is a paired experiment on a granting P4dragon.
"""
import re
import wave

import numpy as np

from hfmodem.shrike import arq as arq_mod
from hfmodem.shrike import onair, p2rx, pactor1, pactor2, placement, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_connected_hush import grid
from hfmodem.tests.shrike.test_grantslot import _log
from hfmodem.tests.shrike.test_p3_offer import MESSAGE, Keyed, cs_event

FS = spec.SAMPLE_RATE
STATUS = 0x1A


def _entry(payload: bytes = b""):
    path = pactor2.PATHS[0]
    field = pactor2.build_field(
        placement.field_info(payload, path.crc_bytes - 3, STATUS), path)
    return pactor2.data_burst(field, path, swapped=False), field, path


def _band_power(audio, lo, hi):
    f = np.fft.rfftfreq(audio.size, 1 / FS)
    x = np.abs(np.fft.rfft(audio * np.hanning(audio.size))) ** 2
    return x[(f >= lo) & (f < hi)].sum() / x.sum()


def test_geometry_is_sl1_short_on_the_pactor1_tones_inside_the_raster():
    audio, _, path = _entry()
    assert path is pactor2.PATHS[0]
    assert (path.n_symbols, path.bits_per_cell) == (72, 1)          # DBPSK, short
    assert abs(onair.P2_ENTRY_END_N / FS - 0.815) < 1e-3
    keyed_s = (audio.size + onair.P2_KEY_LEAD_N) / FS
    assert keyed_s < spec.CYCLE_SHORT_S - 0.210
    assert _band_power(audio, 1350, 1450) > 0.35
    assert _band_power(audio, 1550, 1650) > 0.35
    assert _band_power(audio, 0, 1250) + _band_power(audio, 1750, FS / 2) < 0.01


def test_it_decodes_through_the_p2_receive_path_byte_exact():
    audio, field, path = _entry()
    pad = np.zeros(round(0.2 * FS), np.float32)
    got = p2rx.decode_expected_burst(np.concatenate([pad, audio, pad]), FS)
    assert got is not None
    _, seen_path, seen_field, _ = got
    assert seen_path.level == path.level and seen_path.n_symbols == 72
    assert bytes(seen_field) == field


def test_neither_the_p1_nor_the_p3_decoder_mistakes_it_for_its_own():
    audio, _, _ = _entry()
    pad = np.zeros(round(0.2 * FS), np.float32)
    window = np.concatenate([pad, audio, pad])
    assert rxfront.decode_expected_p1_packet(window) is None
    assert rxfront.decode_expected_packet(window) is None


class _Keyed(Keyed):
    def __init__(self):
        super().__init__()
        self.p3_entries: list[tuple[int, bytes, int]] = []
        self.p2_entries: list[tuple[bytes, int]] = []

    def send_entry_packet(self, sl, payload, status, *, acquire=False):
        self.p3_entries.append((sl, bytes(payload), status))

    def send_p2_entry_packet(self, payload, status):
        self.p2_entries.append((bytes(payload), status))


def _granted(ladder):
    keyed = _Keyed()
    host = PtcHost(peer=keyed, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.arq.on_host_data(MESSAGE)
    host.arq.cfg.entry_ladder = ladder
    host.on_rx_event(rxfront.Event(0.2, "unassigned", f"word {pactor1.CS_59A}",
                                   protocol="PACTOR-1", spare=pactor1.CS_59A,
                                   sense=0))
    host.tick()
    return keyed


def test_the_rung_is_buildable_and_default_off():
    assert arq_mod.ArqConfig().entry_ladder == ("template",)
    assert onair._entry_ladder("template,p2sl1") == ("template", "p2sl1")
    assert "p2sl1" not in arq_mod.ENTRY_RUNGS_GROUNDED


def test_a_grant_on_the_rung_keys_the_p2_entry_and_the_control_is_unchanged():
    keyed = _granted(("p2sl1",))
    assert keyed.p2_entries == [(b"", keyed.p2_entries[0][1])]
    assert keyed.p3_entries == []
    keyed = _granted(("template",))
    assert keyed.p3_entries and keyed.p3_entries[0][:2] == (1, b"")
    assert keyed.p2_entries == []


def test_the_replay_names_the_family_and_the_extent(tmp_path):
    log = _log(tmp_path, "--p1-grant-only", "--p3-entry", "p2sl1")
    assert "P2 SL1 COMPAT ENTRY 0B 815ms 1400/1600 Hz" in log, log
    assert "SL1 ENTRY 0B" not in log
    # The P2 waveform owns its measured extent for this rung.
    assert "entry we keyed ends 815.0 ms" in log, log


def test_a_later_template_rung_owns_its_own_extent(tmp_path):
    g = grid(0)
    g.keyed_slot = 1
    g._entry_extent_override = onair.P2_ENTRY_END_N
    g.keying(spec.Protocol.PACTOR3, extent_n=40000, entry_pending=True,
             entry_variant="p2sl1")
    assert g.entry_end_n == onair.P2_ENTRY_END_N
    assert g._entry_extent_override is None
    g.cycles = 8
    g.keyed_slot = 8
    old_answer = g.entry_key_n + 50000
    g.entry_answers = [(7, old_answer, 1000.)]
    g.answer_unread_ms = 1050.
    g.d_ref_n = g.p1_data_n
    g.keying(spec.Protocol.PACTOR3, extent_n=onair.ENTRY_END_N + 1148,
             entry_pending=True, entry_variant="template")
    assert g.entry_at == 8 and g.entry_key_n == g.boundary(8)
    assert not g.entry_answers
    g.update([old_answer], linked=True)
    assert not g.entry_answers and g.answer_unread_ms == 1050.
    assert g.entry_end_n == onair.ENTRY_END_N + 1148
    # Established traffic must not rewrite the entry measurement.
    g.keying(spec.Protocol.PACTOR3, extent_n=50000)
    assert g.entry_end_n == onair.ENTRY_END_N + 1148


def test_replayed_p2_then_template_ladder_reports_the_later_key(tmp_path):
    log = _log(tmp_path, "--p1-grant-only", "--p3-entry", "p2sl1,template")
    assert "P2 SL1 COMPAT ENTRY" in log and "SL1 ENTRY" in log
    extents = [float(x) for x in re.findall(r"entry we keyed ends ([0-9.]+) ms", log)]
    last = int(re.findall(r"TX\[(\d+)\] SL1 ENTRY", log)[-1])
    with wave.open(str(tmp_path / "out" / f"tx_{last:02d}.wav")) as wav:
        measured_ms = wav.getnframes() / wav.getframerate() * 1000
    assert extents and abs(extents[-1] - measured_ms) < .1
    assert 20 < measured_ms - 815 < 35
