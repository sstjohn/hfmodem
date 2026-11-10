# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--p3-crest-limit`: the rendered peak-to-average ratio, graded against SCS's
own crest table, and what limiting costs the decode, the spectrum and the peak."""
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from hfmodem.core import levels
from hfmodem.shrike import onair, placement, rxfront, spec

TARGET = 6.0
FS = spec.SAMPLE_RATE
STATUS = spec.status_byte(0)


def _payload(sl: int, long_cycle: bool = False) -> bytes:
    s = spec.SPEED_LEVELS[sl]
    n = s.payload_long if long_cycle else s.payload_short
    text = f"SL{sl} SHRIKE CONFORMANT PACKET ".encode()
    return (text * (n // len(text) + 1))[:n]


def _papr(a: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(a).max() / np.sqrt(np.mean(a ** 2))))


def _band_power(a: np.ndarray) -> tuple[float, float]:
    f, p = signal.welch(a, FS, nperseg=8192)
    inband = p[(f >= 400) & (f <= 2600)].sum()
    beside = p[(f >= 2700) & (f <= 3300)].sum()
    return float(inband / p.sum()), float(10 * np.log10(beside / inband))


def _packets(audio: np.ndarray) -> list[tuple]:
    padded = np.concatenate([np.zeros(FS // 10), audio, np.zeros(FS // 10)])
    return [e.packet for e in rxfront.decode_events(padded) if e.packet]


def _render(sl: int, crest: float | None, long_cycle: bool = False, monkeypatch=None):
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", crest)
    return placement.link_packet(sl, _payload(sl, long_cycle), STATUS,
                                 long_cycle=long_cycle)


# Gain the limiter is held to at TARGET, per level, against the unlimited sum.
# Measured 0.2-0.3 dB above these on the payload above; level 1 is two tones and
# has little to give.
GAIN_DB = {1: 0.9, 2: 3.7, 3: 4.7, 4: 5.3, 5: 5.8, 6: 5.8}


@pytest.mark.parametrize("sl", sorted(spec.SPEED_LEVELS))
def test_rendered_papr_per_level_against_the_spec_table(sl, monkeypatch):
    table = spec.SPEED_LEVELS[sl].crest_factor_db
    raw = _papr(_render(sl, None, monkeypatch=monkeypatch))
    limited = _papr(_render(sl, TARGET, monkeypatch=monkeypatch))
    assert raw - limited >= GAIN_DB[sl]
    assert limited <= TARGET + 1.8
    # Still above the table -- SCS's figure is the objective, not the claim --
    # but by less than 4.7 dB where the unlimited sum sits 5.7-10.2 above it.
    assert limited <= table + 4.7


@pytest.mark.parametrize("crest", [TARGET, TARGET - 1])
@pytest.mark.parametrize("sl,long_cycle", [(sl, False) for sl in sorted(spec.SPEED_LEVELS)]
                         + [(1, True), (3, True)])
def test_limited_render_decodes_byte_exact(sl, long_cycle, crest, monkeypatch):
    got = _packets(_render(sl, crest, long_cycle, monkeypatch=monkeypatch))
    assert got, f"SL{sl} at {crest} dB did not decode"
    level, status, payload, crc_ok = got[0]
    assert (level, status, payload, crc_ok) == (sl, STATUS, _payload(sl, long_cycle), True)


@pytest.mark.parametrize("sl", sorted(spec.SPEED_LEVELS))
def test_limited_render_keeps_the_band_and_the_peak(sl, monkeypatch):
    raw = _render(sl, None, monkeypatch=monkeypatch)
    limited = _render(sl, TARGET, monkeypatch=monkeypatch)
    occupied_raw, acpr_raw = _band_power(raw)
    occupied, acpr = _band_power(limited)
    assert occupied >= 0.999 and occupied >= occupied_raw - 1e-4
    assert acpr <= acpr_raw
    # The PACTOR-1 renders overshoot the drive by 1.1 dB between samples; a
    # limited packet keeps the 0.03 dB a band-limited sum at 48 kHz has.
    keyed = levels.at_drive(limited).astype(float)
    true_peak = np.abs(signal.resample_poly(keyed, 8, 1)).max()
    assert 20 * np.log10(true_peak / levels.TX_DRIVE) <= 0.05


def test_control_burst_follows_the_setting_through_its_cache(monkeypatch):
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", None)
    raw = placement.control_signal(5, swapped=True)
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", TARGET)
    limited = placement.control_signal(5, swapped=True)
    assert limited is not raw and _papr(raw) - _papr(limited) >= 1.2
    assert placement.control_pulse_lead(5, swapped=True) == placement.pulse_lead(limited)
    assert not limited.flags.writeable
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", None)
    assert placement.control_signal(5, swapped=True) is raw
    assert placement.control_pulse_lead(5, swapped=True) == placement.pulse_lead(raw)


def test_onair_render_caches_key_on_the_setting(monkeypatch):
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."))
    ident = ("pkt", 3, _payload(3), STATUS, False)
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", None)
    raw = tx._p3_link_packet(ident)
    entry_raw = tx._entry_burst(1, b"", STATUS, False)[0]
    monkeypatch.setattr(placement, "CREST_LIMIT_DB", TARGET)
    limited = tx._p3_link_packet(ident)
    entry = tx._entry_burst(1, b"", STATUS, False)[0]
    assert limited is not raw and _papr(raw) - _papr(limited) >= GAIN_DB[3]
    assert entry is not entry_raw and _papr(entry_raw) - _papr(entry) >= GAIN_DB[1]
    assert tx._p3_link_packet(ident) is limited
