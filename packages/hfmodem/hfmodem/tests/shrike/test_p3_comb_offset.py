# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-III channel probe has to be pointed at the peer, not at the dial.

`rxfront.p3_comb_db` asks whether 1080 and 1920 Hz stand over the 120 Hz comb.
Both numbers are nominal, `p3rx.TONE_HALFWIDTH_HZ` is 45 and the comb is spaced
120, so a peer 85 Hz low has no energy in either anchor and its NEIGHBOURING tone
35 Hz inside one of them: the probe reads a channel the station never lit against
a median it did. Every PACTOR-III peer this station has worked sits tens of hertz
off -- WS8EOC at -85, KB5LZK at +74.6 within the same fortnight -- so nominal is
the case that does not occur.

The knee was re-derived here and left at 8.0. The four copies whose readability
is known do not separate once the probe points correctly, and `P3_COMB_KNEE_DB`
records why; what this file keeps in the tree is the four readings themselves.

Run:  python -m pytest hfmodem/tests/shrike/test_p3_comb_offset.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from hfmodem.shrike import onair, rxfront, spec

FS = onair.FS
REPO = Path(__file__).resolve().parents[5]
FIXTURE = Path(__file__).with_name("fixtures") / "kb5lzk-answer-0915"

WS8EOC = REPO / "captures" / "onair-0914-2252"
"""The 80 m greeting of 2026-09-14, recorded twice: this station's own input and
an independently registered witness receiver 11999 Hz away."""

WS8EOC_PACKETS = ((1848762, 154, -85.75), (2688762, 91, -84.50),
                  (2748762, 75, -84.75))
"""Stream head, registration delta and main-input CFO of the three SL3 packets,
as the v31 known-geometry cut of the 09-14 WS8EOC session took them:
the frame's first data row is head + 4320 + delta."""

WITNESS_LEAD = 691469
WITNESS_CFO = -88.50
"""Witness registration: witness sample = (main + lead) / 4, the lead being the
mean of the two RMS anchors `ws8_witness_v1.py` fitted. It is not sample-exact
over this range, which costs nothing here -- the probe scores every position in
the window and reports its best."""

DATA_N = round(0.81 * FS)


def _cut(path: Path, lo: int, n: int) -> np.ndarray:
    fs, pcm = wavfile.read(path, mmap=True)
    seg = pcm[lo:lo + n]
    seg = seg[:, 0] if seg.ndim == 2 else seg
    return np.asarray(seg, dtype=np.float64) / 32768


def _pair(shift: float) -> np.ndarray:
    """An SL1-like two-carrier burst, `shift` Hz off nominal, in the same noise."""
    rng = np.random.default_rng(20260916)
    t = np.arange(round(0.34 * FS)) / FS
    out = rng.normal(0, .002, t.size)
    for cn in rxfront.P3_ANCHOR_CHANNELS:
        hz = spec.channel_freq_hz(cn) + shift
        out += .05 * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
    return out


def test_the_probe_follows_the_offset_and_not_the_dial():
    """The same signal, read at nominal and at +75 Hz, is the same reading.

    Which is the whole claim: the measure is offset-driven rather than retuned to
    one peer. Without the offset the shifted copy reads as an empty channel.
    """
    nominal, shifted = _pair(0.0), _pair(75.0)
    assert abs(rxfront.p3_comb_db(shifted, offset_hz=75.0)
               - rxfront.p3_comb_db(nominal)) < 1.0
    assert rxfront.p3_comb_db(shifted) < rxfront.p3_comb_db(nominal) - 40.0


def test_a_nominal_signal_read_at_an_offset_is_an_empty_channel():
    """The other direction, so the first test cannot pass by ignoring the offset."""
    nominal = _pair(0.0)
    assert rxfront.p3_comb_db(nominal, offset_hz=75.0) < 5.0


@pytest.fixture(scope="module")
def kb5lzk():
    """The two cycles of the 09-15 KB5LZK answer, as committed cuts."""
    if not (FIXTURE / "metadata.json").exists():
        pytest.skip(f"no {FIXTURE.name} cuts under {FIXTURE.parent}")
    meta = json.loads((FIXTURE / "metadata.json").read_text())
    rows = {}
    for row in meta["rows"]:
        fs, pcm = wavfile.read(FIXTURE / row["file"])
        assert fs == FS
        rows[row["hold"]] = pcm.astype(float) / 32768
    return meta["correction_hz"], rows


def test_the_peers_own_offset_lifts_both_kb5lzk_answers(kb5lzk):
    """Cycle 35 carried a full-strength answer, cycle 23 the same word with
    channel 5 faded. Pointed at +74.6 Hz both rise by about 5 dB, and 23 crosses
    the knee it was under -- it IS a PACTOR-III emission, which is what the flag
    claims to name."""
    hz, rows = kb5lzk
    for hold, nominal, pointed in ((35, 13.2, 17.7), (23, 3.2, 8.2)):
        assert rxfront.p3_comb_db(rows[hold]) == pytest.approx(nominal, abs=0.2)
        assert rxfront.p3_comb_db(rows[hold], offset_hz=hz) \
            == pytest.approx(pointed, abs=0.2)


def test_the_full_strength_answer_outreads_the_faded_one(kb5lzk):
    """The pair that does order by readability, and by nine dB."""
    hz, rows = kb5lzk
    assert (rxfront.p3_comb_db(rows[35], offset_hz=hz)
            > rxfront.p3_comb_db(rows[23], offset_hz=hz) + 5.0)


@pytest.fixture(scope="module")
def ws8eoc():
    """Main and witness copies of the three SL3 greeting packets, by head."""
    if not (WS8EOC / "stream.wav").exists() or not (WS8EOC / "witness.wav").exists():
        pytest.skip(f"no {WS8EOC.name} under {REPO / 'captures'}")
    out = {}
    for head, delta, cfo in WS8EOC_PACKETS:
        main = _cut(WS8EOC / "stream.wav", head + 4320 + delta, DATA_N)
        lo = (head - 4800 + WITNESS_LEAD) // 4
        native = _cut(WS8EOC / "witness.wav", lo, DATA_N // 4 + 8000)
        at = head + WITNESS_LEAD - lo * 4 + 4320 + delta
        out[head] = (main, cfo, resample_poly(native, 4, 1)[at:at + DATA_N])
    return out


def test_the_mispointed_probe_read_the_main_input_as_an_empty_channel(ws8eoc):
    """What the defect costs: four dB of a peer's own comb, on every packet."""
    for head, (main, cfo, _) in ws8eoc.items():
        assert rxfront.p3_comb_db(main) < 2.0
        assert rxfront.p3_comb_db(main, offset_hz=cfo) \
            > rxfront.p3_comb_db(main) + 2.0


def test_no_knee_separates_the_readable_copy_from_the_unreadable_one(ws8eoc):
    """And why 8.0 stayed. The witness copy CRC-decodes this greeting and the
    main input does not, so the witness should read higher; pointed correctly it
    reads 1.1/3.9/0.6 dB against the main input's 5.7/3.4/4.1. The populations
    interleave, so nothing fitted to them would mean anything -- see
    `P3_COMB_KNEE_DB` for the reference artefact behind it."""
    witness = [rxfront.p3_comb_db(w, offset_hz=WITNESS_CFO)
               for _, _, w in ws8eoc.values()]
    main = [rxfront.p3_comb_db(m, offset_hz=cfo) for m, cfo, _ in ws8eoc.values()]
    assert min(witness) < max(main)
    assert min(main) < max(witness)


def _host(monkeypatch, **env):
    for name in ("HOST_FREE_IDENT", "HOST_FREE_AT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    host = type("_Host", (), {})()
    onair._note_host_free(host)
    return host


def test_the_launchers_free_signal_reaches_the_transcript(monkeypatch, capsys):
    """`tools/lib/attempts.sh` exports the pair; the arm records both and says so."""
    host = _host(monkeypatch, HOST_FREE_IDENT="WS8EOC",
                 HOST_FREE_AT=f"{time.time() - 12.5:.0f}")
    assert host.host_free_ident == "WS8EOC"
    line = capsys.readouterr().out
    assert "host free: WS8EOC last burst 1" in line and "s before start" in line


def test_an_arm_the_launcher_said_nothing_about_records_nothing(monkeypatch, capsys):
    host = _host(monkeypatch)
    assert host.host_free_ident is None and host.host_free_at is None
    assert "host free" not in capsys.readouterr().out


@pytest.mark.parametrize("env", ({"HOST_FREE_IDENT": "WS8EOC"},
                                 {"HOST_FREE_AT": "1758000000"},
                                 {"HOST_FREE_IDENT": "WS8EOC",
                                  "HOST_FREE_AT": "not-an-instant"}))
def test_half_a_reading_is_no_reading(monkeypatch, capsys, env):
    """An ident with no instant is a station list, not a go-ahead, and a bad
    export costs the arm a line rather than its slot."""
    host = _host(monkeypatch, **env)
    assert host.host_free_ident is None and host.host_free_at is None
    assert "host free" not in capsys.readouterr().out
