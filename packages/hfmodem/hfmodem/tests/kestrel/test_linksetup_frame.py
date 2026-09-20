# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The link-setup frame: layout, and that it rides the ordinary wideband DATA
burst rather than a separate narrowband one [spec/04 §4.2A].

These pin the 2026-07-21 corrections to §4.2A/§3.5.5, which were established
against the staged capture after an earlier reading described the burst as
narrowband 16-FSK carrying no caller identity, and the 2026-08-15 bench
measurement of the same frame at BW500.
"""
import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf500 as rx500
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.tests.kestrel.corpora import (
    BW2300_CAPTURE,
    harness,
    requires_bw2300_capture,
    wav_mono,
)
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_ofdm as ofdm
from hfmodem.kestrel.vara.vara_frames import (
    caller_from_link_setup,
    crc16_genibus,
    is_link_setup,
    link_setup_frame,
)

_probe = harness("linksetup_probe")
frame, genuine_body, variants = _probe.frame, _probe.genuine_body, _probe.variants

# The link-setup frame as recovered from the staged capture, in full.
_CAPTURED = bytes.fromhex(
    "0410417000000080147f" + "00" * 78 + "0482" + "5387")


def _capture():
    a = wav_mono(BW2300_CAPTURE)
    return a / (np.abs(a).max() or 1.0)


def test_body_layout_is_uniquely_determined_by_its_crc():
    """The tail is load-bearing: only 88=04, 89=0x82 reproduces the CRC."""
    head = _CAPTURED[:10]
    assert crc16_genibus(head + bytes(80)) != 0x5387, "all-zero tail must not fit"
    solutions = [n for n in range(256)
                 if crc16_genibus(head + bytes(78) + bytes([0x04, n])) == 0x5387]
    assert solutions == [0x82]


def test_genuine_body_reconstructs_the_captured_frame():
    assert frame(genuine_body("AAAA1")) == _CAPTURED


def test_zero_region_is_78_bytes():
    assert _CAPTURED[10:88] == bytes(78)


# Heads (off 0..9) of real SSID'd link-setup frames captured from a stock VARA
# on 2026-07-23 — the SSID rides body[5], the base call stays in the 6-bit field,
# and byte 9 CRCs the full "CALL-SSID" literal [spec/04 §4.2A].
@pytest.mark.parametrize("caller,head", [
    ("W9SSJ",    "5e44d3280000008014f6"),   # bare: byte5=0, byte9=crc("W9SSJ")
    ("W9SSJ-5",  "5e44d328000500801443"),   # byte5=0x05, byte9=crc("W9SSJ-5")
    ("W9SSJ-10", "5e44d328000a00801437"),   # byte5=0x0a, byte9=crc("W9SSJ-10")
])
def test_ssid_encoding_is_byte_exact(caller, head):
    fb = link_setup_frame(caller)
    assert fb[:10] == bytes.fromhex(head)
    base, _, ssid = caller.partition("-")
    assert fb[5] == (int(ssid) if ssid else 0), "SSID lives in body[5], not the 6-bit field"
    assert fb[:4] == link_setup_frame(base)[:4], "6-bit field carries the base call only"
    assert caller_from_link_setup(fb) == caller


@pytest.mark.parametrize("name,body", variants("AAAA1"))
def test_probe_variants_round_trip(name, body):
    """Every probe burst must decode back byte-exact, or the measurement it
    produces at a stock VARA would be uninterpretable."""
    fr = frame(body)
    got = rx.decode_burst(tx.synth_frame(fr, over=0), rx.BASE_LEVEL)
    assert got is not None and got.crc_ok
    assert bytes(got.frame_bytes) == fr


@requires_bw2300_capture
def test_linksetup_rides_the_wideband_rec3_burst():
    """§4.2A correction: the first wideband over carries the caller identity.

    If this ever fails, the 'narrowband 16-FSK, caller-invariant training'
    reading was right after all and §4.2A must go back.
    """
    audio = _capture()
    segs = rx.detect_overs(audio)
    s, e = segs[0]
    assert 4.3 < (e - s) / 48000 < 4.5, "leading burst is not the ~4.4 s wideband over"

    frames = rx.decode_overs(audio)
    assert bytes(frames[0].frame_bytes) == _CAPTURED
    assert frames[0].crc_ok


@requires_bw2300_capture
def test_short_over_regenerates_the_same_trailer():
    """A short DATA over pads with the same `14 <crc_hi> … 04 <block>` structure
    and with clean zeros — not stale buffer content."""
    tail = bytes(rx.decode_overs(_capture())[6].frame_bytes)[67:90]
    assert tail == bytes([0x14, 0x7f]) + bytes(19) + bytes([0x04, 0x82])


# --------------------------------------------------------------------------- #
# BW500. Five real VARA HF 4.9.0 -> VARA HF 4.9.0 connects driven on the Wine
# bench 2026-08-15, each one's step-4 burst decoded off the A->B cable with the
# PTT ledger saying who keyed it: 403 columns, c0 = 9, CRC clean, 46 bytes.
# Four base calls of 4, 5 and 6 characters are what pins the layout — the packed
# 6-bit field is many-to-one, so one callsign would fit almost any reading of the
# zero run's length.
_CAPTURED_500 = {
    "W9SSJ":    "5e44d3280000008014f6" + "00" * 32 + "0482" + "1dee",
    "K5ABC":    "2e00420c0000008014ba" + "00" * 32 + "0482" + "2498",
    "KB1A":     "2c270100000000801416" + "00" * 32 + "0482" + "8d19",
    "W9SSJ-10": "5e44d328000a00801437" + "00" * 32 + "0482" + "c016",
    "VE7QRP":   "58589149000000801418" + "00" * 32 + "0482" + "01c4",
}


@pytest.mark.parametrize("caller,hexs", sorted(_CAPTURED_500.items()))
def test_bw500_link_setup_is_byte_exact_against_the_bench(caller, hexs):
    fb = link_setup_frame(caller, bw="500")
    assert fb == bytes.fromhex(hexs)
    assert len(fb) == 46, "the BW500 frame is one level-4 DATA frame"
    assert is_link_setup(fb) and caller_from_link_setup(fb) == caller


def test_bw500_zero_region_is_32_bytes():
    """The two bandwidths differ only in how much zero fill separates the CRC
    byte from the `04 82`; the structure bytes keep their offsets from each end."""
    wide, narrow = link_setup_frame("W9SSJ"), link_setup_frame("W9SSJ", bw="500")
    assert wide[:10] == narrow[:10] and wide[-4:-2] == narrow[-4:-2]
    assert narrow[10:42] == bytes(32) and wide[10:88] == bytes(78)


def test_bw500_link_setup_round_trips_through_the_level4_burst():
    """The burst `link_setup_tx` renders is what a real VARA answered on
    2026-08-15 — CONNECTED W9SSJ / KD9ZZZ / G4ABC-7 <- W1AW at BW500, KD9ZZZ
    being a callsign in no recording — and it comes back byte-exact here."""
    audio = ofdm.link_setup_tx("KD9ZZZ", bw="500")
    # The render stops on its last symbol; a detector reads a stream, and a run
    # still open at the end of one is not a burst.
    (start, stop), = rx500.burst_spans(np.concatenate([audio, np.zeros(512)]))
    assert 4.1 < (stop - start) / rx500.FS < 4.4, "not the one-frame level-4 over"
    # Read at the render onset, not the detected start: kestrel cannot key the
    # nine lead-in columns (see vara_ofdm.ONSET_500), so a detector puts its own
    # burst nine columns late and the c0 lock has no room to take that back.
    fr = rx500.decode_burst(audio, ofdm.ONSET_500)
    assert fr.crc_ok and fr.c0 == 9
    assert caller_from_link_setup(fr.frame_bytes) == "KD9ZZZ"


def test_a_bw500_data_over_is_not_read_as_a_link_setup():
    """The 8-byte counter over from the same bench session, which carries the
    same `04 82` tail — only bytes 7 and 8 separate the two."""
    over = bytes.fromhex("000102030405060714ba" + "00" * 32 + "0482" + "e822")
    assert not is_link_setup(over)
