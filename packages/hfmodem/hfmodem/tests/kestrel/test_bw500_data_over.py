# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW500 DATA over — one body law, two speed levels, off a stock 4.9.0 pair.

A 2026-08-30 bench session between two stock VARA HF 4.9.0 instances, W9SSJ
calling W1AW at BW500, one virtual cable per direction. The responder was handed
200 bytes and delivered them in five overs: four at host ``BITRATE (4)`` carrying
43 payload bytes apiece, then one at ``BITRATE (3)`` carrying the last 28. The
frames below are what this tree's receiver reads off that tape, CRC-clean at
24/24 reference columns and 1.00 self-consistency, and every byte of the 200 is
0x00..0xC7 in order.

They say the body is one law at every bandwidth and level: payload from offset 0,
one per-frame field behind it, and — when the over is short — the trailer
``arq.phy.vara_body`` has always written, 0x14 then the CALLER callsign's CRC-16
high byte, zero fill, and ``04 82`` in the last two bytes. The BW2300 base body
is 90 bytes of that layout; the BW500 level-4 body is 44 and the level-3 body 35.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf500 as rx500
from hfmodem.kestrel.vara import vara_ofdm as OF
from hfmodem.tests.kestrel import corpora

_CALLER = "W9SSJ"

#: The four level-4 frames, as ``(43 payload bytes, per-frame field)``. The field
#: moves per over and is not payload  [see arq.phy.vara_body].
_L4 = ((bytes(range(0x00, 0x2B)), 0x95),
       (bytes(range(0x2B, 0x56)), 0x91),
       (bytes(range(0x56, 0x81)), 0x8D),
       (bytes(range(0x81, 0xAC)), 0x81))

#: The closing level-3 frame's whole 35-byte body: 28 payload bytes and the
#: trailer that ends a delivery.
_L3_BODY = bytes(range(0xAC, 0xC8)) + bytes.fromhex("14 f6 00 00 00 04 82")


def test_body_size_is_the_payload_and_one_field():
    assert phy.body_size("500") == 44
    assert phy.body_size("2300") == 90


def test_a_full_body_is_payload_then_the_field():
    for payload, field in _L4:
        body = phy.vara_body(payload, _CALLER, tail=field, body_len=44)
        assert body == payload + bytes([field])
        assert not phy.over_is_last(body, _CALLER)
        assert phy.vara_payload(body, caller=_CALLER, body_len=44) == payload


def test_the_short_body_is_the_one_the_stock_pair_keyed():
    body = phy.vara_body(_L3_BODY[:28], _CALLER, body_len=35)
    assert body == _L3_BODY
    assert phy.over_is_last(body, _CALLER)


def test_the_delivery_reassembles_to_its_two_hundred_bytes():
    got = b"".join(phy.vara_payload(pl + bytes([f]), caller=_CALLER, body_len=44)
                   for pl, f in _L4)
    got += phy.vara_payload(_L3_BODY, caller=_CALLER, body_len=35)
    assert got == bytes(range(0x00, 0xC8))


def test_a_level_4_over_round_trips_through_the_burst():
    payload, field = _L4[0]
    body = phy.vara_body(payload, _CALLER, tail=field, body_len=44)
    audio = OF.data_over_tx(body, bw="500")
    fr = rx500.decode_burst(audio, OF.ONSET_500)
    assert fr.crc_ok and not fr.is_connect
    assert fr.frame_bytes[:44] == body
    assert fr.payload == payload
    assert fr.marker == field


def test_a_short_level_4_over_carries_the_trailer_over_the_air():
    body = phy.vara_body(b"hello", _CALLER, body_len=44)
    fr = rx500.decode_burst(OF.data_over_tx(body, bw="500"), OF.ONSET_500)
    assert fr.crc_ok
    assert phy.over_is_last(fr.frame_bytes[:44], _CALLER)
    assert phy.vara_payload(fr.frame_bytes[:44], caller=_CALLER,
                            body_len=44) == b"hello"


# --------------------------------------------------------------------------- #
# What the ARQ layer hands the receiver is not a recording: it is one bracket the
# energy segmenter cut, a burst with a little silence either side of it. These say
# `decode_stream` reads one — at either speed level, however much padding the
# bracket carries, and however many frames the burst holds.

_LEAD_TRAIL = ((0, 6000), (2000, 2000), (6000, 6000), (24000, 24000))


def _bracket(x, span, lead, trail):
    a, b = span
    return x[max(0, a - lead):b + trail]


@pytest.fixture(scope="module")
def multiframe():
    return corpora.wav_mono(corpora.MULTIFRAME_SESSION)


@corpora.requires_multiframe_session
@pytest.mark.parametrize("lead,trail", _LEAD_TRAIL)
def test_a_bracket_yields_every_frame_its_burst_holds(multiframe, lead, trail):
    """A 796-column burst carries two frames and a 403-column one carries one.

    The whole session, so the count is the corpus's and not a chosen burst's: the
    link-setup is the one burst that yields no data frame, and no burst the
    detector measured loses a frame to the bracket's edges."""
    seen: dict[int, int] = {}
    for span in rx500.burst_spans(multiframe):
        res = rx500.decode_stream(_bracket(multiframe, span, lead, trail))
        assert not res.frames_skipped
        assert all(f.crc_ok for f in res.data_frames)
        seen[len(res.data_frames)] = seen.get(len(res.data_frames), 0) + 1
    assert seen == {0: 1, 1: 6, 2: 45}


@corpora.requires_multiframe_session
def test_the_link_setup_is_not_a_data_over(multiframe):
    """Gate 3 of the over recogniser, for free: a link-setup decodes CRC-clean and
    is still not an over to answer."""
    span = rx500.burst_spans(multiframe)[0]
    res = rx500.decode_stream(_bracket(multiframe, span, 6000, 6000))
    assert not res.data_frames
    assert [f.is_connect for f in res.frames] == [True]


def test_band_noise_is_not_an_over():
    x = np.random.default_rng(7).normal(0, 0.05, 6 * rx500.FS)
    res = rx500.decode_stream(x)
    assert not res.data_frames and not res.complete


# --------------------------------------------------------------------------- #
# Our own emission, seen the way the receive path sees a peer's: by the span an
# energy detector measured.

def test_our_own_over_decodes_from_a_detected_span():
    """`grid480` reads zero at the nine lead-in columns, so our burst opens with
    them silent and a detector puts its start seven columns late  [vara_ofdm,
    _NCOL_500]. The alignment lock reaches back over the whole preamble and reads
    it anyway, as it reads a peer's burst whose head our own changeover cut
    [test_bw500_cut_head]. So an echo of our own over is a CRC-clean frame at
    this bandwidth, and the echo question is settled by the body it carries,
    which the station keyed  [vara_arq, _keyed_bodies], not by the decode failing."""
    body = phy.vara_body(bytes(range(43)), _CALLER, body_len=44)
    buf = np.concatenate([np.zeros(20000), OF.data_over_tx(body, bw="500"),
                          np.zeros(20000)])
    ours = rx500.decode_stream(buf).data_frames
    assert [f.crc_ok for f in ours] == [True]
    assert ours[0].frame_bytes[:44] == body
    assert all(-8 * rx500.H < o < -6 * rx500.H for o in ours[0].onset), ours[0].onset


@corpora.requires_multiframe_session
def test_a_real_modems_over_decodes_from_its_detected_span(multiframe):
    """The other half of it: a burst that carries its lead-in columns is found
    where it starts, and every frame of the session reads at 1.00."""
    spans = rx500.burst_spans(multiframe)
    for span in spans[1:6]:
        for f in rx500.decode_stream(_bracket(multiframe, span, 6000, 6000)).data_frames:
            assert f.crc_ok and f.self_consistency == 1.0
