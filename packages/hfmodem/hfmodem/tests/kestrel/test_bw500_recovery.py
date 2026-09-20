# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The measured BW500 recovery of a first over the caller could not read.

Two stock VARA HF 4.9.0s on the cables of 2026-09-11, the caller's receive
corrupted over the responder's greeting: the caller keys nothing into that
over's turnaround, the responder keys its idle 1.5 s after its unkey, and
0.24 s after the idle ends the caller keys a 32-symbol session frame — SEED_OFF
60 / PREADV 1117, keyed to the called station — at which the responder re-sends
from speed level 1 and climbs. Twice, to the symbol
[working/vara-evening-en63bc-0910/analysis/stock500-recovery]. The NAK token
keyed into the turnaround instead drew four idles and no resend.

The owed over here is the K5FIT greeting of 2026-09-11 under the same in-band
fault the bench put on the stock greeting: the receiver reads the clean K5FIT
frame now, so the fault is what leaves the over owed. It decodes as one
level-4 frame at self-consistency 0.79 with a failed CRC — the live shape.

The same fault on an over behind one already acknowledged is the chain arm of
the same evening [analysis/stock500-chain]: stock had taken our answer to over
#6 and keyed its seventh as a two-frame level-1 burst our receiver could not
read, and the re-acknowledgement ladder for #6 keyed its next rung behind it —
which stock read as the acknowledgement of #7 (``BUFFER 162, 153``) and
discarded 18 greeting bytes unread. No rung goes out behind an unread over; the
over is owed the recovery instead.
"""
import json
from pathlib import Path
import wave

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import firwin, fftconvolve

from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

FIXTURES = Path(__file__).with_name("fixtures")
STOCK = FIXTURES / "bw500-recovery" / "stock-caller-over-nak-500"
K5FIT = FIXTURES / "k5fit_20260911_cut_head_greeting.wav"
LOW = FIXTURES / "bw500-low-levels"
NAK = VF.for_bw(VF.SESSION_OVER_NAK, "500")
IDLE = VF.for_bw(VF.SESSION_RESPONDER_IDLE, "500")
POLL = 960

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason="the recorded BW500 recovery fixtures require the source checkout "
           "(bw500-recovery/, k5fit_20260911_cut_head_greeting.wav, bw500-low-levels/)")


class _IO(VA.VaraIO):
    def __init__(self):
        self.sent: list[np.ndarray] = []
        self.msgs: list[str] = []
        self.host: list[bytes] = []
        self.fed = 0
        self.keyed_at: list[int] = []

    def key(self, on):
        if on:
            self.keyed_at.append(self.fed)

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def log(self, msg): self.msgs.append(msg)

    def data(self, payload): self.host.append(bytes(payload))


def _station(called="K5FIT"):
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="500")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", called
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    return hs, io


def _k5fit():
    with wave.open(str(K5FIT)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768


def _low(name):
    rate, x = wavfile.read(LOW / f"{name}.wav")
    assert rate == MK.FS
    return x.astype(float) / 32768


def _unreadable(x=None, keep_cols=24, tail_cols=8, seed=910):
    """The greeting with its interior replaced by in-band noise at its own RMS:
    envelope, lead-in and level survive, the DBPSK cells do not
    [stock500-nak/fault_design.py]."""
    x = _k5fit() if x is None else x.copy()
    (start, stop), = RX.burst_spans(x)
    a, b = start + keep_cols * RX.H, stop - tail_cols * RX.H
    rms = np.sqrt(np.mean(x[a:b] ** 2))
    taps = firwin(401, [f / (RX.FS / 2) for f in RX.BURST_BAND], pass_zero=False)
    n = fftconvolve(np.random.default_rng(seed).standard_normal(b - a + 400), taps, "same")
    n = n[200:200 + (b - a)]
    x[a:b] = n / np.sqrt(np.mean(n ** 2)) * rms
    return x


def _idle(called="K5FIT"):
    return MK.synth_burst(called, IDLE)


def _stream(hs, io, x, poll=POLL):
    for i in range(0, len(x), poll):
        io.fed += len(x[i:i + poll])
        hs.on_rx_stream(x[i:i + poll])


def _is(samples, kind, callsign):
    return VF.recognize(MK.demod_burst(samples, kind, MK.band_for("500")), callsign, kind)


def _owed_greeting():
    hs, io = _station()
    assert hs._answer_data_over(_unreadable())
    return hs, io


def test_session_over_nak_regenerates_the_stock_callers_frame():
    meta = json.loads(STOCK.with_suffix(".json").read_text())
    tones = VF.handshake_tones(meta["link"]["called"], NAK)
    assert tones == meta["waveform"]["tones"]
    assert tones != VF.handshake_tones(meta["link"]["caller"], NAK)
    rate, audio = wavfile.read(STOCK.with_suffix(".wav"))
    assert rate == MK.FS
    heard = MK.demod_burst(audio.astype(float)[meta["provenance"]["slice_pad_samples"]:],
                           NAK, MK.band_for("500"))
    assert heard == tones
    assert VF.BURSTS[NAK.name] is VF.SESSION_OVER_NAK
    assert (NAK.seed_off, NAK.preadv, NAK.keyed_by) == (60, 1117, "called")


def test_the_fault_leaves_the_greeting_in_the_live_shape():
    res = RX.decode_stream(_unreadable())
    assert not res.complete
    assert [(f.level, f.crc_ok) for f in res.data_frames] == [(4, False)]
    assert RX.decode_stream(_k5fit()).complete


def test_an_owed_over_that_will_not_decode_keys_nothing_and_stays_owed():
    hs, io = _owed_greeting()
    assert io.sent == [] and io.host == []
    assert hs._owed_recovery and hs._owed_block
    assert hs._answer_owed == VA._OWED_OVER
    assert hs._undecoded == 0 and hs._reacks == 0
    assert hs.state is VA.VaraState.CONNECTED
    assert any("keying nothing into its turnaround" in m for m in io.msgs)


def test_the_responder_idle_draws_session_over_nak_at_the_stock_lead():
    hs, io = _owed_greeting()
    idle = _idle()
    quiet = np.zeros(int(1.5 * MK.FS))
    _stream(hs, io, np.concatenate([quiet, idle, np.zeros(MK.FS)]))
    assert len(io.sent) == 1
    assert _is(io.sent[0], NAK, "K5FIT")
    # The idle's last sample is known to the recogniser's alignment plateau,
    # ~1400 samples, and the keying to the poll: 0.234 s here against the stock
    # caller's 0.242, and the stock's two arms agree to 0.05 s.
    lead = (io.keyed_at[0] - (len(quiet) + len(idle))) / MK.FS
    assert -0.03 <= lead - VA._OVER_NAK_LEAD_S <= 0.03 + POLL / MK.FS, lead
    assert hs._owed_recovery and hs._owed_block
    assert hs._answer_owed == VA._OWED_OVER
    assert hs._reacks == 1
    assert io.host == []


def test_the_bracket_route_answers_the_idle_with_the_same_frame():
    hs, io = _owed_greeting()
    hs.on_rx_audio(np.concatenate([_idle(), np.zeros(int(0.2 * MK.FS))]))
    assert len(io.sent) == 1 and _is(io.sent[0], NAK, "K5FIT")
    assert hs._owed_recovery and hs._reacks == 1


def test_the_idle_of_another_link_draws_nothing():
    hs, io = _owed_greeting()
    _stream(hs, io, np.concatenate([np.zeros(MK.FS), _idle("KC9GHZ"), np.zeros(MK.FS)]))
    assert io.sent == [] and hs._reacks == 0


def test_the_ladder_bounds_the_frames_and_then_the_link_closes():
    hs, io = _owed_greeting()
    cue = np.concatenate([np.zeros(int(1.5 * MK.FS)), _idle(), np.zeros(MK.FS)])
    for n in range(1, VA._REACK_MAX + 1):
        _stream(hs, io, cue)
        assert len(io.sent) == n and _is(io.sent[-1], NAK, "K5FIT")
    _stream(hs, io, cue)
    assert hs.state is VA.VaraState.DISCONNECTED
    assert len(io.sent) == VA._REACK_MAX + 1
    assert _is(io.sent[-1], VF.for_bw(VF.SESSION_DISCONNECT_REQ, "500"), "K5FIT")
    assert sum("tx NAK" in m for m in io.msgs) == VA._REACK_MAX


def test_unreadable_bursts_alone_key_nothing_and_close_nothing():
    hs, io = _owed_greeting()
    for _ in range(VA._OVER_NAK_MAX + 2):
        assert hs._answer_data_over(_unreadable())
    assert io.sent == [] and hs._undecoded == 0
    assert hs.state is VA.VaraState.CONNECTED and hs._owed_recovery


def test_resends_that_will_not_decode_keep_the_ladders_count():
    hs, io = _owed_greeting()
    cue = np.concatenate([_idle(), np.zeros(int(0.2 * MK.FS))])
    for n in range(1, VA._REACK_MAX + 1):
        hs.on_rx_audio(cue)
        assert hs._reacks == n and len(io.sent) == n
        assert hs._answer_data_over(_unreadable())
        assert hs._reacks == n and hs._undecoded == 0
    hs.on_rx_audio(cue)
    assert hs.state is VA.VaraState.DISCONNECTED
    assert [_is(s, NAK, "K5FIT") for s in io.sent] == [True] * VA._REACK_MAX + [False]
    assert _is(io.sent[-1], VF.for_bw(VF.SESSION_DISCONNECT_REQ, "500"), "K5FIT")


def test_a_resend_that_decodes_clears_the_debt():
    hs, io = _owed_greeting()
    hs.on_rx_audio(np.concatenate([_idle(), np.zeros(int(0.2 * MK.FS))]))
    assert hs._answer_data_over(_k5fit())
    assert len(io.host) == 1
    assert not hs._owed_recovery and not hs._owed_block
    assert hs._undecoded == 0 and hs._peer_over == 1
    assert len(io.sent) == 2 and not _is(io.sent[-1], NAK, "K5FIT")


def test_only_the_first_over_with_nothing_of_ours_on_the_air_is_owed():
    x = _unreadable()
    for name, arrange in (
            ("our own DATA keyed", lambda hs: hs._keyed_bodies.add(b"x")),
            ("an over already read", lambda hs: setattr(hs, "_peer_over", 1)),
            ("the delivery open", lambda hs: setattr(hs, "_peer_delivery_open", True)),
            ("the turn ours", lambda hs: setattr(hs, "turn", VA._TURN_OURS))):
        hs, io = _station()
        arrange(hs)
        assert not hs._answer_data_over(x), name
        assert io.sent == [] and not hs._owed_recovery and not hs._owed_block, name
        assert hs._undecoded == 0, name


def test_no_rung_goes_out_behind_an_over_that_will_not_decode():
    hs, io = _station("KC9GHZ")
    assert hs._answer_data_over(_low("bw500-level1-arm1"))
    assert io.host == [b"RMS Trimo"] and len(io.sent) == 1
    assert hs._answer_owed == VA._OWED_OVER and not hs._owed_block
    assert hs._peer_over == 1 and hs._peer_delivery_open
    acked = io.sent[0]

    unread = _unreadable(_low("bw500-level1-two-frames"))
    assert not RX.decode_stream(unread).complete
    assert hs._answer_data_over(unread)
    assert len(io.sent) == 1 and io.host == [b"RMS Trimo"]
    assert hs._owed_recovery and hs._owed_block and hs._reacks == 0
    assert hs._answer_owed == VA._OWED_OVER
    assert any("the acknowledgement ladder stops here" in m for m in io.msgs)

    cue = np.concatenate([_idle("KC9GHZ"), np.zeros(int(0.2 * MK.FS))])
    for n in range(1, VA._REACK_MAX + 1):
        hs.on_rx_audio(cue)
        assert len(io.sent) == n + 1 and hs._reacks == n
    for burst in io.sent[1:]:
        assert _is(burst, NAK, "KC9GHZ")
        assert not any(_is(burst, VF.for_bw(k, "500"), "KC9GHZ") for k in
                       (VF.SESSION_OVER_RESPONSE, VF.SESSION_OVER_RESPONSE_SHORT))
        assert len(burst) != len(acked) or not (burst == acked).all()
    assert not any("re-acknowledging over" in m for m in io.msgs)
    assert hs.state is VA.VaraState.CONNECTED


def test_a_rung_still_goes_out_when_nothing_unread_follows():
    hs, io = _station("KC9GHZ")
    assert hs._answer_data_over(_low("bw500-level1-arm1"))
    hs.on_rx_audio(np.concatenate([_idle("KC9GHZ"), np.zeros(int(0.2 * MK.FS))]))
    assert len(io.sent) == 2 and hs._reacks == 1 and not hs._owed_block
    assert any("re-acknowledging over #1" in m for m in io.msgs)
    assert not _is(io.sent[-1], NAK, "KC9GHZ")
