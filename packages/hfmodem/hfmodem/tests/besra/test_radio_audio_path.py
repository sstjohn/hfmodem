# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The real-radio sound-card DSP path, exercised without a sound card.

`RadioLink._transmit` and `RollingDecoder` are the only besra code the in-memory
integration tests never touch: the 48 kHz edge resample on transmit, the rolling
decode and 48->12 kHz resample on capture, and the half-duplex mute that stops a
station hearing its own signal. Hardware and PTT aside, this is the `--radio`
path, and a BlackHole loopback proved too flaky to exercise it. So `sounddevice`
is stubbed to hand what `_transmit` "plays" straight into a capture path — the
exact wiring `capture_stream` builds — and the frame must survive the round trip,
one station's transmission must reach the other's ARQ session, and a real
high-floor channel must not stall either.
"""

from __future__ import annotations

import json
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

import hfmodem.besra.radio as R
from hfmodem.besra import crc
from hfmodem.besra.frame import frame as F
from hfmodem.tests import evidence
from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host import protocol as P
from hfmodem.besra.host.modem_core import ModemObserver
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.phy.demodulator import (SAMPLE_RATE, DecodedFrame, Demodulator,
                                           decode, frame_span)
from hfmodem.core import audio as audio_mod
from hfmodem.core.resample import from_card, to_card
from hfmodem.core.rigs import RIGS

_BLOCK = int(0.1 * R.FS_RADIO)          # the block `capture_stream`'s callback hands over

#: What the receive path may take between a frame's last sample and the modem
#: having it. `RollingDecoder.STEP_S` plus the block the step lands in — and the
#: protocol allows ~0.6 s: a ConAck ends ~1.4 s after our ConReq does (0.7 s
#: turnaround, 0.705 s frame) and the confirming ConAck has to be keyed before the
#: ConReq repeats at 2.0 s (`ArqSession._connect_interval`).
_LATENCY_S = R.RollingDecoder.STEP_S + 0.1


class _Sink:
    """Stands in for a modem: `RadioLink` hands it each decoded frame, and its
    receive funnel at teardown."""

    def __init__(self):
        self.audio_out = None
        self.frames: list = []
        self.status_lines: list[str] = []

    def receive_frames(self, frames):
        self.frames.extend(frames)

    def expected_session(self):
        return None

    def rx_epoch(self):
        return 0

    def status(self, text):
        self.status_lines.append(text)


def _stub_sounddevice(monkeypatch):
    played: list = []

    class FakeOut:
        """Records the audio AND whether the stream was closed. A stream left open
        holds a data-audio-keyed rig in transmit after CAT has unkeyed, which is
        invisible to a stub that only records samples — so the close is asserted."""

        def __init__(self, *, device=None, **kw):
            self.device, self.closed = device, False

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            self.closed = True

        def write(self, audio):
            played.append((self.device, np.asarray(audio).reshape(-1), self))

    fake = types.ModuleType("sounddevice")
    fake.OutputStream = FakeOut
    fake.play = lambda audio, fs, device=None: played.append(
        (device, np.asarray(audio), None))
    fake.wait = lambda: None
    fake.stop = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(audio_mod, "_WARMED", set())   # a fresh card is a cold one
    return played


def _capture(link, audio48) -> list[float]:
    """Push 48 kHz float32 audio into a link exactly as `capture_stream` does, pumping
    the decoder between blocks as its own thread would, and return the stream time at
    which each frame reached the modem."""
    at = []
    for i in range(0, len(audio48), _BLOCK):
        link._capture(np.ascontiguousarray(audio48[i:i + _BLOCK], dtype=np.float32))
        before = len(link.modem.frames)
        link._rx.pump()
        at += [(i + _BLOCK) / R.FS_RADIO] * (len(link.modem.frames) - before)
    return at


def _block_rms(audio48: np.ndarray) -> np.ndarray:
    """Per-0.1 s-block RMS in int16 units — the measurement the old gate made on
    every block, and the one the station's live floor was quoted in."""
    n = len(audio48) // _BLOCK * _BLOCK
    return np.sqrt((audio48[:n].astype(np.float64).reshape(-1, _BLOCK) ** 2).mean(axis=1)) * 32768


def test_transmit_audio_survives_the_capture_path(monkeypatch):
    played = _stub_sounddevice(monkeypatch)
    payload = b"BESRA RADIOLINK TX PATH 0123456789"
    frame = M.render_frame(0x4A, payload=payload, session_id=0x5E)

    link = R.RadioLink(_Sink(), rig=None, out_device="stub")
    link._transmit(frame)                       # real resample + stubbed play

    audio48 = played[-1][1]                     # the burst; `warm_output` wrote first
    assert audio48.dtype == np.float32
    assert abs(len(audio48) / len(frame) - 4) < 0.01   # 12 kHz -> 48 kHz

    _capture(link, np.concatenate([np.zeros(_BLOCK, dtype=np.float32), audio48]))

    hit = [f for f in link.modem.frames
           if f.ok and f.type == 0x4A and bytes(f.payload) == payload]
    assert hit, (f"TX audio did not decode back; recovered "
                 f"{[(hex(f.type), f.ok) for f in link.modem.frames]}")


def test_a_frame_reaches_the_modem_within_the_arq_budget():
    """ARQ is a turnaround protocol: a ConAck the modem gets 20 s late is a connect
    that never happens, and 20 s is exactly what the gate this replaced took to
    flush a buffer a noisy channel held open. A frame is owed within a step."""
    link = R.RadioLink(_Sink(), rig=None)
    frame = M.render_frame(0x3A, payload=bytes([24, 24, 24]), session_id=0x5E)
    quiet = np.zeros(int(2.0 * SAMPLE_RATE), dtype="<i2")
    ends_at = (len(quiet) + len(frame)) / SAMPLE_RATE

    at = _capture(link, to_card(np.concatenate([quiet, frame, quiet]), SAMPLE_RATE))

    assert len(at) == 1, f"{len(at)} frames delivered for one transmission"
    assert at[0] - ends_at <= _LATENCY_S, f"delivered {at[0] - ends_at:.2f} s after the frame"


#: WM4RB's repeat interval, measured off `logs/onair/20260814T051359Z-besra-7100000`
#: at four anchors (±0.02 s): the gateway keys its retransmission 2.01 s after the
#: frame it is repeating ends. That is a metronome the gateway runs off its own
#: frame end, not an answer to anything, so a late NAK does not cause the repeat
#: and an on-time one would not stop it — only an ACK stops the repeat timer
#: (`ARQ.c:2143`), and a NAK moves the mode (`ARQ.c:2210-2212`). What the interval
#: bounds is the window a DATANAK has to be keyed, played and finished inside if
#: it is to reach the gateway before the next frame. Ours went out at frame-end
#: +2.25 to +2.70 s, four of four, so none of them was inside it.
_PEER_REPEAT_S = 2.01


def _rolling(audio: np.ndarray, session: int | None = None, *,
             rate: int = R.FS_RADIO) -> list:
    """Push a capture through `RollingDecoder` the way the pump thread does, and
    return `(stream time, position, frame)` for everything it hands up."""
    got: list = []
    now = [0.0]
    rx = R.RollingDecoder(Demodulator(expect_session=lambda: session),
                          lambda pos, f: got.append((now[0], pos, f)),
                          resample=rate != SAMPLE_RATE)
    block = int(0.1 * rate)
    for i in range(0, len(audio), block):
        rx.push(np.ascontiguousarray(audio[i:i + block], dtype=np.float32))
        now[0] = (i + block) / rate
        rx.pump()
    return got


def _faded_data_frame(quiet_s: float = 1.0) -> np.ndarray:
    """A 4PSK.500.100 the channel took a bite out of: leader and header clean, a
    quarter of the body replaced by noise at the frame's own level. The header
    decodes — so the frame's extent is known the moment it does — and the body
    fails RS and CRC, which is the frame the ARQ layer owes a NAK."""
    rng = np.random.default_rng(11)
    frame = M.render_frame(0x50, payload=bytes(rng.integers(0, 256, 64)),
                           session_id=0x0E).astype(np.float64)
    body = 2880 + 2400                          # past the leader and the header
    n = (len(frame) - body) // 4
    at = body + (len(frame) - body - n) // 2
    frame[at:at + n] = rng.normal(0.0, np.sqrt(np.mean(frame ** 2)), n)
    quiet = np.zeros(int(quiet_s * SAMPLE_RATE))
    au = np.concatenate([quiet, frame, np.zeros(int(6.0 * SAMPLE_RATE))])
    return to_card(np.clip(np.round(au), -32768, 32767).astype("<i2"), SAMPLE_RATE)


def test_a_failed_frame_reports_at_its_own_length():
    """A CRC-failed frame is the one the peer is waiting to be told about, and the
    hold-back that keeps a truncated sighting out must not outlast the frame it is
    protecting. `frame_span` reads the extent off the header, so the wait ends when
    that frame's own last sample has arrived.

    A flat `OVERLAP_S` is the frame's length over again for nothing, and it is what
    lost the 2026-08-14 WM4RB session: a 4.41 s frame held 6.0 s from its *start*
    reports at end + 1.83 s of decoder time, and the DATANAKs it drove keyed at end
    + 2.25 to 2.70 s — every one after the gateway had begun retransmitting."""
    got = _rolling(_faded_data_frame(), session=0x0E)

    assert len(got) == 1, [(round(t, 2), hex(f.type), f.ok) for t, _, f in got]
    at, pos, f = got[0]
    assert not f.ok and f.type == 0x50 and f.quality, f
    end = (pos + frame_span(f.type)) / SAMPLE_RATE
    assert at - end <= R.RollingDecoder.FRAME_MARGIN_S + _LATENCY_S, (
        f"failed frame reported {at - end:.2f} s after its last sample")
    assert at - end < _PEER_REPEAT_S, "reported after the peer had stopped listening"
    assert at < pos / SAMPLE_RATE + R.RollingDecoder.OVERLAP_S, (
        "the wait is still the flat carry-over, not the frame's own length")


def test_a_frame_acquired_out_of_stream_order_is_still_reported():
    """Acquisition does not find frames in the order they were sent: a weak leader
    surfaces windows after a frame that follows it, so the dedup has to remember
    positions rather than carry a high-water mark. A mark set by the later frame
    swallows the earlier one for good — measured on the 2026-08-05 KE8LVA session,
    where it withheld 13 frames, five of them ConReq2000M."""
    early, late = 1 * SAMPLE_RATE, 3 * SAMPLE_RATE

    class _OutOfOrder:
        """Sights the later frame first, then both — one window on from a leader
        too weak to acquire at its own position."""

        def __init__(self) -> None:
            self.passes = 0

        def decode(self, window, at=0):
            self.passes += 1
            seen = [late] if self.passes == 1 else [early, late]
            return [DecodedFrame(type=0x24, session_id=0x51, ok=True, name="IDLE",
                                 offset=o) for o in seen]

    got: list[int] = []
    rx = R.RollingDecoder(_OutOfOrder(), lambda pos, f: got.append(pos), resample=False)
    rx._buf = np.zeros(int((R.RollingDecoder.OVERLAP_S + R.RollingDecoder.STEP_S)
                           * SAMPLE_RATE), dtype=np.float32)
    for _ in range(2):
        rx._n = rx._buf.size
        rx._decode()

    assert got == [late, early], got


#: A window edge inside the ConReq below: the frame runs 1.00–2.72 s, so this one
#: leaves 0.92 s of it still to come — the same shape as the corpus incident.
_CUT_S = 1.8


def _cut_conreq(quiet_s: float = 1.0) -> np.ndarray:
    """A ConReq on a noisy channel, placed so window edges fall inside it — the
    2.72 s frame is longer than the 0.3 s the decoder advances by, so several
    windows hold nothing but its front."""
    rng = np.random.default_rng(3)
    frame = M.render_frame(0x34, caller="KC3OWM", target="K4PAR-2", session_id=0x5E)
    au = np.zeros(int(quiet_s * SAMPLE_RATE) + len(frame) + int(4.0 * SAMPLE_RATE))
    au[int(quiet_s * SAMPLE_RATE):][:len(frame)] = frame
    au += rng.normal(0.0, 300.0, au.size)
    return to_card(np.clip(np.round(au), -32768, 32767).astype("<i2"), SAMPLE_RATE)


def test_a_frame_the_window_edge_cuts_is_still_held_for_the_whole_one():
    """The other half, and the reason the hold-back exists at all: a frame the newest
    edge cuts is sighted at the same position as the complete one, so the truncated
    sighting wins `DEDUP_S` and suppresses it. Measured on
    `rf-corpus/7102k_065457.wav`, where 3 of the 4 ConReqs are first seen with ~1.1 s
    of themselves still to come; reproduced here, where every window edge from 1.5 s
    to 2.4 s reads this ConReq as ok=False with no callsign at all.

    The per-frame wait has to keep covering that: a ConReq spans 1.48 s from its
    header, so it is held past every one of those edges and released off the first
    window that holds it whole."""
    audio = _cut_conreq()
    cut = from_card(audio[:int(_CUT_S * R.FS_RADIO)], SAMPLE_RATE)
    early = Demodulator().decode(np.concatenate([cut, np.zeros(4800, dtype="<i2")]))
    assert [(f.type, f.ok, f.caller) for f in early] == [(0x34, False, None)], (
        "the fixture no longer offers a truncated sighting — the test is blind")

    got = _rolling(audio)

    assert len(got) == 1, [(round(t, 2), hex(f.type), f.ok) for t, _, f in got]
    at, _, f = got[0]
    assert f.ok and f.caller == "KC3OWM" and f.target == "K4PAR-2", f
    assert at > _CUT_S, "the truncated sighting won the dedup"


#: Eight seconds of W6IDS's 4FSK.200.50S stint, cut at 79.0 s from
#: `logs/onair/20260829T000632Z-besra-7060000.wav` (2026-08-29, 7061.5 kHz, BW500)
#: and kept at the 12 kHz radio rate. Two of the gateway's frames, both read as far
#: as their headers and no further.
_W6IDS_50BAUD = Path(__file__).parent / "fixtures" / "offair_w6ids_50baud.wav"


def test_a_guessed_type_is_still_released_at_its_own_length():
    """The hold-back reads `frame_span` off the type, and for a header-only frame
    that type is a guess — so the tempting rule is to hold every guess for the
    longest frame anyone can send (4FSK.2000.600, 5.26 s), conservative in the
    direction that keeps a still-arriving frame out of the slot.

    The gateway's cadence refuses it. `ArqSession._unreadable` keys the DATANAK
    within a millisecond of the sighting, and here W6IDS's next header lands 4.14 s
    after the first — so a sighting held 5.26 s is answered from inside the very
    transmission it would be answering over. Across the two 2026-08-28 arms that is
    fourteen of the nineteen header-only sightings, on the one rung that was
    delivering. Run over this fixture the flat rule reports the first frame 4.19 s
    past its own last sample and does not report the second at all: the other
    failure, a real frame never sighted, in the same eight seconds.

    What makes the frame's own span the honest one is that the sighting is a frame
    start: `Demodulator._scan_header` reports nothing from its fallback loop
    without leader behind the header, so a header-only frame is a real
    transmission's own header read on the grid its leader set, and the header
    parity separates every pair of valid types at one symbol. This recording also
    holds the other kind — a `16QAM.500.100.E` minted 0.51 s inside the first frame,
    on a rung this station cannot receive — and that one is the sighting whose span
    would mean nothing. It must not be here."""
    with wave.open(str(_W6IDS_50BAUD)) as w:
        assert w.getframerate() == SAMPLE_RATE
        au = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)

    got = _rolling(au, session=0x0D, rate=SAMPLE_RATE)

    assert [(f.name, f.ok, f.header_only) for _, _, f in got] == \
        [("4FSK.200.50S.E", False, True)] * 2, \
        [(round(t, 2), f.name, f.ok, f.header_only) for t, _, f in got]
    for at, pos, f in got:
        end = (pos + frame_span(f.type)) / SAMPLE_RATE
        assert at - end <= R.RollingDecoder.FRAME_MARGIN_S + _LATENCY_S, (
            f"guess at {pos / SAMPLE_RATE:.2f} s reported {at - end:.2f} s "
            f"after its own last sample")

    longest = max(frame_span(t) for t in F.FRAMES)
    first, second = (pos for _, pos, _ in got)
    assert first + longest > second, (
        "the longest-span rule no longer keys the answer to the first frame "
        "inside the second — re-measure before adopting it")


def test_a_decode_displaces_the_guess_that_took_its_position():
    """The hold-back above is only as good as the span it is handed, and for a
    `header_only` frame that span is a guess: the type comes off ten tones that
    satisfied the frame-type parity and nothing else, so guessed as a shorter class
    the sighting is released while the real frame is still arriving. Its position is
    then taken, and the complete decode a window later is dropped as a duplicate of
    a frame that never decoded — the payload NAKed rather than delivered.

    So the slot holds the verdict as well as the position. A decode displaces a
    failure at the same position; nothing displaces a decode."""
    pos = 1 * SAMPLE_RATE
    guess = DecodedFrame(type=0x24, session_id=0x51, ok=False, name="IDLE",
                         offset=pos, header_only=True)
    whole = DecodedFrame(type=0x50, session_id=0x51, ok=True, name="4PSK.500.100.E",
                         payload=b"RMS Trimode 1.4.2.0", offset=pos)

    class _Sequence:
        """One sighting per window, in the order a mis-guessed span produces them:
        the guess, the frame it was a guess at, and both again."""

        def __init__(self, frames) -> None:
            self.frames = list(frames)

        def decode(self, window, at=0):
            return [self.frames.pop(0)] if self.frames else []

    got: list = []
    demod = _Sequence([guess, whole, whole, guess])
    rx = R.RollingDecoder(demod, lambda pos, f: got.append((pos, f)), resample=False)
    rx._buf = np.zeros(int((R.RollingDecoder.OVERLAP_S + R.RollingDecoder.STEP_S)
                           * SAMPLE_RATE), dtype=np.float32)
    while demod.frames:
        rx._n = rx._buf.size
        rx._decode()

    assert [(f.ok, bytes(f.payload)) for _, f in got] == \
        [(False, b""), (True, b"RMS Trimode 1.4.2.0")], got
    assert {p for p, _ in got} == {pos}, "the two sightings are not the one frame"


class _FakeStream:
    """`sd.InputStream` as `capture_stream` builds it, minus PortAudio."""

    def __init__(self, **kw) -> None:
        self.callback = kw["callback"]
        assert kw["samplerate"] == R.FS_RADIO and kw["channels"] == 1
        assert kw["blocksize"] == _BLOCK and kw["dtype"] == "float32"

    def start(self) -> None: pass
    def stop(self) -> None: pass
    def close(self) -> None: pass


def test_the_live_capture_stream_delivers_through_the_pump_thread(monkeypatch):
    """`start` is the wiring the other tests bypass: the real `capture_stream`
    callback shape — PortAudio hands a 2-D float32 block — and the pump thread that
    decodes it off the audio thread. Nothing here calls `pump`; the frame has to
    arrive anyway."""
    fake = types.ModuleType("sounddevice")
    made: dict = {}
    fake.InputStream = lambda **kw: made.setdefault("stream", _FakeStream(**kw))
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    link = R.RadioLink(_Sink(), rig=None)
    frame = M.render_frame(0x3A, payload=bytes([24, 24, 24]), session_id=0x5E)
    quiet = np.zeros(int(0.5 * SAMPLE_RATE), dtype="<i2")
    audio = to_card(np.concatenate([quiet, frame, quiet]), SAMPLE_RATE)

    link.start()
    try:
        cb = made["stream"].callback
        for i in range(0, len(audio), _BLOCK):
            cb(audio[i:i + _BLOCK].reshape(-1, 1), _BLOCK, None, None)
        deadline = time.monotonic() + 5.0
        while not link.modem.frames and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        link.close()

    assert [f.type for f in link.modem.frames] == [0x3A]


def _hissy_channel() -> np.ndarray:
    """A real receiver's output as the sound card sees it: a noise floor an order of
    magnitude over the 200 int16 RMS the old gate opened at, with a ConReq 18 dB
    above it. 18 dB is what this demodulator wants against stationary white noise
    with a multi-second lead-in — well over the ~10 dB it manages when the noise
    starts with the frame, and the reason the floor here is not the 3935 the live
    station measures: a frame 18 dB over 3935 int16 RMS peaks at 1.9× full
    scale, and clipping it that hard is what stops it decoding, not the receive
    path. The measured 3935 case runs on real off-air audio below."""
    floor = 2000.0
    rng = np.random.default_rng(3)
    frame = M.render_frame(0x32, caller="KC3OWM", target="W9SSJ", session_id=0x5E)
    x = rng.normal(0.0, floor, int(50 * SAMPLE_RATE))
    at = int(3.0 * SAMPLE_RATE)
    x[at:at + len(frame)] += frame.astype(np.float64) * (
        floor * 10 ** 0.9 / np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return to_card(np.clip(np.round(x), -32768, 32767).astype("<i2"), SAMPLE_RATE)


def test_a_channel_that_never_goes_quiet_still_delivers():
    """The defect this receive path exists for: the gate opened above an absolute
    200 int16 RMS and closed on quiet, so on a receiver whose own floor sits above
    200 it opened once and never closed. Measured on `rf-corpus/7102k_065457.wav`,
    44 s of real off-air audio: not one 0.1 s block under 200, zero bursts, nothing
    reached the modem at all. Nothing here is gated, so the floor is irrelevant."""
    audio = _hissy_channel()
    link = R.RadioLink(_Sink(), rig=None)

    at = _capture(link, audio)

    want = [f for f in decode(from_card(audio, SAMPLE_RATE)) if f.ok and f.caller == "KC3OWM"]
    assert want, "fixture no longer decodes at all — check the frame level, not the path"
    got = [f for f in link.modem.frames if f.ok and f.caller == "KC3OWM"]
    assert len(got) >= len(want), f"{len(got)} of {len(want)} ConReqs reached the modem"
    assert at and at[0] < 6.0, f"first frame delivered at {at[0]:.1f} s, frame is at 3 s"


_CORPUS = evidence.CORPUS
#: The floor the station measures on the real FT-891 + C-Media dongle at the
#: verified 0.040 codec working point: 3935 int16 RMS, the 10th percentile of 0.1 s
#: frames, with capture peak -7.0 dBFS and nothing railed. Twenty times the 200 the
#: old gate opened at.
_LIVE_FLOOR = 3935.0


def test_real_offair_audio_at_the_live_noise_floor():
    """The exact condition that broke the ARQ receive path, on real audio: an off-air
    capture scaled to the 3935 int16 RMS floor the station measures at its verified
    codec gain. Every 0.1 s block sits far over the old gate — it would have opened
    at the first block and never closed — and the ConReqs must still arrive."""
    wav = _CORPUS / "7102k_065457.wav"
    if not wav.exists():
        pytest.skip("7102k_065457.wav absent")
    with wave.open(str(wav)) as w:
        assert w.getframerate() == R.FS_RADIO
        raw = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    audio = np.clip(raw * (_LIVE_FLOOR / np.percentile(_block_rms(raw), 10)),
                    -1.0, 1.0).astype(np.float32)
    blocks = _block_rms(audio)
    assert abs(np.percentile(blocks, 10) - _LIVE_FLOOR) < 50
    assert blocks.min() > 200, "not the held-open case — some block falls under the old gate"

    link = R.RadioLink(_Sink(), rig=None)
    _capture(link, audio)

    got = [f for f in link.modem.frames if f.ok and f.caller == "KC3OWM"]
    assert len(got) >= 2, f"{len(got)} ConReqs reached the modem off a {_LIVE_FLOOR:.0f} floor"
    assert all(f.target == "K4PAR-2" for f in got)


def _recorded(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == R.FS_RADIO and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768.0


def test_the_recording_spans_our_own_transmission(monkeypatch, tmp_path):
    """The recording is the receive path's audit trail, and the half-duplex mute must
    not reach it. The decoder drops what arrives while we are keyed; the file keeps
    it, because a WAV with the keyed intervals cut out has no timebase and a
    detection logged at a wall clock cannot be found in it afterwards.

    This is also the specific confusion the file has to settle. A run of undecoded
    detections at a fixed offset after each transmission reads as a station
    answering on a cadence until our own transmissions are visible in the same
    recording at their true spacing."""
    _stub_sounddevice(monkeypatch)
    wav = tmp_path / "spans.wav"
    link = R.RadioLink(_Sink(), rig=None, out_device="stub", record=wav)
    frame = M.render_frame(0x4A, payload=b"x" * 8, session_id=1)
    quiet = np.full(_BLOCK * 3, 0.01, dtype=np.float32)     # a floor, not silence

    audio_mod.warm_output("stub")               # unkeyed, and before the spy below
    out_cls = sys.modules["sounddevice"].OutputStream
    real_write = out_cls.write

    def write_spy(self, audio):
        # The station hearing itself: the card keeps delivering while we are keyed.
        _capture(link, np.asarray(audio).reshape(-1))
        return real_write(self, audio)
    monkeypatch.setattr(out_cls, "write", write_spy)

    _capture(link, quiet)
    link._transmit(frame)
    _capture(link, quiet)
    link.recorder.close()

    got = _recorded(wav)
    played = len(to_card(frame, SAMPLE_RATE))
    assert link.modem.frames == [], "the modem decoded our own transmission"
    # Everything the card delivered, in order: floor, our own transmission, floor.
    assert len(got) == 2 * len(quiet) + played, (
        f"recorded {len(got)} samples of {2 * len(quiet) + played} delivered — "
        "the recording is gapped where we were keyed")
    mid = got[len(quiet):len(quiet) + played]
    assert np.abs(mid).max() > 0.05, "our own transmission is missing from the recording"


def test_the_recording_survives_a_failing_teardown(monkeypatch, tmp_path):
    """A session that ends by failing to unkey is exactly the one worth listening
    back to, so the WAV and its sidecar are finalised even when teardown raises."""
    class _AngryRig:
        def stop(self): raise RuntimeError("rigctl pipe wedged")

    wav = tmp_path / "teardown.wav"
    link = R.RadioLink(_Sink(), rig=_AngryRig(), record=wav)
    _capture(link, np.full(_BLOCK, 0.25, dtype=np.float32))

    with pytest.raises(RuntimeError):
        link.close()

    got = _recorded(wav)
    assert len(got) == _BLOCK
    side = json.loads(wav.with_suffix(".json").read_text())
    # Unnormalised, so the level a capture had is still readable off the file.
    assert side["normalised_on_write"] is False
    assert abs(side["peak"] - 0.25) < 1e-3 and side["samples"] == _BLOCK


def test_a_link_without_a_recorder_still_runs(tmp_path, capsys):
    """`record=None` is a supported mode (`--no-record`), not a crash — and it
    is the one shape where there is no artefact to name, so the report says
    that rather than nothing."""
    link = R.RadioLink(_Sink(), rig=None)
    assert link.recorder is None
    _capture(link, np.zeros(_BLOCK, dtype=np.float32))
    link.close()
    assert "session stream: not taken" in capsys.readouterr().out


# -- naming what was recorded ------------------------------------------------


def _stream_line(capsys) -> str:
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("session stream: ")]
    assert len(lines) == 1, lines
    return lines[0]


def test_the_session_names_its_recording_on_the_way_out(tmp_path, capsys):
    """Two ARDOP arms of the 2026-08-20 slot left a 37 s recording apiece and
    not one line saying so, while every PACTOR arm on the same slot named its
    stream. When the question afterwards was what this station actually
    transmitted, the recorded arms were the ones that could not answer it.

    The sample-clock claim is the same one shrike makes and is true here for the
    same reason: `_capture` records ahead of the half-duplex mute, so sample k of
    the file is sample k of what the card delivered — including the intervals we
    were keyed, which is what `tools/rehear --ours` reads back out of it."""
    wav = tmp_path / "named.wav"
    link = R.RadioLink(_Sink(), rig=None, record=wav)
    _capture(link, np.full(_BLOCK, 0.25, dtype=np.float32))
    link.close()

    line = _stream_line(capsys)
    assert str(wav) in line, line
    assert "INCOMPLETE" not in line, line
    assert "sample clock" in line, line


def test_a_recorder_that_stopped_writing_is_named_incomplete(tmp_path, capsys):
    """A short recording that reads as a whole one is the loss this file exists
    to prevent, so the length is stated with the fault that ended it."""
    def boom() -> None:
        raise OSError("no space left on device")

    wav = tmp_path / "short.wav"
    link = R.RadioLink(_Sink(), rig=None, record=wav)
    _capture(link, np.full(_BLOCK, 0.25, dtype=np.float32))
    link.recorder.drain()                    # what did reach the file
    link.recorder.drain = boom
    link.close()

    line = _stream_line(capsys)
    assert "INCOMPLETE" in line and "no space left on device" in line, line
    assert len(_recorded(wav)) == _BLOCK, "the samples that did land were lost too"


def test_a_recorder_that_fails_mid_session_does_not_take_the_receiver_with_it(
        monkeypatch, tmp_path, capsys):
    """A transmission in progress outranks a recording, and so does a link: the
    pump goes on decoding, and the reason the file stopped is kept for the
    report rather than repeated into the log every 50 ms."""
    def boom() -> None:
        raise OSError("no space left on device")

    fake = types.ModuleType("sounddevice")
    made: dict = {}
    fake.InputStream = lambda **kw: made.setdefault("stream", _FakeStream(**kw))
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    link = R.RadioLink(_Sink(), rig=None, record=tmp_path / "faulty.wav")
    link.recorder.drain = boom
    frame = M.render_frame(0x3A, payload=bytes([24, 24, 24]), session_id=0x5E)
    quiet = np.zeros(int(0.5 * SAMPLE_RATE), dtype="<i2")
    audio = to_card(np.concatenate([quiet, frame, quiet]), SAMPLE_RATE)

    link.start()
    try:
        cb = made["stream"].callback
        for i in range(0, len(audio), _BLOCK):
            cb(audio[i:i + _BLOCK].reshape(-1, 1), _BLOCK, None, None)
        deadline = time.monotonic() + 5.0
        while not link.modem.frames and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        link.close()

    assert [f.type for f in link.modem.frames] == [0x3A], \
        "a recorder fault deafened the receiver"
    assert "INCOMPLETE" in _stream_line(capsys)


class _TimedRig:
    """A rig that only remembers when it was keyed, and how long it says to wait."""

    def __init__(self, settle: float) -> None:
        self.settle, self.log = settle, []

    def ptt(self, on: bool) -> None:
        self.log.append((f"ptt{int(on)}", time.monotonic()))

    def unkey(self, why: str = "") -> None:
        self.log.append(("unkey", time.monotonic()))


#: How far over a rig's own settle the keyed lead may run. `time.sleep` never
#: returns early, so the honest bound is one-sided and tight — measured overshoot
#: on this machine is under 5 ms across 50 trials at both 0.04 and 0.40 s. It was
#: `abs=0.05`, which is wider than the whole FT-891 settle: removing the wait
#: altogether passed the 0.04 case and the old hardcoded 0.35 passed the 0.40 one,
#: so neither test could see either of the two faults it was written for.
_LEAD_SLACK_S = 0.02


def _assert_lead_is(lead: float, settle: float) -> None:
    assert lead >= settle, (
        f"{lead * 1000:.0f} ms lead for a rig asking {settle * 1000:.0f} — "
        "the head of the burst never leaves the modulator")
    assert lead - settle < _LEAD_SLACK_S, (
        f"{(lead - settle) * 1000:.0f} ms of dead carrier past the rig's "
        f"{settle * 1000:.0f} ms settle")


def _keyed_transmission(monkeypatch, rig):
    """Run one `_transmit` against a rig, returning the ordered event log."""
    _stub_sounddevice(monkeypatch)
    out_cls = sys.modules["sounddevice"].OutputStream
    real_init, real_write = out_cls.__init__, out_cls.write

    def init_spy(self, **kw):
        rig.log.append(("open", time.monotonic()))
        real_init(self, **kw)

    def write_spy(self, audio):
        rig.log.append(("audio", time.monotonic()))
        return real_write(self, audio)
    monkeypatch.setattr(out_cls, "__init__", init_spy)
    monkeypatch.setattr(out_cls, "write", write_spy)

    link = R.RadioLink(_Sink(), rig=rig, out_device="stub")
    link._transmit(M.render_frame(0x4A, payload=b"lead", session_id=1))
    return rig.log


def test_the_device_is_driven_before_the_key_and_the_lead_is_the_rig_s(monkeypatch):
    """Key-down to first sample is unmodulated carrier on a shared band, so what may
    sit in it is the rig's own settle, and the open of the burst's own stream.

    The device is opened and run BEFORE the key, once per process, so that a device
    which will not open refuses with the transmitter still down and so that the
    slow first open of a session is not paid under a carrier — the warm-up is the
    first `open`/`audio` pair below, and no key stands between them. The burst's
    own stream is opened after it, because `play_drained` opens per burst and must:
    a held-open output stream keeps a rig whose PTT follows data audio in transmit
    after CAT has unkeyed.

    That second open is the whole of what the lead carries beyond the settle, and
    `tools/ptt_tail_check.py` is what sizes it: that instrument keys, plays through
    `play_drained` and unkeys with no settle of its own, so the 0.040 s lead it
    measured on this rig is CAT acknowledgement plus a COLD open plus codec
    latency — a ceiling on a warmed one.

    The lead is visible in each modem's own capture, which goes to digital silence
    from key-down until its first radiated sample. Off the 2026-08-06 recordings,
    same rig and band minutes apart: besra led by 0.43 s (median, n=31), kestrel by
    0.085 s (median, n=19). `tools/ptt_tail_check` instrumented the same rig that
    hour at lead 0.040 s and tail 0.220 s, and kestrel's tail reproduces that 0.220
    exactly — which is what says the silence really is the keyed window."""
    settle = RIGS["ft891"]["settle"]
    log = _keyed_transmission(monkeypatch, _TimedRig(settle))

    assert [e for e, _ in log] == ["open", "audio", "ptt1", "open", "audio",
                                   "ptt0"], log
    at = dict(log)                              # duplicate keys keep the burst's
    lead = at["audio"] - at["ptt1"]
    _assert_lead_is(lead, settle)


def test_a_rig_that_needs_a_long_settle_still_gets_it(monkeypatch):
    """The number is the radio's, not ours, in both directions. The X6100's USB codec
    re-initialises when it is keyed and the head of the burst never reaches the
    modulator without 0.40 s — so a link that clamped the wait short would break the
    rig the wait exists for."""
    settle = RIGS["x6100"]["settle"]
    log = _keyed_transmission(monkeypatch, _TimedRig(settle))

    at = dict(log)
    _assert_lead_is(at["audio"] - at["ptt1"], settle)


def test_the_rig_carries_the_settle_from_the_one_table(monkeypatch):
    """`RIGS` exists because the settle had four copies that disagreed. `RadioLink`
    reads it off the rig, so this is the join that keeps it from growing a fifth.

    The atexit unkey is unhooked first: these rigs are never keyed, and a test that
    leaves handlers behind spawns `rigctl` against a made-up port at interpreter
    exit."""
    monkeypatch.setattr(R.atexit, "register", lambda *a, **k: None)
    for name, spec in RIGS.items():
        # rigctl="/bin/sh": always present, so construction clears the
        # missing-hamlib refusal; nothing here ever spawns it.
        assert R.Rig.named(name, "/dev/null", rigctl="/bin/sh").settle == spec["settle"]


def test_tune_atu_warms_the_device_then_waits_the_rig_s_settle(monkeypatch):
    """The third copy of the same discipline, and the one nothing was watching.

    `tune_atu` keys a bare carrier into an antenna tuner, so everything between the
    key and the first sample radiates as unmodulated carrier exactly as it does on
    the data path. The device is warmed *before* PTT — that is the rule this
    function was written for, after the incident that stuck the finals — and the
    wait after PTT is the rig's own settle from `core.rigs`, not a number of ours.
    """
    played = _stub_sounddevice(monkeypatch)
    rig = _TimedRig(RIGS["x6100"]["settle"])
    out_cls = sys.modules["sounddevice"].OutputStream
    real_write = out_cls.write

    def write_spy(self, audio):
        rig.log.append(("audio", time.monotonic()))
        return real_write(self, audio)
    monkeypatch.setattr(out_cls, "write", write_spy)

    R.tune_atu(rig, out_device="stub", seconds=0.05)

    assert [e for e, _ in rig.log] == ["audio", "ptt1", "audio", "ptt0"], rig.log
    warm, key, tone = (t for _, t in rig.log[:3])
    assert warm < key, "the device was opened into a keyed transmitter"
    _assert_lead_is(tone - key, rig.settle)
    assert all(s.closed for _, _, s in played), \
        "output stream outlived the tune — a data-keyed rig stays keyed"


def test_tune_atu_unkeys_when_the_tone_fails(monkeypatch):
    """A keyed carrier with a raised exception between it and the unkey is how a
    transmitter is left on. The `finally` is the whole safety property."""
    _stub_sounddevice(monkeypatch)
    rig = _TimedRig(0.0)
    out_cls = sys.modules["sounddevice"].OutputStream
    calls = {"n": 0}

    def boom(self, audio):
        calls["n"] += 1
        if calls["n"] > 1:                      # let the pre-key warm-up through
            raise RuntimeError("device went away")
    monkeypatch.setattr(out_cls, "write", boom)

    with pytest.raises(RuntimeError):
        R.tune_atu(rig, out_device="stub", seconds=0.05)

    assert [e for e, _ in rig.log] == ["ptt1", "ptt0"], rig.log


class _RefusingRig(_TimedRig):
    """A rig that answers its keyer honestly: the key-up did not confirm."""

    def ptt(self, on: bool) -> bool:
        super().ptt(on)
        return not on                 # every key-up refused, every unkey taken


@pytest.mark.realtime
def test_a_refused_key_up_keeps_the_frame_out_of_the_codec(monkeypatch, caplog):
    """The kestrel and shrike fix, on this path too: `rig.ptt(True)` saying False
    means the burst cannot have gone out, and settle plus audio into the codec
    then is a transmission in the log and silence on the band. The frame is not
    written; the ARQ's own repeat ladder decides what happens next, and the next
    key-up is decided afresh."""
    caplog.set_level("ERROR", logger="hfmodem.besra.radio")
    played = _stub_sounddevice(monkeypatch)
    rig = _RefusingRig(0.25)
    link = R.RadioLink(_Sink(), rig=rig, out_device="stub")

    t0 = time.monotonic()
    link._transmit(M.render_frame(0x4A, payload=b"refused", session_id=1))

    assert not any(a.any() for _, a, _ in played), (
        "a frame the rig refused to key was written to the codec")
    assert time.monotonic() - t0 < rig.settle, (
        "the settle was paid for a key-up that never happened")
    assert "NOT TRANSMITTING" in caplog.text, caplog.text
    # The drop-to-be-sure and the unmute still run: the receive path comes back.
    assert [e for e, _ in rig.log] == ["ptt1", "ptt0"], rig.log
    assert link.muted is False


def test_a_retired_rig_reports_the_refusal_it_already_makes(monkeypatch):
    """`Rig.ptt` refused a retired rig's key-up and told nobody: the return was
    None, so `_transmit` played the whole frame into an unkeyed transmitter.
    The verdict is the fix -- False, so the frame stays out of the codec."""
    monkeypatch.setattr(R.atexit, "register", lambda *a, **k: None)
    rig = R.Rig(1036, "/dev/null", 38400, rigctl="/bin/sh", rigctld="127.0.0.1:1")
    monkeypatch.setattr(R.Rig, "_write", lambda self, *a: True)
    rig.retire()
    assert rig.ptt(True) is False, "a refused key-up must say so"
    assert rig.ptt(False) is True, "the unkey stays allowed on a retired rig"


def _pipe_dead_rig(tmp_path, monkeypatch):
    """A hamlib-keyed rig whose persistent pipe takes the key-down and is dead
    by the key-up: `_write` answers True for `T 1` and False for `T 0` — and
    False is proof the command cannot have gone (`Rig._write`). The one-shot
    rigctl exits nonzero instantly, the way rigctl reports a dead rigctld, so
    the ladder — `core.ptt.Keyer.unkey` now, which is also where the drop is
    intercepted — has to fall through to `drop_rts` and the retire."""
    from hfmodem.core import ptt as core_ptt

    exe = tmp_path / "rigctl"
    exe.write_text("#!/bin/sh\nexit 2\n")
    exe.chmod(0o755)
    monkeypatch.setattr(R.atexit, "register", lambda *a, **k: None)
    rig = R.Rig(1036, "/dev/null", 38400, rigctl=str(exe), ptt_device="/dev/null")
    monkeypatch.setattr(rig, "_write", lambda *a: list(a) != ["T", "0"])
    dropped: list[str] = []
    monkeypatch.setattr(core_ptt, "drop_rts",
                        lambda dev, log: dropped.append(dev) or False)
    monkeypatch.setattr(core_ptt, "UNKEY_BUDGET_S", 0.2)   # real seconds, hurried
    return rig, dropped


def test_a_key_up_the_pipe_refused_escalates_to_the_emergency_unkey(
        monkeypatch, tmp_path, caplog):
    """`ptt(False)` returning False on the hamlib path is proof `T 0` never
    left, and CAT PTT latches at the rig — so a frame's `finally` must hand the
    rig to `unkey`'s ladder (one-shot, `drop_rts`, alarm, retire) rather than
    cancel the watchdog, clear `muted`, and go back to listening with the
    transmitter keyed. The verdict was added for the key-down; the one place
    False means "the transmitter may still be up" must read it too."""
    caplog.set_level("ERROR", logger="hfmodem.besra.radio")
    played = _stub_sounddevice(monkeypatch)
    rig, dropped = _pipe_dead_rig(tmp_path, monkeypatch)
    link = R.RadioLink(_Sink(), rig=rig, out_device="stub")

    link._transmit(M.render_frame(0x4A, payload=b"stuck", session_id=1))

    assert played, "the burst itself went out — the failure is the key-up"
    assert dropped == ["/dev/null"], (
        "ptt(False)'s False was discarded: the emergency unkey never ran")
    assert rig.retired, ("a transmitter nobody has confirmed down is not one "
                         "this program may key again")
    assert "MAY BE STUCK" in caplog.text, caplog.text
    assert link.muted is False, "the receive path still comes back"


def test_tune_atu_escalates_a_key_up_the_pipe_refused(
        monkeypatch, tmp_path, caplog):
    """The same shape in `tune_atu`'s `finally`: a bare keyed carrier is the
    burst this discipline exists for, and its key-up verdict was discarded the
    same way."""
    caplog.set_level("ERROR", logger="hfmodem.besra.radio")
    _stub_sounddevice(monkeypatch)
    rig, dropped = _pipe_dead_rig(tmp_path, monkeypatch)

    R.tune_atu(rig, out_device="stub", seconds=0.02)

    assert dropped == ["/dev/null"], (
        "ptt(False)'s False was discarded: the emergency unkey never ran")
    assert rig.retired
    assert "MAY BE STUCK" in caplog.text, caplog.text


def test_transmit_mutes_own_capture(monkeypatch):
    """Half-duplex: capture is muted across the play so a station does not decode its
    own transmission, and unmuted afterwards. The mute drops the audio at the
    callback, before the decoder — a station's own signal never enters the window."""
    _stub_sounddevice(monkeypatch)
    link = R.RadioLink(_Sink(), rig=None)
    frame = M.render_frame(0x4A, payload=b"x" * 8, session_id=1)
    seen = {}

    out_cls = sys.modules["sounddevice"].OutputStream
    real_write = out_cls.write

    def write_spy(self, audio):
        seen["muted_during_tx"] = link.muted
        _capture(link, np.asarray(audio).reshape(-1))   # the station hearing itself
        return real_write(self, audio)
    monkeypatch.setattr(out_cls, "write", write_spy)

    link._transmit(frame)
    assert seen["muted_during_tx"] is True
    assert link.muted is False
    assert link.modem.frames == [], "the modem decoded our own transmission"


# --------------------------------------------------------------------------- #
# Two stations, both through the real 48 kHz radio path.
# --------------------------------------------------------------------------- #

class _Recorder(ModemObserver):
    def __init__(self):
        self.connected_to = None
        self.disconnected = False
        self.rx = bytearray()

    def modem_newstate(self, state): pass
    def modem_connected(self, remote, bw): self.connected_to = remote
    def modem_disconnected(self): self.disconnected = True
    def modem_ptt(self, on): pass
    def modem_buffer(self, n): pass
    def modem_data_received(self, kind, blob): self.rx += blob
    def modem_status(self, text): pass


class _Station(BesraModem):
    """A modem that also keeps what the receive path handed it, so a two-station
    session is inspectable frame by frame."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.frames: list = []

    def receive_frames(self, frames) -> None:
        self.frames.extend(frames)
        super().receive_frames(frames)


def _station(call: str, *, listen: bool = False):
    modem = _Station(bandwidth=500)
    modem.set_mycall(call)
    modem.set_listen(listen)
    obs = _Recorder()
    modem.start(obs)
    return R.RadioLink(modem, rig=None, out_device=call), obs


def _relay(links, played) -> int:
    """Run the air: every transmission one station plays is captured by the other,
    in 0.1 s blocks through the real resamplers, with a turnaround of quiet either
    side. Delivering a frame keys the reply, which lands back on `played`."""
    quiet = np.zeros(int(0.4 * R.FS_RADIO), dtype=np.float32)
    sent = 0
    while played:
        device, audio, stream = played.pop(0)
        assert stream is None or stream.closed, \
            "output stream outlived the transmission — a data-keyed rig stays keyed"
        if not audio.any():
            continue                     # `warm_output`'s silence, played unkeyed
        sent += 1
        for link in links:
            if link.out_device != device:
                _capture(link, np.concatenate([quiet, audio.astype(np.float32), quiet]))
    return sent


def test_two_stations_connect_transfer_and_disconnect_over_the_radio_path(monkeypatch):
    """The integration gate for the receive path: a full ARQ session where every
    frame goes out through `_transmit`'s 12->48 kHz edge and comes back through the
    capture callback and the rolling decode. No gate, no bursts, no queue between
    the sound card and the session."""
    played = _stub_sounddevice(monkeypatch)
    caller, ca = _station("W9SSJ")
    responder, ra = _station("K7ABC", listen=True)
    links = [caller, responder]

    caller.modem.connect("K7ABC")
    sent = _relay(links, played)
    assert caller.modem.connected, f"caller state {caller.modem.state}"
    assert responder.modem.connected, f"responder state {responder.modem.state}"
    assert ca.connected_to == "K7ABC" and ra.connected_to == "W9SSJ"

    caller.modem.transmit(b"hello winlink over the sound card")
    sent += _relay(links, played)
    assert bytes(ra.rx) == b"hello winlink over the sound card", f"got {bytes(ra.rx)!r}"

    caller.modem.disconnect()
    sent += _relay(links, played)
    assert not ra.disconnected, "a single DISC ended the session"
    # The IRS answers the DISC repeat and not the first copy
    # (`ArqSession._corroborated`), and nothing else here advances the clock.
    for step in range(1, 13):
        caller.modem.tick(step * 0.5)
        responder.modem.tick(step * 0.5)
        sent += _relay(links, played)
    assert ca.disconnected and ra.disconnected
    assert caller.modem.state == P.ArdopState.DISC
    assert responder.modem.state == P.ArdopState.DISC

    # One transmission in, one frame out, all of them clean. A rolling window sees
    # every frame in several windows, and a DATAACK delivered twice would clear a
    # data frame the peer never acknowledged — the dedup is what makes it safe to
    # decode overlapping audio and hand the result to stop-and-wait ARQ.
    #
    # Counted over the frames addressed to this exchange, which is the whole of
    # what the dedup guards. The frame-type search also locks onto the quiet either
    # side of a burst: this run alone mints a `4PSK.200.100S.E` for session 0x95 and
    # a DATANAK for 0x7f, neither keyed by anybody, both carrying a session id from
    # nowhere. `on_receive`'s address filter is what refuses those, and it is tested
    # where it lives.
    session = crc.session_id("W9SSJ", "K7ABC")
    heard = [f for f in caller.modem.frames + responder.modem.frames
             if f.session_id in (session, 0xFF)]
    assert len(heard) == sent, f"{sent} transmissions, {len(heard)} frames delivered"
    assert all(f.ok for f in heard)
