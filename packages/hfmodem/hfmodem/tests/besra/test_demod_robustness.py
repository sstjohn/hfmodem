# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Receiver robustness: multi-frame scanning, noise rejection, and CFO tolerance.

Covers the failure modes the ground-truth fixtures cannot, because every fixture
is a single clean on-tune frame:

- **multi-frame** — several frames back to back in one buffer must all decode,
  not just the first (the scan must step past a frame's trailing energy before
  re-acquiring);
- **noise** — pure noise must not mint phantom control frames (the header
  quality gate);
- **carrier frequency offset** — a mistuned frame (±200 Hz, spec §4.1/§7) must
  still decode once the leader-estimated offset is compensated;
- **sensitivity** — the link budget in noise and fading, and the same budget
  measured on the *reference's* renders rather than besra's own, so a receiver
  tuned to its own transmitter cannot hide here.
"""

from __future__ import annotations


import numpy as np
import pytest

from hfmodem.besra import monitor
from hfmodem.besra.dsp import detect
from hfmodem.besra.phy import demodulator as D
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.radio import FS_RADIO, RollingDecoder
from hfmodem.besra.frame import frame as F
from hfmodem.tests import evidence
from . import groundtruth as gt

SR = 12000
_LEAD = np.zeros(2400, dtype=np.int16)
_TAIL = np.zeros(4800, dtype=np.int16)


def _framed(*chunks: np.ndarray) -> np.ndarray:
    return np.concatenate([_LEAD, *chunks, _TAIL])


def _pl(ftype: int) -> bytes:
    """A deterministic payload filling the frame's net capacity."""
    return bytes((i * 7 + 3) & 0xFF for i in range(F.FRAMES[ftype].net_payload))


# --------------------------------------------------------------------------- #
# FIX 1 — back-to-back frames in one buffer all decode.
# --------------------------------------------------------------------------- #

def test_multiframe_buffer_decodes_all():
    gap = np.zeros(3000, dtype=np.int16)
    a = M.render_frame(0x40, payload=_pl(0x40), session_id=0x5A)
    b = M.render_frame(0x44, payload=_pl(0x44), session_id=0x33)
    c = M.render_frame(0x34, caller="W9SSJ", target="K7ABC", session_id=0xFF)
    frames = D.decode(_framed(a, gap, b, gap, c))

    assert [f.type for f in frames] == [0x40, 0x44, 0x34]
    assert all(f.ok for f in frames)
    assert (frames[0].session_id, frames[1].session_id) == (0x5A, 0x33)
    assert frames[0].payload == _pl(0x40)
    assert frames[1].payload == _pl(0x44)
    assert (frames[2].caller, frames[2].target) == ("W9SSJ", "K7ABC")
    # Frame positions strictly increase — no frame is re-acquired as the next.
    assert frames[0].offset < frames[1].offset < frames[2].offset


def test_single_frame_still_yields_one():
    """The trailing-energy skip must not conjure a second frame from one frame."""
    au = _framed(M.render_frame(0x48, payload=_pl(0x48), session_id=0x11))
    frames = D.decode(au)
    assert len(frames) == 1 and frames[0].type == 0x48 and frames[0].ok


# --------------------------------------------------------------------------- #
# FIX 2 — the monitor feeds the whole capture straight through, timestamped.
# --------------------------------------------------------------------------- #

def test_monitor_straight_through_prints_every_frame(capsys):
    gap = np.zeros(3000, dtype="<i2")
    frames = [
        M.render_frame(0x31, caller="W9SSJ", target="K7ABC", session_id=0xFF),
        M.render_frame(0x3A, payload=bytes([24, 24, 24]), session_id=0x5E),
        M.render_frame(0x4A, payload=_pl(0x4A), session_id=0x5E),
        M.render_frame(0x30, caller="W9SSJ", grid="EN63", session_id=0xFF),
    ]
    stream = np.concatenate([gap] + sum(([f.astype("<i2"), gap] for f in frames), []))
    monitor._decode_capture(monitor.Demodulator(), stream, 0.0)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 4
    assert [ln.split()[1] for ln in lines] == ["0x31", "0x3A", "0x4A", "0x30"]
    # Timestamps come from each frame's own offset and strictly increase.
    stamps = [float(ln.split()[0]) for ln in lines]
    assert stamps == sorted(stamps) and stamps[0] > 0.0


# --------------------------------------------------------------------------- #
# FIX 3 — the header quality gate rejects noise.
# --------------------------------------------------------------------------- #

def test_noise_yields_almost_no_frames():
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(200):
        total += len(D.decode((rng.standard_normal(24000) * 800).astype(np.int16)))
    # A clean header scores ~9.99/10; noise that satisfies the frame-type parity
    # tops out near 6.5, well under the 8.0 floor — so noise mints ~no frames.
    assert total < 5, f"{total} phantom frames from 200 noise captures"


def test_a_frames_verdict_is_a_verdict_only_where_a_body_can_refute_it():
    """`DecodedFrame.ok` is an integrity verdict for every frame that has a body and
    a tautology for the eight that do not, and `unverified` is the difference.

    Pinned mechanically rather than against a list of names: rewrite everything
    past the frame-type header as noise and decode again. A frame whose `ok` was a
    verdict cannot survive that — its RS, CRC or callsign fails. A frame reported
    `unverified` decodes exactly as before, because there was never anything behind
    the ten header tones to disagree with them.

    That is why a minted control frame cannot be caught here and is caught by its
    missing repeat instead (`ArqSession._corroborated`)."""
    rng = np.random.default_rng(7)
    hdr = D._LEADER_LEN + D._HEADER_LEN
    survived, refuted = [], []
    for ftype, fd in F.FRAMES.items():
        sess = 0xFF if fd.forces_session else 0x5E
        clean = M.render_frame(ftype, payload=_pl(ftype) if fd.k >= 16 else b"",
                               session_id=sess,
                               caller="W9SSJ", target="K7ABC", grid="EN63", timing=240,
                               sn=10, quality=80)
        got = D.decode(_framed(clean))
        assert len(got) == 1 and got[0].ok and got[0].type == ftype, fd.name
        assert got[0].unverified == (fd.name in D._SHORT_CONTROL), fd.name

        gutted = clean.copy()
        gutted[hdr:] = (rng.standard_normal(gutted.size - hdr) * 800).astype(np.int16)
        again = [f for f in D.decode(_framed(gutted)) if f.type == ftype and f.ok
                 and not f.header_only]
        (survived if again else refuted).append(fd.name)

    assert set(survived) == D._SHORT_CONTROL, f"survived without a body: {sorted(survived)}"
    assert set(refuted) == {fd.name for fd in F.FRAMES.values()} - D._SHORT_CONTROL


def test_ten_symbols_of_a_steady_tone_are_reported_as_a_frame_nobody_sent():
    """The frame-type header carries no rejection power against a structured
    signal: ten 50-baud symbols all reading the same 4FSK tone satisfy its parity
    and decode to `16QAM.500.100.O sess=0x00`. That is the exact phantom the
    2026-08-23 WW2MI session logged fifteen times, one of them between our own PTT
    ON and PTT OFF, and the receiver has no second opinion to weigh it against — so
    the type stands and the report must not present it as a sighting."""
    ftype, session = F.decode_header([1] * 10)
    assert F.FRAMES[ftype].name == "16QAM.500.100.O" and session == 0x00
    phantom = D.DecodedFrame(type=ftype, session_id=session, ok=True,
                             name=F.FRAMES[ftype].name)
    assert phantom.unverified is False        # a data type: its body is what refutes it
    bare = D.DecodedFrame(type=next(t for t, d in F.FRAMES.items() if d.name == "DISC"), session_id=0x5E, ok=True, name="DISC")
    assert bare.unverified and "UNVERIFIED" in monitor._render(bare)


def _steady_tone(tone: int = 1, syms: int = 40) -> np.ndarray:
    """A leader followed by ``syms`` symbols of one 4FSK tone — the shape that
    mints the phantom, and nothing a station transmits."""
    lead = np.arange(D._LEADER_LEN) / SR
    body = np.arange(syms * D._SYM_50) / SR
    return _framed(
        (4000 * (np.sin(2 * np.pi * detect.LEADER_TONES[0] * lead)
                 + np.sin(2 * np.pi * detect.LEADER_TONES[1] * lead))).astype(np.int16),
        (8000 * np.sin(2 * np.pi * detect.FSK_TONES[50][tone] * body)).astype(np.int16))


def _heard(au: np.ndarray, session: int | None) -> list[D.DecodedFrame]:
    return D.Demodulator(
        expect_session=None if session is None else (lambda: session)).decode(au)


def test_a_session_reports_no_frame_addressed_to_another_one():
    """The gate on the phantom above: while this station is party to a session, a
    type read off ten tones and bearing an id that is not the session's is a claim
    with no corroboration of any kind behind it, and `ArqSession.on_receive`
    already discards exactly those frames as someone else's traffic.

    A listening receiver is not touched — an unsolicited frame from a station we
    are not in session with is a real thing this receiver exists to see — and
    neither is a session whose own id happens to be zero, which is a legal one."""
    au = _steady_tone()
    assert [(f.name, f.session_id, f.header_only) for f in _heard(au, None)] \
        == [("16QAM.500.100.O", 0x00, True)]
    assert _heard(au, 0x0D) == []
    assert len(_heard(au, 0x00)) == 1, "0x00 is a session id like any other"


def test_a_frame_that_validated_is_a_sighting_whatever_id_it_bears():
    """The gate must not blind us to a third station. Both 2026-08-26 W6IDS
    recordings hold `4PSK.200.100 sess=0x05` frames at quality 62-66 from a QSO
    that was not ours — RS and payload CRC intact, so a real frame on the air —
    and a receiver in session with 0x0d must still report them."""
    stranger = M.render_frame(0x40, payload=_pl(0x40), session_id=0x05)
    got = _heard(_framed(stranger), 0x0D)
    assert [(f.name, f.session_id, f.ok) for f in got] == [("4PSK.200.100.E", 0x05, True)]


def test_a_header_only_frame_of_our_own_session_still_reports():
    """The frames the gate exists not to take: a data frame whose body the channel
    destroyed but whose header carries the session's own id. Eight of them across
    the two 2026-08-26 recordings, at quality 52-63, and they are what the
    unreadable budget and the NAK path are built on."""
    rng = np.random.default_rng(11)
    frame = M.render_frame(0x40, payload=_pl(0x40), session_id=0x0D)
    head = D._LEADER_LEN + D._HEADER_LEN
    frame[head:] = (rng.standard_normal(frame.size - head) * 800).astype(np.int16)
    got = _heard(_framed(frame), 0x0D)
    assert [(f.name, f.session_id, f.ok, f.header_only) for f in got] \
        == [("4PSK.200.100.E", 0x0D, False, True)]


def test_a_type_minted_inside_a_frame_already_arriving_is_not_a_sighting():
    """The 2026-08-28 defect in synthesis: a rolling window that opens *inside* a
    frame the peer is still sending. Acquisition finds no leader before the body
    symbols it lands on, so a type read off ten of them has no corroboration of any
    kind — not RS, not a CRC, and not the leader that precedes every real header —
    and the session id it carries came out of the same ten symbols as the type.

    A window holding the same frame whole reads it for what it is."""
    frame = M.render_frame(0x48, payload=_pl(0x48), session_id=0x0D)
    whole = _framed(frame)
    mid = whole[D._LEADER_LEN + D._HEADER_LEN + 4000:]
    assert [(f.name, f.ok) for f in _heard(whole, 0x0D)] == [("4FSK.200.50S.E", True)]
    assert [f.name for f in _heard(mid, 0x0D) if f.header_only] == []


#: The 2026-08-28 W6IDS arm, ten seconds either side of the `16QAM.500.100.E` the
#: log reports at 19:08:01 — minted 0.52 s into the body of a `4FSK.200.50S.E` the
#: gateway was still transmitting, out of a window that opened past that frame's
#: leader, and bearing this station's own 0x0d. It put a DATANAK on the air 1 ms
#: later and it swallowed the real frame behind it, because the 3.9 s span its
#: guessed type claims carries the acquisition cursor straight over it.
_W6IDS_MINT = (evidence.LOGS / "onair" / "20260829T000632Z-besra-7060000.wav", 80.0, 90.0)


def test_the_w6ids_midframe_mint_goes_and_the_frame_under_it_comes_back():
    """The frame that keyed the NAK, on the recording, through the rolling windows
    the live path decodes in.

    The session-id match is what the 2026-08-26 gate was credited with — and this
    one carries 0x0d, so `_addressed` passes it. What refuses it is the leader the
    audio does not have."""
    wav, t0, t1 = _W6IDS_MINT
    if not wav.exists():
        pytest.skip("station recording not present")
    got = []
    rx = RollingDecoder(D.Demodulator(expect_session=lambda: 0x0D),
                        lambda _pos, f: got.append(f), resample=True)
    for block in _blocks(wav, t0, t1):
        rx.push(block)
        rx.pump()
    rx.flush()

    assert not [f for f in got if f.name.startswith("16QAM")]
    # The gateway's own frame, and the header-only sighting of the next one that
    # the unreadable budget and the NAK path are built on: both kept.
    assert ("4FSK.200.50S.E", True, 73) in [(f.name, f.ok, f.quality) for f in got]
    assert ("4FSK.200.50S.O", True) in [(f.name, f.header_only) for f in got]


#: The two W6IDS fetches of 2026-08-26, and the seconds either side of three of
#: the four `16QAM.500.100.O sess=0x00` sightings they logged — one at the
#: truncation sentinel and one at a quality of 7, so that a reading of q is not
#: what separates them.
_W6IDS = [
    (evidence.LOGS / "onair" / "20260826T133925Z-besra-7060000.wav", 86.0, 96.0),
    (evidence.LOGS / "onair" / "20260826T152426Z-besra-7060000.wav", 78.0, 88.0),
]


@pytest.mark.parametrize("wav,t0,t1", _W6IDS)
def test_the_w6ids_phantoms_go_and_the_session_keeps_its_frames(wav, t0, t1):
    """On the recordings, through the rolling windows the live path decodes in — a
    whole-capture pass does not reproduce these at all, because its silence gate is
    a fraction of the *capture's* peak (`RollingDecoder`).

    Both arms, and that is the change of 2026-08-28. `_addressed` could only take a
    phantom off a receiver that was already in a session, so the listening arm kept
    two of these and the gate was measured on the difference between the arms. The
    leader `_scan_header` now asks every header-only claim for is not addressed to
    anyone, so both arms lose them and the two agree: nothing here was ever a
    frame, whoever was listening."""
    if not wav.exists():
        pytest.skip("station recording not present")
    got = {None: [], 0x0D: []}
    for session in got:
        rx = RollingDecoder(
            D.Demodulator(expect_session=None if session is None else (lambda s=session: s)),
            lambda _pos, f, out=got[session]: out.append(f), resample=True)
        for block in _blocks(wav, t0, t1):
            rx.push(block)
            rx.pump()
        rx.flush()

    assert not [f for f in got[None] if f.name.startswith("16QAM")]
    assert not [f for f in got[0x0D] if f.name.startswith("16QAM")]
    assert {(f.name, f.session_id) for f in got[0x0D]} \
        == {(f.name, f.session_id) for f in got[None]}


def _blocks(wav, t0: float, t1: float):
    """The recording between ``t0`` and ``t1`` as the sound card's own 0.1 s blocks."""
    import wave

    with wave.open(str(wav)) as w:
        assert w.getframerate() == FS_RADIO
        w.setpos(int(t0 * FS_RADIO))
        au = np.frombuffer(w.readframes(int((t1 - t0) * FS_RADIO)), "<i2")
    au = au.astype(np.float32) / 32768.0
    step = int(0.1 * FS_RADIO)
    return [au[i:i + step] for i in range(0, au.size, step)]


@gt.requires_reference
def test_fixtures_survive_the_gate():
    """The gate must not cost a single clean fixture (they sit at the ceiling)."""
    manifest = gt.txframe_manifest()
    for name in gt.wav_names():
        frames = D.decode(np.concatenate([_LEAD, gt.load_wav(name), _TAIL]))
        assert len(frames) == 1 and frames[0].ok, name
        assert frames[0].type == manifest[name]["type"], name


# --------------------------------------------------------------------------- #
# FIX 4 — carrier frequency-offset compensation (spec §4.1/§7, ±200 Hz).
# --------------------------------------------------------------------------- #
#
# A carrier/tuning offset is single-sideband: the whole audio passband slides by
# f. The analytic-signal shift below models exactly that. (Note: mixing a *real*
# signal — ``real(au * exp(2πj f n))`` — is double-sideband, planting a mirror
# copy at −f; at small f that image lands inside ARDOP's own 50 Hz symbol bin and
# no ARDOP receiver can separate it, so that formula does not model a CFO. See
# ``test_dsb_mix_images_at_small_offset`` for the documented boundary.)

def _analytic(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x)
    n = x.size
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def _tune(au: np.ndarray, hz: float) -> np.ndarray:
    n = np.arange(au.size)
    return np.real(_analytic(au.astype(float)) * np.exp(2j * np.pi * hz * n / SR))


# One frame per carrier count and modulation, out to the densest/widest mode.
_CFO_MODES = [0x40, 0x44, 0x46, 0x4A, 0x50, 0x54, 0x64, 0x74]
_CFO_OFFSETS = [10, 50, 100, 200, -100, -200]


@pytest.mark.parametrize("ftype", _CFO_MODES)
def test_cfo_tolerance(ftype):
    pl = _pl(ftype)
    au = np.concatenate([_LEAD, M.render_frame(ftype, payload=pl, session_id=0x2A), _TAIL])
    for hz in _CFO_OFFSETS:
        frames = D.decode(_tune(au, hz))
        assert len(frames) == 1, f"0x{ftype:02X} @ {hz:+d} Hz: {len(frames)} frames"
        f = frames[0]
        assert f.type == ftype and f.ok and f.payload == pl, \
            f"0x{ftype:02X} @ {hz:+d} Hz mis-decoded"


def test_zero_offset_unchanged():
    """On-tune frames fall inside the deadband, so nothing is de-rotated."""
    pl = _pl(0x40)
    au = np.concatenate([_LEAD, M.render_frame(0x40, payload=pl, session_id=0x2A), _TAIL])
    a = D.decode(au)
    b = D.decode(_tune(au, 0.0))
    assert a[0].payload == pl and b[0].payload == pl


def test_dsb_mix_images_at_small_offset():
    """Documented boundary: the double-sideband ``real(au·exp)`` formula decodes
    once its −f image is 200+ Hz clear of the tones, but not when the image sits
    within the symbol bin — a limit of ARDOP's waveform, not of the compensator."""
    pl = _pl(0x40)
    au = np.concatenate([_LEAD, M.render_frame(0x40, payload=pl, session_id=0x2A), _TAIL])
    n = np.arange(au.size)

    def dsb(hz):
        return np.real(au.astype(float) * np.exp(2j * np.pi * hz * n / SR))

    for hz in (100, 200, -200):
        frames = D.decode(dsb(hz))
        assert frames and frames[0].ok and frames[0].payload == pl, f"DSB {hz:+d}"


# --------------------------------------------------------------------------- #
# What the offset estimate is worth in Hz, against a known truth.
#
# Everything above reads the estimate through a decode, which passes as long as one
# of the retry shifts lands — so an estimate that is merely near enough looks the
# same as one that is right, and a report reading the number itself is left with no
# bound on it. These pin the number: a known offset in, a bounded error out.
# --------------------------------------------------------------------------- #

#: `_CFO_DEADBAND`: an estimate inside it is treated as on-tune, so it costs the
#: acquisition nothing.
_CFO_ERROR_HZ = 3.0

_CFO_TRUTHS = [-200.0, -137.4, -100.0, -50.0, -8.3, 0.0, 8.3, 50.0, 100.0, 137.4, 200.0]


def _leader_of(ftype: int, hz: float, leader_ms: int = 240) -> np.ndarray:
    """The leader region of one frame, mistuned by ``hz``, as the walk hands it over."""
    meta = {"payload": _pl(ftype)} if ftype in (0x4A, 0x40, 0x54) else \
           {"timing": 230} if ftype == 0x3C else {}
    au = np.concatenate([_LEAD, M.render_frame(ftype, session_id=0x51,
                                               leader_ms=leader_ms, **meta)])
    return _tune(au, hz)[_LEAD.size:_LEAD.size + 3120]


@pytest.mark.parametrize("hz", _CFO_TRUTHS)
def test_leader_cfo_reads_a_known_offset(hz):
    """The whole ±200 Hz capture range the spec asks a receiver to tolerate."""
    got = detect.estimate_leader_cfo(_leader_of(0x4A, hz))
    assert abs(got - hz) <= _CFO_ERROR_HZ, f"{hz:+.1f} Hz read as {got:+.1f}"


@pytest.mark.parametrize("leader_ms", [100, 120, 160, 200, 240, 400])
def test_leader_cfo_reads_a_short_leader(leader_ms):
    """A peer's leader is 5 to 50 symbols (spec App. B) and auto-timing shortens it:
    KE8LVA's greeting carries ~168 ms and KN4LQN's 80 m frames 120-160 ms, against
    the 240 ms this station sends. A window sized off our own leader runs into the
    4FSK header, whose tones are 50 Hz apart too, and the two-tone product cannot
    tell a sequential pair from a simultaneous one — the 200 ms window this used to
    read missed a 120 ms leader by more than the deadband on 46% of offsets and a
    100 ms leader on 90%, at every SNR from +20 to 0 dB."""
    for ftype in (0x4A, 0x3C, 0xE9, 0x40, 0x54):
        for hz in _CFO_TRUTHS:
            got = detect.estimate_leader_cfo(_leader_of(ftype, hz, leader_ms))
            assert abs(got - hz) <= _CFO_ERROR_HZ, \
                f"0x{ftype:02X} {leader_ms} ms: {hz:+.1f} Hz read as {got:+.1f}"


@pytest.mark.parametrize("snr", [0, 6, 20])
def test_leader_cfo_holds_its_hz_in_noise(snr):
    """The estimate is a narrowband measurement — two 5 Hz bins out of 6 kHz — so it
    holds far below where the frame behind it decodes."""
    for seed, hz in enumerate(_CFO_TRUTHS):
        seg = add_awgn(_leader_of(0x4A, hz), snr, 20260827 + seed)
        got = detect.estimate_leader_cfo(seg)
        assert abs(got - hz) <= _CFO_ERROR_HZ, f"{snr} dB: {hz:+.1f} Hz read as {got:+.1f}"


#: KN4LQN's leaders on 80 m, 2026-08-26: five inbound frames across the three arms,
#: by the sample their two-tone leader starts at in the 12 kHz stream. The gateway's
#: transmitter sits +8.3 Hz high on every one — measured off the tone pair itself,
#: and corroborated here by a line at 2*(1500 + offset) in the squared signal, which
#: shares nothing with the estimate under test. Station audio, so this skips where
#: the logs are absent.
_KN4LQN_HZ = 8.3
_KN4LQN = [("20260826T022425Z-besra-3590500.wav", 340639),
           ("20260826T022425Z-besra-3590500.wav", 357548),
           ("20260826T022631Z-besra-3590500.wav", 300834),
           ("20260826T022801Z-besra-3590500.wav", 1061011),
           ("20260826T022801Z-besra-3590500.wav", 1134786)]


@pytest.mark.parametrize("name,onset", _KN4LQN)
def test_leader_cfo_reads_a_gateway_off_air(name, onset):
    """The estimate against a real transmitter's real offset, which is the reading
    the reports quote and the one nothing else here covers: every synthetic leader
    above is 240 ms because that is what this station sends, and this gateway's are
    120-160 ms."""
    import wave
    from hfmodem.core.resample import from_card

    wav = evidence.LOGS / "onair" / name
    if not wav.exists():
        pytest.skip("station log not present")
    with wave.open(str(wav)) as w:
        assert w.getframerate() == FS_RADIO
        card = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    au = from_card(card, SR).astype(np.float64)

    got = detect.estimate_leader_cfo(au[onset:onset + 3120])
    assert abs(got - _KN4LQN_HZ) <= _CFO_ERROR_HZ, f"{name} @ {onset}: read {got:+.1f} Hz"


# --------------------------------------------------------------------------- #
# Sensitivity — decodes in noise (the link budget), not just rejects it.
# Thresholds sit well inside the measured floors so a sensitivity regression
# trips without run-to-run flakiness. besra/ARDOP SNR convention.
# --------------------------------------------------------------------------- #

from hfmodem.besra.sim.channel import add_awgn, watterson                     # noqa: E402


@pytest.mark.parametrize("snr", [-4, 0, 6])
def test_connect_frame_decodes_in_noise(snr):
    """The 4FSK connect frame is reliable to ~-6 dB; guard at -4 dB. This is why a
    connect reaches almost any audible gateway."""
    clean = M.render_frame(0x32, caller="KC3OWM", target="W9SSJ", session_id=0x5E)
    good = sum(any(f.ok and f.caller == "KC3OWM" and f.target == "W9SSJ"
                   for f in D.decode(add_awgn(clean, snr, seed=s)))
               for s in range(16))
    assert good >= 15, f"connect decode {good}/16 at {snr:+d} dB (besra)"


def test_bw500_data_decodes_in_noise():
    """BW500 4PSK data reliable to ~-2 dB; guard at +2 dB."""
    pl = _pl(0x50)
    clean = M.render_frame(0x50, payload=pl, session_id=0x5E)
    good = sum(any(f.ok and f.payload == pl for f in D.decode(add_awgn(clean, 2, seed=s)))
               for s in range(16))
    assert good >= 15, f"BW500 data decode {good}/16 at +2 dB (besra)"


def test_deep_noise_mints_no_connect():
    """Well below the floor (-16 dB) the connect frame must not decode — sensitivity
    without hallucination."""
    clean = M.render_frame(0x32, caller="KC3OWM", target="W9SSJ", session_id=0x5E)
    minted = sum(any(f.ok for f in D.decode(add_awgn(clean, -16, seed=s))) for s in range(16))
    assert minted <= 1, f"{minted}/16 phantom decodes at -16 dB"


@pytest.mark.parametrize("spread,delay", [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)])
def test_connect_survives_watterson_fading(spread, delay):
    """The connect frame must ride through HF fading — measured 75-100% across
    slow/moderate/poor CCIR-520 profiles. Guard at 50% (well inside) so a fading
    regression trips without flaking. (Data-frame throughput under slow selective
    fading is a separate matter, and one ARQ recovers from.)"""
    clean = M.render_frame(0x32, caller="KC3OWM", target="W9SSJ", session_id=0x5E)
    good = sum(any(f.ok and f.caller == "KC3OWM" for f in
                   D.decode(watterson(clean, 10, spread_hz=spread, delay_ms=delay, seed=s)))
               for s in range(16))
    assert good >= 8, f"connect {good}/16 under Watterson {spread}Hz/{delay}ms"


# --------------------------------------------------------------------------- #
# Foreign waveforms under impairment.
#
# Everything above impairs besra's *own* modulator output, so a receiver quietly
# tuned to its own transmitter would pass it all and still fail on air. These
# drive the same channels with the 59 reference renders instead. Measured with 20
# noise realizations per point over the whole fixture set; the committed points
# use 8 fixed seeds so the suite stays fast and deterministic.
#
# SNR is besra's convention throughout: signal power over the active samples,
# noise across the whole block (``besra.sim.channel``). It is not referenced to a
# 3 kHz noise bandwidth, so these numbers do not compare directly with figures
# quoted that way.
# --------------------------------------------------------------------------- #

# Measured AWGN SNR90 (lowest SNR above which every swept point held >=90%) for
# one representative fixture per mode. The waterfall is the expected shape: the
# 50-baud 4FSK control frames get through at -8 dB, and each step up in
# constellation or bandwidth costs 2-6 dB, out to 16QAM/2000 at +14.
_FOREIGN_SNR90 = {
    "txframe_ConReq500M.wav": -8,
    "txframe_IDFrame.wav": -8,
    "txframe_IDLE.wav": -2,
    "txframe_ConAck2000-t2000.wav": -2,
    "txframe_PingAck-sn12-q80.wav": -2,
    "txframe_4FSK.200.50S.E.wav": -8,
    "txframe_4FSK.500.100.E.wav": -6,
    "txframe_4FSK.2000.600.E.wav": 14,
    "txframe_4FSK.2000.600S.E.wav": 12,
    "txframe_4PSK.200.100.E.wav": -6,
    "txframe_4PSK.500.100.E.wav": -2,
    "txframe_4PSK.1000.100.E.wav": 0,
    "txframe_4PSK.2000.100.E.wav": 4,
    "txframe_8PSK.200.100.E.wav": 0,
    "txframe_8PSK.500.100.E.wav": 4,
    "txframe_8PSK.1000.100.E.wav": 8,
    "txframe_8PSK.2000.100.E.wav": 12,
    "txframe_16QAM.200.100.E.wav": 0,
    "txframe_16QAM.500.100.E.wav": 6,
    "txframe_16QAM.1000.100.E.wav": 10,
    "txframe_16QAM.2000.100.E.wav": 14,
}
# Two SNR steps of guard: every mode measured 8/8 at SNR90+2, so +4 leaves a full
# step of headroom before a sensitivity regression is called.
_GUARD_DB = 4


@gt.requires_reference
@pytest.mark.parametrize("name,snr90", sorted(_FOREIGN_SNR90.items()))
def test_reference_fixture_decodes_in_noise(name, snr90):
    entry = gt.txframe_manifest()[name]
    au = gt.load_wav(name)
    good = sum(any(gt.matches(f, entry)
                   for f in D.decode(_framed(add_awgn(au, snr90 + _GUARD_DB, seed=s))))
               for s in range(8))
    assert good == 8, f"{name}: {good}/8 at {snr90 + _GUARD_DB:+d} dB"


# Modes straddling their decode knee, where a receiver biased toward its own
# transmitter would show up as a success-rate gap. Over the full 27-point sweep
# the two tracked to within one trial in 1080; on these fixed points they are
# identical, so the assertion is equality.
_KNEE = {
    "txframe_ConReq2000M.wav": (-12, -10),
    "txframe_4FSK.2000.600.E.wav": (4, 6),
    "txframe_8PSK.1000.100.E.wav": (4, 6),
    "txframe_16QAM.2000.100.E.wav": (12, 14),
}


@gt.requires_reference
@pytest.mark.parametrize("name,snrs", sorted(_KNEE.items()))
def test_foreign_and_own_waveforms_decode_alike(name, snrs):
    """besra's demodulator is no kinder to besra's own transmitter than to the
    reference's — the seam every other test in this file leaves untouched."""
    entry = gt.txframe_manifest()[name]
    own = M.render_frame(entry["type"], payload=entry.get("payload", b""),
                         session_id=entry["session_id"], **entry.get("meta", {}))

    def score(au):
        return sum(any(gt.matches(f, entry)
                       for f in D.decode(_framed(add_awgn(au, snr, seed=s))))
                   for snr in snrs for s in range(8))

    foreign, mine = score(gt.load_wav(name)), score(own)
    assert foreign == mine, f"{name}: foreign {foreign}/16 vs own {mine}/16"


# A real rig is routinely tens of Hz off frequency, and the fixtures are foreign
# renders with no shared timing or phase reference. Measured noiseless over all 59:
# every one decodes across the whole ±200 Hz spec window and none at ±250, so the
# capture range is exactly the window the leader estimator searches.
def _shift(au: np.ndarray, hz: float) -> np.ndarray:
    # Half scale first: the fixtures peak at 32700 and the analytic shift overshoots
    # to ~43k, which would clip on the way back through int16.
    return _tune(au.astype(float) * 0.5, hz)


@gt.requires_reference
@pytest.mark.parametrize("hz", [-200, -10, 10, 200])
def test_reference_fixtures_survive_cfo(hz):
    manifest = gt.txframe_manifest()
    for name in gt.wav_names():
        frames = D.decode(_framed(_shift(gt.load_wav(name), hz)))
        assert any(gt.matches(f, manifest[name]) for f in frames), f"{name} @ {hz:+d} Hz"


@gt.requires_reference
@pytest.mark.parametrize("hz", [-250, 250])
def test_reference_fixtures_lost_beyond_capture_range(hz):
    """The documented edge: past ±200 Hz the leader estimate falls outside the
    search and nothing decodes — silence, not a mis-decode."""
    manifest = gt.txframe_manifest()
    for name in sorted(_FOREIGN_SNR90):
        frames = D.decode(_framed(_shift(gt.load_wav(name), hz)))
        assert not any(gt.matches(f, manifest[name]) for f in frames), f"{name} @ {hz:+d} Hz"


# Watterson 1 Hz spread / 2 ms delay on the foreign renders. The 2 ms delay puts
# the coherence bandwidth near 500 Hz, so this is flat fading for the 200 Hz modes
# and frequency-selective for the wide ones; besra, like the reference, carries no
# equalizer. Measured over 20 realizations: the fixtures below clear their SNR at
# 100%, while 16QAM/500 and up sit on a selective-fading error floor (5-20%) that
# no amount of SNR lifts. That floor is the waveform's, not a decoder defect, and
# ARQ is what recovers it — so it is recorded here rather than asserted.
_FADE_SNR = {
    "txframe_ConReq500M.wav": 6,
    "txframe_IDFrame.wav": 6,
    "txframe_IDLE.wav": 6,
    "txframe_4FSK.200.50S.E.wav": 6,
    "txframe_4PSK.200.100.E.wav": 6,
    "txframe_4PSK.500.100.E.wav": 10,
}


@gt.requires_reference
@pytest.mark.parametrize("name,snr", sorted(_FADE_SNR.items()))
def test_reference_fixture_survives_fading(name, snr):
    entry = gt.txframe_manifest()[name]
    au = gt.load_wav(name)
    good = sum(any(gt.matches(f, entry) for f in D.decode(_framed(
        watterson(au, snr, spread_hz=1.0, delay_ms=2.0, seed=s))))
        for s in range(8))
    # Measured 8/8; one bad fade draw is tolerated, two is a regression.
    assert good >= 7, f"{name}: {good}/8 under Watterson at {snr:+d} dB"


def test_render_handles_every_frame_type():
    """The live monitor renders whatever it decodes; a format/dispatch bug on any
    one of the 121 frame types would crash it mid-session. Every type must render
    to a non-empty line."""
    for ftype, fd in F.FRAMES.items():
        frame = D.DecodedFrame(type=ftype, session_id=0x5E, ok=True, name=fd.name,
                               caller="W9SSJ", target="K7ABC", grid="EN63", payload=b"test")
        line = monitor._render(frame)
        assert isinstance(line, str) and line, f"_render failed for {fd.name} ({hex(ftype)})"


# --------------------------------------------------------------------------- #
# FIX 5 — clipped-leader acquisition (the ARQ reply path).
# --------------------------------------------------------------------------- #
#
# An ARQ reply's leader arrives while this station's input is still recovering
# from its own transmission, so its front is routinely lost. Measured on air
# (logs/onair/20260802T233047Z-besra-7102000.wav): KE8LVA answered eight of
# eleven ConReqs with a ConAck2000 whose session ID is exactly
# crc8("W9SSJ"+"KE8LVA") and whose timing bytes read 220-230 ms of received
# leader — and the pre-fix receiver, which pinned the header at exactly one full
# leader after the visible leader edge, decoded none of them, because ~80 ms of
# each leader fell inside the recovery window. A leader clipped by N ms carries
# its header N ms *earlier* than the full-leader geometry claims; the receiver
# must find it anyway, for every frame class a peer can answer with.

_CLIP_MS = [40, 120, 200]


@pytest.mark.parametrize("clip_ms", _CLIP_MS)
def test_clipped_leader_conack_decodes(clip_ms):
    au = M.render_frame(0x3C, session_id=0x51, timing=230)
    got = D.decode(_framed(au[12 * clip_ms:]))
    assert [(f.name, f.ok, f.conack_timing_ms) for f in got] == \
        [("ConAck2000", True, 230)]


@pytest.mark.parametrize("clip_ms", _CLIP_MS)
def test_clipped_leader_conreq_decodes(clip_ms):
    au = M.render_frame(0x32, caller="W9SSJ", target="KE8LVA", session_id=0xFF)
    got = D.decode(_framed(au[12 * clip_ms:]))
    assert [(f.name, f.ok, f.caller, f.target) for f in got] == \
        [("ConReq500M", True, "W9SSJ", "KE8LVA")]


@pytest.mark.parametrize("clip_ms", _CLIP_MS)
def test_clipped_leader_data_frame_decodes(clip_ms):
    au = M.render_frame(0x4A, payload=_pl(0x4A), session_id=0x51)
    got = D.decode(_framed(au[12 * clip_ms:]))
    assert [(f.name, f.ok, f.payload) for f in got] == \
        [("4FSK.500.100.E", True, _pl(0x4A))]


@pytest.mark.parametrize("clip_ms", _CLIP_MS)
def test_clipped_leader_dataack_decodes(clip_ms):
    """A bare control has no body to validate, so its clipped-leader acceptance
    rides `_leader_backed` — the surviving leader must abut the header."""
    au = M.render_frame(0xE9, session_id=0x51)
    got = D.decode(_framed(au[12 * clip_ms:]))
    assert [(f.name, f.ok) for f in got] == [("DATAACK", True)]


def test_leader_presence_reads_the_same_off_int16_as_off_floats():
    """`detect.leader_presence` squares its input for the window power, and that is
    the one place in that module with no complex phasor to widen an int16 capture
    first. Left to wrap, ``x*x`` collapses the denominator, the ``1e-9`` clamp turns
    what remains into division by nothing, and the ratio comes back around 1e21 —
    maximal confidence, four thousand times over the 0.25 the acquisition walk
    triggers on, for a capture of pure noise. `Demodulator.decode` casts before it
    gets here, so production was safe by accident; a WAV read straight off disk was
    not, and nothing said so."""
    frame = M.render_frame(0x4A, payload=_pl(0x4A), session_id=1).astype("<i2")
    noise = np.round(np.random.default_rng(0).normal(0, 6000, 24000)).astype("<i2")

    for name, au in (("frame", frame), ("noise", noise)):
        assert np.allclose(detect.leader_presence(au),
                           detect.leader_presence(au.astype(np.float64))), name

    assert detect.leader_presence(frame).max() <= 1.0     # it is a ratio
    assert detect.leader_presence(noise).max() < 0.25     # noise is not a leader


def test_cut_body_mints_no_phantom_controls():
    """The counterexample the widened search must survive: a frame whose body is
    cut (a window edge, in the live path) leaves 10-symbol runs of crisp body
    tones that can spell a valid bare-control header. Without the leader-abutment
    gate this minted DATAACK/DATANAK phantoms; the only acceptable reading is the
    cut frame itself, unvalidated."""
    con = M.render_frame(0x32, caller="W9SSJ", target="KE8LVA", session_id=0xFF)
    got = D.decode(np.concatenate([_LEAD, con[:3 * con.size // 5]]))
    assert [(f.name, f.ok) for f in got] == [("ConReq500M", False)]


# The recording that found the defect, kept as its regression: three connect
# attempts to KE8LVA, eight ConAck2000 answers on tape, four of them clean
# enough to read symbol-by-symbol (the other four are channel-mangled past any
# alignment — no header decodes at any shift). Station-local audio, so this
# skips where the log is absent.
_KE8LVA_WAV = evidence.LOGS / "onair" / "20260802T233047Z-besra-7102000.wav"
_KE8LVA_TX_ENDS = [11.35, 14.87, 18.70, 22.53, 26.38, 30.11,
                   33.93, 37.76, 41.57, 45.40, 49.24]


@pytest.mark.skipif(not _KE8LVA_WAV.exists(), reason="station log not present")
def test_offair_ke8lva_conacks_decode():
    import wave
    from hfmodem.core.resample import from_card

    with wave.open(str(_KE8LVA_WAV)) as w:
        fs = w.getframerate()
        au = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    grants = []
    for te in _KE8LVA_TX_ENDS:
        seg = from_card(au[int((te - 0.35) * fs):int((te + 1.6) * fs)], SR)
        got = [f for f in D.decode(np.concatenate([seg, _TAIL])) if f.ok]
        assert all(f.name == "ConAck2000" and f.session_id == 0x51 for f in got)
        grants += got
    # Measured: cycles 3, 4, 7 and 8 validate; more is improvement, fewer is loss.
    assert len(grants) >= 4
    assert all(f.conack_timing_ms in (220, 230) for f in grants)
