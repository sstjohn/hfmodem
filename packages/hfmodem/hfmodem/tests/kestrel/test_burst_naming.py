# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A burst the handshake never matched must never be named in the log.

``on_rx_audio`` sizes a bracket it could not preamble-lock and looks the symbol
count up in ``_NSYM``. That lookup is a duration, not an identification: 16
symbols is any bracket 0.662-0.705 s long, 17 is 0.705-0.747, 41 is 1.729-1.771.
Every branch of ``on_rx_tones`` earns the name with ``VF.recognize`` before acting
on it — the fall-through did not, and reported the guess.

On 2026-08-06 this station called KD9USW eight times on 7101.8 kHz and the log
carried, once::

    rx session-confirm unexpected in state=CONNECTING role=initiator
                       step=I_CR_SENT — ignored

Scanned afterwards, all 70.7 s of that recording holds no VARA frame but our own
eight connect-requests (31/31 payload tones each, at every one of the eight
positions). The best score anywhere for a connect-response or a session-confirm
keyed to KD9USW is 4 of 15 — chance is 0.43 — at any tuning offset within
+-470 Hz. What arrived at 0.70 s was a QRN swell 4.2 dB over the tracked floor
whose spectrum holds no tone at all (peak-to-mean 3-5 across the burst, against
39.8 median inside our own transmissions). Replaying the bracket segmenter over
that recording at 50 restart phases gives 241 brackets, of which 2 size as a
named frame — so the name is rare, and it is still a name for band noise.

A previous instance of exactly this was patched by teaching the connect tool to
suppress the one line it had seen (``session-disconnect-request unexpected``).
This is the same defect one bin along, so the fix belongs where the name is made.
"""
from __future__ import annotations

import numpy as np

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

FS = MK.FS
_CALLED = "KD9USW"
_MYCALL = "W9SSJ"

#: Every symbol count ``_NSYM`` answers to, and a bracket length that hits it.
_NAMED = {n: (n - 1) * MK.HOP + MK.STRIDE for n in VA._NSYM}


class _IO(VA.VaraIO):
    """Collects the log. Nothing here opens a device or reaches a transmitter."""

    def __init__(self):
        self.msgs: list[str] = []
        self.keys = 0

    def key(self, on): self.keys += bool(on)
    def tx(self, samples): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): self.msgs.append(msg)


def _calling() -> tuple[VA.VaraStationHandshake, _IO]:
    """An initiator that has sent its connect-request and is waiting for an answer
    — the state the KD9USW attempt was in for all eight of its cycles."""
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io)
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_CR_SENT
    return hs, io


def _log_of(audio) -> tuple[list[str], int]:
    """What one bracket of audio makes the handshake say, and whether it keyed."""
    hs, io = _calling()
    hs.on_rx_audio(np.asarray(audio, dtype=float))
    return io.msgs, io.keys


def _noise(n_sym: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed * 100 + n_sym).normal(0.0, 0.05, _NAMED[n_sym])


def test_band_noise_is_never_reported_as_a_frame():
    """Noise cut to each named burst length, in the state the attempt was in.

    Held against what the same handshake says for a *genuine* burst of that kind,
    rather than against the wording of the line the defect produced. The claim is
    that band noise cannot be reported the way an arriving frame is, and two logs
    with nothing in common is that claim with no prose in it — reword the log and
    it still holds, delete the tone test behind it and the two logs converge.
    """
    for n_sym, kind in sorted(VA._NSYM.items()):
        named, _ = _log_of(MK.synth_burst(_CALLED, kind))
        for seed in range(24):
            msgs, keys = _log_of(_noise(n_sym, seed))
            assert keys == 0, f"{n_sym}-symbol noise bracket keyed the transmitter"
            shared = sorted(set(msgs) & set(named))
            assert not shared, (
                f"{n_sym}-symbol band-noise bracket (seed {seed}) was logged exactly "
                f"as a real {kind.name} is: {shared}")


def test_a_real_frame_in_the_wrong_state_is_still_named():
    """The gate is a tone test, not a blanket silencer.

    A keepalive keyed to the station we dialled has no business arriving before
    the connect completes, and that is worth a line naming it — a line noise of the
    same length never draws.
    """
    kind = VF.SESSION_KEEPALIVE_A
    n_sym = len(kind.preamble) + kind.n_payload
    named, _ = _log_of(MK.synth_burst(_CALLED, kind))
    noise, _ = _log_of(_noise(n_sym, 0))
    told_apart = set(named) - set(noise)
    assert any(kind.name in m for m in told_apart), (named, noise)


def test_a_frame_keyed_to_a_stranger_is_not_named():
    """Same burst, someone else's callsign: a real VARA frame, and not ours."""
    named, _ = _log_of(MK.synth_burst(_CALLED, VF.SESSION_KEEPALIVE_A))
    stranger, keys = _log_of(MK.synth_burst("KB9MMT", VF.SESSION_KEEPALIVE_A))
    assert keys == 0
    assert not set(stranger) & set(named), (stranger, named)


def test_the_live_traffic_line_calls_no_undecoded_burst_an_over():
    """The same defect in the connect tool's log, one duration bin along.

    A live connect passes ``decode_wideband=False`` — the turbo decode is seconds
    per burst and the handshake cannot wait for it — so any bracket over 4.4 s was
    labelled "~N s over", naming a VARA DATA over off a length. On a channel whose
    gate never closes every bracket force-closes at the segmenter's cap and lands
    here: five such lines came off 71 s of 80 m band noise on 2026-08-06, each of
    them exactly 6.016 s, the cap to three decimal places.
    """
    monitor = corpora.harness("vara_monitor")
    rng = np.random.default_rng(7)
    r = monitor.classify(rng.normal(0.0, 0.05, int(6.016 * FS)), [_CALLED],
                         decode_wideband=False, level_db=6.4)
    said = f"{r.kind} {r.info}".lower()
    for word in ("over", "vara", "session", "gateway"):
        assert word not in said, f"a burst nothing decoded claimed {word!r}: {r}"
    assert "dB" in r.quality, f"reported without a level: {r.quality!r}"
