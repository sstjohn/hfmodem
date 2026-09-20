# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What each chain's working point is, and whether a capture is usable.

The measured working point at this station — an FT-891 into a C-Media dongle with
the rig's own DATA OUT at 45/100 — is a codec input scalar of **0.040**, which
holds real off-air traffic at −4 to −8 dBFS peak with nothing railed.

That number is here because it was wrong in four places at once — 0.51, 0.118,
0.50 and 0.18 — and one of those re-set the wrong value during a session
handover. Across 617 captures taken under the old settings, 39% peaked at 0 dBFS,
12% clipped more than 1% of their samples, and the worst had 17.8% of its samples
against the rail. **A hot input does not present as an error. It presents as a
working radio that decodes nothing**, which is why it survived so long.

Three rules came out of that, and only the first is code:

  1. Fixing the live setting is not fixing the problem — change the constant
     anywhere a script sets gain on handover, arming or session start.
  2. Verify after setting, every session. `peak_dbfs` and `railed_ppm` are what
     preflight measures; a configuration bound cannot substitute for measuring.
  3. Treat a hot input as a first-class failure mode rather than a warning.

Two peak figures, because one would be wrong. `PEAK_AIM_DBFS` is where the
strongest expected signal should sit; `PEAK_LIMIT_DBFS` is where the headroom has
actually gone. A working point measured at −4 to −8 means a reading of −5 is
inside the spread of a setting that works, and a criterion calling that a fault
is a criterion nobody will keep.

Note the rig's own DATA OUT menu is a second gain stage in series that no
software here can see. It is exactly how two careful people measure the same
chain and get different answers, and both are right.

The transmit side is the same shape of number and had the same shape of bug, so
it lives here beside it: one drive, set once against the meter, and every keyed
burst normalised to it before it reaches the card.
"""
from __future__ import annotations

import numpy as np

from hfmodem.core.busy import occupancy_db

#: The measured codec input scalar for this station's chain.
WORKING_POINT = 0.040

#: The peak every keyed burst leaves at, whatever composed it. On a data
#: interface the audio level is the RF power, so this — not the rig's power
#: setting — is what decides how much goes out.
#:
#: One number because there is one interface and one radio: a drive that belongs
#: to a protocol is wrong by construction, and the four here disagreed by 18 dB,
#: so an operator who set the rig's input gain against one was overdriving on the
#: next. Kept below full scale on purpose — past its ALC an SSB transmitter
#: distorts and splatters, and the station's own measurement is that a steady
#: carrier at 0.6 already reaches this rig's ceiling. Raise it against a
#: wattmeter with the ALC watched, not against a link that felt weak.
TX_DRIVE = 0.6

#: |sample| at or above this is against the rail.
RAIL = 0.997

#: Where the strongest expected signal should sit, and where headroom has gone.
PEAK_AIM_DBFS = -6.0
PEAK_LIMIT_DBFS = -3.0

#: The in-band level below which this station can no longer hear itself, in the
#: currency of :func:`hfmodem.core.busy.occupancy_db`. The session-tap note in
#: `tests/kestrel/test_txwitness.py` quotes the same levels per 200 Hz slice,
#: which reads 12.1 dB lower throughout.
#:
#: WHAT SETS IT. Our own transmission returns into the capture through the codec's
#: DAC-to-ADC crosstalk, and every keyed second of every 2026-08-22 session tap
#: reads -22.3 dBFS of it -- the same figure to a tenth on taps whose receivers
#: are 16 dB apart, because the coupling is electrical and not a setting. With the
#: receiver muted the deepest 200 Hz slice of that return sits at -60.9 to -61.6
#: dBFS, derived twice: `onair-0822-0922` listening at -39.6 sees the slice fall
#: 22.0 dB, and `-0929` listening at -23.5 sees it fall 37.4, on receivers 16 dB
#: apart. The half-duplex mute wants 26 dB of fall, so under -34.9 dBFS -- the
#: stricter of the two -- a receiver cannot produce one at all.
#:
#: The taps say the same thing without the arithmetic: `-0922` and `-0924`
#: listened at -39.6 and -40.1 and read back none of their thirty-odd keyings,
#: minutes before `-0929`, `-0933`, `-0936` and `-0938` listened at -23.5 to
#: -24.3 on the same rig and read all of theirs. Nothing has been recorded
#: between those two populations, and this bound lies in the gap.
#:
#: IT IS A LEVEL AT ONE CAPTURE GAIN, which is a hardware setting on the dongle
#: rather than :data:`WORKING_POINT`: on the 2026-08-19 and -20 taps the same
#: crosstalk reads -37.3 dBFS and every level above moves with it. Re-gain the
#: capture side and both figures have to be measured again.
RX_FLOOR_DBFS = -34.9

#: The broadband RMS at which this chain starts putting samples on the rail. A
#: VARA session on 2026-08-23 climbed 0.055 to 0.169 to 0.283 and stayed clean;
#: 0.327 on 2026-08-22 railed 16% of its samples. The session taps fill in
#: between -- 0.274 rails 1.0%, 0.290 rails 1.3%, 0.364 rails 3.9% and 0.571
#: rails 12.1% -- so the bound sits between the highest clean reading and the
#: lowest railing one, where every threshold in `core.busy` sits.
RX_HOT_RMS = 0.30

#: Below this the codec is delivering nothing. A dead one reads about 0.00007
#: against a live receiver's 0.05 to 0.20, and the decade between the two is a
#: codec delivering quiet audio -- a different fault, at the other end of the
#: station, and naming the wrong one sends the operator to the keying line when
#: the answer was the radio's own gain.
RX_SILENT_RMS = 0.001

#: What that reading means, in one place, so the launcher's teardown and its
#: pre-flight say it the same way.
NO_AUDIO_CAUSE = ("at this level the codec is not delivering samples at all. The "
                  "interface is unplugged or asleep, or a transmitter is holding "
                  "the receiver muted.")

#: The knob that reaches this converter, and the one that does not. `AF` drives
#: the speaker output, which on this station is the separate `KT USB Audio` tap;
#: the modems are fed from `USB Audio Device` off the data port, and an operator
#: sent to `AF` for a level on this feed spent a recalibration on 2026-08-22
#: finding that out.
_CAPTURE_GAIN = ("the modem's own capture device (tools/codec_gain.py --device "
                 "<name> --set, or [audio] input_gain in the station file), NOT "
                 "the rig's AF, which drives the speaker tap and not this feed")

#: What to turn when the band is quiet, in the order that has worked. RF gain
#: leads because it is the only one of these that is per band: on 2026-08-20 a
#: 20 m arm read 0.0046 RMS against the 0.055-0.065 the same codec setting gave
#: on 40 m, and raising RF gain recovered 18 dB and took the connect candidates
#: from 1 to 5. Raising the capture gain to lift 20 m would have overdriven 40 m,
#: which is why a deficit on ONE band is never the codec's. The wide-open case is
#: last because it is the one an operator most needs to recognise fast: there is
#: no control left, and hunting for one costs the slot.
QUIET_CONTROLS = (
    "Raise the rig's RF gain first (front panel, or hamlib's `RF`) -- it is the "
    "only per-band control here, and a band down on its own is RF gain or the "
    "antenna, never the codec, whose gain is shared and would overdrive the bands "
    "that read right. Then the rig's DATA OUT level, or the gain on "
    f"{_CAPTURE_GAIN}. RF gain already wide open and still this quiet: the "
    "antenna, or a band that is simply closed, and no control fixes either.")


def at_drive(audio: np.ndarray, drive: float = TX_DRIVE) -> np.ndarray:
    """Every burst out at the same peak, whatever the protocol composed in.

    Normalised rather than merely limited. The renderers pick amplitudes to suit
    an offline decode from a file — the PACTOR-1 connect comes out at peak 0.11
    and sabir hands over 1.415 — so a limiter alone leaves the two protocols 22 dB
    apart with only one of them bounded. Peak is the right measure because
    clipping is what the radio punishes: ALC on a squared-off OFDM peak splatters
    across the band.
    """
    a = np.asarray(audio, dtype=np.float64)
    peak = float(np.abs(a).max()) if a.size else 0.0
    return (a if peak == 0.0 else a * (drive / peak)).astype(np.float32)


def peak_dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS. `-inf` for digital silence, which is a real reading —
    a capture that stalls returns exactly this and it must not look like a
    quiet band."""
    a = np.asarray(audio, float)
    peak = float(np.abs(a).max()) if a.size else 0.0
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def railed_ppm(audio: np.ndarray) -> float:
    """Parts per million of samples against full scale.

    Measured against full scale rather than against the capture's own peak. A
    normalising WAV writer destroys exactly this measurement: 9.60% railed off
    the codec reads 0.00% from a file scaled to 0.8 peak.
    """
    a = np.asarray(audio, float)
    if not a.size:
        return 0.0
    return 1e6 * float(np.mean(np.abs(a) >= RAIL))


def usable(audio: np.ndarray) -> tuple[bool, str]:
    """Whether a capture is worth decoding, and why not if not.

    Returns a reason rather than a bare False because every symptom of a hot
    input looks like a radio fault, and the operator needs to be told which.
    """
    p, r = peak_dbfs(audio), railed_ppm(audio)
    if p == float("-inf"):
        return False, "digital silence — the capture stalled or the device is wrong"
    if r > 0:
        return False, (f"{r/1e4:.2f}% of samples against the rail at {p:.1f} dBFS — "
                       f"the input is hot; the measured working point is {WORKING_POINT}")
    if p > PEAK_LIMIT_DBFS:
        return False, (f"peak {p:.1f} dBFS is above {PEAK_LIMIT_DBFS} — headroom has "
                       f"gone; aim for {PEAK_AIM_DBFS}")
    return True, f"peak {p:.1f} dBFS, nothing railed"


def rx_verdict(audio: np.ndarray, fs: int = 48000) -> tuple[str, str]:
    """What the receiver is delivering, before anything is keyed against it.

    Returns ``("live" | "quiet" | "hot" | "silent", one line for the operator)``.

    Four states rather than a usable/not, because the three failures are told
    apart by the reading and answered at different ends of the station. Silence is
    the codec, and no gain fixes it. Hot is the capture gain, and the answer to the
    call arrives clipped. Quiet is the one that is not a single control: the arm
    will transmit into a recording nothing can be read out of, and what fixes that
    is RF gain, the antenna, or nothing at all if the band is closed.

    The level is the loudest second in the window: a receiver that stands up once
    is a receiver, and what this refuses is a chain delivering nothing but its own
    floor. Quiet is asked last of the three, so a window loud enough to rail is
    never reported as a quiet one.
    """
    a = np.asarray(audio, float).ravel()
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    if rms < RX_SILENT_RMS:
        return "silent", f"receiver rms {rms:.5f} -> NO AUDIO: {NO_AUDIO_CAUSE}"
    # ANY sample on the rail, because the level at which this station stops
    # decoding barely moves the railed fraction. 2026-08-29 flew one arm twice
    # three minutes apart with only the capture gain changed: at 0.209 every 2 s
    # of the session had samples against the rail -- 0.001% to 0.056% of them --
    # and it read 3 control signals; at 0.099 none railed and it read 19. The 1%
    # this carried, off the 617 captures behind WORKING_POINT, passed the
    # clipping arm on all 59 of its windows. The cost of the zero is measured
    # over 748 sense-length windows of nine arms that went on to decode: one
    # refuses, a 0.2 s static crash railing 0.094% -- harder than the clipping
    # arm ever did, so no allowance keeps that arm and still catches this one.
    pct = railed_ppm(a) / 1e4
    if pct > 0 or rms > RX_HOT_RMS:
        return "hot", (f"receiver rms {rms:.3f}, {pct:.3f}% of samples on the rail "
                       f"-> CLIPPING: the far end's answer would arrive clipped and "
                       f"undecodable. Lower the gain on {_CAPTURE_GAIN}.")
    db = occupancy_db(a, fs)
    level = float(db.max()) if db.size else float("-inf")
    if level < RX_FLOOR_DBFS:
        return "quiet", (f"receiver {level:.1f} dBFS in band -> TOO QUIET: "
                         f"{RX_FLOOR_DBFS - level:.1f} dB under the {RX_FLOOR_DBFS} "
                         f"dBFS at which this chain still hears its own transmission "
                         f"come back, so nothing keyed from here can be read out of the "
                         f"recording. {QUIET_CONTROLS}")
    return "live", (f"receiver {level:.1f} dBFS in band, rms {rms:.3f}, nothing "
                    f"railed -> LIVE")
