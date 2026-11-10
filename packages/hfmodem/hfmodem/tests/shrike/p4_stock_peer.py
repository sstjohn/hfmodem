# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stock peer audio, independent of this station's transmit renderer."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from hfmodem.shrike import p4breakintx, p4frame, p4rx, p4shorttx, p4sig, p4words


def render(chips, fs, cfo):
    chips = chips * np.exp(2j*np.pi*cfo*np.arange(len(chips))/p4rx.CHIP_RATE)
    impulses = np.zeros(p4rx.SAMPLES_PER_CHIP*(len(chips)-1)+1, complex)
    impulses[::p4rx.SAMPLES_PER_CHIP] = chips
    g = gcd(fs, p4rx.SAMPLE_RATE)
    bb = resample_poly(np.convolve(impulses, p4rx.RRC_TAPS), fs//g, p4rx.SAMPLE_RATE//g)
    audio = (bb*np.exp(2j*np.pi*p4rx.CENTRE_HZ*np.arange(len(bb))/fs)).real
    return audio/np.max(abs(audio))


def word_chips(name):
    chips = (p4words.BANK[name][:, None]*p4sig.SPREAD16).ravel()
    return chips*np.exp(2j*np.pi*p4words.CARRIER_HZ.get(name, 0.)*np.arange(len(chips))/p4rx.CHIP_RATE)


def word(name, fs=48000, *, cfo=0.):
    return render(word_chips(name), fs, cfo)


def body(payload, status, fs=48000, *, cfo=0.):
    field = p4frame.encode_field(payload[:31].ljust(31, b'\x1e'), p4frame.Status.from_byte(status))
    chips = np.r_[word_chips('Y27')*p4breakintx.WORD_GAIN,
                  p4breakintx.body_symbols(p4breakintx.coded_bits(field))]
    return render(chips, fs, cfo)


def normal(payload, status, fs=48000, *, cfo=0.):
    field = p4frame.encode_field(payload[:40].ljust(40, b'\x1e'), p4frame.Status.from_byte(status))
    chips = np.r_[p4shorttx.HEADER, p4shorttx.body_symbols(p4shorttx.coded_bits(field))]
    return render(chips, fs, cfo)
