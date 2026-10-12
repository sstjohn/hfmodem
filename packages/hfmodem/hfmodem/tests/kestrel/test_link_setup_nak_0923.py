# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K0SI 80 m and KC9GHZ 40 m, 2026-09-23: the link-setup NAK and the lower offer.

289/31 behind our link-setup is the peer's CRC-reject, not a connect confirmation:
it is scored soft, so a co-channel carrier cannot hide it, and the link-setup goes
again at once. A lower-level offer is read as soon as its payload has ended.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.tests.evidence import require_input
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

FS = MK.FS
FIXTURES = Path(__file__).with_name('fixtures') / 'link-setup-nak-0923'
# Where each peer's offer on the same arm put it, as the handshake measured it.
OFFSET_HZ = {'K0SI': -4.72, 'KC9GHZ': 2.21, 'KD7UHR': 3.73}


class IO:
    def __init__(self):
        self.msgs, self.sent = [], []

    def key(self, on): ...
    def tx(self, samples): self.sent.append(len(samples))
    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): ...
    def log(self, msg): self.msgs.append(msg)


def clip(name):
    path = FIXTURES / f'{name}.wav'
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == FS and x.dtype == np.int16
    return x.astype(float) / 32768


def station(called, bw, step=VA._I_LINKSETUP_SENT, offset=True):
    io = IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw=bw)
    hs.role, hs.called, hs.caller = 'initiator', called, 'W9SSJ'
    hs.state, hs.step = VA.VaraState.CONNECTING, step
    if step == VA._I_LINKSETUP_SENT:
        hs._linksetup_tx = 1
    if offset:
        hs._peer_offset = OFFSET_HZ[called] / (FS / MK.NFFT)
    return hs, io


def drive(hs, io, x, block=960):
    """Receive time of the first keying, fed as the transport polls."""
    for at in range(0, len(x), block):
        hs.on_rx_stream(x[at:at + block])
        if io.sent or hs.state != VA.VaraState.CONNECTING:
            return (at + block) / FS
    return None


def nak_end(x, called, bw, offset):
    """Where the NAK's last symbol ends in ``x``, off its soft fit."""
    band = MK.band_for(bw)
    kind = VF.for_bw(VF.SESSION_LINK_SETUP_NAK, bw)
    tones = [(t, t) for t in VF.payload_bins(called, kind)]
    _, at = VA._frame_fit(VA._band_power(x, band, offset / (FS / MK.NFFT)), tones, band)
    return (at + (kind.n_payload - 1) * MK.HOP + MK.STRIDE) / FS


@pytest.mark.parametrize('name,called,bw', [('k0si-nak', 'K0SI', '2300'),
                                            ('kc9ghz-nak-under-carrier', 'KC9GHZ', '2750')])
@pytest.mark.parametrize('offset', [True, False])
@pytest.mark.parametrize('block', [960, 4800])
def test_the_nak_resends_the_link_setup_at_once(name, called, bw, offset, block):
    x = clip(name)
    hs, io = station(called, bw, offset=offset)
    hs._setup_level = 4
    keyed = drive(hs, io, x, block)
    assert keyed is not None, io.msgs
    assert hs.state == VA.VaraState.CONNECTING and hs.step == VA._I_LINKSETUP_SENT
    assert len(io.sent) == 1 and hs._linksetup_tx == 2
    assert not any('link is up' in m for m in io.msgs)
    assert any(m.startswith(f'rx link-setup NAK (289/31) from {called}') for m in io.msgs)
    assert any('level 4' in m and 'unverified' in m for m in io.msgs), io.msgs
    end = nak_end(x, called, bw, OFFSET_HZ[called])
    assert end + MK.HOP / FS <= keyed <= end + 0.3, (end, keyed)


def test_the_carrier_hides_the_nak_from_an_argmax():
    x = clip('kc9ghz-nak-under-carrier')
    kind = VF.for_bw(VF.SESSION_LINK_SETUP_NAK, '2750')
    want = VF.payload_bins('KC9GHZ', kind)
    heard, _ = VA._payload_fit(x, kind, np.asarray(VF.handshake_tones('KC9GHZ', kind)),
                               band=MK.band_for('2750'))
    assert sum(int(h) == w for h, w in zip(heard[1:], want)) < VA._RESP_MIN_TONES


def test_a_synthetic_carrier_at_bin_22_does_not_hide_it():
    kind = VF.for_bw(VF.SESSION_LINK_SETUP_NAK, '2750')
    burst = MK.synth_burst('KC9GHZ', kind, amplitude=0.1)
    x = np.concatenate([np.zeros(FS // 10), burst, np.zeros(FS // 2)])
    x += 0.3 * np.sin(2 * np.pi * 22 * FS / MK.NFFT * np.arange(len(x)) / FS)
    x += np.random.default_rng(23).normal(0, 0.01, len(x))
    hs, io = station('KC9GHZ', '2750', offset=False)
    assert drive(hs, io, x) is not None
    assert hs._linksetup_tx == 2 and hs.state == VA.VaraState.CONNECTING


def test_a_spent_budget_reads_the_nak_and_keys_nothing():
    hs, io = station('K0SI', '2300')
    hs._linksetup_tx = hs.max_link_setups
    drive(hs, io, clip('k0si-nak'))
    assert not io.sent and hs.state == VA.VaraState.CONNECTING
    assert any('budget spent' in m for m in io.msgs)


@pytest.mark.parametrize('called,bw', [('N0XYZ', '2300'), ('KC9GHZ', '2750')])
def test_another_stations_nak_is_not_read(called, bw):
    hs, io = station('K0SI', bw, offset=False)
    hs.called = called
    drive(hs, io, clip('k0si-nak'))
    assert not io.sent and not any('NAK' in m for m in io.msgs)


@pytest.mark.parametrize('name,called,bw', [('kc9ghz-busy', 'KC9GHZ', '2750'),
                                            ('k0si-level1-offer', 'K0SI', '2300')])
def test_a_busy_band_and_an_offer_are_not_a_nak(name, called, bw):
    hs, io = station(called, bw)
    hs._linksetup_tx = hs.max_link_setups       # a repeated offer keys nothing either
    drive(hs, io, clip(name))
    assert not io.sent and hs.state == VA.VaraState.CONNECTING
    assert not any('NAK' in m for m in io.msgs)


def test_noise_is_not_a_nak():
    x = np.random.default_rng(9).normal(0, 0.05, 2 * FS)
    for bw in ('2300', '2750'):
        hs, io = station('K0SI', bw, offset=False)
        drive(hs, io, x)
        assert not io.sent and not any('NAK' in m for m in io.msgs)


def test_a_connected_ack_still_brings_the_link_up():
    hs, io = station('KD7UHR', '2750')
    drive(hs, io, clip('kd7uhr-connected-ack'))
    assert hs.state == VA.VaraState.CONNECTED
    assert any('connected-ack' in m for m in io.msgs)
    assert not any('NAK' in m for m in io.msgs)


def test_the_nak_is_not_read_before_a_link_setup():
    hs, io = station('K0SI', '2300', step=VA._I_CR_SENT)
    drive(hs, io, clip('k0si-nak'))
    assert not io.sent and not any('NAK' in m for m in io.msgs)


@pytest.mark.parametrize('block', [960, 4800])
def test_a_level_one_offer_keys_the_link_setup_at_once(block):
    x = clip('k0si-level1-offer')
    hs, io = station('K0SI', '2300', step=VA._I_CR_SENT, offset=False)
    keyed = drive(hs, io, x, block)
    assert keyed is not None and hs.step == VA._I_LINKSETUP_SENT and hs._setup_level == 1
    kind = VF.connect_response('2300', 1)
    band = MK.band_for('2300')
    tones = [(t, t) for t in VF.handshake_tones('K0SI', kind)]
    _, at = VA._frame_fit(VA._band_power(x, band), tones, band)
    end = (at + (len(tones) - 1) * MK.HOP + MK.STRIDE) / FS
    assert end <= keyed <= end + 0.25, (end, keyed)


def test_provenance_matches_the_fixtures():
    prov = json.loads(require_input(FIXTURES / 'provenance.json').read_text())
    for name, facts in prov.items():
        fs, x = wavfile.read(require_input(FIXTURES / name))
        assert fs == facts['sample_rate'] and len(x) == facts['samples']
