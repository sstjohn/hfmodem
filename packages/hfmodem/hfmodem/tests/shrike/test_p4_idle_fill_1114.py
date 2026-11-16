# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Our ISS normal shorts against stock: IDLE fill, header A then B.

N9VC's empty 0x40 normal in k0003 (seg-0003, 1488.06 s) carries forty 0x1E on
header A. Stock keys header B on every later copy of a packet: 177 of 178
repeats the gateways keyed to us, against 18 of 19 first copies on A."""
import numpy as np
import pytest

from hfmodem.shrike import onair, p4rx, p4short, p4shorttx, spec
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.test_p4_data_turn_0930 import T, sending
from hfmodem.tests.shrike.test_p4_normal_short_0928 import FS, FX, read_copy, receiver
from hfmodem.tests.shrike.test_p4turn import word

SILENCE = 4800
TRAINING = SILENCE*p4rx.SAMPLE_RATE//48000 + p4shorttx.HEADER_CHIPS*p4rx.SAMPLES_PER_CHIP


@pytest.fixture(scope='module')
def n9vc_idle():
    audio = onair.session.load_wav(str(FX/'n9vc-idle-normal.wav'), FS)
    audio = np.r_[np.tile(audio[:2400], 20), audio]
    r = receiver(round(1.08*FS), 0.)
    frames = [x.observation.frame for x in read_copy(r, audio, round(1.08*FS))
              if x.observation.frame]
    headers = [e['header'] for e in r.live_events if e.get('event') == 'normal-header']
    return frames, headers


def baseband(audio):
    return p4rx.audio_to_baseband(np.r_[np.zeros(SILENCE), audio, np.zeros(SILENCE)], 48000)


def test_our_empty_normal_is_the_one_n9vc_keyed(n9vc_idle):
    (stock,), headers = n9vc_idle
    assert stock.status.byte == 0x40 and stock.data == bytes([spec.IDLE])*40
    assert headers == ['A']
    ours = p4short.decode_body(baseband(p4shorttx.short_packet(b'', 0x40)),
                               first_training_start=TRAINING)
    assert ours == stock


@pytest.mark.parametrize('header', ['A', 'B'])
def test_rendered_header_reads_back_and_the_body_still_decodes(header):
    audio = p4shorttx.short_packet(b'W9SSJ', 1, header=header)
    z = baseband(audio)
    assert p4shorttx.read_header(z, TRAINING, 0.)[0] == header
    frame = p4short.decode_body(z, first_training_start=TRAINING)
    assert frame.data == b'W9SSJ'.ljust(40, bytes([spec.IDLE])) and frame.status.byte == 1


def test_a_repeated_data_normal_keys_header_b():
    m, _ = sending(bytes(range(32, 122)))
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    first = m.keyed()
    assert (first.kind, first.status, first.header) == (Kind.NORMAL, 1, 'A')
    m.observe(word('X27', first.stream_sample+T, responds_to=first.stream_sample),
              now=first.stream_sample+T+12000)
    again = m.decide(first.stream_sample+round(1.25*FS)-m.lead)
    assert (again.kind, again.status, again.payload) == (Kind.NORMAL, 1, first.payload)
    assert again.header == 'B'
    m.keyed()
    m.observe(word('V27', again.stream_sample+T, responds_to=again.stream_sample),
              now=again.stream_sample+T+12000)
    nxt = m.keyed()
    assert (nxt.kind, nxt.status, nxt.header) == (Kind.NORMAL, 2, 'A')


def test_an_unacknowledged_idle_normal_repeats_on_header_b():
    m, _ = sending()
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    idle = m.keyed()
    assert (idle.kind, idle.status, idle.payload, idle.header) == (Kind.NORMAL, 1, b'', 'A')
    m.observe(word('X27', idle.stream_sample+T, responds_to=idle.stream_sample),
              now=idle.stream_sample+T+12000)
    again = m.decide(idle.stream_sample+round(1.25*FS)-m.lead)
    assert (again.kind, again.status, again.header) == (Kind.NORMAL, 1, 'B')
