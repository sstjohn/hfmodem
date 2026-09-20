# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stock's terminal BW500 window must survive a one-column envelope error."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.winlink import Message, decompress

FIXTURES=Path(__file__).with_name('fixtures')/'bw500-terminal-window'
requires_window=pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason='the stock terminal-window recording requires the source checkout '
           '(fixtures/bw500-terminal-window)')
TAIL=bytes.fromhex('7a25719b680427')
EXPECTED_BODY='''Saul, here is the Thursday net traffic for W9SSJ.

Net control logged 14 check-ins on 3.985 MHz at 0100Z: KB3AC-10 5x7,
WA4QJM 4x4, K9ZTV 5x9, N0BXQ 3x3, VE3KPG 4x6, KF0OIC 5x5, AJ4GU 2x2,
W0LON 4x7, KE8LVA 5x8, N9TVP 3x5, KD8WHQ 4x3, W5RMB 5x6, KC1JXB 2x4,
NX7UD 3x7. Two pieces of formal traffic went to the Illinois section
net. Test your gateway path before the drill on the 14th.

The next session is at 0100Z Thursday, net control W5RMB, alternate KC1JXB.

73 de W1AW
'''.replace('\n','\r\n').encode('ascii')


@pytest.fixture(scope='module')
def recovered():
    rate,audio=wavfile.read(FIXTURES/'stock-final-window.wav')
    assert rate==48000 and audio.dtype==np.float32 and audio.ndim==1
    result=RX.decode_stream(audio)
    assert result.complete, 'A measured envelope one column early lost the last seven host bytes'
    assert len(result.data_frames)==2
    assert [f.crc_ok for f in result.data_frames]==[True,True]
    return [phy.vara_payload(bytes(f.payload)+bytes([f.marker]),caller='W9SSJ',
                             body_len=len(f.payload)+1) for f in result.data_frames]


@requires_window
def test_stock_final_window_delivers_last_seven_bytes_and_empty_partner(recovered):
    assert recovered==[TAIL,b'']


@requires_window
def test_recovered_tail_completes_b2_checksums_and_exact_bench_message(recovered):
    prefix=(FIXTURES/'received-prefix.lzh').read_bytes()
    tail=b''.join(recovered)
    assert len(prefix)==487 and tail[-2]==4
    compressed=prefix+tail[:-2]
    assert len(compressed)==492
    assert (sum(compressed)+tail[-1])&0xFF==0, 'B2 EOT checksum'
    rendered=decompress(compressed,expected_size=645)  # also checks LZH CRC
    message=Message.parse(rendered)
    assert len(rendered)==645 and message.render()==rendered
    assert message.mid=='D01ZP51L7PHN'
    assert message.subject=='Thursday net traffic'
    assert message.get('Mbo')=='KC9GHZ'
    assert message.body==EXPECTED_BODY


class _IO(VA.VaraIO):
    def __init__(self): self.sent=[]; self.payloads=[]; self.messages=[]
    def tx(self,samples): self.sent.append(samples)
    def key(self,on): pass
    def log(self,message): self.messages.append(message)
    def data(self,payload): self.payloads.append(bytes(payload))


def _waiting():
    io=_IO(); hs=VA.VaraStationHandshake(['W9SSJ'],io,bw='500')
    hs.role,hs.caller,hs.called='initiator','W9SSJ','KC9GHZ'
    hs.state,hs.step,hs.turn=VA.VaraState.CONNECTED,VA._I_CONNECTED,VA._TURN_PEER
    hs._peer_delivery_open=True; hs._answer_owed=VA._OWED_OVER
    hs._reack_frame=VA.OVER_CONTINUE_SHORT
    return hs,io


def _partial_result(good=True):
    body=phy.vara_body(b'A'*43,'W9SSJ',body_len=44)
    first=RX.FrameResult(100,(0,0),9,1.0,good,body+b'\0\0',frames_in_burst=2)
    bad=RX.FrameResult(100,(0,0),9,.75,False,b'\0'*46,frames_in_burst=2)
    return RX.DecodeResult([first,bad],[first,bad]),body


@pytest.mark.parametrize('missing',['crc_failed','not_demodulated'])
def test_identified_incomplete_peer_window_owes_data_not_previous_ack(missing,monkeypatch):
    hs,io=_waiting(); result,_=_partial_result()
    if missing=='not_demodulated':
        result=RX.DecodeResult(result.frames[:1],result.data_frames[:1])
    monkeypatch.setattr(VA._RX500,'decode_stream',lambda _:result)
    assert hs._answer_data_over(np.zeros(100))
    assert hs._owed_block and hs._answer_owed==VA._OWED_OVER
    assert io.payloads==[]
    assert any('tx NAK' in m for m in io.messages)
    first=io.sent[-1].copy()
    assert hs._reack()
    np.testing.assert_array_equal(io.sent[-1],first)
    assert hs._owed_block


@pytest.mark.parametrize('case',['all_bad','our_echo','no_peer_delivery'])
def test_uncertain_or_own_failed_frame_does_not_get_a_nak(case,monkeypatch):
    hs,io=_waiting(); result,body=_partial_result(good=case!='all_bad')
    if case=='our_echo': hs._keyed_bodies.add(body)
    if case=='no_peer_delivery': hs._peer_delivery_open=False
    monkeypatch.setattr(VA._RX500,'decode_stream',lambda _:result)
    assert not hs._answer_data_over(np.zeros(100))
    assert io.sent==[] and not hs._owed_block
