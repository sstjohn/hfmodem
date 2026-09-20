"""An emitted CS3 still owns its ACK across listening and recovered slots."""
import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _Session

CAP = evidence.CAPTURES/'onair-0919-2300'
CYCLE = 60000
START, END = 5252865, 5292866  # Last physically emitted CS3, TX61.


@pytest.fixture(scope='module')
def recording():
    if not (CAP/'stream.wav').exists():
        pytest.skip('local WS8EOC September 19 23:00 capture unavailable')
    fs, raw = wavfile.read(CAP/'stream.wav')
    assert fs == onair.FS
    return raw.astype(np.float32)/32768


def pending():
    s = _Session(role=arq.IRS)
    a = s.host.arq
    a.cfg.speed_up = 'hold'
    a.cfg.traffic_sl = 1
    a.on_host_data(b'ABCDEF')
    a.on_host_breakin()
    a.on_rx_packet(1, b'', 0, True)
    a.on_cycle()
    assert a.unconfirmed_breakin
    g = onair._MasterGrid(32690, CYCLE, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.d_n, g.d_ref_n = 5040, 46080
    g.keyed_slot = 87
    g._p3_keyed_reply = (87, START, END, CYCLE)
    s.host.peer.raster = g
    s.rx.p3_receive_offset_hz = -19.8
    assert g.rx_due(87) == 5303810
    return s, g


def test_recorded_late_cs1_settles_actual_pending_breakin(recording):
    s, g = pending()
    answers = []
    for n in (1, 2):
        lo, hi = END+n*CYCLE, START+(n+1)*CYCLE
        s.rx.new_cycle()
        heard, _ = s.rx.control_signal_in(recording[lo:hi], lo, g)
        answers.append(heard)
    assert arq.CS_ACK in answers
    assert not s.host.arq.unconfirmed_breakin
    assert s.host.arq._inflight.seq == 1
    assert abs(s.rx._p3_answer_at - 5417580) < 480


@pytest.mark.parametrize('reason', ['ordinary', 'unkeyed', 'old_buffer'])
def test_late_window_does_not_invent_other_acknowledgements(recording, reason, monkeypatch):
    s, g = pending()
    lo, hi = END+CYCLE, START+2*CYCLE
    if reason == 'ordinary':
        s.host.arq._inflight.breakin = False
    elif reason == 'unkeyed':
        g._p3_keyed_reply = None
    else:
        lo, hi = START-20000, START
    asked = []
    monkeypatch.setattr(s.rx, '_p3_cs', lambda *a, **kw: asked.append(True))
    heard, _ = s.rx.control_signal_in(recording[lo:hi], lo, g)
    assert heard is None and not asked


def test_recovered_multicycle_buffer_gets_one_bounded_read(recording, monkeypatch):
    s, g = pending()
    asked = []
    monkeypatch.setattr(s.rx, 'control_signal',
                        lambda seg, start, at, **kw: asked.append((len(seg), start, at)))
    s.rx.control_signal_in(recording[5313024:5549218], 5313024, g)
    assert len(asked) == 1
    size, start, at = asked[0]
    assert size <= CYCLE-(END-START)
    assert start >= END and start <= at < start+size
