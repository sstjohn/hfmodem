"""Keep the P1 grant clock independent of the physically keyed P2 reply."""
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, pactor1, pactor2, spec
from hfmodem.tests.shrike.test_p2_granted_retry import granted
from hfmodem.tests.shrike.test_p3_offer import cs_event


def receiver():
    host, _ = granted()
    host.peer.invert = False
    rx = onair._SessionRx(host)
    grid = onair._MasterGrid(0, 60000, 0, packet_n=46080, cs_n=5760,
                             d_max_n=6240)
    grid.d_n = 3504.
    grid.d_ref_n = 46080
    grid.keyed_slot = 0
    grid.keying(spec.Protocol.PACTOR2, extent_n=39994)
    grid._p2_keyed_reply = (0, 0, 39994, 60000)
    return host, rx, grid


@pytest.mark.parametrize('phase_delay', [0, 900])
@pytest.mark.parametrize('index', range(6))
@pytest.mark.parametrize('swap', [False, True])
def test_transition_reads_p2_at_its_own_epoch(index, swap, phase_delay):
    host, rx, grid = receiver()
    host.peer.invert = swap
    grid._p2_keyed_reply = (0, phase_delay, 39994 + phase_delay, 60000)
    audio = np.zeros(60000, np.float32)
    at = round(.880 * onair.FS) + phase_delay
    wave = pactor2.control_signal(index, swapped=swap)
    start = at - pactor2.pulse_lead()
    audio[start:start+len(wave)] = wave
    assert grid.rx_due(0) == 49584  # P1 grants remain at 1.033 seconds.
    events = []
    rx._on = lambda ev, **kw: events.append(ev)
    heard, where = rx.control_signal_in(audio, 0, grid)
    assert heard == index and where == at
    assert len(events) == 1 and events[0].protocol == spec.Protocol.PACTOR2


@pytest.mark.parametrize('stale', [False, True])
def test_silence_noise_and_unkeyed_replies_do_not_ack(stale):
    _, rx, grid = receiver()
    if stale:
        grid.keyed_slot = 1
    events = []
    rx._on = lambda ev, **kw: events.append(ev)
    rng = np.random.default_rng(20260920)
    for audio in [np.zeros(60000)] + [rng.normal(0, .1, 60000) for _ in range(32)]:
        rx.control_signal_in(audio, 0, grid)
    assert not events


def test_native_entry_holds_handover_until_confirmed():
    host, _ = granted()
    assert host.arq.p2_entry_pending
    assert not host.arq.entry_pending  # Does not select the P3 entry renderer.
    assert not host.arq._inflight.status & 0x40
    host.arq.on_host_data(b'mail')
    host.on_rx_event(cs_event(arq.CS_ACK, spec.Protocol.PACTOR2))
    assert not host.arq.p2_entry_pending


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('status,marker', [(0x18, 4), (0x19, 5), (0x1a, 4), (0x1b, 5)])
def test_header_matches_stock_counter_parity(status, marker, swap):
    # HB9AK stock SL3 recording: header-flag.json contains 30 matched fields.
    from hfmodem.shrike import p2rx, placement
    path = pactor2.PATHS[2]
    field = pactor2.build_field(placement.field_info(b'', path.crc_bytes-3, status), path)
    wave = pactor2.data_burst(field, path, swapped=swap)
    audio = np.concatenate([np.zeros(24000), wave, np.zeros(24000)])
    found = p2rx.find_markers(audio, onair.FS, .95)
    assert any(k == marker and swapped == swap for _, _, k, _, _, swapped in found)


def test_repeated_fsk_grant_is_still_read_at_p1_epoch():
    _, rx, grid = receiver()
    audio = np.zeros(60000)
    wave = pactor1.control_signal(pactor1.CS_6A9)
    start = grid.rx_due(0)
    audio[start:start+len(wave)] = wave
    events = []
    rx._on = lambda ev, **kw: events.append(ev)
    rx.control_signal_in(audio, 0, grid)
    assert len(events) == 1 and events[0].protocol == spec.Protocol.PACTOR1


def test_mail_and_over_wait_behind_native_p2_entry():
    from hfmodem.tests.shrike.test_p2_policy import linked, grant
    host, _ = linked()
    host.arq.on_host_data(b'queued mail')
    host.arq.on_host_over()
    host.on_rx_event(grant(pactor1.CS_6A9))
    assert host.arq.p2_entry_pending
    assert host.arq._inflight.payload == b''
    assert not host.arq._inflight.status & 0x40
    assert bytes(host.arq._outbuf) == b'queued mail'


@pytest.mark.parametrize('delay_ms', [0, 18.75])
@pytest.mark.parametrize('long', [False, True])
def test_real_transmit_path_records_p2_phase_and_reply(tmp_path, monkeypatch, long, delay_ms):
    host, _, grid = receiver()
    grid.regear(long)
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    tx.host = host
    tx.p2_phase_delay_n = round(delay_ms * onair.FS / 1000)
    monkeypatch.setattr(onair.session, 'write_wav', lambda *args: None)
    tx.aim(grid, 4)
    phase = grid.boundary(4)
    send = tx.send_p2_long_packet if long else tx.send_p2_packet
    send(1, b'', 0x1a)
    slot, recorded_phase, end, cycle = grid._p2_keyed_reply
    assert slot == 4 and recorded_phase == phase + tx.p2_phase_delay_n
    assert phase < end < phase + cycle
    assert cycle == round((3.75 if long else 1.25) * onair.FS)


@pytest.mark.parametrize('keyed', [None, (1, 0, 39994, 60000)])
def test_valid_control_without_matching_keying_cannot_ack(keyed):
    _, rx, grid = receiver()
    grid._p2_keyed_reply = keyed
    audio = np.zeros(60000)
    wave = pactor2.control_signal(0)
    start = round(.880 * onair.FS) - pactor2.pulse_lead()
    audio[start:start+len(wave)] = wave
    events = []
    rx._on = lambda ev, **kw: events.append(ev)
    rx.control_signal_in(audio, 0, grid)
    assert not events


def test_p3_compatibility_renderer_ignores_native_p2_delay(tmp_path):
    host, _ = granted()
    host.protocol = spec.Protocol.PACTOR3
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    tx.host = host
    wave = pactor2.control_signal(0)
    calls = []
    tx._tx = lambda *args, **kwargs: calls.append(kwargs['lead_n'])
    tx._tx_p2(wave, 'baseline')
    tx.p2_phase_delay_n = 900
    tx._tx_p2(wave, 'P3 compatibility')
    assert calls[0] == calls[1]


def test_p2_mail_phase_does_not_move_other_presets():
    from types import SimpleNamespace
    args = SimpleNamespace(p2_mail=False, p1_6a9=False)
    onair._p1_response_defaults(args)
    assert args.p2_phase_delay_ms == 0
