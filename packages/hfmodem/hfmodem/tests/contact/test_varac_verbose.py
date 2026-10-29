# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.tests.contact.test_varac import Desk


def test_verbose_defaults_off_and_toggles_even_during_close():
    desk = Desk()
    assert desk.session.verbose is False
    start = desk.said[:]
    desk.session.diagnostic('rx_protocol', text='turn granted')
    assert desk.said == start
    desk.session.command('/verbose')
    assert desk.session.verbose and desk.said[-1].startswith('verbose on:')
    desk.session._closing = True
    desk.session.command('/verbose off')
    assert not desk.session.verbose and desk.said[-1] == 'verbose off'
    desk.session.command('/verbose on')
    desk.session.command('/verbose nonsense')
    assert desk.session.verbose and desk.said[-1] == 'usage: /verbose [on|off]'


def test_queued_text_is_not_sent_or_acknowledged_and_buffer_clear_does_not_claim_delivery():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ KE8PLM 500')
    desk.session.command('/verbose on')
    desk.session.command('hello')
    assert desk.said[-1] == 'TX queued (not yet acknowledged): hello ↵ '
    desk.host.client._line('BUFFER 10')
    desk.host.client._line('BUFFER 0')
    assert desk.said[-2:] == ['TX buffer: 10 bytes pending', 'TX buffer empty']
    assert not any('delivered' in line for line in desk.said)


def test_carrier_does_not_claim_a_decoded_frame_but_actual_rx_protocol_is_named():
    desk = Desk()
    desk.session.command('/verbose on')
    desk.host.client._line('BUSY ON')
    assert desk.said[-1] == 'RX: channel busy; no frame identified by BUSY alone'
    desk.session.diagnostic('rx_protocol', text='turn grant from KE8PLM')
    assert desk.said[-1] == 'RX: turn grant from KE8PLM'
    desk.hear('CONNECTED W9SSJ KE8PLM 500')
    desk.receive('hello from the other station')
    assert 'RX KE8PLM: hello from the other station' in desk.said


def test_verbose_tx_describes_actual_metadata_and_distinguishes_failure():
    desk = Desk()
    desk.session.command('/verbose on')
    metadata = dict(packet_kind='data', speed_level=4, over_id=3, retry=True)
    desk.session.diagnostic('tx_start', keyed=True, **metadata)
    desk.session.diagnostic('tx_end', outcome='sent', **metadata)
    assert desk.said[-2:] == ['TX sending: data (over 3, level 4, retry)',
                             'TX sent: data (over 3, level 4, retry)']
    desk.session.diagnostic('tx_end', outcome='refused', packet_kind='session-turn-request-short')
    assert desk.said[-1] == 'TX refused; not sent: turn request short'
    desk.session.diagnostic('tx_start', keyed=False, identification=True)
    assert desk.said[-1] == 'TX playing audio; key not confirmed: CW identification'


def test_runtime_queue_never_writes_terminal_from_worker_and_is_bounded(tools):
    import varac_chat
    desk = Desk()
    desk.session.command('/verbose on')
    before = desk.said[:]
    events = varac_chat.Diagnostics()
    def producer():
        for _ in range(300):
            events.put('rx_protocol', text='over acknowledged')
    worker = threading.Thread(target=producer)
    worker.start()
    worker.join()
    assert desk.said == before and events.events.qsize() == 256
    assert events.dropped == 44
    events.drain(desk.session)
    assert '44 runtime observations omitted' in desk.said[len(before)]
    assert len(desk.said) == len(before) + 257
    assert events.events.empty() and events.dropped == 0


@pytest.fixture
def audio(tools, monkeypatch):
    import kestrel_connect as kc
    fake = SimpleNamespace(query_devices=lambda *a: {'max_output_channels': 2},
                           InputStream=lambda **kw: SimpleNamespace(start=lambda: None, stop=lambda: None,
                                                                    close=lambda: None))
    monkeypatch.setitem(sys.modules, 'sounddevice', fake)
    monkeypatch.setattr(kc, '_bounded', lambda seconds, fn: fn())
    monkeypatch.setattr(kc, 'play_drained', lambda *a, **kw: None)
    rig = SimpleNamespace(armed=True, retired=False)
    io = kc.AudioVaraIO('stub-out', 'stub-in', rig=rig, tx_tail=0)
    events = []
    io.diagnostic = lambda kind, **details: events.append((kind, details))
    io._tx_key_confirmed = True
    return kc, io, events


def test_audio_reports_start_at_playback_and_completion_from_actual_verdict(audio, monkeypatch):
    kc, io, events = audio
    def played(*args, **kwargs):
        assert [kind for kind, _ in events] == ['tx_start']
    monkeypatch.setattr(kc, 'play_drained', played)
    io.tx_metadata(packet_kind='session-connect-request', speed_level=None)
    io.tx(np.zeros(480, np.float32))
    assert [kind for kind, _ in events] == ['tx_start', 'tx_end']
    assert events[0][1]['keyed'] and events[1][1]['outcome'] == 'sent'
    assert events[0][1]['packet_kind'] == events[1][1]['packet_kind'] == 'session-connect-request'


def test_refused_audio_has_no_sending_event(audio, monkeypatch):
    kc, io, events = audio
    io._refused = True
    monkeypatch.setattr(kc, 'play_drained', lambda *a, **k: pytest.fail('played refused frame'))
    io.tx(np.zeros(480, np.float32))
    assert [kind for kind, _ in events] == ['tx_end']
    assert events[0][1]['outcome'] == 'refused'


def test_playback_error_is_reported_and_diagnostic_failure_cannot_interrupt_tx(audio, monkeypatch):
    kc, io, events = audio
    def fail(*args, **kwargs):
        raise RuntimeError('playback failed')
    monkeypatch.setattr(kc, 'play_drained', fail)
    with pytest.raises(RuntimeError, match='playback failed'):
        io.tx(np.zeros(480, np.float32))
    assert events[-1][1]['outcome'] == 'error'
    io.diagnostic = lambda *a, **kw: fail()
    monkeypatch.setattr(kc, 'play_drained', lambda *a, **kw: None)
    io.tx(np.zeros(480, np.float32))


def test_only_existing_rx_decoder_logs_feed_rx_diagnostics(audio):
    _, io, events = audio
    io.log('rx control burst acknowledged our confirm')
    io.log('tx DATA over #1')
    assert events == [('rx_protocol', {'text': 'control burst acknowledged our confirm'})]


def test_recovery_and_closing_reasons_reach_default_terminal_and_event_log(audio, tools):
    import varac_chat
    _, io, _ = audio
    desk = Desk()
    events = varac_chat.Diagnostics()
    io.diagnostic = events.put
    before = list(desk.said)
    # Verbose observations cannot crowd out the reason for closing.
    for _ in range(300):
        events.put('rx_protocol', text='ordinary frame')
    io.log('link-recovery requested: Recovery: retry requested at level 4; receipt unknown.')
    io.log('link-recovery closing: Closing link locally: no valid retry accepted; 8 bytes unacknowledged.')
    assert desk.said == before
    events.drain(desk.session)
    assert desk.said[len(before):] == [
        'Recovery: retry requested at level 4; receipt unknown.',
        'Closing link locally: no valid retry accepted; 8 bytes unacknowledged.']
    assert [r['state'] for r in desk.kinds('link_recovery')] == ['requested', 'closing']


def test_a_capture_rate_warning_reaches_default_terminal_and_event_log(tools):
    import varac_chat
    desk = Desk()
    events = varac_chat.Diagnostics()
    for _ in range(300):
        events.put('rx_protocol', text='ordinary frame')
    text = '!! CAPTURE RATE: USB Audio Device is delivering 52% of real time'
    events.put('capture_rate', text=text, device='USB Audio Device', share=.52, slow=True)
    events.drain(desk.session)
    assert desk.said[-1] == text
    [record] = desk.kinds('capture_rate')
    assert record['device'] == 'USB Audio Device' and record['share'] == .52


def test_bounded_recovery_queue_preserves_newest_closing_reason(tools):
    import varac_chat
    desk = Desk()
    events = varac_chat.Diagnostics()
    for n in range(40):
        events.put('link_recovery', state='waiting', text=f'wait {n}')
    events.put('link_recovery', state='closing', text='Closing locally: recovery exhausted')
    assert events.recovery.qsize() == 16
    events.drain(desk.session)
    assert desk.said[-1] == 'Closing locally: recovery exhausted'
    assert len(desk.kinds('link_recovery')) == 16



def test_runtime_ack_credit_is_distinct_from_discard_and_reported_once():
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from hfmodem.kestrel.host.vara_modem import VaraHostModem
    desk = Desk()
    desk.session.command('/verbose on')
    transport = SimpleNamespace(diagnostic=desk.session.diagnostic)
    runtime = AudioHostRuntime(transport)
    runtime.modem = VaraHostModem(transport, bandwidth='500')
    ledger = runtime.modem.station.payload_ledger
    # Compressed wire bytes must credit the original text only after the whole
    # encoded span is confirmed, rather than interpreting BUFFER changes.
    ledger.queue(b'ab', b'hello peer', atomic=True)
    runtime._report_payload_credit()
    start = len(desk.said)
    ledger.settle(1)
    runtime._report_payload_credit()
    assert len(desk.said) == start
    ledger.settle(1)
    runtime._report_payload_credit()
    runtime._report_payload_credit()
    ledger.queue(b'unsent', b'unsent')
    runtime.modem._replace_station(reason='operator closed')
    runtime._report_payload_credit()
    assert desk.said[start:] == ['TX acknowledged by peer modem: 10 original bytes',
                                 'TX discarded without acknowledgement: 6 original bytes']
    # A new attachment starts its own totals; it cannot lose its first ACK.
    runtime.modem = VaraHostModem(transport, bandwidth='500')
    runtime.modem.station.payload_ledger.queue(b'new', b'new')
    runtime.modem.station.payload_ledger.settle(3)
    runtime._report_payload_credit()
    assert desk.said[-1] == 'TX acknowledged by peer modem: 3 original bytes'
