# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Passive display tests only: fake monitor and already-decoded payload bytes."""
from types import SimpleNamespace
import pytest
from hfmodem.contact import slots as S, varac as v
from hfmodem.tests.contact.test_slots import watched


def record(d, w, slot, text, kind='text', station=''):
    w.found = [('qso', slot, S.Said(slot, d.host.now, kind, text, station))]
    d.tick()


def test_follow_all_replays_and_streams_current_passband_with_frequencies():
    d, w = watched()
    record(d, w, 1, 'earlier', 'stage')
    record(d, w, None, 'de N0XYZ', station='N0XYZ')
    record(d, w, 11, 'idle or polling', 'stage')
    before = len(d.sent), len(d.texts), d.tunes[:]
    d.do('/follow all')
    lines = [line for line in d.said if line.startswith('[')]
    assert len(lines) == 3
    assert '14104.250 kHz' in lines[0] and 'station unverified' in lines[0]
    assert '14105.000 kHz' in lines[1] and 'N0XYZ (signed)' in lines[1]
    assert '14105.750 kHz' in lines[2] and 'idle or polling' in lines[2]
    record(d, w, 1, 'live')
    assert d.said[-1].endswith('live')
    assert (len(d.sent), len(d.texts), d.tunes) == before


def test_follow_all_prints_off_channel_idle_without_feeding_the_live_link():
    d, w = watched()
    d.do('/follow all')
    d.flow.dial = v.slot_frequency(d.flow.band, 1)
    before = len(d.sent), len(d.texts), d.tunes[:], list(d.host.client.messages)
    record(d, w, None, 'idle reply addressed to W9SSJ', kind='control')
    assert '[CF 14105.000 kHz]' in d.said[-1]
    assert 'station unverified: idle reply addressed to W9SSJ' in d.said[-1]
    assert (len(d.sent), len(d.texts), d.tunes, list(d.host.client.messages)) == before


def test_cf_only_off_and_passband_changes_filter_live_and_replay():
    d, w = watched()
    d.do('/follow cf')
    record(d, w, 1, 'other slot')
    assert not any(line.endswith('other slot') for line in d.said)
    record(d, w, None, 'CF traffic')
    assert d.said[-1].endswith('CF traffic')
    w.rows = [S.Activity(1, 750)]
    record(d, w, None, 'outside current passband')
    assert not any(line.endswith('outside current passband') for line in d.said)
    d.do('/follow all')
    assert d.said[-1].endswith('other slot')
    d.do('/follow off')
    record(d, w, 1, 'not displayed')
    assert d.said[-1] == 'follow off'


def test_replay_has_time_and_count_limits():
    d, w = watched()
    record(d, w, 1, 'expired')
    d.host.now += v.REPLAY_S + 1
    for i in range(v.REPLAY_LINES + 10):
        record(d, w, 1, f'row {i}')
    assert len(d.session._transcripts[1]) == v.REPLAY_LINES
    start = len(d.said)
    d.do('/follow all')
    replay = d.said[start + 1:]
    assert len(replay) == v.REPLAY_LINES
    assert not any('expired' in line for line in replay)
    assert replay[-1].endswith(f'row {v.REPLAY_LINES + 9}')


@pytest.mark.parametrize('payload', [b'plain ASCII\r\n', 'café on HF'.encode()])
def test_printable_utf8_or_ascii_is_retained(payload):
    assert S.payload_text(payload) == (payload.decode(), None)


def test_text_flag_requires_successful_decode():
    data = b'Hello from W9SSJ\r\n'
    assert S.payload_text(S.compression.encode_text(data), text_encoded=True) == (data.decode(), None)
    text, detail = S.payload_text(b'\0' * 10, text_encoded=True)
    assert text is None and 'not resolved' in detail


def test_binary_payload_summary_is_bounded_and_never_claims_undecodable_rf():
    raw = bytes.fromhex('2164c23c598e5b43873984aad410204081020f140000')
    text, detail = S.payload_text(raw)
    assert text is None and '22 bytes' in detail and raw.hex() in detail
    text, detail = S.payload_text(b'\x00' * 1000)
    assert text is None and len(detail) < 320 and '1000 bytes' in detail


def test_binary_frame_never_extracts_callsigns_or_chat(monkeypatch):
    m = object.__new__(S.SlotMonitor)
    m.samples = S.FS
    m._link = lambda ch, ev: False
    m._keyed = lambda ev, body: None
    heard = []
    m.heard = heard.append
    monkeypatch.setattr(S.B, 'parse_payload', lambda body: None)
    monkeypatch.setattr(S.VF, 'is_link_setup', lambda body: False)
    monkeypatch.setattr(S.phy, 'vara_payload', lambda *args, **kwargs: b'\x00de N0XYZ <DISC>')
    frame = SimpleNamespace(frame_bytes=b'\x00'*6, is_connect=False, level=4)
    ch = SimpleNamespace(slot=1, excess=[])
    ev = S._Evidence(text='prior text', level=4)
    result = m._frame(ch, ev, frame)
    assert result[-1][2].kind == 'payload'
    assert not heard and not ev.calls and not ev.closed and ev.text == ''
    assert not any(item[2].kind == 'text' for item in result)


def test_unresolved_live_qsy_bytes_have_escaped_preview_without_becoming_text():
    payload = bytes.fromhex('36203c513a37353e05')
    text, detail = S.payload_text(payload)
    assert text is None
    assert r"preview '6 <Q:75>\x05'" in detail
    assert payload.hex() in detail
    assert '\x05' not in detail, 'never emit the raw terminal control byte'


def test_monitoring_hint_is_once_and_does_not_enable_follow():
    d, w = watched()
    d.session.monitor()
    hints = [line for line in d.said if '/follow cf shows' in line]
    assert len(hints) == 1 and d.session.following is None
    d.session.monitor()
    record(d, w, None, 'decoded but hidden')
    assert len([line for line in d.said if '/follow cf shows' in line]) == 1
    assert not any(line.startswith('[') for line in d.said)


def test_follow_outside_passband_retunes_without_transmitting_and_returns():
    d, w = watched()
    d.session.set_beacon(15)
    d.do('/follow 2')
    assert d.tunes == [v.slot_frequency(20, 2)]
    assert w.tuned == d.tunes
    assert d.armed[-1] is False and d.sent[-1] == 'LISTEN OFF'
    assert d.session.following == 2
    assert any('CF activity outside the passband will be missed' in line for line in d.said)
    sent = d.sent[:]
    d.host.now += 1000
    d.tick()
    assert d.sent == sent
    d.do('/follow cf')
    assert d.tunes[-1] == v.CALLING_HZ[20]
    assert d.armed[-1] is True and d.session.beacon == 15
    assert not d.session._follow_tuned


def test_follow_non_slot_frequency_streams_and_off_returns():
    d, w = watched()
    d.do('/follow 14101.125')
    assert d.tunes[-1] == 14101125
    key = 'hz:14101125'
    assert d.session.following == key
    w.rows = [S.Activity(key, 1500)]
    record(d, w, key, 'already in progress')
    assert d.said[-1] == '[14101.125 kHz] station unverified: already in progress'
    d.do('/follow off')
    assert d.tunes[-1] == v.CALLING_HZ[20] and d.session.following is None


@pytest.mark.parametrize('command', ['/follow 0', '/follow 99', '/follow nan', '/follow -5'])
def test_follow_invalid_target_does_not_tune(command):
    d, w = watched()
    d.do(command)
    assert d.said[-1].startswith('error:') and not d.tunes


@pytest.mark.parametrize('busy', ['connected', 'pending', 'ptt'])
def test_follow_retune_refuses_active_modem(busy):
    d, w = watched()
    setattr(d.host.client, busy, True)
    d.do('/follow 2')
    assert d.said[-1].startswith('error: cannot')
    assert not d.tunes


def test_non_slot_monitor_keeps_actual_cf_and_adds_tuned_channel():
    dial, cf = 14101125, v.CALLING_HZ[20]
    m = S.SlotMonitor(dial, cf)
    assert any(a.slot == f'hz:{dial}' and a.audio_hz == 1500 for a in m.table())
    assert m.calling == cf
    m.retune(cf)
    assert all(not isinstance(a.slot, str) for a in m.table())


@pytest.mark.parametrize('slot', v.DEFAULT_CODEC.slots)
def test_every_slot_can_be_selected(slot):
    d, w = watched()
    d.do(f'/follow {slot}')
    assert d.session.following == slot
    if slot not in (1, 11):
        assert d.tunes[-1] == v.slot_frequency(20, slot)


@pytest.mark.parametrize('offset', [0, -350])
def test_non_slot_monitor_respects_wide_offset(offset):
    dial = 14101125
    held = dict(S.channels(dial, v.CALLING_HZ[20], offset=offset))
    assert held[f'hz:{dial}'] == 1500 - offset


def test_follow_tune_failure_never_claims_success_or_sends_beacon():
    d, w = watched()
    def fail(dial):
        raise v.HostError('tuner failed')
    d.flow.tuner = fail
    d.do('/follow 2')
    assert d.said[-1] == 'error: tuner failed'
    assert d.session.following is None and d.armed[-1] is False
    assert d.session._follow_tuned


def test_follow_off_cannot_interrupt_an_outgoing_operation():
    d, w = watched()
    d.do('/follow 2')
    d.session.steps = iter([1])
    before = d.tunes[:]
    d.do('/follow off')
    assert d.said[-1].startswith('error:') and d.tunes == before


def test_retune_discards_old_passband_table_until_worker_confirms_new_one():
    import queue
    watch = S.SlotWatch.__new__(S.SlotWatch)
    watch._in, watch._out = queue.Queue(), queue.Queue()
    watch._calling, watch._generation = v.CALLING_HZ[20], 0
    watch._table, watch._found, watch._now = [S.Activity(None, 1500)], [], 3
    watch._out.put((0, ('table', [S.Activity(None, 1500)], 4)))
    watch.retune(v.slot_frequency(20, 2))
    assert watch.table()[0] == []
    watch._out.put((1, ('table', [S.Activity(2, 1500)], 5)))
    assert watch.table()[0][0].slot == 2


def test_audio_gap_notes_are_logged_but_not_printed_under_follow_all():
    d, w = watched()
    d.do('/follow all')
    start = len(d.said)
    for _ in range(5):
        record(d, w, 1, 'passive audio gap: 0.12 seconds skipped', 'coverage')
        d.host.now += 2
    assert not [line for line in d.said[start:] if line.startswith('[') or 'gap' in line]
    assert not d.session._transcripts.get(1)
    assert sum(r.get('qso') == 'coverage' for r in d.kinds('slot')) == 5
    assert v.follow_line(S.Said(1, 0, 'coverage', 'x')) == '[slot 1] coverage: x'


def test_heavy_audio_loss_gets_one_summary_per_interval():
    d, w = watched()
    d.do('/follow all')
    start = len(d.said)
    for _ in range(40):
        record(d, w, 1, 'passive audio gap: 0.50 seconds skipped', 'coverage')
        d.host.now += 2
    summaries = [line for line in d.said[start:] if 'dropping audio' in line]
    assert len(summaries) == 1 and 'decodes may be missed' in summaries[0]
    d.host.now += v.COVERAGE_NOTICE_S
    for _ in range(10):
        record(d, w, 1, 'passive audio gap: 0.50 seconds skipped', 'coverage')
        d.host.now += 2
    assert sum('dropping audio' in line for line in d.said[start:]) == 2


def test_light_loss_stays_silent_unless_verbose():
    d, w = watched()
    d.do('/follow all')
    start = len(d.said)
    record(d, w, 1, 'passive audio gap: 0.12 seconds skipped', 'coverage')
    assert not any('dropping' in line for line in d.said[start:])
    d.do('/verbose on')
    record(d, w, 1, 'passive audio gap: 0.12 seconds skipped', 'coverage')
    assert any('dropping audio' in line for line in d.said[start:])


def test_a_shed_channel_is_said_once_each_way():
    d, w = watched()
    d.do('/follow all')
    start = len(d.said)
    record(d, w, 11, v.SHED, 'coverage')
    record(d, w, 11, v.RESUMED, 'coverage')
    lines = [line for line in d.said[start:] if 'passive follower' in line]
    assert len(lines) == 2 and lines[0].startswith('[slot 11') and lines[0].endswith(f'coverage: {v.SHED}')
    assert lines[1].endswith(f'coverage: {v.RESUMED}')
