# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The slot monitor on the 10-23 40 m calling frequency, and what the session shows of it."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from hfmodem.contact import slots as S, varac as v
from hfmodem.core.wav import source
from hfmodem.tests.contact.test_varac import Desk
from hfmodem.tests.evidence import CORPUS, require_input

FIXTURES = Path(__file__).parent / 'fixtures'
PROVENANCE = {name: (tape, prov) for tape in ('varac-slots-1023', 'varac-slots-1024')
              for name, prov in json.loads((FIXTURES / tape / 'provenance.json').read_text()).items()}
CF = v.CALLING_HZ[40]
#: The dial the 10-23 evaluation flew --monitor-wide from.
WIDE = -350


def recording(name):
    tape, prov = PROVENANCE[name]
    path = require_input(CORPUS / tape / f'{name}.flac')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov['flac_sha256']
    rate, x = wavfile.read(source(path))
    assert rate == S.FS and len(x) == prov['samples'][1] - prov['samples'][0]
    return x / 32768


def replay(x, heard=(), **monitor):
    """`heard`: `(seconds, call)` the host reported, as the session passes them on."""
    m = S.SlotMonitor(CF, CF, mycall='W9SSJ', **monitor)
    heard, found = list(heard), []
    for i in range(0, len(x), 4800):
        while heard and heard[0][0] * S.FS <= i:
            m.heard(heard.pop(0)[1])
        found += m.feed(x[i:i + 4800])
    return m, found


def states(found, slot):
    return [record[2] for record in found if record[0] == 'state' and record[1] == slot]


def said(found, slot):
    return [record[2] for record in found if record[0] == 'qso' and record[1] == slot]


def test_the_channels_are_the_ones_whose_500_hz_band_the_passband_holds():
    assert S.channels(CF, CF) == [(1, 750), (None, 1500), (11, 2250)]
    assert S.channels(CF, CF, offset=WIDE) == [(2, 350), (1, 1100), (None, 1850), (11, 2600)]
    assert S.channels(CF - 3000, CF) == [(5, 750), (4, 1500), (3, 2250)]
    assert S.channels(CF + 3750, CF) == [(14, 750), (15, 1500)]


def test_forwarded_mail_names_remain_unidentified_and_slot_11_stays_quiet():
    m, found = replay(recording('slot1-qso-nc3z-dl4vcw'))
    seen = states(found, 1)
    assert [a.state for a in seen] == ['busy'] * len(seen) and seen
    # These names occur inside forwarded Cluster/VMail text, not a decoded
    # handshake or a standalone station signature (attribution fix e6f5c865c).
    assert seen[-1].calls == ()
    assert any('NC3Z' in s.text for s in said(found, 1))
    assert any('DL4VCW' in s.text for s in said(found, 1))
    # Text metadata updates the table without inventing a new RF identity event.
    final = next(a for a in m.table() if a.slot == 1)
    assert final.level in (1, 2, 3, 4) and {'VW', 'SM'} <= set(final.tags)
    assert states(found, 11) == []
    assert [r for r in found if r[0] in ('cq', 'beacon')] == []
    table = {a.slot: a for a in m.table()}
    assert table[1].state == 'busy' and table[11].state == 'quiet' and table[None].home


def test_a_link_on_the_slot_a_cq_named_reads_busy_once_with_both_stations():
    # The host heard KD5LEE's beacon 7.9 s in; AI7EQ's CQ for slot 1 the monitor reads
    # itself. The link's setup is unread, but its short frames close on KD5LEE's CRC byte.
    _, found = replay(recording('slot1-ai7eq-link'), heard=[(7.9, 'KD5LEE')])
    seen = states(found, 1)
    assert [(a.state, a.caller, a.called) for a in seen] == [('busy', '', 'AI7EQ'), ('busy', 'KD5LEE?', 'AI7EQ')]
    assert v.slot_line(seen[0]) == 'slot 1 busy: ? ↔ AI7EQ chatting, level 4'
    # The level-3 frame now identifies the caller 1.5 seconds after link evidence,
    # before the later level-1 frame and idle reply at the one-minute boundary.
    assert seen[1].at - seen[1].since == pytest.approx(1.5)
    assert v.slot_line(seen[1]) == 'slot 1 busy: KD5LEE? ↔ AI7EQ chatting, level 3'
    lines = [(s.kind, s.text) for s in said(found, 1)]
    assert lines[:4] == [('stage', 'link heard at level 4'), ('text', '73 My information:'),
                         ('text', '<NAME:Sean>'), ('text', '<QTH:Crooked River')]
    assert ('text', '21 Hi Lee.') in lines and ('stage', 'idle or polling') in lines
    level = next(text for kind, text in lines if kind == 'level')
    assert level.startswith('level 3, +') and 3 <= int(level[10:-3]) <= 12
    assert ('level', 'level 1, +8 dB') in lines


def test_one_side_of_a_link_holds_it_busy_through_the_far_sides_overs():
    _, found = replay(recording('slot1-ai7eq-gap'), heard=[(0, 'KD5LEE')])
    seen = states(found, 1)
    assert [a.state for a in seen] == ['busy'] * len(seen) and seen
    lines = said(found, 1)
    assert [(s.station, s.text) for s in lines if s.station] == [('AI7EQ', '15 de AI7EQ <R-10>')]
    texts = [s.at for s in lines if s.kind == 'text']
    assert max(b - a for a, b in zip(texts, texts[1:])) > 180
    last = seen[-1]
    assert (last.caller, last.called, last.one_side) == ('KD5LEE?', 'AI7EQ', True)
    assert v.slot_line(last).endswith('(one side heard)')


def stock_ident(call, at_hz):
    """Stock's closing ident: a 1500 Hz mark and 750 Hz space at 28 WPM, as bench 2
    keyed it (tests/kestrel/test_bw500_stock_ident_1021), moved to a channel."""
    from hfmodem.tests.kestrel.test_bw500_stock_ident_1021 import stock_ident as bench
    return S.shift_burst(bench(call), at_hz - S.CENTRE_HZ)


def receiver_noise(seconds, seed=23):
    rng = np.random.default_rng(seed)
    return sosfilt(butter(8, (100, 2950), 'bandpass', fs=S.FS, output='sos'), rng.normal(0, .05, seconds * S.FS))


def test_a_cw_ident_names_its_station_and_closes_the_link():
    x = receiver_noise(90)
    ident = stock_ident('WI9YZA', 2250)
    x[40 * S.FS:40 * S.FS + len(ident)] += 0.3 * ident
    _, found = replay(x)
    assert [s.text for s in said(found, 11)] == ['CW ident WI9YZA', 'quiet; the link is over']
    assert [(a.state, a.calls) for a in states(found, 11)] == [('busy', ('WI9YZA',)), ('quiet', ())]
    assert said(found, 1) == [] and said(found, None) == []


def test_the_slot_11_session_no_reader_decodes_shows_as_a_signal():
    _, found = replay(recording('slot11-unread-session'))
    assert [a.state for a in states(found, 11)][:1] == ['signal']
    assert 'busy' not in [a.state for a in states(found, 11)]
    assert 'busy' in [a.state for a in states(found, 1)]


def test_the_qso_reads_the_same_from_a_dial_350_hz_low():
    x = S.shift_burst(recording('slot1-qso-nc3z-dl4vcw'), -WIDE)
    m, found = replay(x, offset=WIDE)
    # The moved tape carries its passband 350 Hz up with it, and once the monitor has
    # measured that, slot 2 at 350 Hz is outside.
    assert [(a.slot, a.audio_hz) for a in m.table()] == [(1, 1100), (None, 1850), (11, 2600)]
    assert states(found, 1)[-1].calls == ()  # Embedded mail authors are not RF endpoints.
    assert states(found, 11) == [] and states(found, 2) == []


def test_receiver_noise_with_broadband_steps_raises_nothing():
    rng = np.random.default_rng(5)
    steps = np.repeat(rng.choice([1, 1.78], size=300), S.FS // 2)
    _, found = replay(receiver_noise(150) * steps)
    assert [record[0] for record in found] == ['passband']


def spectra(x):
    return [abs(np.fft.rfft(x[i:i + S.BLOCK] * np.hanning(S.BLOCK))) ** 2
            for i in range(0, S.MEASURE_S * S.FS, S.BLOCK)]


def test_the_passband_is_measured_from_the_noise_of_either_filter():
    narrow, opened = (S.measure(spectra(recording(name))) for name in ('noise-width3000', 'noise-opened'))
    # WIDTH 3000 behind the 3 kHz roofing filter ends in a cliff; opened by hand it is a skirt.
    assert 2800 < narrow.edges[6][1] < 2900 and 2900 < narrow.usable[1] < narrow.edges[10][1] < 3000
    assert opened.edges[6][1] > narrow.edges[6][1] + 100 and opened.edges[10][1] > 3300
    for passband in (narrow, opened):
        assert passband.usable[0] == S.CUTOFF_HZ and 70 < passband.edges[10][0] < 90
        assert [slot for slot, _ in S.channels(CF, CF, passband=passband.usable)] == [1, None, 11]
    assert narrow.moved(opened) and not narrow.moved(narrow)
    assert str(narrow).startswith('-3 dB ') and str(narrow).endswith(' Hz') and '-8 dB' not in str(narrow)


@pytest.mark.parametrize('passband,offset,held', [
    ((100, 2854), -350, [2, 1, None, 11]),
    ((100, 2952), 350, [1, None, 11, 12]),
    ((100, 3201), 200, [1, None, 11, 12]),
    ((117, 2273), 300, [1, None, 11]),
])
def test_the_wide_dial_fits_the_most_channels_with_ours_inside_the_tx_bandpass(passband, offset, held):
    assert S.wide_offset(passband) == offset
    assert [slot for slot, _ in S.channels(CF, CF, offset=offset, passband=passband)] == held
    ours = S.CENTRE_HZ - offset
    assert S.TX_BPF_HZ[0] <= ours - S.HALF_HZ and ours + S.HALF_HZ <= S.TX_BPF_HZ[1]


def test_a_tx_bandpass_too_narrow_for_any_offset_keeps_the_dial():
    assert S.wide_offset((100, 3500), tx=(1300, 1700)) == 0


def test_the_monitor_measures_at_the_start_and_after_a_retune_and_says_what_changed():
    m, found = replay(recording('noise-width3000'), offset=WIDE)
    said = [r[2] for r in found if r[0] == 'passband']
    assert len(said) == 1 and said[0].slots == (2, 1, None, 11)
    assert said[0].offset == WIDE and said[0].wide == 350
    assert said[0].line().startswith('passband -3 dB ')
    assert said[0].line().endswith('watching slot 2, slot 1, CF, slot 11 '
                                   '(--monitor-wide would sit the dial 350 Hz above the channel)')
    m.retune(CF)
    found = m.feed(recording('noise-opened')[:S.MEASURE_S * S.FS])
    again = [r[2] for r in found if r[0] == 'passband']
    assert len(again) == 1 and again[0].passband.edges[10][1] > 3300
    assert m.passband == again[0].passband.usable


def test_silence_while_keyed_is_never_a_passband():
    m = S.SlotMonitor(CF, CF)
    assert m.feed(np.zeros((S.MEASURE_S + 5) * S.FS)) == []
    assert m.passband == S.PASSBAND_HZ


def test_the_session_says_the_measured_passband():
    desk, watch = watched()
    measured = S.Measured(S.Passband({3: (117, 2531), 6: (94, 3023), 8: (88, 3290), 10: (82, 3562)}),
                          (1, None, 11, 12), 0, 300)
    watch.found = [('passband', None, measured)]
    desk.tick()
    assert desk.said[-1] == ('passband -3 dB 117-2531, -6 dB 94-3023, -10 dB 82-3562 Hz; '
                             'watching slot 1, CF, slot 11, slot 12')
    assert desk.kinds('passband')[-1]['slots'] == [1, None, 11, 12]


def test_moved_audio_streams_as_a_burst_moves_and_comes_back():
    t = np.arange(S.FS) / S.FS
    x = np.cos(2 * np.pi * 1500 * t) * np.hanning(S.FS)
    sent = S.shift_burst(x, -WIDE)
    spectrum = abs(np.fft.rfft(sent))
    assert np.argmax(spectrum) == 1850
    shift = S.AudioShift(WIDE)
    heard = np.concatenate([shift(sent[i:i + 512]) for i in range(0, len(sent), 512)])
    delay = S.AudioShift.TAPS // 2
    assert np.max(abs(heard[delay:] - x[:-delay])) < .02


class Watch:
    def __init__(self, *table):
        self.rows, self.found, self.tuned, self.calls = list(table), [], [], []

    def poll(self):
        found, self.found = self.found, []
        return found

    def table(self):
        return self.rows, 100.

    def retune(self, dial, calling=None):
        self.tuned.append(dial if calling is None else (dial, calling))

    def heard(self, call, slot=None):
        self.calls.append(call)


def watched(**replies):
    desk = Desk(**replies)
    desk.session.slots = watch = Watch(S.Activity(1, 750), S.Activity(None, 1500, home=True), S.Activity(11, 2250))
    return desk, watch


def test_slot_changes_are_said_once_and_ours_only_when_busy():
    desk, watch = watched()
    busy = S.Activity(1, 750, 'busy', 90., ('NC3Z',), 3)
    watch.found = [('state', 1, S.Activity(1, 750, 'quiet')), ('state', 1, busy), ('state', 1, busy),
                   ('state', None, S.Activity(None, 1500, 'signal', 99., home=True)),
                   ('state', 1, S.Activity(1, 750, 'busy', 95., ('NC3Z', 'DL4VCW'), 3)),
                   ('state', 1, S.Activity(1, 750, 'busy', 97., ('NC3Z', 'DL4VCW'), 4)),
                   ('state', 11, S.Activity(11, 2250, 'signal', 98.)),
                   ('state', 1, S.Activity(1, 750, 'quiet', 97.))]
    desk.tick()
    assert desk.said[1:] == ['slot 1 busy: NC3Z ↔ ?, level 3', 'slot 1 busy: NC3Z ↔ DL4VCW, level 3',
                             'slot 11 signal', 'slot 1 quiet']
    assert len(desk.kinds('slot')) == 8


def test_slots_lists_each_channel_and_the_monitoring_line_names_a_busy_one():
    desk, watch = watched()
    watch.rows[0] = S.Activity(1, 750, 'busy', 88., ('NC3Z', 'DL4VCW'), 3, ('VW', 'SM'))
    desk.do('/slots')
    assert desk.said[-3:] == ['slot 1    750 Hz  busy    heard 12 s ago  NC3Z ↔ DL4VCW  level 3  <VW> <SM>',
                              'CF       1500 Hz  quiet   nothing heard  (ours)',
                              'slot 11  2250 Hz  quiet   nothing heard']
    desk.session.monitor()
    assert desk.said[-2:] == [
        'monitoring 14105.000 kHz at 500 Hz, auto-answer on; slot 1 busy: NC3Z ↔ DL4VCW, level 3',
        'Passive transcripts are logged; /follow cf shows calling-frequency activity, '
        '/follow all shows the current passband, /follow off hides it. '
        '/follow SLOT or KHZ retunes if needed; /follow cf returns to CF.']


def test_a_busy_slot_says_its_stations_stage_level_and_age_or_that_none_is_heard():
    link = S.Activity(1, 750, 'busy', 200., ('KD5LEE', 'AI7EQ'), 3, caller='KD5LEE?', called='AI7EQ',
                      stage='chatting', since=100., at=230., one_side=True)
    assert v.slot_line(link) == 'slot 1 busy: KD5LEE? ↔ AI7EQ chatting, level 3, 2 min (one side heard)'
    signed = S.Activity(1, 750, 'busy', 200., ('AI7EQ',), 3, stage='chatting', since=100., at=230.)
    assert v.slot_line(signed) == 'slot 1 busy: AI7EQ ↔ ? chatting, level 3, 2 min'
    assert v.slot_line(S.Activity(1, 750, 'busy', 200., level=4, stage='chatting', since=190., at=200.)) == \
        'slot 1 busy (VARA link; stations not yet heard)'


def test_follow_prints_a_slots_qso_and_replays_the_last_two_minutes():
    desk, watch = watched()
    clock = [1000.]
    desk.session.client.clock = lambda: clock[0]
    lines = [S.Said(1, 10., 'stage', 'link heard at level 4'), S.Said(1, 11., 'text', '21 Hi Lee.'),
             S.Said(11, 12., 'stage', 'idle or polling'), S.Said(1, 13., 'text', '15 de AI7EQ <R-10>', 'AI7EQ')]
    watch.found = [('qso', said.slot, said) for said in lines[:3]]
    desk.tick()
    clock[0] += 100
    watch.found = [('qso', 1, lines[3])]
    desk.tick()
    assert not [line for line in desk.said if line.startswith('[')]
    assert [(e['slot'], e['qso'], e['text'], e['station']) for e in desk.kinds('slot')] == \
        [(1, 'stage', 'link heard at level 4', None), (1, 'text', '21 Hi Lee.', None),
         (11, 'stage', 'idle or polling', None), (1, 'text', '15 de AI7EQ <R-10>', 'AI7EQ')]
    clock[0] += 30
    desk.do('/follow 1')
    assert desk.said[-2:] == ['following slot 1; passive decoded activity only, text may be incomplete',
                                    '[slot 1 14104.250 kHz] AI7EQ (signed): 15 de AI7EQ <R-10>']
    watch.found = [('qso', 11, lines[2]), ('qso', 1, S.Said(1, 50., 'stage', 'CW ident AI7EQ'))]
    desk.tick()
    assert desk.said[-1] == '[slot 1 14104.250 kHz] station unverified: CW ident AI7EQ'
    desk.do('/follow off')
    watch.found = [('qso', 1, S.Said(1, 51., 'stage', 'quiet; the link is over'))]
    desk.tick()
    assert desk.said[-1] == 'follow off'
    desk.do('/follow 4')
    assert desk.tunes[-1] == v.slot_frequency(20, 4)
    assert desk.session.following == 4


def test_cqs_and_beacons_off_our_channel_join_the_heard_list_marked_with_it(monkeypatch):
    from hfmodem.kestrel.rx.beacon import BeaconResult
    from hfmodem.kestrel.rx.cq import CQResult
    monkeypatch.setattr(v.time, 'time', lambda: 1_761_264_000)
    desk, watch = watched()
    watch.found = [('cq', None, CQResult(b'', 'K7QB-2', 500, sn=-3.0)),
                   ('beacon', 11, BeaconResult('20OEN54FLKG9B', sn=4.0))]
    desk.tick()
    assert desk.said[-2:] == ['  1  K7QB     slot 2   14103.500 kHz   500 Hz  SN  -3.0  2025-10-24 00:00:00Z  via CF',
                              '  2  KG9B     beacon   14105.750 kHz   500 Hz  SN  +4.0  2025-10-24 00:00:00Z  EN54FL VMail email AI BBS  via slot 11']
    assert watch.calls == ['K7QB', 'KG9B']


def test_cq_without_a_slot_takes_one_the_monitor_holds_quiet():
    desk, watch = watched()
    watch.rows[0] = S.Activity(1, 750, 'busy', 99., ('NC3Z',))
    assert desk.session.free_slot() == 11
    watch.rows[2] = S.Activity(11, 2250, 'signal', 99.)
    assert desk.session.free_slot() == 2
    desk.hear('CQFRAME K4ABC-2 500')
    assert desk.session.free_slot() == 3


def test_a_tune_moves_the_watch():
    desk, watch = watched(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.do('/call K7QB 4')
    assert watch.tuned == [14_102_000]


def test_an_offset_dial_moves_what_the_modem_hears_and_keys(monkeypatch):
    import sys
    import types
    from hfmodem.tests.kestrel import corpora
    kc = corpora.harness('kestrel_connect')
    stream = types.SimpleNamespace(start=lambda: None, stop=lambda: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, 'sounddevice', types.SimpleNamespace(
        InputStream=lambda **_: stream, query_devices=lambda *_: {'max_output_channels': 2}))
    io = kc.AudioVaraIO('out', 'in', offset_hz=WIDE)
    raw = []
    io.taps.append(raw.append)
    t = np.arange(S.FS) / S.FS
    air = np.cos(2 * np.pi * 1850 * t).astype(np.float32)
    status = types.SimpleNamespace(input_overflow=False, input_underflow=False)
    for i in range(0, S.FS, 480):
        io._on_rx(air[i:i + 480, None], 480, types.SimpleNamespace(inputBufferAdcTime=i / S.FS), status)
    assert np.array_equal(np.concatenate(raw), air)
    assert np.argmax(abs(np.fft.rfft(np.concatenate(io._buf)))) == 1500
    played = []
    monkeypatch.setattr(kc, 'play_drained', lambda block, fs, device: played.append(block))
    io.tx(np.cos(2 * np.pi * 1500 * t) * np.hanning(S.FS))
    assert np.argmax(abs(np.fft.rfft(played[0][:, 0]))) == 1850


def test_the_watch_runs_apart_and_reports_its_table():
    import time
    watch = S.SlotWatch(CF, CF, mycall='W9SSJ')
    try:
        watch.put(np.zeros(S.BLOCK, np.float32))
        end = time.monotonic() + 60
        while watch.table()[1] < .5 and time.monotonic() < end:
            time.sleep(.05)
        table, now = watch.table()
        assert now == .5 and [a.slot for a in table] == [1, None, 11]
        assert watch.poll() == []
    finally:
        watch.close()
    assert not watch._process.is_alive()
