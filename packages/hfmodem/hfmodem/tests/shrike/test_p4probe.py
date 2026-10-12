"""P4 probes preserve full RX windows and cannot inherit P3 retry behaviour."""
import contextlib
import io
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from hfmodem.shrike import onair, p4probe
from hfmodem.tests.shrike.test_grantslot import _peer_wav

FS = onair.FS


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(onair, '_load', lambda _: np.zeros(90 * FS, np.float32))
    live = onair._ReplayInput('synthetic')
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live = live
    tx.p4_probe = p4probe.EntryProbe()
    grid = onair._MasterGrid(0, 60000, 0, packet_n=46080, cs_n=5760, d_max_n=6240)
    tx.aim(grid, 0)
    # Any invocation would reproduce the expensive P3 pre-key path's bug.
    tx.sessrx = SimpleNamespace(bridge=lambda _: pytest.fail('P3 bridge during probe'))
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **k: pytest.fail('P3 scan'))
    monkeypatch.setattr(onair, '_scan_previous_window', lambda *a, **k: pytest.fail('P3 scan'))
    tx.send_p4_entry_packet(b'', 0x31)
    return tx, live, grid


def test_grant_queues_no_immediate_transmission(setup):
    tx, live, _ = setup
    assert tx.p4_probe.requested and not tx.slots_used
    tx.send_p4_entry_packet(b'new', 0x32)
    assert tx.p4_probe.status == 0x31 and tx.p4_probe.payload == b''
    tx._tx(np.zeros(100), 'old ARQ retry')
    assert tx.refused and not tx.slots_used


def test_two_chirps_with_complete_receive_windows(setup, capsys):
    tx, live, grid = setup
    grid_before = (grid.protocol, grid.data_n, grid.slot_n)
    onair._run_p4_probe(tx, live, grid)
    probe = tx.p4_probe
    assert len(probe.emissions) == 2
    assert 'final receive window complete' in probe.reason
    for emitted in probe.emissions:
        window = next(w for w in probe.windows if w['start'] == emitted['audio_end'])
        assert window['end'] - window['start'] == 4 * FS
    assert probe.emissions[1]['audio_start'] - probe.emissions[0]['audio_end'] >= 4 * FS
    assert (grid.protocol, grid.data_n, grid.slot_n) == grid_before
    assert not probe.emitting and tx.sessrx is not None
    text = capsys.readouterr().out
    assert 'IS GONE' not in text and 'LATE TO THE KEY' not in text
    assert 'wrong rate' not in text and '106%' not in text


def test_guard_never_stands_down(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(grid, 'key_refusal', lambda *a, **k: 'recorded P1 reply conflict')
    tx.guard_drops = 100  # Previous refusals must not grant permission.
    onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emissions
    assert 'refused' in tx.p4_probe.reason
    assert tx.n == 1


def test_explicit_guard_override(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(grid, 'key_refusal', lambda *a, **k: 'recorded P1 reply conflict')
    tx.qrm_guard = False
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 2


@pytest.mark.parametrize('experimental,expected', [(False, 0), (True, 3)])
@pytest.mark.parametrize('grant', ['0x59A', '0x6A9'])
def test_real_grant_raster_allows_only_experimental_transition(setup, experimental, expected, grant):
    tx, live, grid = setup
    tx.p4_probe.counter_experiment = experimental
    tx.p4_probe.attempts = 3 if experimental else 2
    tx.p4_probe.trigger_word = grant
    grid.sending = True
    grid.note_peer_codeword(round(1.03 * FS), round(.12 * FS),
                            grant + '/unassigned', 'WS8EOC', protocol=onair.Protocol.PACTOR1)
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == expected
    assert tx.qrm_guard
    assert bool(tx.p4_probe.guard_exceptions) == experimental


@pytest.mark.parametrize('word,protocol', [
    ('CS1', onair.Protocol.PACTOR1), ('0x6A9/unassigned', onair.Protocol.PACTOR1),
    ('0x59A/unassigned', onair.Protocol.PACTOR3),
])
def test_other_controls_cannot_release_short_cycle_guard(setup, word, protocol):
    tx, _, grid = setup
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.emitting = True
    grid.sending = True
    grid.note_peer_codeword(49440, 5760, word, 'WS8EOC', protocol=protocol)
    assert tx._refused(58080, 160000, 'P4 challenge')
    assert not tx.p4_probe.guard_exceptions


def test_transition_does_not_key_inside_current_reply(setup):
    tx, _, grid = setup
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.emitting = True
    grid.sending = True
    grid.note_peer_codeword(49440, 5760, '0x59A/unassigned', 'WS8EOC',
                            protocol=onair.Protocol.PACTOR1)
    assert tx._refused(50000, 160000, 'P4 challenge')
    assert not tx.p4_probe.guard_exceptions


@pytest.mark.parametrize('attempts', [1, 4])
def test_burst_cap(setup, attempts):
    tx, live, grid = setup
    tx.p4_probe.attempts = attempts
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == attempts


def test_deadline_reserves_receive_window(setup):
    tx, live, grid = setup
    tx.p4_probe.timeout = 10
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 1
    assert 'deadline' in tx.p4_probe.reason
    assert live.pos <= 10 * FS


def test_wall_clock_deadline_even_with_stalled_sample_clock(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(p4probe.time, 'monotonic', lambda: tx.p4_probe.grant_wall + 61)
    onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emissions and 'deadline' in tx.p4_probe.reason


def test_interrupt_restores_state(setup, monkeypatch):
    tx, live, grid = setup
    def interrupt(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(tx, '_tx', interrupt)
    with pytest.raises(KeyboardInterrupt):
        onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emitting and tx.sessrx is not None


def test_truncated_capture_stops(setup):
    tx, live, grid = setup
    live.audio = live.audio[:5 * FS]
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 1
    assert 'capture ended' in tx.p4_probe.reason


def test_counter_experiment_emits_a_a_b_and_retains_each_reply(setup, tmp_path, monkeypatch):
    from hfmodem.shrike import p4experiment, p4chirprx
    tx, live, grid = setup
    emitted = []
    original = tx._tx
    def record(audio, label):
        emitted.append(audio.copy())
        return original(audio, label)
    monkeypatch.setattr(tx, '_tx', record)
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.attempts = 3
    onair._run_p4_probe(tx, live, grid)
    probe = tx.p4_probe
    assert [e['status'] for e in probe.emissions] == [0x61, 0x61, 0x62]
    hashes = [e['waveform_sha256'] for e in probe.emissions]
    assert hashes[0] == hashes[1] and hashes[2] != hashes[1]
    frames = [p4chirprx.decode_packet(audio).frame for audio in emitted]
    assert [f.status.byte for f in frames] == [0x61, 0x61, 0x62]
    assert frames[0].data == frames[1].data == frames[2].data
    assert len(probe.received) == 3
    assert all(len(x) == 4 * FS for _, _, x in probe.received)
    result = p4experiment.write_report(probe, tmp_path, mode='replay')
    assert result['complete'] and result['counter_verdict'] == 'inconclusive'
    assert result['setup_verdict'] == 'inconclusive'
    assert not result['mail_delivered'] and result['arq_acknowledgements_sent'] == 0
    assert (tmp_path / 'p4-mail-experiment.json').exists()
    with np.load(tmp_path / result['receive_artifact']) as saved:
        assert saved.files == ['probe0', 'probe1', 'probe2']


def test_analysis_failure_keeps_audio_and_inconclusive_report(setup, tmp_path, monkeypatch):
    from hfmodem.shrike import p4experiment
    tx, live, grid = setup
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.attempts = 3
    onair._run_p4_probe(tx, live, grid)
    def fail(*args):
        raise ValueError('analysis failure')
    monkeypatch.setattr(p4experiment, 'observe', fail)
    result = p4experiment.write_report(tx.p4_probe, tmp_path, mode='replay')
    assert len(result['analysis_errors']) == 3 and not result['complete']
    assert result['counter_verdict'] == 'inconclusive'
    assert (tmp_path / 'p4-mail-rx.npz').exists()


def test_full_counter_challenge_with_synthetic_reply_changes(setup, tmp_path, monkeypatch):
    from hfmodem.shrike import p4experiment
    from hfmodem.tests.shrike.test_p4sig import _synth_header
    tx, live, grid = setup
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.attempts = 3
    original = tx._tx
    roots = iter((14, 14, 7))
    def respond(audio, label):
        result = original(audio, label)
        header = _synth_header(next(roots), 3, cfo=6)
        start = tx.tx_end + FS // 10
        live.audio[start:start + len(header)] = header
        return result
    monkeypatch.setattr(tx, '_tx', respond)
    onair._run_p4_probe(tx, live, grid)
    result = p4experiment.write_report(tx.p4_probe, tmp_path, mode='replay')
    assert result['complete']
    assert result['counter_verdict'] == 'candidate-pattern-consistent-with-counter-sensitive-replies'
    assert [o['p4_candidate']['root'] for o in result['observations']] == [14, 14, 7]
    assert result['arq_acknowledgements_sent'] == 0 and not result['mail_delivered']


def test_counter_experiment_guard_refusal_is_inconclusive(setup, tmp_path, monkeypatch):
    from hfmodem.shrike import p4experiment
    tx, live, grid = setup
    tx.p4_probe.counter_experiment = True
    tx.p4_probe.attempts = 3
    monkeypatch.setattr(grid, 'key_refusal', lambda *a, **k: 'reply conflict')
    onair._run_p4_probe(tx, live, grid)
    result = p4experiment.write_report(tx.p4_probe, tmp_path, mode='replay')
    assert not result['complete'] and not result['emissions']
    assert result['setup_verdict'] == result['counter_verdict'] == 'inconclusive'


def _air(tmp_path, *flags):
    """One whole session against a station that answers the call and grants nothing."""
    wav = tmp_path / 'peer.wav'
    if not wav.exists():
        _peer_wav(wav, slots=16, grant=False)
    save = onair._save_capture_async
    onair._save_capture_async = lambda *a, **kw: None
    argv, log = sys.argv, io.StringIO()
    sys.argv = ['shrike.onair', '--replay', str(wav), '--hold', '10',
                '--max-cycles', '4', '--mycall', 'W9SSJ', '--dxcall', 'K7ABC',
                '--dial', '7100000', '--outdir', str(tmp_path / 'out'), *flags]
    try:
        with contextlib.redirect_stdout(log):
            onair.main()
    finally:
        onair._save_capture_async = save
        sys.argv = argv
    return log.getvalue()


def test_a_listening_arm_reads_its_cycles_out_without_keying(tmp_path):
    log = _air(tmp_path, '--after-connect', 'listen', '--after-connect-cycles', '3')
    assert '[after-connect] listening 3 cycles; keying nothing' in log, log
    assert log.count('OBSERVING, not keying') == 3, log
    after = log[log.index('** CONNECTED'):]
    assert 'TX[' not in after, after
    assert 'the after-connect listen read 3 cycles without keying' in log, log
    # The link is over, and a goodbye queued on a machine nothing ticks again
    # would leave the summary reporting one that is still up.
    assert log.rindex('state CONNECTED -> DISCONNECTED') > log.rindex('OBSERVING'), log


def test_the_connect_answer_starts_the_probe_without_a_grant(tmp_path):
    log = _air(tmp_path, '--after-connect', 'chirp', '--p4-entry-attempts', '2',
               '--no-qrm-guard')
    assert '0x59A' not in log, log
    after = log[log.index('** CONNECTED'):]
    assert 'P4 PROBE CHIRP 1/2' in after and 'P4 PROBE CHIRP 2/2' in after, after
    assert 'P4 probe: 2/2 chirps' in log, log
    assert 'SL1 ENTRY' not in log and 'P1 DATA' not in log, log


def test_a_schedule_keys_the_operators_bytes_and_records_the_pairs(setup, capsys):
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.schedule = (0x63, 0x60, 0x61)
    probe.attempts = 3
    onair._run_p4_probe(tx, live, grid)
    assert [e['status'] for e in probe.emissions] == [0x63, 0x60, 0x61]
    assert [q['status_hex'] for q in probe.queries] == ['0x63', '0x60', '0x61']
    assert [q['counter'] for q in probe.queries] == [3, 0, 1]
    assert all(q['reply_class'] == 'none' for q in probe.queries)
    out = capsys.readouterr().out
    assert '[p4 probe] query schedule 0x63,0x60,0x61' in out
    assert out.count('[p4 query]') == 3
    assert probe.result(FS)['queries'] == probe.queries


def test_the_field_the_operator_named_is_the_one_keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(onair, '_load', lambda _: np.zeros(20 * FS, np.float32))
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live = onair._ReplayInput('synthetic')
    tx.p4_probe = p4probe.EntryProbe(chirp_field=b'W9SSJ')
    tx.send_p4_entry_packet(b'', 0x31)
    assert tx.p4_probe.payload == b'W9SSJ'
    assert tx.p4_probe.result(FS)['payload_len'] == 5


def test_the_counter_experiment_keeps_its_preregistered_schedule():
    probe = p4probe.EntryProbe(3, counter_experiment=True)
    assert probe.statuses() == [0x61, 0x61, 0x62]
    probe.schedule = (0x63,)
    assert probe.statuses() == [0x63]


@pytest.mark.parametrize('flags,message', [
    (['--p4-mail', '--p4-probe-statuses', '0x63'], 'preregistered'),
    (['--p4-mail', '--p4-probe-payload', 'W9SSJ'], 'preregistered'),
    (['--p4-probe-statuses', '0x63'], 'which is off'),
    (['--p4-entry', '--p4-probe-statuses', '0x63,0x60,0x61,0x62,0x63'], 'one to four'),
    (['--p4-entry', '--p4-probe-statuses', '0x100'], 'one to four'),
    (['--p4-entry', '--p4-probe-statuses', 'CS1'], 'byte literals'),
])
def test_the_schedule_is_refused_before_the_rig_opens(monkeypatch, capsys, flags, message):
    monkeypatch.setattr(onair, 'run', lambda args: pytest.fail('opened run'))
    monkeypatch.setattr(sys, 'argv', ['onair', *flags])
    with pytest.raises(SystemExit) as refusal:
        onair.main()
    said = capsys.readouterr()
    assert message in f'{refusal.value}{said.err}{said.out}'


def test_the_schedule_sets_the_burst_count(monkeypatch):
    got = []
    monkeypatch.setattr(onair, 'run', lambda args: got.append(args) or 0)
    monkeypatch.setattr(sys, 'argv', ['onair', '--p4-entry', '--p4-probe-statuses',
                                      '0x63,0x60,0x61', '--p4-probe-payload', 'W9SSJ'])
    assert onair.main() == 0
    assert got[0].p4_probe_statuses == (0x63, 0x60, 0x61)
    assert got[0].p4_entry_attempts == 3 and got[0].p4_probe_payload == b'W9SSJ'


@pytest.mark.parametrize('packet,expected', [
    ('chirp', 0), ('short', 2), ('robust', 2)])
def test_a_cycle_sized_packet_fits_where_the_chirp_overruns(setup, capsys,
                                                            packet, expected):
    """The real grant raster: 3.3 s crosses the peer's slot and 0.76-0.79 s does not."""
    from hfmodem.shrike import p4robusttx, p4shorttx
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet = packet
    if packet != 'robust':
        probe.schedule = (0x63, 0x60)
    grid.sending = True
    grid.note_peer_codeword(round(1.03 * FS), round(.12 * FS), '0x59A/unassigned',
                            'WS8EOC', protocol=onair.Protocol.PACTOR1)
    onair._run_p4_probe(tx, live, grid)
    assert tx.qrm_guard and not probe.guard_exceptions
    assert len(probe.emissions) == expected
    if not expected:
        assert 'refused' in probe.reason
        return
    clone = packet == 'robust'
    assert [e['status'] for e in probe.emissions] == (
        [None, None] if clone else [0x63, 0x60])
    length = p4robusttx.EMISSION_S if clone else p4shorttx.EMISSION_S
    keyed = [e['audio_end'] - e['audio_start'] for e in probe.emissions]
    assert all(abs(n / FS - length) < .01 for n in keyed), keyed
    out = capsys.readouterr().out
    name = 'ROBUST' if clone else 'SHORT'
    assert f'P4 PROBE {name} 1/2' in out and f'P4 PROBE {name} 2/2' in out
    assert f'{packet} packet' in out and 'CHIRP' not in out
    assert [q['status_hex'] for q in probe.queries] == (
        ['clone', 'clone'] if clone else ['0x63', '0x60'])
    assert all(q['payload_len'] == 0 for q in probe.queries) or not clone
    assert probe.result(FS)['packet'] == packet


def test_a_locked_peer_offset_is_kept_for_the_reader(setup):
    tx, live, grid = setup
    tx.sessrx = SimpleNamespace(bridge=lambda _: None, p3_receive_offset_hz=8.0)
    onair._run_p4_probe(tx, live, grid)
    assert tx.p4_probe.cfo_hz == 8.0
    assert all(q['cfo_locked'] for q in tx.p4_probe.queries)
    assert tx.p4_probe.result(FS)['peer_offset_hz'] == 8.0


def test_an_unmeasured_peer_offset_leaves_the_search_free(setup):
    tx, live, grid = setup
    onair._run_p4_probe(tx, live, grid)
    assert tx.p4_probe.cfo_hz is None
    assert not any(q['cfo_locked'] for q in tx.p4_probe.queries)


def test_the_short_packet_is_refused_where_the_probe_is_not_running(monkeypatch, capsys):
    monkeypatch.setattr(onair, 'run', lambda args: pytest.fail('opened run'))
    monkeypatch.setattr(sys, 'argv', ['onair', '--p4-probe-packet', 'short'])
    with pytest.raises(SystemExit) as refusal:
        onair.main()
    said = capsys.readouterr()
    assert 'which is off' in f'{refusal.value}{said.err}{said.out}'


def test_a_short_arm_answers_the_connect_with_a_normal_short_packet(tmp_path):
    log = _air(tmp_path, '--after-connect', 'chirp', '--p4-entry-attempts', '2',
               '--p4-probe-packet', 'short')
    after = log[log.index('** CONNECTED'):]
    assert 'P4 PROBE SHORT 1/2' in after and 'P4 PROBE SHORT 2/2' in after, after
    assert 'P1 DATA' not in after and 'SL1 ENTRY' not in after, after
    assert 'P4 PROBE CHIRP' not in log, log
    assert 'P4 probe: 2/2 chirps' in log, log


def test_the_live_pass_defers_the_root_and_the_teardown_names_it(setup, monkeypatch, capsys):
    from hfmodem.tests.shrike.test_p4oracle import _robust
    tx, live, grid = setup
    probe = tx.p4_probe
    original = tx._tx
    def respond(audio, label):
        result = original(audio, label)
        burst = _robust(85, cfo=6.0)
        start = tx.tx_end + FS // 2
        live.audio[start:start + burst.size] = burst
        return result
    monkeypatch.setattr(tx, '_tx', respond)
    spent = []
    for module, name in ((onair.p4sig, 'header_score'),
                         (onair.p4oracle, 'despread_word'),
                         (onair.p4oracle, 'training_lock')):
        real = getattr(module, name)
        monkeypatch.setattr(module, name,
                            lambda *a, _r=real, _n=name, **k: (spent.append(_n)
                                                               or _r(*a, **k)))
    onair._run_p4_probe(tx, live, grid)
    # NOTHING EXPENSIVE ON THE PROBE'S OWN CLOCK: the family, and no more.
    assert not spent
    assert [q['reply_class'] for q in probe.queries] == ['p4-robust?'] * 2
    assert all(r['root'] is None and r['word'] is None
               for q in probe.queries for r in q['replies'])
    assert capsys.readouterr().out.count('p4-robust?') == 2
    onair._resolve_p4_queries(probe)
    assert 'header_score' in spent and 'despread_word' in spent
    assert [q['reply_class'] for q in probe.queries] == ['p4-robust-short'] * 2
    assert all(r['root'] is not None for q in probe.queries for r in q['replies'])
    assert [q['index'] for q in probe.queries] == [1, 2]
    assert all(q['headers_read'] for q in probe.queries)
    assert 'root' in capsys.readouterr().out


def test_a_window_cut_short_is_read_once_at_teardown(setup, capsys):
    tx, live, grid = setup
    live.audio = live.audio[:5 * FS]
    onair._run_p4_probe(tx, live, grid)
    assert 'capture ended' in tx.p4_probe.reason
    assert not tx.p4_probe.queries
    assert '[p4 query]' not in capsys.readouterr().out
    onair._resolve_p4_queries(tx.p4_probe)
    assert len(tx.p4_probe.queries) == 1
    assert '[p4 query] 1 ' in capsys.readouterr().out


def test_the_prekey_intervals_are_the_floor_a_filled_window_is_read_against(setup):
    tx, live, grid = setup
    onair._run_p4_probe(tx, live, grid)
    assert tx.p4_probe.floor_db is not None
    assert tx.p4_probe.result(FS)['quiet_floor_db'] == tx.p4_probe.floor_db
    onair._resolve_p4_queries(tx.p4_probe)
    rows = tx.p4_probe.queries
    assert all(q['window_floor_db'] == round(tx.p4_probe.floor_db, 1) for q in rows)
    assert all(q['reply_class'] == 'none' for q in rows)


def test_a_clone_query_carries_no_status_byte_of_ours(setup):
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet = 'robust'
    probe.payload = b'W9SSJ'
    onair._run_p4_probe(tx, live, grid)
    onair._resolve_p4_queries(probe)
    assert probe.statuses() == [None, None]
    assert [q['status_hex'] for q in probe.queries] == ['clone', 'clone']
    assert all(q['counter'] is None and q['payload_len'] == 0
               for q in probe.queries)
    assert probe.result(FS)['statuses'] == [None, None]


@pytest.mark.parametrize('extra', [
    ['--p4-probe-statuses', '0x63'], ['--p4-probe-payload', 'W9SSJ']])
def test_a_robust_packet_takes_no_status_schedule_of_ours(monkeypatch, capsys, extra):
    monkeypatch.setattr(onair, 'run', lambda args: pytest.fail('opened run'))
    monkeypatch.setattr(sys, 'argv', ['onair', '--p4-entry',
                                      '--p4-probe-packet', 'robust', *extra])
    with pytest.raises(SystemExit) as refusal:
        onair.main()
    said = capsys.readouterr()
    assert 'datablock value' in f'{refusal.value}{said.err}{said.out}'


def test_a_clone_arm_answers_the_connect_with_the_tapes_own_packet(tmp_path):
    log = _air(tmp_path, '--after-connect', 'chirp', '--p4-entry-attempts', '2',
               '--p4-probe-packet', 'robust')
    after = log[log.index('** CONNECTED'):]
    assert 'P4 PROBE ROBUST 1/2' in after and 'P4 PROBE ROBUST 2/2' in after, after
    assert 'P1 DATA' not in after and 'SL1 ENTRY' not in after, after
    assert 'P4 PROBE CHIRP' not in log and 'P4 PROBE SHORT' not in log, log
    assert 'status=clone' in log, log


def test_a_clone_reports_the_tapes_field_on_both_paths_out(setup, monkeypatch):
    """A reading that fails must not report a field length the success never does."""
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet = 'robust'
    probe.payload = b'W9SSJ'
    onair._run_p4_probe(tx, live, grid)
    monkeypatch.setattr(onair.p4oracle, 'query',
                        lambda *a, **k: (_ for _ in ()).throw(ValueError('nope')))
    onair._resolve_p4_queries(probe)
    assert all(q['reply_class'] == 'unread' for q in probe.queries)
    assert all(q['payload_len'] == 0 and q['status_hex'] == 'clone'
               for q in probe.queries)


def test_each_value_keys_its_own_packet(setup, capsys):
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet, probe.attempts = 'robust', 3
    probe.values, probe.headers = (6, 9, 8), ('A', 'A', 'A')
    onair._run_p4_probe(tx, live, grid)
    assert len(probe.emissions) == 3
    assert [e['value'] for e in probe.emissions] == [6, 9, 8]
    # A value is the whole of what one packet says, so three values are three
    # different emissions and not one repeated.
    assert len({e['waveform_sha256'] for e in probe.emissions}) == 3
    out = capsys.readouterr().out
    for n, v in enumerate((6, 9, 8), 1):
        assert f'P4 PROBE ROBUST {n}/3 v={v} hdr=A' in out, out
    assert [q['status_hex'] for q in probe.queries] == ['v6/A', 'v9/A', 'v8/A']
    assert all(q['counter'] is None and q['payload_len'] == 0
               for q in probe.queries)
    assert probe.result(FS)['values'] == [6, 9, 8]


def test_the_header_word_rides_beside_the_value(setup, capsys):
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet, probe.attempts = 'robust', 2
    probe.values, probe.headers = (6, 6), ('A', 'B')
    onair._run_p4_probe(tx, live, grid)
    assert [e['header'] for e in probe.emissions] == ['A', 'B']
    assert len({e['waveform_sha256'] for e in probe.emissions}) == 2
    assert [q['status_hex'] for q in probe.queries] == ['v6/A', 'v6/B']
    assert 'v=6 hdr=B' in capsys.readouterr().out
    assert probe.result(FS)['headers'] == ['A', 'B']


def test_without_values_the_probe_still_repeats_the_tape(setup, capsys):
    tx, live, grid = setup
    probe = tx.p4_probe
    probe.packet = 'robust'
    onair._run_p4_probe(tx, live, grid)
    assert len({e['waveform_sha256'] for e in probe.emissions}) == 1
    assert [q['status_hex'] for q in probe.queries] == ['clone', 'clone']
    assert 'v=' not in capsys.readouterr().out


def test_a_value_arm_answers_the_connect_with_one_packet_each(tmp_path):
    log = _air(tmp_path, '--after-connect', 'chirp', '--p4-probe-packet', 'robust',
               '--p4-probe-values', '6,9,8')
    after = log[log.index('** CONNECTED'):]
    assert 'P4 PROBE ROBUST 1/3 v=6 hdr=A' in after, after
    assert 'P4 PROBE ROBUST 2/3 v=9 hdr=A' in after, after
    assert 'P4 PROBE ROBUST 3/3 v=8 hdr=A' in after, after
    assert 'status=v6/A' in log and 'status=v8/A' in log, log


@pytest.mark.parametrize('flags,message', [
    (['--p4-entry', '--p4-probe-values', '6'], 'robust datablock'),
    (['--p4-entry', '--p4-probe-packet', 'short', '--p4-probe-values', '6'],
     'robust datablock'),
    (['--p4-entry', '--p4-probe-packet', 'robust', '--p4-probe-headers', 'A'],
     'names none'),
    (['--p4-entry', '--p4-probe-packet', 'robust', '--p4-probe-values', '64'],
     '0..63'),
    (['--p4-entry', '--p4-probe-packet', 'robust', '--p4-probe-values', '1,2,3,4,5'],
     'one to four'),
    (['--p4-entry', '--p4-probe-packet', 'robust', '--p4-probe-values', '6,9',
      '--p4-probe-headers', 'A,B,A'], 'one header per'),
    (['--p4-entry', '--p4-probe-packet', 'robust', '--p4-probe-values', '6',
      '--p4-probe-headers', 'C'], "'alt' or one"),
    (['--p4-mail', '--p4-probe-values', '6'], 'preregistered'),
])
def test_a_datablock_query_is_refused_before_the_rig_opens(monkeypatch, capsys,
                                                           flags, message):
    monkeypatch.setattr(onair, 'run', lambda args: pytest.fail('opened run'))
    monkeypatch.setattr(sys, 'argv', ['onair', *flags])
    with pytest.raises(SystemExit) as refusal:
        onair.main()
    said = capsys.readouterr()
    assert message in f'{refusal.value}{said.err}{said.out}'


def test_alt_alternates_the_header_words(monkeypatch):
    got = []
    monkeypatch.setattr(onair, 'run', lambda args: got.append(args) or 0)
    monkeypatch.setattr(sys, 'argv', ['onair', '--p4-entry', '--p4-probe-packet',
                                      'robust', '--p4-probe-values', '6,6,6,6',
                                      '--p4-probe-headers', 'alt'])
    assert onair.main() == 0
    assert got[0].p4_probe_headers == ('A', 'B', 'A', 'B')
    assert got[0].p4_entry_attempts == 4
