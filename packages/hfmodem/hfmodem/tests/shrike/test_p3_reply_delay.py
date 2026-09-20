"""Intentional B delay shifts the schedule before RX/PTT, not a late clamp."""
import json

import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p3trial import TimingTrial, CYCLE, ROTATION
from hfmodem.tests.shrike.test_entry_delay import _parsed
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive


@pytest.mark.parametrize('ci', [0, 1])
def test_reply_delay_moves_dac_ptt_and_deadline_together(tmp_path, capsys, ci):
    cases = []
    for delay in (0.0, 3.125):
        directory = tmp_path/str(delay)
        directory.mkdir()
        s, tx, live, grid = setup(directory, 'B', delay=234)
        trial = tx.timing_trial
        trial.reply_delay_ms = delay
        trial.unbounded = True
        entry = trial.entry_phase
        receive(s, tx, live, grid, entry+44400)
        tx.aim(grid, 1)
        tx._pending_p3_cs = ci
        onair._p3_place_reply(grid, tx, 1)
        key = tx.key_instant(grid, 1)
        deadline = onair._p3_decode_deadline(live, key, 1920)
        tx.emit_pending_cs()
        assert not tx.refused and tx.slot == 1
        pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
        assert (pulse-entry)%CYCLE == ROTATION+round(delay*48)
        assert trial.replies[-1]['reply_delay_ms'] == delay
        sidecar = json.loads((directory/'tx_02.json').read_text())
        assert sidecar['timing_reply_delay_ms'] == delay
        trial.write(directory)
        report = json.loads((directory/trial.filename).read_text())
        assert report['target_entry_rotation_ms'] == 600+delay
        cases.append(dict(entry=entry, key=key, deadline=deadline,
                          audio=tx.tx_audio_start, ptt=tx.tx_key_up,
                          first_reply=trial.first_reply,
                          waveform=(directory/'tx_02.wav').read_bytes()))
    a,b = cases
    assert a['entry'] == b['entry']
    assert a['waveform'] == b['waveform']
    for key in ('key', 'deadline', 'audio', 'ptt', 'first_reply'):
        assert b[key]-a[key] == 150, (key,a,b)
    assert b['audio']-b['ptt'] == a['audio']-a['ptt']
    log = capsys.readouterr().out
    assert 'KEYED INTO ITS BOUNDARY' not in log and 'LATE TO THE KEY' not in log


def test_delayed_trial_retries_do_not_walk_clock_or_reset_first_opportunity():
    t = TimingTrial('B', unbounded=True, reply_delay_ms=3.125)
    t.entry(1000,1)
    t.packet(45000,0,b'RMS',breakin=True,long_cycle=False)
    epoch = t.target(45000,852)
    first = t.first_reply
    for k in range(1,30):
        phase = 45000+k*CYCLE
        t.packet(phase,k%4,b'abcde',breakin=False,long_cycle=False)
        assert t.target(phase,852) == epoch
        assert t.first_reply == first
        assert t.attempt(first+k*CYCLE)


@pytest.mark.parametrize('flags', [(), ('--p3-timing-trial','A'),
                                  ('--p3-entry-timing-trial','B')])
def test_reply_delay_requires_b_before_io(flags):
    with pytest.raises(SystemExit,match='requires --p3-timing-trial B'):
        onair._timing_trial_defaults(_parsed(*flags,'--p3-timing-reply-delay','3.125'))


def test_reply_delay_configuration_reaches_defaults(capsys):
    args = _parsed('--p3-timing-trial','B','--p3-timing-unbounded',
                   '--p3-timing-reply-delay','3.125')
    onair._timing_trial_defaults(args)
    assert args.p3_entry_delay == 4.875 and args.p3_timing_reply_delay == 3.125
    assert '+603.125 ms' in capsys.readouterr().out
    with pytest.raises(ValueError):
        TimingTrial('A',reply_delay_ms=3.125)
    with pytest.raises(ValueError):
        TimingTrial('B',reply_delay_ms=float('nan'))
