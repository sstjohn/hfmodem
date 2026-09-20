"""Extended receive attempts preserve B timing without experiment cutoffs."""
import json
import pytest
import sys
import numpy as np

from hfmodem.shrike import onair, placement
from hfmodem.shrike.p3trial import TimingTrial, FS, CYCLE, ROTATION
from hfmodem.tests.shrike.test_entry_delay import _parsed
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive
from hfmodem.tests.shrike.archive import requires_ws8eoc_p3


def test_unbounded_counters_wrap_duplicates_do_not_advance_and_text_can_repeat(tmp_path):
    t = TimingTrial('B', unbounded=True)
    t.entry(1000, 1)
    t.packet(45000, 0, b'RMS', breakin=True, long_cycle=False)
    t.target(45000, 852)
    for k in range(1, 42):
        phase = 45000 + k*CYCLE
        # Matching payload bytes are legal on new packet counters.
        t.packet(phase, k % 4, b'aaaaa', breakin=False, long_cycle=False)
        t.packet(phase+30000, k % 4, b'aaaaa', breakin=False, long_cycle=False)
        assert t.attempt(t.first_reply+k*CYCLE)
    assert len(t.progress) == 41 and t.reason is None
    assert t.check(3600*FS) is None
    assert t.opportunities > 12
    t.write(tmp_path)
    result = json.loads((tmp_path/t.filename).read_text())
    assert result['progression'] and result['unbounded']
    assert all(value is None for value in result['limits'].values())


def test_actual_b_reply_after_old_deadline_still_uses_entry_epoch(tmp_path):
    s, tx, live, g = setup(tmp_path, 'B', delay=234)
    t = tx.timing_trial
    t.unbounded = True
    receive(s, tx, live, g, t.entry_phase+round(.925*FS))
    # Supply a fresh peer CRC after the old 20-second deadline. Stale-frame
    # permission is intentionally not bypassed by the unbounded option.
    head = t.opened + 20*CYCLE
    raw = placement.changeover_packet(b'RMS', 0, swapped=False)
    center = int(np.argmax(placement.protocol_config().pulse()))
    audio = np.pad(raw, (9600-center, FS//10))
    s.rx.new_cycle()
    live.now = live.pos = head + round(.84*FS)
    tx.aim(g, 21)
    s.rx.control_signal(audio, head-9600, head)
    onair._p3_place_reply(g, tx, 21)
    tx.emit_pending_cs()
    assert not tx.refused and t.reason is None
    pulse = tx.tx_audio_start+min(tx.tx_pulse_offsets)
    assert pulse > t.opened+20*FS
    assert (pulse-t.entry_phase) % CYCLE == ROTATION
    assert onair._timing_trial_close(tx, s.host, live.now) is None


@pytest.mark.parametrize('status,long,reason', [
    (0x80, False, 'QRT'), (0x41, False, 'role reversal'), (1, True, 'long cycle')])
def test_peer_end_and_geometry_changes_still_stop(status, long, reason):
    t = TimingTrial('B', unbounded=True)
    t.entry(1000,1)
    t.packet(45000,0,b'RMS',breakin=True,long_cycle=False)
    t.packet(105000,status,b'hello',breakin=False,long_cycle=long)
    assert reason in t.reason


def test_hold_is_unbounded_but_explicit_close_still_works():
    b = onair._HoldBudget(32, unbounded=True)
    b.spend(10000)
    b.moved(10000)
    assert b.deadline == float('inf')
    b.close(10001)
    assert b.closed and b.deadline == 10001
    ordinary = onair._HoldBudget(32)
    assert ordinary.deadline == 32


def test_unbounded_requires_reply_trial_before_io():
    for flags in [(), ('--p3-entry-timing-trial','B')]:
        with pytest.raises(SystemExit, match='requires --p3-timing-trial'):
            onair._timing_trial_defaults(_parsed('--p3-timing-unbounded', *flags))
    args = _parsed('--p3-timing-trial','B','--p3-timing-unbounded')
    onair._timing_trial_defaults(args)
    assert args.p3_entry_delay == 4.875 and args.hold and not args.mail_fetch


@requires_ws8eoc_p3
def test_recorded_loop_continues_after_three_fields_then_ends_on_link_loss(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_late_entry_loop as loop
    main = onair.main
    def extended():
        sys.argv += ['--p3-timing-trial','B','--p3-timing-unbounded']
        return main()
    monkeypatch.setattr(onair, 'main', extended)
    got = loop.run_late(monkeypatch, tmp_path, 1, False)
    report = json.loads((tmp_path/'p3-timing-trial.json').read_text())
    assert report['unbounded'] and len(report['progress']) == 3, got['log']
    assert report['opportunities'] > 12
    assert report['replies'][-1]['audio_start'] > report['progress'][-1]['phase']+CYCLE
    assert 'limit' not in report['reason'] and 'PTT off.' in got['log']
    assert 'without an experiment/hold cutoff' in got['log']
