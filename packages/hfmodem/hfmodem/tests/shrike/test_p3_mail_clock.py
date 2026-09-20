"""Mail keeps the proven first-turn timing without trial termination rules."""
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p3trial import ReplyClock, CYCLE
from hfmodem.tests.shrike.test_entry_delay import _parsed
from hfmodem.tests.shrike.test_p3_timing_trial import setup, receive


@pytest.mark.parametrize('ci', [0, 1])
def test_mail_first_reply_matches_delayed_trial(tmp_path, ci):
    results = []
    for mail in (False, True):
        directory = tmp_path/str(mail)
        directory.mkdir()
        s, tx, live, grid = setup(directory, 'B', delay=234)
        entry = tx.timing_trial.entry_phase
        if mail:
            tx.timing_trial = None
            tx.reply_clock = ReplyClock()
            tx.reply_clock.entry(entry)
        else:
            tx.timing_trial.reply_delay_ms = 3.125
        receive(s, tx, live, grid, entry+44400)
        tx.aim(grid, 1)
        tx._pending_p3_cs = ci
        onair._p3_place_reply(grid, tx, 1)
        key = tx.key_instant(grid, 1)
        tx.emit_pending_cs()
        assert not tx.refused
        pulse = tx.tx_audio_start + min(tx.tx_pulse_offsets)
        assert (pulse-entry) % CYCLE == 28950
        assert onair._timing_trial_close(tx, s.host, live.now) is None
        results.append((key, tx.tx_audio_start, tx.tx_key_up,
                        (directory/'tx_02.wav').read_bytes()))
    assert results[0] == results[1]


def test_clock_rotates_only_after_accepted_turns(tmp_path):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock()
    tx.reply_clock.entry(entry)
    receive(s, tx, live, grid, entry+44400)
    tx.aim(grid, 1)
    onair._p3_place_reply(grid, tx, 1)
    first = tx.reply_clock.pulse_epoch
    assert first == entry+28950  # Initial IRS reversal was not counted twice.
    grid.reverse(to_iss=True)
    assert tx.reply_clock.pulse_epoch == first
    grid.turn_accepted = False
    grid.reverse(to_iss=False)
    assert tx.reply_clock.pulse_epoch == first
    grid.reverse(to_iss=True)
    grid.turn_accepted = True
    grid.reverse(to_iss=False)
    assert tx.reply_clock.pulse_epoch == first+28800
    tx.aim(grid, 3)
    onair._p3_place_reply(grid, tx, 3)
    assert (grid.boundary(3)+852-entry) % CYCLE == (28950+28800) % CYCLE


def test_only_emitted_entry_establishes_mail_clock(tmp_path):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    tx.timing_trial = None
    tx.reply_clock = ReplyClock()
    tx.listening = True
    tx.send_entry_packet(1, b'', 0x1a)
    assert tx.refused and tx.reply_clock.entry_phase is None
    assert tx.reply_clock.target(852) is None
    tx.listening = False
    tx.aim(grid, 2)
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused
    assert tx.reply_clock.entry_phase == tx.tx_audio_start+min(tx.tx_pulse_offsets)


@pytest.mark.parametrize('flags', [
    ('--p3-timing-trial', 'B'), ('--p3-timing-unbounded',),
    ('--pactor1-only',), ('--p4-entry',), ('--p3-changeover-p1-cs',),
    ('--p3-control-placement', 'pulse-center'), ('--p3-entry-delay', '0'),
])
def test_mail_profile_refuses_conflicts_before_io(flags):
    with pytest.raises(SystemExit):
        _parsed('--p3-mail', *flags)


def test_mail_profile_uses_normal_application_limits():
    a = _parsed('--p3-mail')
    assert a.mail_fetch and a.mail_wait_greeting
    assert a.p3_speed_up == 'auto' and a.p3_repeat_gear == 0
    assert a.p1_grant_only and a.no_p3_fallback and a.hold > 0
    assert not a.p3_timing_trial and not a.p3_timing_unbounded


def test_mail_profile_preserves_explicit_speed_hold():
    a = _parsed('--p3-mail', '--p3-speed-up', 'hold')
    onair._p3_mail_defaults(a)
    assert a.p3_speed_up == 'hold' and a.p3_repeat_gear == 0


@pytest.mark.parametrize('policy', ['auto', 'hold'])
def test_mail_speed_policy_drives_real_counter_mapped_replies(policy):
    from hfmodem.shrike import arq
    from hfmodem.tests.shrike.test_gear_gate_speed_up import linked_at, delivers
    from hfmodem.tests.shrike.test_repeat_gear_stall import arrives, status_at

    args = _parsed('--p3-mail', '--p3-speed-up', policy)
    host, seam = linked_at(args.p3_speed_up, repeat_gear=args.p3_repeat_gear,
                           speed_up_after=3)
    for seq in (1, 2, 3):
        delivers(host, seq)
    assert seam.words == [arq.CS_REQUEST, arq.CS_ACK,
                          arq.CS_SPEED_UP if policy == 'auto' else arq.CS_REQUEST]
    if policy == 'auto':
        # The next received packet declares the higher level and counter zero;
        # the alternating ACK resumes, without another speed command.
        arrives(host, sl=2, status=status_at(0, long_cycle=False), field=b'next')
        host.arq.on_cycle()
        assert seam.words[-1] == arq.CS_ACK
        assert host.arq.rx_progress == 4
