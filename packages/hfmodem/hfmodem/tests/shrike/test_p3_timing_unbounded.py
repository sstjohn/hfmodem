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
    from hfmodem.shrike import arq
    peers = []
    original_peer = loop.LatePeer

    class LifetimePeer(original_peer):
        def __init__(self, *args):
            super().__init__(*args)
            peers.append(self)

        def start_p3(self, entry_at):
            super().start_p3(entry_at)
            end = max(end for _, end in self.responses)
            needed = end + (arq.P3_CRC_WAIT_TICKS + 2) * CYCLE
            if needed > self.limit:
                self.audio = np.pad(self.audio, (0, needed - self.limit))
                self.limit = len(self.audio)

    monkeypatch.setattr(loop, "LatePeer", LifetimePeer)
    lifetime = _trace_p3_lifetime(monkeypatch, lambda: peers[0] if peers else None)
    main = onair.main
    def extended():
        sys.argv += ['--p3-timing-trial','B','--p3-timing-unbounded']
        return main()
    monkeypatch.setattr(onair, 'main', extended)
    got = loop.run_late(monkeypatch, tmp_path, 1, False)
    _assert_p3_lifetime(lifetime, got, charged_phase_extra=2560)
    report = json.loads((got['outdir']/'p3-timing-trial.json').read_text())
    assert report['unbounded'] and len(report['progress']) == 3, got['log']
    assert report['opportunities'] > 12
    assert report['replies'][-1]['audio_start'] > report['progress'][-1]['phase']+CYCLE
    assert 'limit' not in report['reason'] and 'PTT off.' in got['log']
    assert 'without an experiment/hold cutoff' in got['log']


def _trace_p3_lifetime(monkeypatch, peer):
    """Observe real settlement; do not replace qualification or timing rules."""
    from hfmodem.shrike import arq
    rows = []
    settle = arq.PactorArq.settle_peer_presence

    def observed(a, *args, **kwargs):
        op = a._receive_opportunity
        current = op is not None and not op.presence_settled
        crc = bool(op and op.crc)
        result = settle(a, *args, **kwargs)
        if current and peer() is not None:
            p = peer()
            rows.append(dict(sample=p.samples, now=p.now, crc=crc,
                             ticks=a._p3_crc_wait_ticks, dark=a._dark_windows,
                             quiet=a.p3_receive_only, state=a.state,
                             keys=len(p.emissions), acked=a.tx_acked_bytes,
                             rx_seq=a._expected_seq, tx_seq=a._next_seq,
                             queued=bytes(a._outbuf)))
        return result

    monkeypatch.setattr(arq.PactorArq, "settle_peer_presence", observed)
    return rows


def _assert_p3_lifetime(rows, got, *, expect_current=True, charged_phase_extra=0):
    from hfmodem.shrike import arq
    current = [row for row in rows if row["state"] in
               (arq.State.CONNECTED, arq.State.DISCONNECTING, arq.State.DISCONNECTED)]
    # Response permission follows the last current qualified observation.
    # CRC-free lifetime follows current accepted CRC, not any CRC flag from
    # the deliberately substituted historical previous-row reader.
    last_support = max(i for i, row in enumerate(current) if row["dark"] == 0)
    tail = current[last_support + 1:]
    assert tail and tail[-1]["state"] == arq.State.DISCONNECTED
    assert tail[-1]["ticks"] == arq.P3_CRC_WAIT_TICKS == 96
    quiet = next(row for row in tail if row["quiet"])
    assert quiet["dark"] == arq.P3_DARK_WINDOWS == 6
    assert tail[0]["keys"] == current[last_support]["keys"] + 1
    assert quiet["keys"] - tail[0]["keys"] == arq.P3_DARK_WINDOWS - 1 == 5
    assert all(row["keys"] == quiet["keys"] for row in tail
               if row["ticks"] >= quiet["ticks"])
    snapshot = (quiet["acked"], quiet["rx_seq"], quiet["tx_seq"], quiet["queued"])
    assert all((row["acked"], row["rx_seq"], row["tx_seq"], row["queued"]) == snapshot
               for row in tail if row["state"] != arq.State.DISCONNECTED and
               row["ticks"] >= quiet["ticks"])
    if expect_current:
        last_crc = max(i for i, row in enumerate(current) if row["crc"] and row["ticks"] == 0)
        assert last_crc == last_support
        elapsed = tail[-1]["sample"] - current[last_crc]["sample"]
        # The three-field fixture has a measured 20-block charged receive
        # phase shift. Pin it exactly; this is no production deadline change.
        if charged_phase_extra:
            assert elapsed == arq.P3_CRC_WAIT_TICKS * CYCLE + charged_phase_extra
        else:
            assert abs(elapsed - arq.P3_CRC_WAIT_TICKS * CYCLE) <= 384
    else:
        stale_crc = [row for row in current if row["crc"] and row["ticks"] > 0]
        assert stale_crc
        assert all(row["ticks"] > 0 for row in current[current.index(stale_crc[0]):])
    a = got["host"].arq
    assert a.state == arq.State.DISCONNECTED and not a.said_goodbye
    assert a.transfer_unconfirmed["suffix"] == quiet["queued"]
    assert a.tx_acked_bytes == quiet["acked"]
    assert got["peer"].now <= got["peer"].limit
