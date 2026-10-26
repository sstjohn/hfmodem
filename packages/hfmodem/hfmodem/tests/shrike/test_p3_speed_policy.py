# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""--p3-speed-policy stock beside the flown ladder, and the arms replayed."""
import json
import re
from pathlib import Path

import pytest

from hfmodem.shrike import arq
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_long_sl1 import PAYLOAD, linked, row
from hfmodem.tests.shrike.test_tx_gear_hold import BODY, control, sender

WORKING = evidence.WORKING
LOGIN = (b': W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 52852319\r; WS8EOC DE W9SSJ\r'
         b'FC EM 9MD0TG4OI2FJ 635 421 0\rF> 4A\r')
CS = {name: index for index, name in arq._CS_NAMES.items()}


def stock(queue=b'', inflight=None, policy='stock'):
    prefix = inflight.payload if inflight is not None else b'\x01/W'
    host, seam = sender(enabled=False, payload=prefix + queue)
    host.arq.cfg.p3_speed_policy = policy
    host.arq._outbuf = bytearray(queue)
    host.arq._buffer_raw = len(prefix) + len(queue)
    if inflight is not None:
        host.arq._inflight = inflight
    return host, seam


def after_entry(policy):
    """The scene every sense arm opened on: the entry packet keyed, nothing queued."""
    host, seam = stock(inflight=arq._Packet(0x1a, b'', 1, entry=True), policy=policy)
    host.arq.cfg.entry_sl = host.arq.cfg.traffic_sl = 1   # the mail preset's
    host.arq._next_seq = 3
    host.arq._entry_keyed = True
    return host, seam


def test_config_names_the_two_policies():
    assert arq.ArqConfig().p3_speed_policy == 'current'
    assert arq.ArqConfig(p3_speed_policy='stock').p3_max_try == 2
    with pytest.raises(ValueError, match='current or stock'):
        arq.ArqConfig(p3_speed_policy='fast')


def test_stock_cs4_climbs_only_on_a_queue_above_the_field():
    host, seam = after_entry('stock')
    control(host, 4)
    host.tick()                              # an idle packet waits for the boundary
    sl, status, payload = seam.frames[-1]
    assert host.arq.speed_level == 1 and (sl, status & 3, payload) == (1, 3, b'')
    assert '[speed] peer advises SL2; queue 0 B, staying at SL1' in host.log_lines
    host.arq.on_host_data(b'abcde')          # exactly the SL1 field: not more
    control(host, 4)
    assert host.arq.speed_level == 1 and seam.frames[-1][:1] + seam.frames[-1][2:] == (1, b'abcde')
    host.arq.on_host_data(b'fghijk')         # one more than the field
    control(host, 4)
    assert host.arq.speed_level == 2 and seam.frames[-1][0] == 2
    assert host.arq._inflight.trial_from == 1



def test_p3_stock_idle_cs4_after_offer_is_one_trial():
    """KB5LZK 30 m 0926-1949: CS4 answered the entry and then our offered 0x5b."""
    host, seam = after_entry('stock')
    a = host.arq
    a.on_host_over()
    control(host, 4)                         # answers the entry: keep_speed
    assert seam.frames[-1] == (1, 0x5b, b'')
    control(host, 4)                         # answers the offer: one idle trial
    sl, status, payload = seam.frames[-1]
    assert (sl, payload) == (2, b'') and status & 0x40
    assert a.speed_level == 2 and a._inflight.trial_from == 1
    sent = len(seam.frames)
    control(host, 2 - (a.tx_seq & 1))        # a repeat request for the trial's counter
    assert len(seam.frames) == sent + 1 and seam.frames[-1] == (1, status, b'')
    assert f'P3 MAXTry=1: seq={status & 3} unacknowledged after 1 transmissions at SL2 -> return to SL1' in host.log_lines
    assert not any('handover recovery' in line for line in host.log_lines)
    assert a.speed_level == 1 and a._p3_tx_gear_command is None
    control(host, 4)                         # a third CS4 is advice
    assert a.speed_level == 1
    assert host.log_lines.count('[speed] peer advises SL2; queue 0 B, staying at SL1') == 2
    sl, status, payload = seam.frames[-1]
    assert (sl, payload) == (1, b'') and not status & 0x40

def test_current_cs4_climbs_on_an_idle_packet():
    host, seam = after_entry('current')
    control(host, 4)
    host.tick()
    sl, status, payload = seam.frames[-1]
    assert host.arq.speed_level == 2 and (sl, status & 3, payload) == (2, 3, b'')


def test_stock_repeated_cs4_acknowledges_and_advances_the_counter():
    host, seam = stock(BODY)
    control(host, 4)
    assert host.arq.speed_level == 2
    seq = host.arq.tx_seq
    control(host, 4)                       # VE3KPG 0924-0033: CS4 carries the ACK
    assert host.arq.tx_seq == (seq + 1) % 4
    assert not any('GEAR HOLD' in line for line in host.log_lines)
    sent = b''.join(payload for _, _, payload in seam.frames)
    assert BODY.startswith(sent)


def test_stock_max_try_is_one_and_current_keeps_its_own():
    host, seam = stock(BODY)
    control(host, 4)
    assert host.arq.p3_max_try == 1
    host.arq._on_nak(False, answered=False)
    assert host.arq.speed_level == 1 and seam.frames[-1][0] == 1
    host, seam = sender(enabled=False)
    control(host, 4)
    assert host.arq.p3_max_try == 2
    host.arq._on_nak(False, answered=False)
    assert host.arq.speed_level == 2
    host.arq._on_nak(False, answered=False)
    assert host.arq.speed_level == 1


@pytest.mark.parametrize('policy', ['current', 'stock'])
def test_repeated_cs6_hold_is_bounded_only_under_stock(policy):
    host, tx = linked()
    a = host.arq
    a.cfg.p3_speed_policy = policy
    a.on_host_data(PAYLOAD * 4)
    host.tick()
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    seq = a.tx_seq
    pending = a._inflight.payload + bytes(a._outbuf)
    for _ in range(arq.P3_CS6_HOLD_REPEATS - 1):
        a.on_rx_cs(arq.CS_CYCLE_TOG)
        assert a.cycle_long and a.tx_seq == seq
    a.on_rx_cs(arq.CS_CYCLE_TOG)
    assert a.cycle_long == (policy == 'current')
    assert a.tx_seq == seq
    assert a._inflight.payload + bytes(a._outbuf) == pending
    row(f'{policy}-cs6-hold', host, tx, pending)
    if policy == 'stock':
        assert a._p3_tx_gear_command is None
        a.on_rx_cs(arq.CS_CYCLE_TOG)       # a fresh grant: acknowledges and toggles
        assert a.cycle_long and a.tx_seq != seq


# -- the arms -------------------------------------------------------------

ARMS = {
    'WS8EOC-40 1002-1132': ('onair-1002-1132/pactor-current-ws8eoc-40-force-20261002T163245Z.log', 'changeover'),
    'WS8EOC-40 1002-1137': ('onair-1002-1137/pactor-current-ws8eoc-40-force-20261002T163715Z.log', 'changeover'),
    'VE3KPG-80 0921-0001': ('onair-0921-0001/pactor-current-ve3kpg-80-sense-20260921T050049Z.log', 'entry'),
    'VE3KPG-80 0920-1926': ('onair-0920-1926/pactor-current-ve3kpg-80-sense-20260921T002604Z.log', 'entry'),
    'K0NTS-40 0921-0017': ('onair-0921-0017/pactor-current-k0nts-40-sense-20260921T051645Z.log', 'entry'),
    'WS8EOC-40 0921-1209': ('onair-0921-1209/pactor-current-ws8eoc-40-force-20260921T170929Z.log', 'changeover'),
}
TX = re.compile(r'\[traffic\] TX P3 SL(\d) DATA .*seq=(\d)')
RX = re.compile(r'\[traffic\] RX P3 CS(\d) ')
HOST = re.compile(r'\[host\] rx CS (\w+)')
RESET = 'the peer answered the entry packet -> SL1'
"""The driver's entry-answer bookkeeping, when it lands AFTER the CS4 (0920-1926)
rather than before it (0921-0001): `PtcHost._entry_answered` puts the level back
to `traffic_level`. Not the ARQ's policy, but it keyed that arm's first idle
packet at SL1 and the ladder is reproduced only with it."""


def stint(name, limit=24):
    """(events, frames) of the arm's ISS stint from the log.

    An event is a physical codeword index, None for a silent cycle, or RESET;
    a frame is (sl, seq) as keyed. The answer to a codeword is printed before
    the codeword's own [host] line, so the TX lines pending at that line are
    its predecessors' silent repeats and, last, its answer.
    """
    path, opening = ARMS[name]
    log = WORKING / path
    if not log.is_file():
        fixture = Path(__file__).with_name('fixtures') / 'ws8eoc-speed-1002' / f'{name[-9:]}.json'
        if fixture.is_file():
            recorded = json.loads(fixture.read_text())
            return recorded['events'][:limit], [tuple(f) for f in recorded['frames'][:limit]]
        pytest.skip(f'arm log not on this machine: {log}')
    start = 'link upgraded to PACTOR-3' if opening == 'entry' else 'changeover -> ISS'
    events, frames, pending = [], [], []
    physical = on = None
    openings = 2 if name == 'WS8EOC-40 1002-1137' else 1
    for line in log.read_text(errors='replace').splitlines():
        if not on:
            if start in line:
                openings -= 1
                on = openings == 0
            continue
        if (tx := TX.search(line)):
            pending.append((int(tx[1]), int(tx[2])))
        elif (rx := RX.search(line)):
            physical = int(rx[1]) - 1
        elif (host := HOST.search(line)):
            if host[1] == 'BRK':
                break
            assert arq._CS_NAMES[physical] == host[1] or host[1] in ('ACK', 'REQ')
            answer = pending.pop()
            for silent in pending:
                events.append(None)
                frames.append(silent)
            events.append(physical)
            frames.append(answer)
            pending = []
            if len(frames) >= limit:
                break
        elif line.endswith(RESET):
            events.append(RESET)
        elif 'changeover -> IRS' in line or 'link down' in line:
            break
    assert events, name
    return events, frames


def replay(name, policy):
    _, opening = ARMS[name]
    if opening == 'entry':
        host, seam = after_entry(policy)
    else:
        host, seam = stock(LOGIN, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                           policy=policy)
    events, frames = stint(name)
    owed = False                           # the boundary after an answered cycle
    for event in events:
        if event == RESET:
            host.arq.speed_level = host.arq.traffic_level
            continue
        if owed:
            host.tick()
        if event is None:
            host.tick()
        else:
            control(host, event + 1)
        owed = event is not None
    if owed:
        host.tick()
    keyed = [(sl, status & 3) for sl, status, _ in seam.frames]
    return host, seam, frames, keyed


@pytest.mark.parametrize('name', list(ARMS))
def test_current_replays_the_recorded_controls(name):
    _, _, frames, keyed = replay(name, 'current')
    if ARMS[name][1] == 'changeover':
        assert frames[:15] == [(2, 1)] + [(3, 2)]*2 + [(2, 2)]*5 + [(1, 2)]*7
        assert keyed[:8] == [(2, 1)] + [(3, 2)]*2 + [(1, 2)]*5
        assert keyed[8:15] == [(2, 2)] + [(1, 2)]*6
        assert keyed[15:] == frames[15:]
        assert keyed != frames
        return
    if name in ('VE3KPG-80 0921-0001', 'K0NTS-40 0921-0017'):
        top = frames.index((6, 3))
        frames = frames[:top + 1] + [(sl, 3) for sl, _ in frames[top + 1:]]
        frames[top + 2] = (1, 3)
    assert keyed == frames


@pytest.mark.parametrize('name', list(ARMS))
def test_stock_never_leaves_sl1_on_an_idle_packet(name):
    host, seam, frames, keyed = replay(name, 'stock')
    events, _ = stint(name)
    events = [event for event in events if event != RESET]
    assert all(sl == 1 or (sl == 2 and arq.CS_NAK in events[:i + 1])
               for i, (sl, _, payload) in enumerate(seam.frames) if not payload)
    flown = sorted({sl for sl, _ in frames})
    reached = sorted({sl for sl, _ in keyed})
    idle_stock = sum(1 for _, _, payload in seam.frames if not payload)
    entry = ARMS[name][1] == 'entry'
    print(f'{name}: idle transmissions current {len(frames) if entry else 0} '
          f'/ stock {idle_stock}; levels current {flown} / stock {reached}')
    if entry:
        assert reached == [1] and idle_stock == len(frames)
    else:
        assert reached == flown == [1, 2, 3]
        assert LOGIN.endswith(host.arq._inflight.payload + bytes(host.arq._outbuf))


@pytest.mark.parametrize('policy', ['current', 'stock'])
def test_cs5_steps_from_the_command_after_maxtry_returns_below_it(policy):
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                       policy=policy)
    control(host, 4)
    control(host, 4)
    for _ in range(host.arq.p3_max_try):
        control(host, 2)
    assert seam.frames[-1][0] == 1
    pending = host.arq._inflight.payload + bytes(host.arq._outbuf)
    control(host, 5)
    assert seam.frames[-1][0] == 2 and host.arq.tx_seq == 2
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == pending
    control(host, 5)
    assert seam.frames[-1][0] == (2 if policy == 'stock' else 1)


def test_1137_peer_trial_returns_to_three_plainly_acknowledged_sl2_packets():
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True))
    control(host, 4)
    for cs in (2, 1, 2):
        control(host, cs)
    assert host.arq.speed_level == 2 and host.arq.tx_seq == 0
    control(host, 4)
    assert seam.frames[-1][:2] == (3, 1)
    assert host.arq._inflight.trial_from == 2
    pending = host.arq._inflight.payload + bytes(host.arq._outbuf)
    control(host, 1)
    assert seam.frames[-1][0] == 2 and host.arq.tx_seq == 1
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == pending


def test_cs5_keeps_the_plainly_established_sl2_after_a_failed_sl3_trial():
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                       policy='current')
    control(host, 4)
    control(host, 2)
    control(host, 4)
    assert host.arq._inflight.trial_from == 2
    for _ in range(host.arq.p3_max_try):
        control(host, 1)
    assert seam.frames[-1][0] == 2 and host.arq.tx_seq == 3
    pending = host.arq._inflight.payload + bytes(host.arq._outbuf)
    control(host, 5)
    assert seam.frames[-1][0] == 2 and host.arq.tx_seq == 3
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == pending


def test_plain_ack_of_the_returned_packet_establishes_the_next_trial():
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                       policy='current')
    control(host, 4)
    control(host, 4)
    control(host, 2)
    control(host, 2)
    control(host, 1)
    control(host, 4)
    assert seam.frames[-1][0] == 2 and host.arq._inflight.trial_from == 1
    control(host, 5)
    assert seam.frames[-1][0] == 1


def test_cs6_does_not_plainly_establish_the_trial_level():
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                       policy='current')
    control(host, 4)
    control(host, 6)
    control(host, 4)
    assert host.arq.speed_level == 3 and host.arq._inflight.trial_from == 1


@pytest.mark.parametrize('minimum', [1, 2])
def test_new_turn_forgets_the_previously_established_level(minimum):
    host, seam = stock(BODY, inflight=arq._Packet(0, b';FW', 1, breakin=True),
                       policy='current')
    control(host, 4)
    control(host, 4)
    control(host, 1)
    host.arq.cfg.min_sl = minimum
    host.arq._take_link()
    host.arq._inflight = arq._Packet(0, b';FW', minimum, breakin=True)
    control(host, 4)
    assert seam.frames[-1][0] == minimum + 1
    assert host.arq._inflight.trial_from == minimum
