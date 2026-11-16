# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json

import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.p3trial import CYCLE, ReplyClock
from hfmodem.tests.shrike.p3_reply_support import disarm_reply, setup


def recorded_copies(tx):
    return {scope: (record.cs, record.policy) if record is not None else None
            for scope, record in tx._p3_changeover_copies.items()}


def receiver(tmp_path, policy):
    s, tx, live, grid = setup(tmp_path, 'B')
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry, pulse_epoch=entry)
    s.host.arq._yield_link(acked=True)
    s.host.arq.entry_pending = False
    grid.reverse(to_iss=False)
    s.host.p3_changeover_answer = policy
    s.host.arq.cfg.repeat_gear = 0
    delivered, emitted = [], []
    s.host.deliver = delivered.append
    notify = s.host.on_cs_emitted

    def record(word):
        emitted.append(word)
        notify(word)
    s.host.on_cs_emitted = record
    return s, tx, live, grid, delivered, emitted


def arm(tx, live, grid, slot):
    grid.cycles = slot
    tx.aim(grid, slot)
    live.now = live.pos = tx.boundary - CYCLE // 2
    tx.arm_p3_reply()
    assert tx._p3_armed is not None
    return tx._p3_armed.cs


def decode(s, *, seq=0, payload=b'RMS', breakin=True, phase=None):
    tx = s.host.peer
    if phase is None:
        phase = tx.boundary + tx._p3_control_lead(arq.CS_ACK) - tx.p3_answer_n
    s.host.on_rx_event(rxfront.Event(phase / onair.FS, 'packet', '',
        protocol=spec.Protocol.PACTOR3, packet=(1, seq, payload, True), breakin=breakin))


def emit(tx, live, *, late=False):
    live.now = live.pos = tx.boundary - (0 if late else 5000)
    tx.emit_pending_cs()
    assert not tx.refused
    return tx._p3_emitted_cs, tx.tx_audio_start


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
def test_policy_counts_only_aired_in_time_copies(tmp_path, policy, word, capsys):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    words, instants = [], []
    for slot, kind in enumerate(('late', 'unread', 'timely', 'timely', 'timely', 'timely'), 1):
        held = arm(tx, live, grid, slot)
        assert held == (arq.CS_REQUEST if slot == 1 else
                        arq.CS_ACK if slot <= 3 or policy == 'cs4' else word)
        if kind != 'unread':
            decode(s)
            decode(s)
        else:
            tx.send_cs(held)
        actual, instant = emit(tx, live, late=kind == 'late')
        words.append(actual)
        instants.append(instant)
    assert words == [arq.CS_REQUEST, arq.CS_ACK, word, word, word, arq.CS_ACK]
    assert emitted == words and delivered == [b'RMS']
    assert len(tx._p3_changeover_copies) == 5
    assert tx._p3_changeover_emissions == 3
    assert tx.slots_used[-6:] == list(range(1, 7))
    assert all(b - a == CYCLE for a, b in zip(instants, instants[1:]))
    events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('tx_event ')]
    assert [int(event['label'][2]) - 1 for event in events
            if event['event'] == 'completed' and event.get('outcome') == 'sent'][-6:] == words
    assert sum(event.get('event') == 'completed' and event.get('outcome') == 'sent'
               for event in events) >= 6


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
def test_same_word_dac_misses_preserve_policy_budget(tmp_path, policy, word):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    scopes = []
    for slot, late in enumerate((False, True, True, False), 1):
        held = arm(tx, live, grid, slot)
        if slot > 1:
            assert held == (arq.CS_ACK if policy == 'cs4' else word)
        decode(s)
        scopes.append(tx._p3_reply_scope())
        assert emit(tx, live, late=late)[0] == (held if late else word)
        assert tx._p3_changeover_emissions == (1 if slot < 4 else 2)
    unread = arq.CS_ACK if policy == 'cs4' else word
    assert emitted == [word, unread, unread, word] and delivered == [b'RMS']
    assert tx._p3_changeover_emissions == 2
    assert recorded_copies(tx) == dict(zip(
        scopes, [(word, True), (unread, False), (unread, False), (word, True)]))


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
def test_missed_commit_expires_before_unread_counter_one(tmp_path, policy):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    held = arm(tx, live, grid, 1)
    decode(s)
    assert emit(tx, live, late=True)[0] == held
    assert tx._pending_p3_cs not in (arq.CS_SPEED_UP, arq.CS_NAK)
    arm(tx, live, grid, 2)
    tx.send_cs(s.host._counter_cs_for(s.host.arq._p3_error_reply(count=False)))
    assert emit(tx, live)[0] == arq.CS_ACK
    assert delivered == [b'RMS'] and s.host.arq.rx_seq == 0
    arm(tx, live, grid, 3)
    decode(s, seq=1, payload=b'hello', breakin=False)
    emit(tx, live, late=True)
    assert delivered == [b'RMS', b'hello']
    arm(tx, live, grid, 4)
    decode(s, seq=1, payload=b'hello', breakin=False)
    emit(tx, live)
    assert delivered == [b'RMS', b'hello'] and s.host.arq.rx_seq == 1


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
def test_policy_answer_does_not_acknowledge_unread_data(tmp_path, policy):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    arm(tx, live, grid, 1)
    decode(s)
    word = arq.CS_SPEED_UP if policy == 'cs4' else arq.CS_NAK
    assert emit(tx, live)[0] == word
    held = arm(tx, live, grid, 2)
    tx.send_cs(held)
    assert emit(tx, live)[0] == held == (arq.CS_ACK if policy == 'cs4' else word)
    assert delivered == [b'RMS'] and s.host.arq.rx_seq == 0
    arm(tx, live, grid, 3)
    s.host.arq._rx_this_cycle = False
    s.host.tick()
    assert emit(tx, live)[0] == held
    assert emitted == [word, held, held]
    assert s.host.arq.rx_seq == 0
    arm(tx, live, grid, 4)
    decode(s, seq=1, payload=b'hello', breakin=False)
    emit(tx, live)
    assert delivered == [b'RMS', b'hello']


@pytest.mark.parametrize('expiry', ['carry', 'regrid', 'refusal'])
def test_policy_decision_expires_on_slot_loss(tmp_path, monkeypatch, expiry):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, 'cs4')
    arm(tx, live, grid, 1)
    decode(s)
    assert tx._pending_p3_cs == arq.CS_SPEED_UP
    if expiry == 'carry':
        disarm_reply(tx, live)
        tx.aim(grid, 2)
    elif expiry == 'regrid':
        transmit = live.transmit
        first = True

        def refuse_once(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise onair._MissedTxSlot(1000)
            return transmit(*args, **kwargs)
        disarm_reply(tx, live)
        monkeypatch.setattr(live, 'transmit', refuse_once)
    else:
        key_guard = tx._refused

        def refuse(*args, **kwargs):
            tx.refused = True
            return True

        monkeypatch.setattr(tx, '_refused', refuse)
        live.now = live.pos = tx.boundary - 5000
        tx.emit_pending_cs()
        monkeypatch.setattr(tx, '_refused', key_guard)
        tx.aim(grid, 2)
    if expiry == 'refusal':
        tx.send_cs(s.host._counter_cs_for(s.host.arq._p3_error_reply(count=False)))
    assert emit(tx, live)[0] == arq.CS_ACK
    assert arq.CS_SPEED_UP not in emitted


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
@pytest.mark.parametrize('retry_again', [False, True])
@pytest.mark.parametrize('accepted', [False, True])
def test_retry_decode_adopts_policy_scope_and_expires_it(tmp_path, monkeypatch,
                                                        policy, word, retry_again, accepted):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    if accepted:
        arm(tx, live, grid, 1)
        decode(s)
        assert emit(tx, live)[0] == word
    spent = tx._p3_changeover_emissions if accepted else 0
    held = arm(tx, live, grid, 2 if accepted else 1)
    assert held == ((arq.CS_ACK if policy == 'cs4' else word)
                    if accepted else arq.CS_REQUEST)
    disarm_reply(tx, live)
    tx.send_cs(held)
    owner = s.host.arq.p3_tx_owner
    render = tx._p3_cs_render
    transmit = live.transmit
    calls, scans = [], []

    def record_render(cs, grid_swap):
        calls.append(cs)
        return render(cs, grid_swap)

    def miss_deadline(*args, **kwargs):
        if len(calls) == 1 or (retry_again and len(calls) == 2):
            raise onair._MissedTxSlot(1000)
        return transmit(*args, **kwargs)

    def read_body(sessrx, audio, origin, *, tracked_only):
        assert sessrx is s.rx and audio.size and tracked_only
        scans.append(tx._p3_reply_scope())
        if len(scans) == 1:
            decode(s)
            assert (s.host.arq.p3_tx_owner == owner) == accepted
            assert tx._pending_p3_cs == word
            assert tx._pending_p3_cs_policy == scans[0]

    monkeypatch.setattr(live, 'transmit', miss_deadline)
    monkeypatch.setattr(tx, '_p3_cs_render', record_render)
    monkeypatch.setattr(onair, '_scan_frame', read_body)
    actual, _ = emit(tx, live)
    # An expired policy proposal cannot acknowledge the next unread packet.
    fallback = word if accepted and policy != 'cs4' else arq.CS_ACK
    assert calls == [held, word] + ([fallback] if retry_again else [])
    assert actual == (fallback if retry_again else word)
    assert emitted == ([word] if accepted else []) + [actual]
    assert delivered == [b'RMS'] and s.host.arq.rx_seq == 0
    assert tx._p3_changeover_emissions == spent + (0 if retry_again else 1)
    assert tx._pending_p3_cs is None and tx._pending_p3_cs_policy is None
    assert tx._p3_render_owner is None


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
@pytest.mark.parametrize('placement', ['audio-start', 'pulse-center'])
@pytest.mark.parametrize('waveform', ['historical', 'current'])
def test_timely_policy_budget_survives_control_placement(tmp_path, policy, word,
                                                        placement, waveform, capsys):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    tx.p3_control_placement = placement
    tx.p3_control_waveform = waveform
    scopes = []
    for slot in range(1, 5):
        arm(tx, live, grid, slot)
        decode(s)
        scopes.append(tx._p3_reply_scope())
        emit(tx, live)
    assert emitted == [word] * 3 + [arq.CS_ACK]
    assert tx._p3_changeover_emissions == 3
    expected_copies = [(word, True)] * 3 + [(arq.CS_ACK, False)]
    assert recorded_copies(tx) == dict(zip(scopes, expected_copies))
    assert (tx._p3_keyed_word.scope, tx._p3_keyed_word.cs) == (scopes[-1], arq.CS_ACK)
    assert delivered == [b'RMS']
    events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('tx_event ')]
    assert [int(event['label'][2]) - 1 for event in events
            if event['event'] == 'completed' and event.get('outcome') == 'sent'][-4:] == emitted


def test_changeover_flags_and_mail_timing_presets():
    parser = onair.build_parser()
    args = parser.parse_args(['--p3-changeover-answer', 'cs4', '--p3-mail'])
    onair._p3_mail_defaults(args)
    assert args.p3_changeover_answer == 'cs4'
    args = parser.parse_args(['--p3-changeover-answer', 'cs5-every', '--p3-mail'])
    onair._p3_mail_defaults(args)
    args = parser.parse_args(['--p3-changeover-answer', 'cs4', '--p3-timing-trial', 'B'])
    with pytest.raises(SystemExit):
        onair._timing_trial_defaults(args)


@pytest.mark.parametrize('policy', [None, 'cs1', 'cs5-every', 'legacy'])
def test_default_and_legacy_words_and_sample_instants(tmp_path, policy):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy or 'cs1')
    if policy is None:
        del s.host.p3_changeover_answer
    if policy == 'legacy':
        s.host.p3_changeover_answer = onair.build_parser().parse_args(
            ['--p3-changeover-cs5']).p3_changeover_answer
    instants = []
    for slot in range(1, 7):
        arm(tx, live, grid, slot)
        decode(s)
        instants.append(emit(tx, live, late=slot == 1)[1])
    word = arq.CS_NAK if policy in ('cs5-every', 'legacy') else arq.CS_ACK
    assert emitted == [arq.CS_REQUEST] + [word] * 5
    assert instants == [568800 + i * CYCLE for i in range(6)]


@pytest.mark.parametrize('policy', [None, 'cs1', 'cs5-every', 'legacy'])
@pytest.mark.parametrize('accepted', [False, True])
@pytest.mark.parametrize('misses', [1, 2])
def test_default_and_legacy_retry_preserves_queued_changeover(tmp_path, monkeypatch,
                                                            policy, accepted, misses):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy or 'cs1')
    if policy is None:
        del s.host.p3_changeover_answer
    if policy == 'legacy':
        s.host.p3_changeover_answer = onair.build_parser().parse_args(
            ['--p3-changeover-cs5']).p3_changeover_answer
    word = arq.CS_NAK if policy in ('cs5-every', 'legacy') else arq.CS_ACK
    if accepted:
        arm(tx, live, grid, 1)
        decode(s)
        assert emit(tx, live)[0] == word
    slot = 2 if accepted else 1
    held = arm(tx, live, grid, slot)
    # The held word is the word that actually keyed, including policy words.
    assert held == (word if accepted else arq.CS_REQUEST)
    disarm_reply(tx, live)
    tx.send_cs(held)
    owner = s.host.arq.p3_tx_owner
    render = tx._p3_cs_render
    transmit = live.transmit
    calls, scans = [], []

    def record_render(cs, grid_swap):
        calls.append(cs)
        return render(cs, grid_swap)

    def miss_deadline(*args, **kwargs):
        if len(calls) <= misses:
            raise onair._MissedTxSlot(1000)
        return transmit(*args, **kwargs)

    def read_body(sessrx, audio, origin, *, tracked_only):
        assert sessrx is s.rx and audio.size and tracked_only
        scans.append(tx.slot)
        decode(s)
        assert (s.host.arq.p3_tx_owner == owner) == accepted
        assert tx._pending_p3_cs == word

    monkeypatch.setattr(live, 'transmit', miss_deadline)
    monkeypatch.setattr(tx, '_p3_cs_render', record_render)
    monkeypatch.setattr(onair, '_scan_frame', read_body)
    actual, instant = emit(tx, live)
    expected = held if accepted else word
    assert calls == [held] + [expected] * misses
    assert actual == expected
    assert scans == list(range(slot + 1, slot + misses + 1))
    assert instant == 568800 + (slot + misses - 1) * CYCLE
    retained = accepted and word != held
    assert tx._pending_p3_cs == (word if retained else None)
    if retained:
        arm(tx, live, grid, slot + misses + 1)
        assert emit(tx, live) == (word, instant + CYCLE)
    assert emitted == ([word] if accepted else []) + [expected] + ([word] if retained else [])
    assert delivered == [b'RMS'] and s.host.arq.rx_seq == 0


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
@pytest.mark.parametrize('reset', ['data', 'turn', 'link'])
def test_policy_budget_restarts_with_a_new_changeover(tmp_path, policy, reset):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    for slot in range(1, 5):
        arm(tx, live, grid, slot)
        decode(s)
        emit(tx, live)
    assert emitted[-1] == arq.CS_ACK
    a = s.host.arq
    if reset == 'data':
        arm(tx, live, grid, 5)
        decode(s, seq=1, payload=b'data', breakin=False)
        emit(tx, live)
    elif reset == 'turn':
        a._take_link()
        a._give_link()
    else:
        a._finish_disconnected()
        a.role = arq.IRS
        a._enter_connected()
        s.host.protocol = spec.Protocol.PACTOR3
    arm(tx, live, grid, 6)
    decode(s)
    assert emit(tx, live)[0] == (arq.CS_SPEED_UP if policy == 'cs4' else arq.CS_NAK)


@pytest.mark.parametrize('policy', ['cs4', 'cs5', 'cs5-every'])
@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2])
def test_changeover_policy_leaves_other_protocols_unchanged(policy, protocol):
    from hfmodem.tests.shrike.p3_arq_support import Sink

    def run(value):
        io = Sink()
        io.protocol = protocol
        io.p3_changeover_answer = value
        a = arq.PactorArq(io)
        a.role = arq.IRS
        a._enter_connected()
        a.on_rx_packet(1, b'RMS', 0, True, breakin=True, protocol=protocol)
        return io.sent, io.delivered
    assert run(policy) == run('cs1')


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
def test_recorded_ws8eoc_repeated_changeover_timing(tmp_path, policy):
    import re
    from hfmodem.tests import evidence

    directory = evidence.RECORD / 'operator-data/working/onair-20261108T234755210382Z-e11d8b1e-3a27-4752-b2ef-859bfc8941cc'
    logs = sorted(directory.glob('*.log'))
    if not logs:
        pytest.skip('optional WS8EOC 11-08 23:48Z operator recording is absent')
    copies, rows, late = 0, [], False
    for line in logs[0].read_text().splitlines():
        if '[traffic] RX P3 CHANGEOVER' in line:
            copies += 1
        if "past the DAC's notice" in line:
            late = True
        if line.startswith('tx_event '):
            event = json.loads(line.split(' ', 1)[1])
            if (event['event'] == 'completed' and event.get('outcome') == 'sent'
                    and re.match(r'CS[12] ', event.get('label', '')) and event['slot'] >= 10):
                rows.append((event['slot'], event['audio_start_sample'], copies, late))
                copies, late = 0, False
    assert sum(row[2] for row in rows) == 36
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    import numpy as np
    live.audio = np.pad(live.audio, (0, 60 * onair.FS))
    live.limit = live.audio.size
    grid.anchor = rows[0][1] - rows[0][0] * CYCLE
    grid._p3_yield_placed = True
    spent = 0
    for slot, instant, count, late in rows:
        held = arm(tx, live, grid, slot)
        for _ in range(count):
            decode(s)
        if not count:
            tx.send_cs(held)
        word, actual = emit(tx, live, late=late)
        expected = arq.CS_SPEED_UP if policy == 'cs4' else arq.CS_NAK
        if not count or late:
            expected = held
        elif spent == 3:
            expected = arq.CS_ACK
        else:
            spent += 1
        assert word == expected and actual == instant
    assert spent == 3 and delivered == [b'RMS']
    assert tx.slots_used[1:] == [row[0] for row in rows]


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
def test_decode_after_armed_emission_is_counted_once(tmp_path, policy):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    held = arm(tx, live, grid, 1)
    tx.send_cs(held)
    emit(tx, live, late=True)
    decode(s)
    decode(s)
    records = [line.split('P3 changeover copy ', 1)[1]
               for line in s.host.log_lines
               if 'P3 changeover copy ' in line]
    assert records == [f'1 slot=1 aired CS{held + 1}; policy not applied (0/3)']
    assert recorded_copies(tx) == {(grid, 1): (held, False)}
    assert len(tx._p3_changeover_copies) == 1
    assert tx._p3_changeover_emissions == 0
    assert tx._pending_p3_cs is None
    assert delivered == [b'RMS']
    arm(tx, live, grid, 2)
    decode(s)
    assert emit(tx, live)[0] == (arq.CS_SPEED_UP if policy == 'cs4' else arq.CS_NAK)


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
def test_previous_window_copy_keeps_ordinary_answer(tmp_path, policy, word):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    arm(tx, live, grid, 1)
    decode(s)
    assert emit(tx, live)[0] == word
    held = arm(tx, live, grid, 2)
    phase = tx.boundary + tx._p3_control_lead(arq.CS_ACK) - tx.p3_answer_n
    tx.send_cs(held)
    assert emit(tx, live)[0] == held == (arq.CS_ACK if policy == 'cs4' else word)
    arm(tx, live, grid, 3)
    decode(s, phase=phase)
    decode(s, phase=phase)
    assert emit(tx, live)[0] == held
    decode(s, phase=phase)
    assert emitted == [word, held, held]
    assert delivered == [b'RMS'] and s.host.arq.rx_seq == 0
    assert tx._p3_changeover_emissions == 1
    records = [line.split('P3 changeover copy ', 1)[1]
               for line in s.host.log_lines
               if 'P3 changeover copy ' in line]
    assert records == [f'1 slot=1 aired CS{word + 1}; policy applied (1/3)',
                       f'2 slot=2 aired CS{held + 1}; policy not applied (1/3)']
    assert recorded_copies(tx) == {(grid, 1): (word, True), (grid, 2): (held, False)}


@pytest.mark.parametrize('policy,word', [('cs4', arq.CS_SPEED_UP), ('cs5', arq.CS_NAK)])
def test_subslot_replacement_keeps_policy_answer(tmp_path, monkeypatch, policy, word):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    arm(tx, live, grid, 1)
    disarm_reply(tx, live)
    decode(s)
    before = tx.boundary

    def shift(slot, cs_index=None):
        grid.anchor += 1
        return 'waveform lead'

    monkeypatch.setattr(tx, 'p3_reply_shift', shift)
    assert emit(tx, live)[0] == word
    assert tx.boundary == before + 1
    assert emitted == [word] and tx._p3_changeover_emissions == 1


@pytest.mark.parametrize('policy', ['cs4', 'cs5'])
def test_spent_policy_budget_preserves_repeat_gear(tmp_path, monkeypatch, policy):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, policy)
    for slot in range(1, 4):
        arm(tx, live, grid, slot)
        decode(s)
        emit(tx, live)
    monkeypatch.setattr(s.host.arq, '_repeat_gear_cs',
                        lambda *args, **kwargs: arq.CS_SPEED_UP)
    arm(tx, live, grid, 4)
    decode(s)
    assert emit(tx, live)[0] == arq.CS_SPEED_UP
    assert tx._p3_changeover_emissions == 3


@pytest.mark.parametrize('options,answer', [
    (['--p3-changeover-cs5'], 'cs5-every'),
    (['--p3-changeover-cs5', '--p3-changeover-answer', 'cs4'], None),
    (['--p3-changeover-answer', 'cs4', '--p3-changeover-cs5'], None),
])
def test_changeover_alias_sets_only_the_policy(options, answer):
    if answer is None:
        with pytest.raises(SystemExit) as error:
            onair.build_parser().parse_args(options + ['--p3-mail'])
        assert error.value.code == 2
        return
    args = onair.build_parser().parse_args(options + ['--p3-mail'])
    onair._p3_mail_defaults(args)
    assert args.p3_changeover_answer == answer
    assert not hasattr(args, 'p3_changeover_cs5')


def test_changeover_alias_is_incompatible_with_timing_trials():
    args = onair.build_parser().parse_args(['--p3-changeover-cs5', '--p3-timing-trial', 'B'])
    with pytest.raises(SystemExit):
        onair._timing_trial_defaults(args)


def test_new_changeover_releases_previous_control_records(tmp_path):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, 'cs4')
    arm(tx, live, grid, 1)
    decode(s)
    assert emit(tx, live)[0] == arq.CS_SPEED_UP
    old_scope = tx._p3_reply_scope()
    arm(tx, live, grid, 2)
    decode(s, seq=1, payload=b'hello', breakin=False)
    emit(tx, live)
    s.host.arq._take_link()
    s.host.arq._give_link()
    arm(tx, live, grid, 3)
    decode(s, payload=b'new')
    assert emit(tx, live)[0] == arq.CS_SPEED_UP
    assert old_scope not in tx._p3_control_emissions
    assert set(tx._p3_control_emissions) == {tx._p3_reply_scope()}
    assert tx._p3_changeover_emissions == 1


@pytest.mark.parametrize('ordinary', [arq.CS_ACK, arq.CS_REQUEST])
def test_changeover_scope_uses_one_lead_near_slot_edge(tmp_path, monkeypatch, ordinary):
    s, tx, live, grid, delivered, emitted = receiver(tmp_path, 'cs4')
    arm(tx, live, grid, 1)
    monkeypatch.setattr(tx, '_p3_control_lead',
                        lambda word: 168 if word == arq.CS_ACK else 0)
    phase = (tx.boundary + CYCLE // 2 + 84 - tx.p3_answer_n) / onair.FS
    answer = tx.choose_changeover_answer(object(), 'cs4', ordinary, phase)
    assert answer == arq.CS_SPEED_UP
    tx.send_cs(answer)
    assert emit(tx, live)[0] == arq.CS_SPEED_UP
    assert tx._p3_changeover_emissions == 1
