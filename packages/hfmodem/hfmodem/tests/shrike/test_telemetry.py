# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded diagnostics cannot become delivery, timing or admission evidence."""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair, telemetry
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol


def test_history_retirement_preserves_append_detection_and_saved_hold_indexes():
    history = telemetry.History(4)
    for number in range(2):
        history.append(number)
    held_from = len(history)
    for number in range(2, 10000):
        before = len(history)
        history.append(number)
        assert len(history) == before + 1
        assert history.retained <= 4
    assert history.first == 9996
    assert history.tail() == history[held_from:] == [9996, 9997, 9998, 9999]
    assert history[9998:] == [9998, 9999]
    assert history[-2] == 9998 and history[-2:] == [9998, 9999]
    assert history[::-1] == [9999, 9998, 9997, 9996]
    assert history[:held_from] == []
    with pytest.raises(IndexError, match='retired'):
        history[held_from]
    history.clear()
    assert not history and history.retained == history.first == 0


def test_breakin_marks_keep_absolute_indexes_without_growing_with_the_contact():
    history = telemetry.History(4)
    marks = telemetry.Marks(history)
    for number in range(10000):
        marks.add(len(history))
        history.append(number & 3)
        assert len(marks) <= 5
    assert marks == {9995, 9996, 9997, 9998, 9999}
    assert history[9998:] == [2, 3]


def test_retired_p1_words_still_constrain_later_p3_frequency_following(tmp_path):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    host = PtcHost(tx)
    rx = onair._SessionRx(host)
    telemetry.bind(tx, rx, capacity=2)
    rx.cs_log.append(SimpleNamespace(protocol=Protocol.PACTOR1, cs=0, spare=None, text='CS1'))
    for _ in range(10000):
        rx.cs_log.append(SimpleNamespace(protocol=Protocol.PACTOR3, cs=0, spare=None, text='ACK'))
    assert rx.cs_log.retained == 2 and len(rx.cs_log) == 10001
    assert all(event.protocol == Protocol.PACTOR3 for event in rx.cs_log)
    assert rx.cs_log.p1_words == 1
    rx._follow_p3_offset(75.)
    assert rx._p3_offset_contradicts_p1


def test_summary_labels_retained_observations_and_rebases_breakin_marks(capsys):
    slots, keyed, seq, controls = (telemetry.History(3) for _ in range(4))
    for number in range(8):
        slots.append(number)
        keyed.append((number, .96))
        seq.append(number % 4)
    onair._summary(controls, slots, 8, keyed, 1.25, seq, onair._ConnectEvidence(),
                   'fixture complete', breakin_at={7})
    text = capsys.readouterr().out
    assert '5 retired of 8 events' in text
    assert 'counts, cadence and verdict below describe retained observations' in text
    assert 'packet counters sent: #1, #2, #3' in text
    assert len(seq) == 8 and seq.retained == 3


@pytest.mark.parametrize('level,needed', [(2.2, 2), (1.5, 3)])
def test_station_acquisition_retires_only_evidence_the_rule_already_ignores(level, needed):
    bounded = onair._ConnectEvidence(bounded=True)
    reference = onair._ConnectEvidence()
    for search in range(10000):
        for evidence in (bounded, reference):
            evidence.note_search(.04 if search % 2 else .15)
        if search % 13 == 0:
            assert bounded.offer(search, .09, level) == reference.offer(search, .09, level)
        assert len(bounded.candidates) <= bounded.SPAN
        assert len(bounded.levels) == len(bounded.candidates)
        assert bounded.count == reference.count
        assert bounded.at == reference.at is None
        assert bounded.by_chance == reference.by_chance
    for search in range(10000, 10000 + needed):
        for evidence in (bounded, reference):
            evidence.note_search(.15)
        result = bounded.offer(search, .12, level)
        assert result == reference.offer(search, .12, level)
        assert result == (search == 9999 + needed)
    assert bounded.at == reference.at == 9999 + needed
    assert bounded.retired > 700
    assert bounded.count == reference.count
    assert 'expired candidate details retired; retained:' in bounded.report()


def test_expired_acquisition_details_do_not_turn_historical_accepts_into_silence(capsys):
    evidence = onair._ConnectEvidence(bounded=True)
    evidence.note_search(.15)
    evidence.offer(0, .09)
    for _ in range(evidence.SPAN):
        evidence.note_search(.15)
    assert evidence.count == evidence.retired == 1
    assert not evidence.candidates and not evidence.levels
    assert '1 of 13 cycles' in evidence.report()
    assert 'retained: none' in evidence.report()
    onair._summary([], [], 13, [], 1.25, [], evidence, 'fixture complete')
    text = capsys.readouterr().out
    assert '1 zero-error codeword(s) accepted' in text
    assert 'fewer than two retained accepts' in text
    assert 'verdict: no PACTOR-1 control signal' not in text


def grid():
    return onair._MasterGrid(0, 60000, 12000, packet_n=46080, cs_n=5760, d_max_n=4800)


def test_entry_position_retention_labels_its_scope_without_changing_geometry():
    raster = grid()
    telemetry.bind_grid(raster, capacity=4)
    raster.entry_at, raster.entry_key_n = 1, raster.boundary(1)
    raster.answer_unread_ms = 110.
    raster.entry_end_n = raster.p1_data_n + round(3 * onair.MAX_PULL_S * onair.FS)
    read_ms = raster.entry_read_ms
    before = (raster.anchor, raster.protocol, raster.rx_ref_n, raster.entry_key_n, read_ms)
    for cycle in range(10000):
        # Earlier off-position readings must not become a whole-session pass
        # when the bounded summary later contains only on-position readings.
        ms = read_ms if cycle >= 9996 else 500.
        raster.entry_answers.append((cycle, raster.boundary(cycle), ms))
        raster.entry_tracked[cycle] = ms
        assert raster.entry_answers.retained <= 4 and len(raster.entry_tracked) <= 4
    raster.cycles = 9999
    message = raster.entry_verdict()
    assert '9996 earlier entry-answer details retired' in message
    assert 'position verdict describes retained answers only' in message
    assert 'cycles 9996-9999' in message and '4 cycle(s)' in message
    assert 'THE ANSWER MOVED TO THE ENTRY POSITION' in message
    assert 'cycle 9999' in raster.answer_position()
    assert before == (raster.anchor, raster.protocol, raster.rx_ref_n, raster.entry_key_n,
                      raster.entry_read_ms)
    raster.entry_answers.clear()
    raster.entry_tracked.clear()
    assert raster.entry_answers.first == len(raster.entry_answers) == 0
    assert 'NO TIMED ENTRY ANSWER RECORDED' in raster.entry_verdict()


def test_retired_p2_key_details_preserve_first_cycle_and_total_count():
    raster = grid()
    telemetry.bind_grid(raster, capacity=4)
    for cycle in range(10000):
        raster.cycles = cycle
        raster.keying(Protocol.PACTOR2)
    assert raster.p2_keyed_cycles.retained == 4
    assert len(raster.p2_keyed_cycles) == 10000
    message = raster.entry_verdict()
    assert '10000 burst(s), first in cycle 0' in message
    assert 'does not confirm the peer received' in message


def test_binding_grid_diagnostics_cannot_drop_existing_entry_observations():
    raster = grid()
    raster.entry_answers.append((1, 60000, 120.))
    with pytest.raises(ValueError, match='fresh grid'):
        telemetry.bind_grid(raster, capacity=4)
    assert raster.entry_answers == [(1, 60000, 120.)]
