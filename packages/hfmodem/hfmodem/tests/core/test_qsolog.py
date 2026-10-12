# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The mode-agnostic QSO log: ADIF the logbook viewer reads, and serials that resume."""
from __future__ import annotations

import pytest

from hfmodem.station.qsolog import QsoLog, read_adif


def test_adif_round_trips_through_the_reader_the_logbook_viewer_uses(tmp_path):
    path = tmp_path / 'qso-log.adi'
    log = QsoLog(path, 'w9ssj', grid='EN63bc', contest='ARRL-RTTY')
    log.start('k1abc')
    log.commit(mode='RTTY', freq_hz=14_082_295, rst_sent='599', rst_rcvd='599', srx='042')
    log.start('N0XYZ')
    log.commit(mode='RTTY', freq_hz=7_082_295, rst_sent='599', rst_rcvd='579', srx='7')
    first, second = read_adif(path.read_text())
    assert first == dict(CALL='K1ABC', QSO_DATE=first['QSO_DATE'], TIME_ON=first['TIME_ON'],
                         FREQ='14.082295', MODE='RTTY', RST_SENT='599', RST_RCVD='599',
                         STATION_CALLSIGN='W9SSJ', MY_GRIDSQUARE='EN63bc',
                         CONTEST_ID='ARRL-RTTY', STX='1', SRX='042')
    assert len(first['QSO_DATE']) == 8 and len(first['TIME_ON']) == 6
    assert (second['CALL'], second['STX'], second['RST_RCVD']) == ('N0XYZ', '2', '579')
    assert path.read_text().count('<EOH>') == 1


def test_serials_resume_per_contest_id(tmp_path):
    path = tmp_path / 'log.adi'
    for contest, calls in (('CQ-WW-RTTY', ['A1A', 'B1B', 'C1C']), ('NAQP-RTTY', ['D1D'])):
        log = QsoLog(path, 'W9SSJ', contest=contest)
        for call in calls:
            log.start(call)
            log.commit(mode='RTTY', freq_hz=14_080_000, rst_sent='599', rst_rcvd='599')
    assert QsoLog(path, 'W9SSJ', contest='CQ-WW-RTTY').serial == 4
    assert QsoLog(path, 'W9SSJ', contest='NAQP-RTTY').serial == 2
    assert QsoLog(path, 'W9SSJ', contest='NEW').serial == 1
    assert QsoLog(path, 'W9SSJ').serial == 1


def test_a_byte_counted_field_does_not_hide_the_contest_after_it(tmp_path):
    path = tmp_path / 'log.adi'
    path.write_text('<CALL:4>K1AB<NAME:4>Zo\u00eb<CONTEST_ID:4>TEST<STX:2>41<EOR>\n', encoding='utf-8')
    assert read_adif(path.read_text(encoding='utf-8'))[0]['CONTEST_ID'] == 'TEST'
    assert QsoLog(path, 'W9SSJ', contest='TEST').serial == 42


def test_call_and_serial_substitution(tmp_path):
    log = QsoLog(tmp_path / 'log.adi', 'W9SSJ', contest='TEST')
    assert log.substitute('{CALL} 599 {SERIAL}') == ' 599 001'
    log.start('k9xyz')
    assert log.substitute('{CALL} 599 {SERIAL} {SERIAL}') == 'K9XYZ 599 001 001'
    log.commit(mode='RTTY', freq_hz=14_080_000, rst_sent='599', rst_rcvd='599')
    assert log.current is None
    assert log.substitute('{SERIAL}') == '002'


def test_a_draft_never_logged_leaves_the_file_untouched(tmp_path):
    path = tmp_path / 'log.adi'
    log = QsoLog(path, 'W9SSJ')
    log.start('K1ABC')
    assert not path.exists()
    path.write_text('kept\n')
    QsoLog(path, 'W9SSJ').start('K2ABC')
    assert path.read_text() == 'kept\n'
    with pytest.raises(ValueError):
        QsoLog(path, 'W9SSJ').commit(mode='RTTY', freq_hz=1, rst_sent='599', rst_rcvd='599')


def test_records_set_aside_count_toward_the_serial_until_they_are_written(tmp_path):
    path = tmp_path / 'log.adi'
    log = QsoLog(path, 'W9SSJ', contest='TEST')
    log.start('K1ABC')
    _, record = log.record(mode='RTTY', freq_hz=14_080_000, rst_sent='599', rst_rcvd='599')
    assert log.keep([record]) and log.kept() == [record]
    assert QsoLog(path, 'W9SSJ', contest='TEST').serial == 2
    log.write(record)
    assert log.keep([]) and log.kept() == [] and not log.held.exists()
    assert QsoLog(path, 'W9SSJ', contest='TEST').serial == 2
