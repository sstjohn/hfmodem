# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hfmodem.core.ftdx10 import filter_commands, normal, read_filters, session_frequency
from hfmodem.core.rigs import RIGS


def test_profile_defaults_and_configurable_cat_fields():
    assert RIGS['ftdx10']['rx_width_hz'] == 4000
    assert RIGS['ftdx10']['roofing_hz'] == 12000
    assert RIGS['ftdx10']['if_shift_hz'] == 500
    assert filter_commands() == ('RF01;', 'SH0021;', 'IS00+0500;')
    assert filter_commands(dict(rx_width_hz=3000, roofing_hz=3000)) == ('RF02;', 'SH0018;')
    assert filter_commands(dict(rx_width_hz=3000, roofing_hz=3000, if_shift_hz=-250))[2] == 'IS00-0250;'


def test_the_if_shift_reads_from_the_rigs_own_answer():
    from hfmodem.core.ftdx10 import read_shift

    assert read_shift('Reply: IS00+0500;') == 500
    assert read_shift('IS00-0120;') == -120
    assert read_shift('send_cmd: IS00+0500;\nRPRT 0') is None
    assert read_shift('IS00+0500; IS00+0000;') is None


def test_a_profile_that_owns_its_width_asks_hamlib_for_no_passband_change():
    from hfmodem.core.rigs import mode_passband

    assert mode_passband(RIGS['ftdx10'], 3000) == -1
    assert mode_passband(RIGS['ft891'], 3000) == 3000


def test_set_values_are_not_readback_evidence():
    assert read_filters('SH0021; RF01;') == (4000, None)
    assert read_filters('SH0021; SH0018; RF06; RF09;') == (None, None)


def test_ssb_only_index_is_unverified_in_data():
    assert read_filters('SH0023; RF06;') == (None, 12000)


def test_frequency_echo_and_vfo_b_reply_are_not_dial_evidence():
    from hfmodem.core.ftdx10 import session_frequency

    reply = ('send_cmd: FA007101500;\nRPRT 0\n'
             'send_cmd: FA;\nReply: FB007101500;\nRPRT 0')
    assert session_frequency(reply, daemon=True) == ''
    assert session_frequency(reply.replace('FB007101500;', 'FA007101500;'), daemon=True) == '7101500'
    assert session_frequency(reply.replace('RPRT 0', 'RPRT -6', 1), daemon=True) == ''


def test_frequency_is_required_before_serial_level_readings():
    from hfmodem.core.ftdx10 import session_frequency

    assert session_frequency('error = dial failed\n0\n0\nSH0021;\nRF06;\nVFOA\nMD0C;') == ''


# Hamlib rigctld's own answer to `+\\send_cmd MD0; SH0; RF0; IS0;` then `q`, byte for byte
# (rigctl_parse.c): each query's answer ends in a NUL, and q has an RPRT 0 of its own.
READBACK = ('send_cmd: MD0;\nReply: MD0C;\0RPRT 0\nsend_cmd: SH0;\nReply: SH0021;\0RPRT 0\n'
            'send_cmd: RF0;\nReply: RF06;\0RPRT 0\nsend_cmd: IS0;\nReply: IS00+0500;\0RPRT 0\n'
            'RPRT 0\n')


def test_rigctlds_readback_verifies_mode_and_filters(capsys):
    from hfmodem.core.ftdx10 import readback_report

    readback_report(normal(READBACK), daemon=True)
    assert capsys.readouterr().out == (
        '  DATA-U verified\n  width 4000 verified\n  roofing filter 12000 verified\n'
        '  IF shift +500 Hz verified\n')


def test_the_quits_status_is_not_part_of_the_last_answer():
    from hfmodem.core.ftdx10 import daemon_part

    assert daemon_part(normal(READBACK), 'send_cmd: IS0;') == (
        'send_cmd: IS0;\nReply: IS00+0500;\nRPRT 0')
    assert daemon_part('get_vfo:\nVFO: VFOA\nRPRT 0\nRPRT 0\n', 'get_vfo:') == (
        'get_vfo:\nVFO: VFOA\nRPRT 0')


def test_mode_set_status_and_other_receiver_are_never_mode_evidence(capsys):
    from hfmodem.core.ftdx10 import readback_report

    readback_report(normal(READBACK.replace('Reply: MD0C;\0', 'Reply: ')), daemon=True)
    readback_report(normal(READBACK.replace('MD0C;', 'MD1C;')), daemon=True)
    readback_report(normal(READBACK.replace('MD0C;', 'MD02;')), daemon=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if 'mode' in line]
    assert lines == ['  !! mode unverified (missing or unknown CAT reply)'] * 2 + [
        '  !! mode reads 2 (not DATA-U)']


def test_serial_readback_reads_the_one_shot_answers(capsys):
    from hfmodem.core.ftdx10 import readback_report

    readback_report('MD0C;\nSH0018;\nRF07;\nIS00+0000;\n')
    readback_report('SH0021;\nRF06;\nIS00+0500;\n')
    assert capsys.readouterr().out.splitlines() == [
        '  DATA-U verified', '  !! width reads 3000', '  !! roofing filter reads 3000',
        '  !! IF shift reads +0 Hz',
        '  !! mode unverified (missing or unknown CAT reply)', '  width 4000 verified',
        '  roofing filter 12000 verified', '  IF shift +500 Hz verified']


def test_nul_terminated_daemon_replies_read_as_the_plain_form():
    # Debian's Hamlib on shackbook, 2026-10-10: the dial read back as nothing.
    reply = ('send_cmd: FA003589500;\nReply: RPRT 0\n'
             'send_cmd: FA;\nReply: FA003589500;\0RPRT 0\n')
    assert session_frequency(reply.replace('\0', ''), daemon=True) == ''
    assert session_frequency(normal(reply), daemon=True) == '3589500'
