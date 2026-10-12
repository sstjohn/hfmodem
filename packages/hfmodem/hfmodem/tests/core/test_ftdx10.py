from hfmodem.core.ftdx10 import filter_commands, normal, read_filters, session_frequency
from hfmodem.core.rigs import RIGS


def test_profile_defaults_and_configurable_cat_fields():
    assert RIGS['ftdx10']['rx_width_hz'] == 4000
    assert RIGS['ftdx10']['roofing_hz'] == 12000
    assert filter_commands() == ('SH0021;', 'RF01;')
    assert filter_commands(dict(rx_width_hz=3000, roofing_hz=3000)) == ('SH0018;', 'RF02;')


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


def test_selection_set_echo_is_never_mode_evidence(capsys):
    from hfmodem.core.ftdx10 import selection_report

    selection_report('set_vfo: VFOA\nRPRT 0\nsend_cmd: MD0C;\nReply: MD0C;\nRPRT 0\n'
                     'get_vfo:\nVFO: VFOA\nRPRT 0\nsend_cmd: MD0;\nRPRT 0', daemon=True)
    output = capsys.readouterr().out
    assert 'VFO-A verified' in output
    assert 'mode unverified' in output and 'DATA-U verified' not in output


def test_serial_mode_set_reply_cannot_verify_missing_mode_query(capsys):
    from hfmodem.core.ftdx10 import session_report

    session_report('MD0C;\n7101500\n0\n0\nSH0021;\nRF06;\nVFOA')
    output = capsys.readouterr().out
    assert 'VFO-A verified' in output
    assert 'mode unverified' in output and 'DATA-U verified' not in output


def test_nul_terminated_daemon_replies_read_as_the_plain_form():
    # Debian's Hamlib on shackbook, 2026-10-10: the dial read back as nothing.
    reply = ('send_cmd: FA003589500;\nReply: RPRT 0\n'
             'send_cmd: FA;\nReply: FA003589500;\0RPRT 0\n')
    assert session_frequency(reply.replace('\0', ''), daemon=True) == ''
    assert session_frequency(normal(reply), daemon=True) == '3589500'
