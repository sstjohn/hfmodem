# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p4turn import machine

RETIRED = ('p4-mail p4-entry p4-entry-attempts p4-probe-packet p4-probe-statuses '
           'p4-probe-payload p4-probe-values p4-probe-headers p4-probe-cadence p4-probe-reply '
           'p4-probe-listen p4-ack-53 p4-ack-53-b p4-ack-53-rotate p4-v59-answer '
           'p4-ladder p4-irs-silent-cycles p4-text p1-6a9 p1-robust-stall-then').split()

@pytest.mark.parametrize('flag', RETIRED)
def test_retired_flags_rejected(flag):
    with pytest.raises(SystemExit):
        onair.build_parser().parse_args(['--'+flag])

@pytest.mark.parametrize('flag,values', [('--p4-normal-ack', ['v27','x27-u27']),
    ('--p4-long-ack', ['fixed-x27','counter-odd-t27','alternate-packet','toggle-v27']),
    ('--p4-chirp-ack', ['x27','alternate']), ('--after-connect', ['chirp']),
    ('--p4-spug', ['first']), ('--p4-chain-reset', ['none', '', 'turn,none'])])
def test_retired_values_rejected(flag, values):
    for value in values:
        with pytest.raises(SystemExit):
            onair.build_parser().parse_args([flag,value])

@pytest.mark.parametrize('opening', ['','64','-1','1,2,3,4,5','x','7,,38'])
def test_invalid_opening(opening):
    with pytest.raises(SystemExit):
        onair.build_parser().parse_args(['--p4-opening',opening])

def test_peer_ack_fold_and_empty_defaults():
    p = onair.build_parser()
    args = p.parse_args(['--p4-peer-ack','53=X27,53/B=V27,59=W27'])
    assert args.p4_peer_ack == {53:'X27','53/B':'V27',59:'W27'}
    assert p.parse_args(['--p4-peer-ack','']).p4_peer_ack == {}
    assert machine(peer_ack={})[0].peer_ack == machine()[0].peer_ack
    assert machine(peer_ack={53:'X27','53/B':'V27',59:'W27'})[0].peer_ack[59] == 'W27'

@pytest.mark.parametrize('entry,opening', [(['--p4-link'],(7,38)),
    (['--connect','robust','--after-robust','p4-robust'],(6,9,8))])
def test_internal_entry(entry, opening):
    args = onair.build_parser().parse_args(entry)
    onair._connect_defaults(args)
    onair._p4_entry_defaults(args)
    assert args.p4_entry and args.p4_opening == opening and args.p4_entry_timeout == 20


def test_p1_hold_and_p3_profile_do_not_shape_p4_driver(monkeypatch):
    import numpy as np
    from hfmodem.shrike import p4turn
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    original = p4turn.P4Turn
    captured = []
    def turn(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(p4turn, 'P4Turn', turn)
    for profile in [[], ['--hold','96','--p3-max-try','2','--p3-max-down','6',
                         '--p3-entry-sl','3','--p3-traffic-sl','1','--p3-repeat-gear','0',
                         '--p3-speed-up','hold','--p3-entry-stagger','--p3-entry-rise']]:
        args = onair.build_parser().parse_args(['--p4-link','--p1-robust-stall','12',
            '--p4-data','body','--p4-link-timeout','1',*profile])
        onair._connect_defaults(args)
        onair._p4_entry_defaults(args)
        assert args.hold == 13 and args.p3_entry == ('p4chirp',)
        live = Stream(np.zeros(2*onair.FS))
        tx = Tx(live)
        tx.p4_link_options = args
        onair._run_p4_link(tx, live, onair._MasterGrid(
            0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    assert captured[0] == captured[1]

@pytest.mark.parametrize('options', [dict(spug='first'), dict(long_ack='toggle-v27'),
    dict(chain_reset='none'), dict(long_level=1)])
def test_turn_rejects_cut_policies(options):
    with pytest.raises(ValueError):
        machine(**options)

@pytest.mark.parametrize('level', [2, 3, 4])
def test_long_level_is_fixed_integer(level):
    assert machine(long_level=level)[0].long_level == level
