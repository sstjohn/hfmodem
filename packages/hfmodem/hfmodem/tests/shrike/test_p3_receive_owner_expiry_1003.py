# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Expired mail retires its transfer and keys the goodbye, in every role."""
import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import presence_session
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid


def _offline_key_cycle(session, outdir):
    """Use the normal renderer/admission with the existing offline key stub."""
    from hfmodem.tests.shrike.test_p3_reply_placement import driver
    outdir.mkdir(parents=True, exist_ok=True)
    grid = irs_grid()
    if session.host.arq.role == arq.ISS:
        grid.reverse(to_iss=True)
    radio = driver(grid, outdir)
    radio.host = session.host
    previous, session.host.peer = session.host.peer, radio
    radio.aim(grid, 30)
    radio.live.now = radio.live.pos = radio.key_instant(grid, 30)-6000
    attempts = []
    send = radio.send_packet

    def record(sl, payload, status, breakin=False):
        attempts.append((len(payload), bool(status & arq.spec.STATUS_QRT), breakin))
        return send(sl, payload, status, breakin=breakin)

    radio.send_packet = record
    try:
        session.host.arq.on_cycle()
    finally:
        session.host.peer = previous
    return attempts, radio.n, radio.refused


@pytest.mark.parametrize('qualified', [False, True], ids=['unsupported', 'qualified'])
@pytest.mark.parametrize('role', ['irs', 'unconfirmed_iss', 'confirmed_iss'])
def test_mail_expiry_retires_the_transfer_and_keys_the_goodbye(role, qualified, tmp_path):
    from hfmodem.tests.shrike.test_p3_receive_only_1003 import _application_expirer
    session = presence_session(1000)
    a = session.host.arq
    a.cfg.long_cycle = False
    a.role = arq.IRS if role == 'irs' else arq.ISS
    expire = _application_expirer(a)
    if role != 'irs':
        a._inflight = arq._Packet(status=0, payload=b'retained field', sl=1,
                                  breakin=role == 'unconfirmed_iss')
    a._outbuf.extend(b'retained suffix')
    credit = a.acked, a.tx_acked_bytes
    token = a.begin_receive_opportunity()
    assert a.qualify_peer(token, a._link_generation, 'current CRC' if qualified else None)
    assert a.settle_peer_presence()
    expire()
    # Whatever the peer's support, the expiry asks the goodbye at once.
    assert a._qrt_pending and not a._outbuf and a._inflight is None
    retained = dict(a.transfer_unconfirmed)
    assert retained['field'] == (b'' if role == 'irs' else b'retained field')
    assert retained['suffix'] == b'retained suffix'
    a.finish_receive_opportunity(token)
    attempts, keys, refused = _offline_key_cycle(session, tmp_path)
    assert attempts == [(0, True, role == 'irs')] and keys == 1 and not refused
    assert (a.acked, a.tx_acked_bytes) == credit
    assert a.transfer_unconfirmed == retained
