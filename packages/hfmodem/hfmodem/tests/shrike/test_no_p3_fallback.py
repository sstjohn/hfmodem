# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""No lower-mode transmissions after P3 entry, including bounded teardown.

These are decoded-event/renderer-seam checks; no waveform or stock-peer
acceptance is simulated. Every emission records its actual routed protocol.
"""
import pytest

from hfmodem.shrike import arq, pactor1, rxfront, spec
from hfmodem.shrike.ptc import PtcHost


class Recorder:
    def __init__(self):
        self.emissions = []

    def attach(self, host):
        self.host = host

    def pump(self):
        pass

    def cycle(self):
        pass

    def connect_burst(self, mycall, dxcall):
        self.emissions.append((spec.Protocol.PACTOR1, "connect", b"", 0))

    def send_p1_packet(self, payload, baud, seq, **kwargs):
        self.emissions.append((spec.Protocol.PACTOR1, "packet", bytes(payload), seq))
        return len(payload)

    def send_p1_breakin(self, payload, baud, seq, **kwargs):
        self.emissions.append((spec.Protocol.PACTOR1, "breakin", bytes(payload), seq))
        return len(payload)

    def send_p1_cs(self, index):
        self.emissions.append((spec.Protocol.PACTOR1, "cs", b"", index))

    def send_packet(self, sl, payload, status, breakin=False):
        self.emissions.append((spec.Protocol.PACTOR3,
                               "breakin" if breakin else "packet",
                               bytes(payload), status))
        return len(payload)

    def send_entry_packet(self, sl, payload, status, acquire=False):
        self.emissions.append((spec.Protocol.PACTOR3, "entry", bytes(payload), status))
        return len(payload)

    def send_cs(self, index):
        self.emissions.append((spec.Protocol.PACTOR3, "cs", b"", index))


def cs(index, protocol=spec.Protocol.PACTOR3):
    return rxfront.Event(0.1, "cs", "control", protocol=protocol, cs=index, sense=0)


def grant(host):
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=spec.Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))


def entered(*, enabled=True, payload=b""):
    tx = Recorder()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.no_p3_fallback = enabled
    host.p1_grant_only = True
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(cs(pactor1.CS_SPEED, spec.Protocol.PACTOR1))
    host.tick()
    if payload:
        host.arq.on_host_data(payload)
    mark = len(tx.emissions)
    grant(host)
    host.tick()
    assert host.protocol is spec.Protocol.PACTOR3 and host.arq.entry_pending
    assert tx.emissions[mark][1] == "entry"
    return host, tx, mark


def acknowledge(host):
    seq = host.arq.tx_seq
    assert seq is not None
    host.on_rx_event(cs(arq.CS_REQUEST if seq & 1 else arq.CS_ACK))
    host.tick()


def only_p3(tx, mark):
    assert tx.emissions[mark:]
    assert all(row[0] is spec.Protocol.PACTOR3 for row in tx.emissions[mark:])


def entry_limit(host):
    # The original baseline used the ordinary retry allowance for this budget.
    return getattr(arq, "ENTRY_GRANT_CYCLES", host.arq.cfg.max_retries)


def test_repeated_grants_keep_entry_and_retry_monitoring_past_twice_the_limit():
    host, tx, mark = entered(payload=b"application data")
    pending = host.arq._inflight
    queued = bytes(host.arq._outbuf)
    repeats = 2 * entry_limit(host) + 3
    for _ in range(repeats):
        grant(host)
        host.tick()
        assert host.arq.state is arq.State.CONNECTED
        assert host.arq.entry_pending and host.arq._inflight is pending
        assert host.arq.upgrade_unanswered and pending.retries == 1
    assert host.arq._upgrade_requests == repeats
    assert bytes(host.arq._outbuf) == queued
    assert len(tx.emissions[mark:]) == repeats + 1
    assert all(row[1] == "entry" for row in tx.emissions[mark:])
    only_p3(tx, mark)


def test_silent_unconfirmed_entry_ends_without_p1_even_during_teardown():
    host, tx, mark = entered()
    for _ in range(64):  # Also covers the original baseline's teardown budget.
        host.tick()
        if host.arq.state not in (arq.State.CONNECTED, arq.State.DISCONNECTING):
            break
    else:
        pytest.fail("silent unconfirmed entry did not reach bounded teardown")
    only_p3(tx, mark)
    count = len(tx.emissions)
    host.tick()
    assert len(tx.emissions) == count
    assert not host.arq.said_goodbye  # Entry was never confirmed for a P3 QRT.


@pytest.mark.parametrize("confirmed,closing", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("kind", ["ack", "cs3", "packet", "breakin"])
def test_p1_events_cannot_acknowledge_yield_or_close_p3(confirmed, closing, kind):
    host, tx, mark = entered(payload=b"" if closing else b"keep a real packet pending")
    if confirmed:
        acknowledge(host)
        assert not host.arq.entry_pending
    if closing:
        # Baseline preserves the announcement behind entry; HEAD settles it.
        # Drain either legitimate queue before requiring an on-air QRT.
        for _ in range(10):
            if host._txbuf == 0:
                break
            acknowledge(host)
        assert host._txbuf == 0
        host.arq.on_host_disconnect()
        host.tick()
        assert host.arq.state is arq.State.DISCONNECTING
    if kind in ("ack", "cs3"):
        event = cs(pactor1.CS_ACK_A if kind == "ack" else pactor1.CS_CHANGEOVER,
                   spec.Protocol.PACTOR1)
    else:
        event = rxfront.Event(0.1, "packet", "P1 data", protocol=spec.Protocol.PACTOR1,
                               packet=(arq.P1_SPEED_LEVEL, 0x80, b"RMS", True),
                               breakin=kind == "breakin")
    before = (host.arq.state, host.arq.role, host.arq._inflight,
              host.arq.tx_seq, host.arq.rx_seq, host.arq.entry_pending,
              host.arq.upgrade_unanswered, host.sent_total, host.rcvd_total,
              len(tx.emissions))
    host.on_rx_event(event)
    after = (host.arq.state, host.arq.role, host.arq._inflight,
             host.arq.tx_seq, host.arq.rx_seq, host.arq.entry_pending,
             host.arq.upgrade_unanswered, host.sent_total, host.rcvd_total,
             len(tx.emissions))
    assert after == before
    assert host.protocol is spec.Protocol.PACTOR3
    only_p3(tx, mark)


def test_confirmed_p3_transfers_payload_and_acknowledges_its_goodbye():
    payload = b"bench payload " * 6
    host, tx, mark = entered(payload=payload)
    # Preserve each snapshot's independent grant policy: baseline requeues its
    # announcement, HEAD settles it. Both must carry every queued byte exactly.
    expected = bytes(host.arq._outbuf)
    assert expected.endswith(payload)
    acknowledge(host)
    assert not host.arq.entry_pending and not host.arq.upgrade_unanswered
    for _ in range(20):
        acknowledge(host)
        if host._txbuf == 0:
            break
    assert host._txbuf == 0
    sent = b"".join(row[2] for row in tx.emissions[mark:] if row[1] == "packet")
    assert sent == expected
    host.arq.on_host_disconnect()
    host.tick()
    assert host.arq.said_goodbye and tx.emissions[-1][3] & spec.STATUS_QRT
    acknowledge(host)
    assert host.arq.goodbye_acked
    assert host.arq.state in (arq.State.DISCONNECTED, arq.State.LISTENING)
    only_p3(tx, mark)


def test_unread_confirmed_rung_qrts_at_the_bound_rather_than_dropping_to_p1():
    """`--no-p3-fallback` closes the only retreat `arq.UNREAD_RUNG_REPEATS` had,
    and what it used to do at the bound was nothing: the branch was one-shot, the
    fallback was refused, and the repeat train carried on -- 22 further cycles
    and 19 s of carrier at KB5LZK 40 m on 2026-09-15. The link ends instead, in
    PACTOR-3, and the log says so before it does."""
    said = []
    host, tx, mark = entered(payload=b"payload awaiting acknowledgement")
    host.log = said.append
    acknowledge(host)
    pending = host.arq._inflight
    assert pending is not None and not pending.entry and host.arq._entry_keyed
    repeat_start = len(tx.emissions)
    repeated_packet = (pending.payload, pending.status)
    for _ in range(arq.UNREAD_RUNG_REPEATS + 2):
        seq = host.arq.tx_seq
        host.on_rx_event(cs(arq.CS_ACK if seq is not None and seq & 1
                            else arq.CS_REQUEST))
        host.tick()
    assert host.protocol is spec.Protocol.PACTOR3
    assert not host.arq._entry_keyed  # The unread-rung branch ran.
    # Every request up to the bound was answered with the same packet again...
    repeated = tx.emissions[repeat_start:]
    asked = [row for row in repeated if (row[2], row[3]) == repeated_packet]
    assert len(asked) == arq.UNREAD_RUNG_REPEATS
    # ...and the one past it ended the link, saying what it was doing.
    assert any(f"asked {arq.UNREAD_RUNG_REPEATS + 1} times" in line
               and "--no-p3-fallback leaves no rung to retreat to" in line
               and "QRT" in line for line in said), said
    assert host.arq.state in (arq.State.DISCONNECTING, arq.State.DISCONNECTED,
                              arq.State.LISTENING)
    assert any(row[3] & spec.STATUS_QRT for row in repeated
               if row[1] == "packet")
    assert not host.arq._outbuf
    only_p3(tx, mark)


def test_abort_then_fresh_contact_restarts_p1_without_old_grant_state():
    host, tx, mark = entered()
    host.arq.on_host_abort()
    host.tick()
    only_p3(tx, mark)
    assert host.protocol is spec.Protocol.PACTOR1
    assert not host._grant_taken and not host._grant_pending
    assert host.no_p3_fallback
    restart = len(tx.emissions)
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.on_rx_event(cs(pactor1.CS_SPEED, spec.Protocol.PACTOR1))
    host.tick()  # Setup queues the announcement; the raster emits it.
    assert tx.emissions[restart][1] == "connect"
    assert tx.emissions[-1][0] is spec.Protocol.PACTOR1
    assert tx.emissions[-1][1] == "packet"
    assert tx.emissions[-1][3] == 1
    grant(host)
    assert host.protocol is spec.Protocol.PACTOR3 and host.arq.entry_pending


def test_unconfirmed_disconnect_closes_without_any_additional_transmission():
    host, tx, mark = entered(payload=b"unsent application bytes")
    count = len(tx.emissions)
    host.arq.on_host_disconnect()
    for _ in range(16):
        host.tick()
    assert len(tx.emissions) == count
    assert host.arq.state in (arq.State.DISCONNECTED, arq.State.LISTENING)
    assert not host.arq.said_goodbye
    only_p3(tx, mark)


@pytest.mark.parametrize("reason", ["silence", "grants", "breakin"])
def test_option_off_preserves_existing_p1_fallback(reason):
    host, tx, mark = entered(enabled=False)
    if reason == "breakin":
        host.on_rx_event(cs(pactor1.CS_CHANGEOVER, spec.Protocol.PACTOR1))
        host.tick()
    else:
        for _ in range(entry_limit(host) if reason == "grants"
                       else arq.UPGRADE_SILENCE_CYCLES):
            if reason == "grants":
                grant(host)
            host.tick()
    assert host.protocol is spec.Protocol.PACTOR1
    assert any(row[0] is spec.Protocol.PACTOR1 for row in tx.emissions[mark:])
