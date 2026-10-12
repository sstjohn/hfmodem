# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Receiver ownership and SIS queue accounting across the integrated fixes."""
from types import SimpleNamespace

import numpy as np

from hfmodem.bateleur import spec, stream
from hfmodem.bateleur.modem import BateleurModem, Observer
from hfmodem.bateleur.s5066 import primitives as P
from hfmodem.bateleur.s5066.node import Node, Timing
from hfmodem.tests.bateleur.s5066.ether import Client


def test_failed_receive_owner_ends_before_another_waveform_delivers():
    events = []
    observer = Observer()
    observer.rx_started = lambda standard, *args: events.append(("start", standard))
    observer.rx_data = lambda data: events.append(("data", data))
    observer.rx_ended = lambda complete, snr, why: events.append(("end", complete, why))
    modem = BateleurModem(lambda *_: "", card_rate=spec.FS, max_key_s=30,
                         observer=observer)
    modem._rx110.push = lambda _: [stream.Started("110", "600S", 0, 0)]
    modem._rx4539.push = lambda _: []
    modem._rx39.push = lambda _: []
    modem._audio(np.zeros(1024))

    def fail(_):
        raise ValueError("receive failure")

    modem._rx110.push = fail
    modem._rx4539.push = lambda _: [stream.Started("4539", "3200S", 0, 0),
                                    stream.Data(b"overlap")]
    modem._audio(np.zeros(1024))
    assert events == [("start", "110"), ("end", False, "receiver reset")]
    modem._rx110.push = lambda _: []
    modem._audio(np.zeros(1024))
    assert events[-2:] == [("start", "4539"), ("data", b"overlap")]


def test_unbind_purges_waiting_data_but_counts_an_active_transmit_window():
    node = Node(1, Timing(1200), segment=16)
    node.sis.max_queued = 1
    node.dts.connection_made(2)
    node.cas.made = lambda remote: True
    old = Client(node)
    old.bind(6, confirm=P.CONFIRM_NONE)
    old.unidata(2, 6, bytes(100))
    node.sis.tick(0)
    active = node.sis.awaiting[0]
    assert node.dts.compose(0, 50)
    assert node.dts.links[2].arq.frames
    old.att.close()
    assert node.sis.awaiting == [active]
    new = Client(node)
    new.bind(6)
    new.unidata(2, 6, b"blocked")
    assert new.of(P.UNIDATA_REQUEST_REJECTED)
    node.sis.spdu_rejected(active, "link lost")
    assert not node.sis.awaiting
    new.unidata(2, 6, b"queued")
    node.sis.tick(1)
    assert node.dts.links[2].arq.pending
    new.att.close()
    assert not node.dts.links[2].arq.pending
    assert not node.sis.awaiting


def test_abort_survives_rendered_size_reduction():
    rendered, sent = [], []
    modem = BateleurModem(lambda *_: "", card_rate=spec.FS, max_key_s=1)

    def render(piece, *args):
        rendered.append(piece)
        return np.zeros(24000 if len(piece) > 2 else 120)

    def send(*args):
        sent.append(args)
        modem.abort()
        return ""

    modem.mode = SimpleNamespace(most_bytes=lambda _: 4, render=render)
    modem._send = send
    modem.transmit(b"abcdefgh")
    assert rendered == [b"abcd", b"ab"]
    assert len(sent) == 1
