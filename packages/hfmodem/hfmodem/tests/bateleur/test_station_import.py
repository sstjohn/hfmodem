# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Review sequences at the station's thread and protocol boundaries."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest


def controller(**kw):
    from hfmodem.bateleur.ale.controller import Config, Controller
    sent = []
    c = Controller(Config(["JOE"], ["A", "B"], listen_s=0, **kw),
                   tune=lambda ch: True, send=lambda ws: sent.append(ws) or len(ws) * .392)
    return c, sent


@pytest.mark.parametrize("value", [-1, 1, float("nan"), float("inf")])
def test_sound_interval_is_bounded(value):
    with pytest.raises(ValueError, match="sound_s"):
        controller(sound_s=value)


@pytest.mark.parametrize("reply", [False, None])
def test_refused_tune_does_not_change_channel_or_send(reply):
    c, sent = controller()
    c._tune = lambda ch: reply
    c.call("SAM", 0, channel=1)
    c.tick(1)
    assert c.channel == 0
    assert not sent
    assert any(e.kind == "CALL_FAILED" and "tune" in e.text for e in c.events)


def test_refusal_detail_reaches_call_event():
    c, _ = controller()
    c._send = lambda ws: "identification due"
    c.call("SAM", 0)
    c.tick(1)
    assert any(e.kind == "CALL_FAILED" and "identification due" in e.text for e in c.events)


def test_oversize_ale_frame_never_reaches_send():
    from hfmodem.bateleur.ale import words as W
    c, sent = controller()
    assert c._transmit(W.call("SAM", "JOE", scan_words=100), 0) is None
    assert not sent
    assert any("max_key_s" in e.text for e in c.events)


def test_carrier_detection_episode_cannot_hold_scan_forever():
    c, _ = controller()
    for now in np.arange(0, 10, .1):
        c.signal(now)
    assert c._pause_until < 2


def test_stray_words_cannot_hold_scan_forever():
    from hfmodem.bateleur.ale import fsk, words as W
    c, _ = controller()
    for now in range(120):
        c.heard(fsk.Heard(W.word(W.TO, "XYZ"), 0, 0, 48, 1), now)
    assert c._pause_until <= 30


def test_dtm_pause_requests_cannot_extend_exchange_forever():
    from hfmodem.bateleur.ale import words as W
    c, _ = controller()
    c.linked.add("SAM")
    c._set_state(c.LINKED)
    c.send_dtm("SAM", "HELLO", 0)
    c.tick(0)
    fields = W.dtm_fields(c._dtm_out[0][2][0][0])
    pause = W.parse(W.call("JOE", "SAM", message=[W.dtm_arq(
        kd1=fields["kd1"], kd2=fields["kd2"], pause=True)])).dtm[0]
    for now in range(55, 301, 55):
        c._dtm_arq("SAM", pause, now)
    c.tick(301)
    assert not c._dtm_out
    assert any(e.kind == "DTM_FAILED" for e in c.events)


@pytest.mark.parametrize("kind", ["dtm", "dbm"])
def test_ale_outbound_messages_are_bounded(kind):
    c, _ = controller()
    send = getattr(c, "send_" + kind)
    for i in range(16):
        send("SAM", "HELLO", 0)
    with pytest.raises(ValueError, match="queue"):
        send("SAM", "HELLO", 0)


def test_ale_noise_estimator_recovers_after_nan():
    from hfmodem.bateleur.ale import fsk
    rx = fsk.Receiver()
    rx.push(np.r_[np.nan, np.ones(8000)])
    assert np.isfinite(rx._noise)
    assert np.isfinite(rx.llr).all()


def test_single_tone_is_not_ale_signal():
    from hfmodem.bateleur.ale import fsk
    rx = fsk.Receiver()
    rx.push(np.cos(2 * np.pi * 1500 * np.arange(16000) / fsk.FS))
    assert not rx.signal


def test_oversize_dbm_cannot_lock_receiver():
    from hfmodem.bateleur.ale import fsk, words as W
    rx = fsk.Receiver()
    rx._lock = 0
    rx._frame = [fsk.Heard(W.word(W.TO, "JOE"), 0, 0, 48, 1)]
    command = W.dbm_word(445)
    rx._accept(fsk.Heard(command, 3136, 0, 48, 1), [])
    rx._dbm_begin(147)
    assert rx._dbm is None and not rx.locked


def received(words):
    from hfmodem.bateleur.ale import fsk, words as W
    return fsk.Received(W.parse(words), [], 0, 0, 0, 20, 0)


def test_repeated_allcall_does_not_refresh_activity():
    from hfmodem.bateleur.ale import words as W
    c, _ = controller()
    frame = received(W.call(W.ALLCALL, "SAM"))
    c.heard(frame, 1)
    c.heard(frame, 25)
    c.tick(32)
    assert c.state == c.AVAILABLE


def test_individual_response_throttle():
    from hfmodem.bateleur.ale import words as W
    c, sent = controller()
    frame = received(W.call("JOE", "SAM"))
    c.heard(frame, 1)
    c.tick(20)
    c.heard(frame, 21)
    assert len(sent) == 1


@pytest.mark.parametrize("dialect", ["bateleur", "ale"])
def test_timed_out_command_is_cancelled(dialect):
    from hfmodem.bateleur.host import BateleurHost
    from hfmodem.bateleur.ale.host import AleHost
    cls = BateleurHost if dialect == "bateleur" else AleHost
    host = object.__new__(cls)
    pending, ran = [], []
    host.modem = SimpleNamespace(post=pending.append)
    with pytest.raises(Exception):
        host._on_modem(lambda: ran.append(True), timeout=.001)
    pending[0]()
    assert not ran


def test_abort_reaches_posted_message():
    from hfmodem.bateleur.host import BateleurHost
    from hfmodem.bateleur.modem import BateleurModem
    sent, pending = [], []
    modem = BateleurModem(lambda *args: sent.append(args) or "", card_rate=48000, max_key_s=30)
    modem.post = pending.append
    host = BateleurHost(modem)
    host._txbuf.extend(b"test")
    assert host.command("SEND") == "OK SEND 4"
    assert host.command("ABORT").startswith("OK")
    pending[0]()
    assert not sent


def test_abort_stops_between_pieces():
    from hfmodem.bateleur.modem import BateleurModem
    sent = []
    modem = BateleurModem(lambda *a: "", card_rate=48000, max_key_s=30)
    modem.mode = SimpleNamespace(most_bytes=lambda s: 2, render=lambda *args: np.zeros(10))

    def send(*args):
        sent.append(args)
        modem.abort()
        return ""

    modem._send = send
    modem.transmit(b"abcdef")
    assert len(sent) == 1


def test_sis_bad_primitive_preserves_neighbours():
    from hfmodem.bateleur.s5066 import primitives as P
    good = P.encode(P.prim(P.KEEP_ALIVE))
    bad = P.PREAMBLE + bytes([P.VERSION, 0, 1, 255])
    result = P.Framer().push(good + bad + good)
    assert [p.type for p in result] == [P.KEEP_ALIVE, P.KEEP_ALIVE]


def test_sis_eviction_clears_attachment_and_pending_data():
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.tests.bateleur.s5066.ether import Client
    from hfmodem.bateleur.s5066.node import Node, Timing
    node = Node(1, Timing(1200))
    node.sis.max_expedited = 1
    client = Client(node)
    client.bind(6)
    client.unidata(2, 6, b"a", expedited=True)
    client.unidata(2, 6, b"b", expedited=True)
    assert 6 not in node.sis.clients
    assert not node.sis.queue
    client.unidata(2, 6, b"c")
    assert not node.sis.queue
    assert client.of(P.UNBIND_INDICATION)


def test_selcall_repeated_format_is_received():
    from hfmodem.bateleur.selcall import code as C
    expected = C.Call("selcall", "5678", "1234")
    dx = C.encode(expected)
    result = C.decode(dx[:1] + dx)
    assert result is not None and (result.to, result.frm) == (expected.to, expected.frm)


def test_selcall_m493_phasing_is_received():
    from hfmodem.bateleur.selcall import code as C, fsk
    dx = [120] + C.encode(C.Call("selcall", "5678", "1234"))
    air = [v for n in range(111, 105, -1) for v in (125, n)]
    rx = [105, 104] + dx[:-2]
    air += [v for pair in zip(dx, rx) for v in pair]
    heard = fsk.receive(np.r_[fsk.modulate(air), np.zeros(4000)])
    assert any(h.call and h.call.to == "5678" and h.call.frm == "1234" for h in heard)


def test_selcall_no_unread_history():
    from hfmodem.bateleur.selcall.modem import SelcallModem
    modem = SelcallModem(["1234"], send=lambda *a: None)
    assert not hasattr(modem, "heard")


def test_sounded_host_line_has_no_empty_peer():
    from hfmodem.bateleur.ale.host import AleHost
    from hfmodem.bateleur.ale.controller import Event
    ctl, _ = controller()
    host = AleHost(SimpleNamespace(ctl=ctl))
    host.event(Event("SOUNDED", channel=0))
    assert host.log[-1] == "SOUNDED CHANNEL A"


def test_http_body_cannot_reach_dialect():
    from hfmodem.core.hostio import lines
    chunks = iter([b"POST / HTTP/1.1\r\nHost: localhost\r\n\r\nCALL SAM\r\n", b""])
    conn = SimpleNamespace(recv=lambda n: next(chunks))
    assert list(lines(conn, lambda: True)) == []


def test_shared_decimator_matches_full_fir_at_chunk_edges():
    from hfmodem.core.resample import FirDecimator
    from scipy.signal import lfilter
    dec = FirDecimator(48000)
    x = np.random.default_rng(4).normal(size=5000)
    expected = lfilter(dec.h, [1], x)[::6]
    actual = np.concatenate([dec.push(x[i:i + 7]) for i in range(0, len(x), 7)])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_emergency_needs_config():
    from hfmodem.bateleur.selcall.host import SelcallHost
    modem = SimpleNamespace(address="1234", post=lambda fn: fn(), call=lambda *a, **kw: None)
    host = SelcallHost(modem)
    assert host.command("EMERGENCY").startswith("ERR")


def test_audio_backlog_keeps_newest_samples_and_resets_at_the_gap():
    from hfmodem.bateleur.selcall.modem import SelcallModem
    modem = SelcallModem(["1234"], send=lambda *a: None)
    seen, calls = [], []
    done = threading.Event()
    modem.reset_receive = lambda: seen.append("reset")
    def receive(at, samples):
        seen.append((at, len(samples)))
        if at == 239 * 2400:
            done.set()
    modem._audio = receive
    for i in range(240):
        modem.feed(i * 2400, np.zeros(2400))
    assert modem._audio_samples <= modem._audio_limit
    assert modem.audio_dropped == 112
    modem.post(lambda: calls.append("command"))
    modem.start()
    try:
        assert done.wait(2)
        modem.stop()
        assert calls == ["command"]
        assert seen == ["reset"] + [(i * 2400, 2400) for i in range(112, 240)]
    finally:
        modem.stop()


def test_directory_bad_refresh_keeps_good_cache(tmp_path, monkeypatch):
    from hfmodem.bateleur.ale import directory as D
    from hfmodem.tests.bateleur.ale.test_directory import SAMPLE
    import io
    cache = tmp_path / "directory"
    cache.write_text(SAMPLE)
    monkeypatch.setattr(D, "_cache_for", lambda url: cache)
    monkeypatch.setattr(D.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad page"))
    assert D.load_one("https://example.invalid/list", refresh=True).who("CW12")
    assert cache.read_text() == SAMPLE


def test_ale_and_selcall_transmit_edges_are_ramped():
    from hfmodem.bateleur.ale import fsk as A, words as W
    from hfmodem.bateleur.selcall import fsk as S
    for audio in (A.modulate(W.call("SAM", "JOE")), S.modulate([120, 117])):
        assert audio[0] == 0 and audio[-1] == 0


def test_repeated_data_preamble_is_not_amd_text():
    from hfmodem.bateleur.ale import words as W
    words = W.call("SAM", "JOE", message=[W.word(W.CMD, "ABC"),
                                          W.word(W.DATA, "DEF"), W.word(W.DATA, "GHI")])
    assert W.parse(words).amd == ["ABCDEF"]


def test_selcall_phasing_search_reuses_tone_windows(monkeypatch):
    from hfmodem.bateleur.selcall import fsk
    receiver = fsk.Receiver()
    original = receiver._soft
    processed = 0

    def count(audio, df):
        nonlocal processed
        processed += len(audio)
        return original(audio, df)

    monkeypatch.setattr(receiver, "_soft", count)
    audio = np.random.default_rng(6).normal(size=3 * fsk.FS)
    for i in range(0, audio.size, 80):
        receiver.push(audio[i:i + 80])
    assert processed < 10 * audio.size


def test_sis_inflight_unconfirmed_data_counts_toward_limit():
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.tests.bateleur.s5066.ether import Client
    from hfmodem.bateleur.s5066.node import Node, Timing
    node = Node(1, Timing(1200))
    client = Client(node)
    client.bind(6, confirm=P.CONFIRM_NONE)
    node.cas.made = lambda remote: True
    node.cas.send = lambda *args, **kwargs: None
    for i in range(65):
        client.unidata(2, 6, bytes([i]))
        node.sis.tick(i)
    assert len(node.sis.awaiting) == 64
    assert client.of(P.UNIDATA_REQUEST_REJECTED)


def test_sis_old_attachment_cannot_requeue_after_rebind():
    from hfmodem.tests.bateleur.s5066.ether import Client
    from hfmodem.bateleur.s5066.node import Node, Timing
    node = Node(1, Timing(1200))
    old = Client(node)
    old.bind(6)
    old.unidata(2, 6, b"old")
    outgoing = node.sis.queue[0]
    old.att.close()
    new = Client(node)
    new.bind(6)
    node.sis.spdu_rejected(outgoing, "link lost")
    assert not node.sis.queue


def test_sis_rebinding_cannot_escape_inflight_volume_limit():
    from hfmodem.bateleur.s5066 import primitives as P
    from hfmodem.tests.bateleur.s5066.ether import Client
    from hfmodem.bateleur.s5066.node import Node, Timing
    node = Node(1, Timing(1200))
    node.cas.made = lambda remote: True
    node.cas.send = lambda *args, **kwargs: None
    for _ in range(4):
        client = Client(node)
        client.bind(6, confirm=P.CONFIRM_NONE)
        for i in range(64):
            client.unidata(2, 6, bytes([i]))
            node.sis.tick(0)
        client.att.close()
    assert len(node.sis.awaiting) == 256
    client = Client(node)
    client.bind(6)
    client.unidata(2, 6, b"over limit")
    assert client.of(P.UNIDATA_REQUEST_REJECTED)
    outgoing = node.sis.awaiting[0]
    node.sis.spdu_rejected(outgoing, "link lost")
    assert len(node.sis.awaiting) == 255 and not node.sis.queue


def test_send_refusal_keeps_host_buffer():
    from hfmodem.bateleur.host import BateleurHost
    def refuse(data):
        raise RuntimeError("queue full")
    host = BateleurHost(SimpleNamespace(queue_transmit=refuse))
    host._txbuf.extend(b"pending")
    assert host.command("SEND").startswith("ERR")
    assert host._txbuf == b"pending"


def test_full_audio_queue_does_not_block_attachment_cleanup():
    from hfmodem.bateleur.selcall.modem import SelcallModem
    modem = SelcallModem(["1234"], send=lambda *a: None)
    for i in range(128):
        modem.feed(i, np.zeros(1))
    closed = []
    modem.post_cleanup(lambda: closed.append(True))
    modem.stop()
    assert closed == [True]


@pytest.mark.parametrize("kind", ["DTM", "DBM"])
def test_data_blocks_fit_key_limit_with_full_addresses(kind):
    from hfmodem.bateleur.ale import words as W, fsk
    from hfmodem.bateleur.ale.controller import Config, Controller
    own, peer = "ABCDEFGHIJKLMNO", "PQRSTUVWXYZ1234"
    c = Controller(Config([own], ["A"], max_key_s=10), tune=lambda ch: True, send=lambda ws: 0)
    c.linked.add(peer)
    c._send_blocks(kind, peer, b"payload" * 100, 0, True)
    blocks = c._dtm_out[0][2]
    assert len(blocks) > 1
    assert all(fsk.air_seconds(W.call(peer, own, message=b)) <= 10 for b in blocks)


def test_own_ale_address_cannot_be_wildcard():
    from hfmodem.bateleur.ale.controller import Config, Controller
    with pytest.raises(ValueError, match="own addresses"):
        Controller(Config(["J?E"], ["A"]), tune=lambda ch: True, send=lambda ws: 0)


def test_net_command_uses_preset_slots_when_members_omitted():
    from hfmodem.bateleur.ale.host import AleHost
    c, _ = controller()
    host = AleHost(SimpleNamespace(ctl=c, now=0, post=lambda fn: fn()))
    assert host.command("NET HFN") == "OK NET HFN"
    assert len(c._slots) == 3


def test_ale_frame_refusal_line_retains_reason():
    from hfmodem.bateleur.ale.host import AleHost
    from hfmodem.bateleur.ale.controller import Event
    c, _ = controller()
    host = AleHost(SimpleNamespace(ctl=c))
    host.event(Event("TX_REFUSED", channel=0, text="frame exceeds max_key_s"))
    assert host.log[-1] == "TX_REFUSED CHANNEL A frame exceeds max_key_s"


def test_rx_end_retains_loss_detail():
    from hfmodem.bateleur.host import BateleurHost
    host = BateleurHost(SimpleNamespace())
    got = []
    host._event = got.append
    host.rx_ended(False, 12, "signal lost before EOM")
    assert got == ["RX END LOST SNR 12 signal lost before EOM"]


def test_data_socket_receives_only_the_current_rx_owner():
    from hfmodem.bateleur import stream
    from hfmodem.bateleur.modem import BateleurModem
    got, starts, ends = [], [], []
    observer = SimpleNamespace(rx_started=lambda *a: starts.append(a), rx_data=got.append,
                               rx_ended=lambda *a: ends.append(a))
    modem = BateleurModem(lambda *a: "", card_rate=48000, max_key_s=30, observer=observer)
    modem._fe = SimpleNamespace(push=lambda x: x, audio=np.zeros(1))
    def receiver(batches):
        return SimpleNamespace(push=lambda x: batches.pop(0) if batches else [],
                               mode=None, receiving=True)
    modem._rx110 = receiver([[stream.Started("110", "600S", 0, 0), stream.Data(b"one")],
                             [stream.Data(b"two"), stream.Ended(True, 10, "EOM")]])
    modem._rx4539 = receiver([[stream.Started("4539", "3200S", 0, 0), stream.Data(b"wrong")],
                              [stream.Data(b"also wrong"), stream.Ended(True, 10, "EOM")]])
    modem._rx39 = receiver([[], []])
    modem._audio(np.zeros(1))
    modem._audio(np.zeros(1))
    assert got == [b"one", b"two"]
    assert starts == [("110", "600S", 0)] and len(ends) == 1


def test_valid_dbm_announcement_holds_scan_for_its_payload():
    from hfmodem.bateleur.ale import fsk, words as W
    c, _ = controller()
    c.heard(fsk.Heard(W.dbm_word(4), 0, 0, 48, 1), 1)
    c.tick(5)
    assert c.channel == 0


def test_disconnected_host_command_is_cancelled_before_running():
    from hfmodem.bateleur.host import BateleurHost
    ready = threading.Event()
    pending, replies = [], []
    def post(fn):
        pending.append(fn)
        ready.set()
    modem = SimpleNamespace(post=post, listen=True)
    host = BateleurHost(modem)
    host._running = True
    chunks = iter([b"LISTEN OFF\r\n", b""])
    conn = SimpleNamespace(closed=False, recv=lambda n: next(chunks), sendall=replies.append,
                           close=lambda: None, settimeout=lambda n: None)
    thread = threading.Thread(target=host._serve_cmd, args=(conn,))
    thread.start()
    assert ready.wait(1)
    conn.closed = True
    pending[0]()
    thread.join(2)
    assert not thread.is_alive()
    assert modem.listen
    assert b"command cancelled" in replies[0]


def test_stop_during_accept_does_not_attach_client(monkeypatch):
    from hfmodem.bateleur.host import BateleurHost
    host = BateleurHost(SimpleNamespace())
    host._running = True
    closed = []
    conn = SimpleNamespace(close=lambda: closed.append(True), settimeout=lambda n: None,
                           setsockopt=lambda *args: None)
    accepted = False
    def accept():
        nonlocal accepted
        if accepted:
            raise OSError("listener closed")
        accepted = True
        host._running = False
        return conn, None
    assert list(host._take(SimpleNamespace(accept=accept))) == []
    assert closed


@pytest.mark.parametrize("kind", ["bateleur", "ale"])
def test_long_burst_preserves_commands_and_station_activity(kind):
    from hfmodem.bateleur.ale.controller import Config
    from hfmodem.bateleur.ale.modem import AleModem
    from hfmodem.bateleur.ale.host import AleHost
    from hfmodem.bateleur.host import BateleurHost
    from hfmodem.bateleur.modem import BateleurModem
    from hfmodem.station.link import AleLink

    playing, release, replied = (threading.Event() for _ in range(3))
    errors, replies, calls = [], [], []

    from hfmodem.core.regulatory import centred
    from hfmodem.station.arbiter import ERRAND, TxArbiter, TxRequest

    def wait_until(at, **kwargs):
        if at:
            playing.set()
            if not release.wait(10):
                errors.append("burst was not released")

    audio = SimpleNamespace(sample_now=lambda: 0, arm_burst=lambda samples, at: at,
                            wait_until=wait_until, finish_burst=lambda: 576000,
                            alive=lambda: None, cancel_burst=lambda: None, tx_latency_n=0)
    rig = SimpleNamespace(max_key_s=60, key=lambda *a, **kw: 1,
                          unkey=lambda token: None, retired=False, line_state="low")
    arbiter = TxArbiter(audio, rig)

    def burst():
        arbiter.submit(TxRequest(ERRAND, 0, kind, np.zeros(576000), 48000,
                                 emission=centred(7100000, 3000)))

    if kind == "bateleur":
        modem = BateleurModem(lambda *a: "", card_rate=48000, max_key_s=60, log=errors.append)
        host, command = BateleurHost(modem), "LISTEN OFF"
    else:
        modem = AleModem(Config(["JOE"], ["A"], scanning=False),
                         send=lambda *a: 0, log=errors.append)
        host, command = AleHost(modem), "SCAN OFF"
    modem._audio = lambda *args: None
    modem.start()
    modem.post(burst)
    worker = threading.Thread(target=lambda: (replies.append(host.command(command)), replied.set()))
    try:
        assert playing.wait(2)
        # Twelve seconds of 50 ms polls while the transmitter holds the thread.
        for i in range(240):
            audio = np.zeros(2400, dtype=np.float32)
            if kind == "bateleur":
                modem.feed(audio)
            else:
                modem.feed(i * 2400, audio)
        worker.start()
        for i in range(32):
            modem.post(lambda i=i: calls.append(i))
        if kind == "ale":
            lane = object.__new__(AleLink)
            lane.lane = SimpleNamespace(poll=lambda: [])
            lane.station = SimpleNamespace(arbiter=SimpleNamespace(last_sent={"bateleur": 576000}))
            lane.modem, lane._others_sent, lane._expect_audio = modem, 0, None
            lane.poll()
            assert lane._others_sent == 576000
        assert not replied.wait(5.2), replies
        release.set()
        worker.join(2)
        done = threading.Event()
        modem.post(done.set)
        assert done.wait(2)
        assert replies == [f"OK {command}"]
        assert calls == list(range(32))
        assert not errors
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        modem.stop()


def test_oversized_audio_keeps_the_tail_with_its_card_index():
    from hfmodem.bateleur.selcall.modem import SelcallModem
    modem = SelcallModem(["1234"], send=lambda *a: None)
    seen, done = [], threading.Event()
    modem.reset_receive = lambda: seen.append("reset")

    def receive(at, samples):
        seen.append((at, samples))
        done.set()

    modem._audio = receive
    samples = np.arange(modem._audio_limit + 100, dtype=np.float32)
    modem.feed(200, samples)
    modem.start()
    try:
        assert done.wait(2)
        assert seen[0] == "reset"
        assert seen[1][0] == 300
        np.testing.assert_array_equal(seen[1][1], samples[100:])
        assert modem.audio_dropped == 1
    finally:
        modem.stop()
