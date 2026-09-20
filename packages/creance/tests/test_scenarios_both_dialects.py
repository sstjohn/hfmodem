# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The same CWP session, unchanged, over both host dialects.

The Link seam exists on one claim: a scenario is written once and runs over
either dialect. That claim was asserted when the seam was built and never
tested — every other end-to-end session test goes over VARA fakes. This file is
the proof, and it is parameterized rather than duplicated so the two cannot
drift: if a scenario ever needs to know which dialect it is on, a test here
fails rather than a comment going quietly stale.
"""

import threading

import pytest

from creance import metrics
from creance.scenarios import CwpSession, initiate, respond
from hfhost.transcript import read as read_transcript
from hfhost.testing.bridge import Bridge
from hfhost.testing.hostapi_bridge import HostApiBridge


class VaraPair:
    dialect = "vara"

    def __init__(self, tmp_path, **kw):
        self.bridge = Bridge(tmp_path, **kw).start()

    def links(self):
        return self.bridge.link("a", "init"), self.bridge.link("b", "resp")

    def connect(self):
        self.bridge.connect(a_call="N0AAA", b_call="N0BBB")

    def stop(self):
        self.bridge.stop()


class HostApiPair:
    dialect = "hostapi"

    def __init__(self, tmp_path, **kw):
        self.bridge = HostApiBridge(tmp_path, **kw).start()

    def links(self):
        return self.bridge.link("a", "init"), self.bridge.link("b", "resp")

    def connect(self):
        self.bridge.connect(a_call="N0AAA", b_call="N0BBB")

    def stop(self):
        self.bridge.stop()


@pytest.fixture(params=[VaraPair, HostApiPair], ids=["vara", "hostapi"])
def pair(request, tmp_path):
    p = request.param(tmp_path)
    yield p
    p.stop()


def _session(pair, timeout=20.0):
    ci, cr = pair.links()
    for link, call in ((ci, "N0AAA"), (cr, "N0BBB")):
        link.configure(call)
        link.attach()
    pair.connect()
    si = CwpSession(ci, ci.transcript, sid="SID", mycall="N0AAA",
                    recv_timeout_s=timeout)
    sr = CwpSession(cr, cr.transcript, sid="SID", mycall="N0BBB",
                    recv_timeout_s=timeout)
    return ci, cr, si, sr


def _run_pair(si, sr, scenario, params):
    """Drive both halves the way the responder daemon does: await the HELLO,
    then hand it to respond()."""
    out = {}

    def serve():
        hello = sr.await_hello(timeout=20.0)
        if hello is not None:
            out["r"] = respond(sr, hello)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    result = initiate(si, scenario, params)
    t.join(timeout=25.0)
    return result, out.get("r")


def test_unidir_completes_over_either_dialect(pair):
    ci, cr, si, sr = _session(pair)
    try:
        result, served = _run_pair(si, sr, "unidir",
                                   {"size": "4k", "payload": "prbs9"})
        assert result.outcome == "ok"
        assert served is not None and served.outcome == "ok"
    finally:
        ci.close()
        cr.close()


def test_payload_arrives_byte_exact_over_either_dialect(pair):
    ci, cr, si, sr = _session(pair)
    try:
        result, served = _run_pair(si, sr, "unidir",
                                   {"size": "8k", "payload": "prbs9"})
        assert result.outcome == "ok"
        # the far end's END carries the sha of what it actually received
        assert served.stats.get("bytes") == 8192
        # the sha the receiver computed over what it actually got must equal
        # the one the sender put in its END
        assert served.stats.get("sha256") == result.stats.get("sha256")
        assert served.stats.get("sha256")
    finally:
        ci.close()
        cr.close()


def test_echo_round_trip_over_either_dialect(pair):
    ci, cr, si, sr = _session(pair)
    try:
        result, served = _run_pair(si, sr, "echo",
                                   {"size": "2k", "payload": "counter"})
        assert result.outcome == "ok"
        assert result.stats.get("integrity") is not False
    finally:
        ci.close()
        cr.close()


def test_reverse_direction_over_either_dialect(pair):
    ci, cr, si, sr = _session(pair)
    try:
        result, served = _run_pair(si, sr, "reverse",
                                   {"size": "4k", "payload": "prbs9"})
        assert result.outcome == "ok"
    finally:
        ci.close()
        cr.close()


def test_a_plain_peer_is_detected_over_either_dialect(pair):
    """No HELLO_ACK back: the initiator must report plain_peer rather than
    hanging, whichever dialect carried the silence."""
    ci, cr, si, sr = _session(pair, timeout=3.0)
    try:
        result = initiate(si, "unidir", {"size": "1k", "payload": "prbs9"},
                          on_plain_peer="disconnect")
        assert result.outcome in ("plain_peer", "report_missing")
    finally:
        ci.close()
        cr.close()


def test_the_scenario_code_never_learns_the_dialect(pair):
    """The seam's whole point. CwpSession is handed a Link and must not reach
    for anything dialect-specific on it."""
    ci, cr, si, sr = _session(pair)
    try:
        assert not hasattr(si.client, "command")        # VARA-only verb path
        assert not hasattr(si.client, "send_data")      # VARA-only method name
        assert hasattr(si.client, "send") and hasattr(si.client, "recv")
    finally:
        ci.close()
        cr.close()


def test_metrics_are_computable_over_either_dialect(pair):
    """metrics.compute is a pure function over the transcript, and it was
    written against VARA notification names. A structured session records
    different ones, so without normalization a whole report comes out empty —
    and silently, which is worse than wrong."""
    ci, cr, si, sr = _session(pair)
    try:
        ci.connect("N0BBB")             # a real connect, so connect_s is timeable
        pair.connect()
        result, _ = _run_pair(si, sr, "unidir", {"size": "8k", "payload": "prbs9"})
        assert result.outcome == "ok"
    finally:
        path = ci.transcript.path
        ci.transcript.close()
        ci.close()
        cr.close()

    recs, _ = read_transcript(str(path))
    m = metrics.compute(recs, {"sid": "SID", "modem": "a",
                               "scenario": "unidir", "outcome": result.outcome})
    assert m.bytes_tx >= 8192, "payload bytes must be counted on both dialects"
    assert m.buffer_curve, "queue depth must be recoverable on both dialects"
    assert m.connect_s is not None
    assert m.handshake_s is not None
    # Every warning except one, and the exception is structural rather than
    # tolerated. `drain_bps_local` needs a buffer that rises and then reaches
    # zero over more than a millisecond; this pair is a loopback with no radio
    # between the ends, so it drains in less time than that whenever the machine
    # is not busy. Asserting no warnings at all made the test a measurement of
    # how loaded the laptop was -- it passed under a running oracle and failed on
    # an idle one. What this test is for is dialect normalisation, and that is
    # what the remaining warnings would report.
    drain = [w for w in m.warnings if w.startswith("drain_bps_local unavailable")]
    assert m.warnings == drain, [w for w in m.warnings if w not in drain]


def test_a_transcript_with_no_recognized_notifications_warns():
    """The failure this guards against is silence: a dialect the metrics layer
    cannot read must say so, not return an empty report that looks like a modem
    which never keyed and never queued."""
    from hfhost.transcript import Kind, Record
    recs = [Record(seq=0, t=0.0, wall=0.0, sid="S", modem="a", chan="cmd",
                   dir="rx", kind=Kind.CMD_RX, fields={"text": "Blorp"})]
    m = metrics.compute(recs, {"sid": "S", "modem": "a", "scenario": "unidir",
                               "outcome": "ok"})
    assert any("no recognized notifications" in w for w in m.warnings)
