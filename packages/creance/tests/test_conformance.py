# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Conformance framework and probe catalog against scripted FakeModems:
conformant scripts must earn the golden verdicts, violating scripts must be
caught with the offending lines as evidence."""

import threading
import time

import pytest

from creance import conformance as conf
from hfhost.client import ModemClient
from creance.config import ModemConfig, Quirks
from creance.conformance import ABSENT, EXTRA, FAIL, PASS, WARN, Finding
from hfhost.transcript import Transcript
from hfhost.wire import classify

from hfhost.testing.fakemodem import FakeModem

BANDS = ("500", "2300", "2750")


@pytest.fixture
def fake():
    fm = FakeModem().start()
    yield fm
    fm.stop()


@pytest.fixture
def tr(tmp_path):
    t = Transcript(str(tmp_path / "conf.jsonl"), "sid-conf")
    yield t
    t.close()


def make_client(fake, tr, **kw):
    cfg = ModemConfig(name="fake", cmd_port=fake.cmd_port,
                      data_port=fake.data_port,
                      quirks=kw.pop("quirks", Quirks()))
    kw.setdefault("attach_timeout_s", 2.0)
    return ModemClient(cfg, "127.0.0.1", tr, **kw)


def make_ctx(client, **kw):
    kw.setdefault("settle_s", 0.15)
    kw.setdefault("reply_timeout_s", 1.0)
    kw.setdefault("connect_timeout_s", 1.0)
    kw.setdefault("drain_timeout_s", 1.0)
    kw.setdefault("data_timeout_s", 1.0)
    kw.setdefault("abort_timeout_s", 1.0)
    kw.setdefault("payload_n", 64)
    return conf.ProbeContext(client, **kw)


def probe(pid):
    return conf.PROBES[pid].fn


class ConformantModem:
    """Scripts a FakeModem into a spec-conformant modem: session verbs with
    the documented choreography, BUFFER rise/drain with loopback echo, and
    optionally an IAMALIVE heartbeat."""

    def __init__(self, fake, *, iamalive_s=0.0):
        self.fake = fake
        self.connected = False
        self._consumed = 0
        self._running = True
        fake.on_command = self
        threading.Thread(target=self._watch, daemon=True).start()
        if iamalive_s:
            threading.Thread(target=self._beat, args=(iamalive_s,),
                             daemon=True).start()

    def stop(self):
        self._running = False

    def __call__(self, cmd):
        parts = cmd.split()
        verb = parts[0].upper()
        if verb == "MYCALL":
            return "OK" if 1 <= len(parts) - 1 <= 5 else "WRONG"
        if verb == "BW":
            return "OK" if parts[1:] and parts[1] in BANDS else "WRONG"
        if verb.startswith("BW") and verb[2:].isdigit():
            return "OK" if verb[2:] in BANDS else "WRONG"
        if verb == "COMPRESSION":
            return "OK" if parts[1:] and parts[1] in ("OFF", "TEXT", "FILES") \
                else "WRONG"
        if verb == "CONNECT":
            if len(parts) < 3:
                return "WRONG"
            self.connected = True
            return ["OK", "PTT ON", "PTT OFF",
                    f"CONNECTED {parts[1]} {parts[2]} 500"]
        if verb == "DISCONNECT":
            self.connected = False
            return ["OK", "BUFFER 0", "DISCONNECTED"]
        if verb == "ABORT":
            self.connected = False
            return ["OK", "DISCONNECTED"]
        return None      # fake defaults: known verbs OK, unknown WRONG, VERSION

    def _watch(self):
        while self._running:
            data = bytes(self.fake.received_data)
            pend = data[self._consumed:]
            if pend and self.connected:
                self._consumed = len(data)
                try:
                    self.fake.notify(f"BUFFER {len(pend)}")
                    self.fake.notify("PTT ON")
                    self.fake.send_data(pend)
                    self.fake.notify("BUFFER 0")
                    self.fake.notify("PTT OFF")
                except RuntimeError:
                    pass
            time.sleep(0.02)

    def _beat(self, interval):
        while self._running:
            time.sleep(interval)
            try:
                self.fake.notify("IAMALIVE")
            except RuntimeError:
                pass


# -- full single-modem suite against a conformant modem ----------------------

def test_run_single_conformant_earns_golden_verdicts(fake, tr):
    script = ConformantModem(fake, iamalive_s=0.25)
    c = make_client(fake, tr, quirks=Quirks(iamalive_s=0.25))
    c.attach()
    try:
        findings = conf.run_single(make_ctx(c, iamalive_window_s=0.8))
    finally:
        script.stop()
        c.close()
    verdicts = {f.probe: f.verdict for f in findings}
    assert verdicts == {
        "reply_discipline": PASS, "unknown_verb_wrong": PASS,
        "mycall_arity": PASS, "bw_forms": PASS, "compression_args": PASS,
        "listen_semantics": PASS, "version_shape": PASS,
        "preconnect_queue": PASS, "buffer_cadence": PASS,
        "disconnect_flush_order": PASS, "abort_prompt": PASS,
        "bitrate_absent": ABSENT, "sn_absent": ABSENT,
        "iamalive_cadence": PASS, "vocabulary_sweep": PASS,
    }, conf.render(findings)


# -- individual active probes ------------------------------------------------

def attached(fake, tr, **kw):
    c = make_client(fake, tr, **kw)
    c.attach()
    return c


def test_reply_discipline_catches_double_ok(fake, tr):
    fake.on_command = lambda cmd: ["OK", "OK"] if cmd.startswith("MYCALL") else None
    c = attached(fake, tr)
    try:
        f = probe("reply_discipline")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert any("2 replies" in e and "OK + OK" in e for e in f.evidence)


def test_reply_discipline_passes_and_records_evidence(fake, tr):
    c = attached(fake, tr)
    try:
        f = probe("reply_discipline")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == PASS
    assert any("'NOSUCHVERB' -> WRONG" in e for e in f.evidence)


def test_unknown_verb_accepted_is_fail(fake, tr):
    fake.on_command = lambda cmd: "OK"
    c = attached(fake, tr)
    try:
        f = probe("unknown_verb_wrong")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert "'XYZZY QUUX' -> OK" in f.evidence


def test_mycall_arity_catches_lax_modem(fake, tr):
    # the default fake OKs MYCALL regardless of arity
    c = attached(fake, tr)
    try:
        f = probe("mycall_arity")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert any("'MYCALL' -> OK (expected WRONG)" in e for e in f.evidence)


def test_bw_forms_requires_both_spellings(fake, tr):
    # the default fake rejects the "BW 500" spelling
    c = attached(fake, tr)
    try:
        f = probe("bw_forms")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert any("'BW 500' -> WRONG (expected OK)" in e for e in f.evidence)


def test_version_shape_quirk_gate_and_fail(fake, tr):
    c = make_client(fake, tr, quirks=Quirks(version_reply=False))
    assert probe("version_shape")(make_ctx(c)).verdict == ABSENT
    c.close()

    fake.on_command = lambda cmd: "WRONG" if cmd == "VERSION" else None
    c = attached(fake, tr)
    try:
        assert probe("version_shape")(make_ctx(c)).verdict == FAIL
    finally:
        c.close()


def test_version_shape_pass(fake, tr):
    c = attached(fake, tr)
    try:
        f = probe("version_shape")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == PASS
    assert f.evidence == ("VERSION 1.0-fake",)


def test_preconnect_queue_quirk_gated_absent(fake, tr):
    c = make_client(fake, tr, quirks=Quirks(preconnect_queue=False))
    f = probe("preconnect_queue")(make_ctx(c))
    c.close()
    assert f.verdict == ABSENT
    assert "not probed" in f.detail


def test_preconnect_queue_discarding_modem_fails(fake, tr):
    # a discarding modem connects fine but never delivers the queued bytes
    fake.on_command = lambda cmd: ["OK", "CONNECTED N0CRE K7CRE 500"] \
        if cmd.startswith("CONNECT") else None
    c = attached(fake, tr)
    try:
        f = probe("preconnect_queue")(make_ctx(c, data_timeout_s=0.4))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert any("0 B arrived" in e for e in f.evidence)


def test_buffer_never_drains_is_fail(fake, tr):
    def script(cmd):
        if cmd.startswith("CONNECT"):
            return ["OK", "CONNECTED N0CRE K7CRE 500"]
        return None
    fake.on_command = script
    c = attached(fake, tr)

    def stuck_buffer():
        fake.wait_data(1, timeout=2.0)
        fake.notify("BUFFER 64")
    threading.Thread(target=stuck_buffer, daemon=True).start()
    try:
        f = probe("buffer_cadence")(make_ctx(c, drain_timeout_s=0.5))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert "never drained" in f.detail
    assert "BUFFER 64" in f.evidence[0]


def test_buffer_silence_is_fail(fake, tr):
    fake.on_command = lambda cmd: ["OK", "CONNECTED N0CRE K7CRE 500"] \
        if cmd.startswith("CONNECT") else None
    c = attached(fake, tr)
    try:
        f = probe("buffer_cadence")(make_ctx(c, drain_timeout_s=0.3))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert "no BUFFER notifications" in f.detail


def test_disconnect_without_flush_is_fail(fake, tr):
    def script(cmd):
        if cmd.startswith("CONNECT"):
            return ["OK", "CONNECTED N0CRE K7CRE 500"]
        if cmd == "DISCONNECT":
            return ["OK", "BUFFER 7", "DISCONNECTED"]
        return None
    fake.on_command = script
    c = attached(fake, tr)
    try:
        f = probe("disconnect_flush_order")(make_ctx(c))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert "was 7, not 0" in f.detail


def test_abort_that_never_disconnects_is_fail(fake, tr):
    fake.on_command = lambda cmd: ["OK", "CONNECTED N0CRE K7CRE 500"] \
        if cmd.startswith("CONNECT") else None
    c = attached(fake, tr)
    try:
        f = probe("abort_prompt")(make_ctx(c, abort_timeout_s=0.3))
    finally:
        c.close()
    assert f.verdict == FAIL
    assert "still connected" in f.detail


# -- passive probes (evaluated over synthetic line streams) ------------------

def lines_at(*specs):
    return [(t, classify(text)) for t, text in specs]


@pytest.fixture
def idle_ctx(fake, tr):
    c = make_client(fake, tr)
    yield make_ctx(c)
    c.close()


def test_vocabulary_sweep_flags_unknown_lines(idle_ctx):
    lines = lines_at((0.1, "BUFFER 5"), (0.2, "XYZZY plugh"), (0.3, "OK"))
    f = probe("vocabulary_sweep")(idle_ctx, lines, 1.0)
    assert f.verdict == EXTRA
    assert f.evidence == ("XYZZY plugh",)
    assert probe("vocabulary_sweep")(idle_ctx, lines[:1], 1.0).verdict == PASS


def test_bitrate_probe_absent_pass_and_malformed(idle_ctx):
    fn = probe("bitrate_absent")
    assert fn(idle_ctx, [], 1.0).verdict == ABSENT
    good = lines_at((0.5, "BITRATE (4) 1200 bps TX"))
    assert fn(idle_ctx, good, 1.0).verdict == PASS
    bad = lines_at((0.5, "BITRATE utter garbage"))
    f = fn(idle_ctx, bad, 1.0)
    assert f.verdict == FAIL
    assert f.evidence == ("BITRATE utter garbage",)


def test_sn_probe(idle_ctx):
    fn = probe("sn_absent")
    assert fn(idle_ctx, [], 1.0).verdict == ABSENT
    assert fn(idle_ctx, lines_at((0.5, "SN 12.5")), 1.0).verdict == PASS


def iamalive_ctx(fake, tr, interval):
    c = make_client(fake, tr, quirks=Quirks(iamalive_s=interval))
    return c, make_ctx(c)


def test_iamalive_cadence_verdicts(fake, tr):
    c, ctx = iamalive_ctx(fake, tr, 1.0)
    try:
        fn = probe("iamalive_cadence")
        beats = lines_at((0.9, "IAMALIVE"), (1.9, "IAMALIVE"))
        assert fn(ctx, beats, 2.5).verdict == PASS
        short = fn(ctx, [], 0.5)
        assert short.verdict == ABSENT and "too short" in short.detail
        missing = fn(ctx, [], 2.0)
        assert missing.verdict == ABSENT and "no IAMALIVE within" in missing.detail
        ragged = fn(ctx, lines_at((0.5, "IAMALIVE"), (3.0, "IAMALIVE")), 3.5)
        assert ragged.verdict == WARN
    finally:
        c.close()


# -- paired probes -----------------------------------------------------------

def paired_setup(tr, wire_a, deliver=False):
    """Two fakes wired as a linked pair: A's command script may poke B."""
    fa, fb = FakeModem().start(), FakeModem().start()
    fa.on_command = wire_a(fa, fb)
    ca = make_client(fa, tr)
    cb = make_client(fb, tr)
    ca.attach()
    cb.attach()
    if deliver:
        def echo():
            if fa.wait_data(64, timeout=3.0):
                fa.notify("PTT ON")
                fa.notify("BUFFER 64")
                fa.notify("BUFFER 0")
                fa.notify("PTT OFF")
                fb.send_data(bytes(fa.received_data[:64]))
                fb.notify("PTT ON")
                fb.notify("PTT OFF")
        threading.Thread(target=echo, daemon=True).start()
    return fa, fb, ca, cb


def run_pair(ca, cb, **kw):
    ctx_a = make_ctx(ca, mycall="N0CRE", **kw)
    ctx_b = make_ctx(cb, mycall="K7CRE", **kw)
    findings = conf.run_paired(ctx_a, ctx_b)
    return {f.probe: f for f in findings}


def test_paired_conformant_choreography(tr):
    def wire_a(fa, fb):
        def on_a(cmd):
            parts = cmd.split()
            if parts[0] == "CONNECT":
                fb.notify("PENDING")
                fb.notify("PTT ON")
                fb.notify("PTT OFF")
                fb.notify(f"CONNECTED {parts[2]} {parts[1]} 500")
                return ["OK", "PTT ON", "PTT OFF",
                        f"CONNECTED {parts[1]} {parts[2]} 500"]
            if parts[0] == "DISCONNECT":
                fb.notify("DISCONNECTED")
                return ["OK", "BUFFER 0", "DISCONNECTED"]
            return None
        return on_a

    fa, fb, ca, cb = paired_setup(tr, wire_a, deliver=True)
    try:
        by = run_pair(ca, cb)
    finally:
        ca.close(); cb.close(); fa.stop(); fb.stop()
    assert {p: f.verdict for p, f in by.items()} == {
        "connected_fields": PASS, "ptt_alternation": PASS,
        "both_ends_disconnected": PASS, "pending_before_connected": PASS,
        "listen_off_side_effect_free": PASS,
    }, conf.render(by.values())


def test_paired_violations_are_caught(tr):
    def wire_a(fa, fb):
        def on_a(cmd):
            parts = cmd.split()
            if parts[0] == "CONNECT":
                # wrong callsign at B, double-keyed PTT at A, and B will
                # never hear the disconnect
                fb.notify("CONNECTED X9XXX N0CRE 500")
                return ["OK", "PTT ON", "PTT ON", "PTT OFF",
                        f"CONNECTED {parts[1]} {parts[2]} 500"]
            if parts[0] == "DISCONNECT":
                return ["OK", "BUFFER 0", "DISCONNECTED"]
            return None
        return on_a

    fa, fb, ca, cb = paired_setup(tr, wire_a)
    try:
        by = run_pair(ca, cb, data_timeout_s=0.3, drain_timeout_s=0.3,
                      abort_timeout_s=0.3)
    finally:
        ca.close(); cb.close(); fa.stop(); fb.stop()
    assert by["connected_fields"].verdict == FAIL
    assert any("B saw src/dst X9XXX" in e for e in by["connected_fields"].evidence)
    assert by["ptt_alternation"].verdict == FAIL
    assert any("A: PTT did not strictly alternate" in e
               for e in by["ptt_alternation"].evidence)
    assert by["both_ends_disconnected"].verdict == FAIL
    assert by["pending_before_connected"].verdict == ABSENT
    assert by["listen_off_side_effect_free"].verdict == PASS


def test_paired_no_session(tr):
    fa, fb, ca, cb = paired_setup(tr, lambda fa, fb: (lambda cmd: None))
    try:
        by = run_pair(ca, cb, connect_timeout_s=0.3)
    finally:
        ca.close(); cb.close(); fa.stop(); fb.stop()
    assert by["connected_fields"].verdict == FAIL
    assert "no session" in by["connected_fields"].detail
    assert by["ptt_alternation"].verdict == ABSENT
    assert by["both_ends_disconnected"].verdict == FAIL


def test_ptt_leading_off_is_tolerated(idle_ctx):
    # a burst straddling the observation-window start leaves a stray unkey
    la = lines_at((0.1, "PTT ON"), (0.2, "PTT OFF"))
    lb = lines_at((0.05, "PTT OFF"), (0.3, "PTT ON"), (0.4, "PTT OFF"))
    ev = {"connected": True, "conn_a": None, "conn_b": None,
          "delivered": 0, "a_disc": True, "b_disc": True}
    f = probe("ptt_alternation")(idle_ctx, idle_ctx, ev, la, lb)
    assert f.verdict == PASS
    assert any("leading OFF dropped" in e for e in f.evidence)


# -- probe crash containment -------------------------------------------------

def test_probe_crash_is_a_finding(fake, tr, monkeypatch):
    c = attached(fake, tr)

    def boom(ctx):
        raise RuntimeError("kaboom")
    monkeypatch.setitem(conf.PROBES, "unknown_verb_wrong",
                        conf.Probe("unknown_verb_wrong", "single", boom))
    try:
        findings = conf.run_single(make_ctx(c))
    finally:
        c.close()
    f = next(x for x in findings if x.probe == "unknown_verb_wrong")
    assert f.verdict == FAIL
    assert "kaboom" in f.detail


# -- golden table ------------------------------------------------------------

def test_golden_table_ids_and_verdicts_are_valid():
    for target, table in conf.GOLDEN.items():
        for pid, verdict in table.items():
            assert pid in conf.PROBES, (target, pid)
            assert verdict in conf.VERDICTS


def test_diff_golden():
    table = conf.GOLDEN["kestrel-loopback"]
    findings = [Finding(pid, v) for pid, v in table.items()]
    assert conf.diff_golden("kestrel-loopback", findings) == []

    flipped = [Finding(f.probe, FAIL if f.probe == "bw_forms" else f.verdict)
               for f in findings]
    assert conf.diff_golden("kestrel-loopback", flipped) == \
        ["bw_forms: expected PASS, got FAIL"]

    missing = [f for f in findings if f.probe != "abort_prompt"]
    assert conf.diff_golden("kestrel-loopback", missing) == \
        ["abort_prompt: expected PASS, not run"]

    extra = findings + [Finding("mystery_probe", PASS)]
    assert conf.diff_golden("kestrel-loopback", extra) == \
        ["mystery_probe: unexpected probe (not in golden table)"]


def test_render_shape():
    findings = [
        Finding("bw_forms", PASS, ("'BW500' -> OK",), "grammar as specified"),
        Finding("vocabulary_sweep", EXTRA, ("XYZZY plugh",), "1 unknown line"),
    ]
    out = conf.render(findings)
    assert "PASS   bw_forms" in out
    assert "| 'BW500' -> OK" in out
    assert out.endswith("2 probes: 1 PASS, 1 EXTRA")
    assert conf.render([]) == "no findings"
