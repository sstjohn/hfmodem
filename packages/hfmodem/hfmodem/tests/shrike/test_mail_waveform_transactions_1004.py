# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Self-implementation waveform integration; no stock-peer or RF claim."""
import json
import os
from pathlib import Path
import sys

from hfmodem.tests.evidence import require_input

BENCH = Path(__file__).resolve().parents[5] / "working/p4-plan-0930/mail-waveform-bench-1004"
require_input(BENCH, allow_module_level=True)
sys.path.insert(0, str(BENCH))

def save_gate(gate, name, tmp_path):
    output = Path(os.environ.get("PACTOR_MAIL_EVIDENCE_DIR", str(tmp_path)))
    output.mkdir(exist_ok=True)
    with (output / (name + ".json")).open("x") as handle:
        json.dump({"fields": gate.rows, "metrics": getattr(gate, "bench_metrics", {}), "controls": getattr(gate, "control_rows", [])}, handle, sort_keys=True)

def test_p3_inbound_message_waveforms(monkeypatch, tmp_path):
    from p3_transaction import instrument
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as original
    gate = instrument(monkeypatch)
    try:
        original.test_scene1_a_message_reaches_disk_over_the_whole_pactor3_exchange(tmp_path)
    finally:
        save_gate(gate, "p3-inbound", tmp_path)

def test_p3_outbound_message_waveforms(monkeypatch, tmp_path):
    from p3_transaction import instrument
    from hfmodem.tests.shrike import test_b2f_loopback_send_ws8eoc as original
    gate = instrument(monkeypatch)
    try:
        original.test_scene7_a_message_of_ours_reaches_the_gateway_byte_exact()
    finally:
        save_gate(gate, "p3-outbound", tmp_path)

def test_p4_outbound_waveform_pilot(monkeypatch, capsys, tmp_path):
    from p4_transaction import instrument
    from hfmodem.tests.shrike import test_p4_mail_0927 as original
    gate = instrument(monkeypatch)
    endings = next(mark.args[1] for mark in original.test_mail_exchange.pytestmark
                   if mark.name == "parametrize" and mark.args[0] == "ending")
    try:
        original.test_mail_exchange(monkeypatch, capsys, endings[0], 0)
    finally:
        save_gate(gate, "p4-outbound-pilot", tmp_path)

def test_p3_clean_two_way_same_host_reconnect(monkeypatch, tmp_path):
    from p3_transaction import instrument
    from hfmodem.shrike import arq
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as receive
    from hfmodem.tests.shrike import test_b2f_loopback_send_ws8eoc as send
    gate = instrument(monkeypatch)
    constructor = receive.PtcHost
    hosts = []
    generations = []

    def reuse(peer, **kwargs):
        if hosts:
            host = hosts[0]
            host.arq._finish_disconnected()
            assert host.arq.state is arq.State.DISCONNECTED
            gate.retire()
            host.peer = peer
        else:
            host = constructor(peer, **kwargs)
        hosts.append(host)
        return host

    monkeypatch.setattr(receive, "PtcHost", reuse)
    try:
        receive.test_scene1_a_message_reaches_disk_over_the_whole_pactor3_exchange(tmp_path)
        generations.append(hosts[-1].arq._link_generation)
        send.test_scene7_a_message_of_ours_reaches_the_gateway_byte_exact()
        generations.append(hosts[-1].arq._link_generation)
        assert len(hosts) == 2 and hosts[0] is hosts[1]
        assert generations[1] == generations[0] + 1
        assert gate.rows
    finally:
        save_gate(gate, "p3-two-way-reconnect", tmp_path)

def test_p3_loss_repeat_supported_cms_wait(monkeypatch, tmp_path):
    from p3_transaction import instrument
    from hfmodem.shrike import arq, p3progress
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as original
    gate = instrument(monkeypatch)
    original_key = original.Ws8eoc.key
    original_send = original.Ws8eoc._send
    original_timer_init = p3progress.MailProgress.__init__
    clock = [0.0]
    delayed = []

    def timer_init(self, *args, **kwargs):
        kwargs["now"] = lambda: clock[0]
        return original_timer_init(self, *args, **kwargs)

    def key(self):
        clock[0] += 1.25
        return original_key(self)

    def send(self):
        previous = getattr(self, "_last", None)
        if (len(delayed) < 50 and previous is not None
                and len(previous[2]) > 3 and self.host.arq.role == arq.IRS):
            sl, status, field = previous
            delayed.append(clock[0])
            return self._arrives(sl, status, field)
        return original_send(self)

    monkeypatch.setattr(p3progress.MailProgress, "__init__", timer_init)
    monkeypatch.setattr(original.Ws8eoc, "key", key)
    monkeypatch.setattr(original.Ws8eoc, "_send", send)
    try:
        original.test_scene2_a_repeating_gateway_still_completes_the_exchange()
        assert len(delayed) == 50
        assert delayed[-1] - delayed[0] + 1.25 == 62.5
        original.test_a_gateway_returning_after_35_unread_copies_completes_the_message()
        assert gate.rows
    finally:
        save_gate(gate, "p3-loss-repeat-cms-wait", tmp_path)

def test_p4_inbound_full_b2f_pilot(monkeypatch, tmp_path):
    from p4_b2f import run
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as model
    message, blob = model.offered()
    remote = model.Ws8eoc(model.cms_script(blob, message.mid, len(message.render())))
    try:
        host, session, gate = run(monkeypatch, remote)
        assert session.done and not session.failure
        assert [item.render() for item in session.inbox] == [message.render()]
        assert not session.sent_mids
    finally:
        save_gate(run.last_gate, "p4-inbound-b2f-pilot", tmp_path)


def test_p4_outbound_full_b2f_pilot(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_b2f_loopback_send_ws8eoc as outgoing
    from p4_b2f import run
    msg = outgoing.binary_outbound()
    expected = msg.render()
    blob = outgoing.compress(expected)
    remote = outgoing.Ws8eoc(outgoing.send_script(blob, msg.subject.encode("ascii")))
    try:
        host, session, gate = run(monkeypatch, remote, outbox=[msg])
    finally:
        save_gate(run.last_gate, "p4-outbound-b2f-pilot", tmp_path)
    assert session.done and not session.failure
    assert session.sent_mids == [msg.mid]
    title, offset, delivered = outgoing.gateway_read(bytes(remote.rx))
    assert title == msg.subject.encode("ascii") and offset == 0
    assert delivered == blob
    assert outgoing.decompress(delivered, expected_size=len(expected)) == expected


def test_p4_clean_two_way_same_host_reconnect(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as incoming
    from hfmodem.tests.shrike import test_b2f_loopback_send_ws8eoc as outgoing
    from p4_b2f import run
    message, blob = incoming.offered()
    remote = incoming.Ws8eoc(incoming.cms_script(blob, message.mid, len(message.render())))
    with monkeypatch.context() as patch:
        try:
            host, session, first_gate = run(patch, remote)
        finally:
            save_gate(run.last_gate, "p4-clean-reconnect-inbound", tmp_path)
        assert session.done and not session.failure
        assert [m.render() for m in session.inbox] == [message.render()]
        generation = host.arq._link_generation
    msg = outgoing.binary_outbound()
    expected = msg.render()
    blob = outgoing.compress(expected)
    remote = outgoing.Ws8eoc(outgoing.send_script(blob, msg.subject.encode("ascii")))
    with monkeypatch.context() as patch:
        try:
            again, session, gate = run(patch, remote, outbox=[msg], reuse_host=host)
        finally:
            save_gate(run.last_gate, "p4-clean-reconnect-outbound", tmp_path)
        assert again is host
        assert host.arq._link_generation == generation + 1
        assert session.done and not session.failure
        assert session.sent_mids == [msg.mid]
        title, offset, delivered = outgoing.gateway_read(bytes(remote.rx))
        assert title == msg.subject.encode("ascii") and offset == 0
        assert delivered == blob
        assert outgoing.decompress(delivered, expected_size=len(expected)) == expected


def test_p4_radio_tx_complete_inbound_b2f(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as model
    from p4_b2f import run
    message, blob = model.offered()
    remote = model.Ws8eoc(model.cms_script(blob, message.mid, len(message.render())))
    try:
        host, session, gate = run(monkeypatch, remote, radio_outdir=tmp_path)
    finally:
        save_gate(run.last_gate, "p4-radio-inbound", tmp_path)
    assert session.done and not session.failure
    assert [m.render() for m in session.inbox] == [message.render()]


def test_p4_loss_repeat_final_admission_refusal(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_b2f_loopback_send_ws8eoc as outgoing
    from p4_b2f import run
    msg = outgoing.binary_outbound()
    expected = msg.render()
    blob = outgoing.compress(expected)
    for refuse in (False, True):
        remote = outgoing.Ws8eoc(outgoing.send_script(blob, msg.subject.encode("ascii")))
        directory = tmp_path / ("refused" if refuse else "positive")
        directory.mkdir()
        with monkeypatch.context() as patch:
            try:
                host, session, gate = run(patch, remote, outbox=[msg], radio_outdir=directory,
                                          drop_ack=True, refuse_after_repeat=refuse)
            finally:
                save_gate(run.last_gate, "p4-loss-refusal" if refuse else "p4-loss-positive", tmp_path)
            metrics = gate.bench_metrics
            assert metrics["radio_tx"] and metrics["dropped_ack"]
            assert metrics["repeated_after_loss"] and metrics["duplicate_fields"] >= 1
            if not refuse:
                assert session.done and not session.failure
                assert session.sent_mids == [msg.mid]
                title, offset, delivered = outgoing.gateway_read(bytes(remote.rx))
                assert title == msg.subject.encode("ascii") and offset == 0
                assert delivered == blob
                assert outgoing.decompress(delivered, expected_size=len(expected)) == expected
            else:
                assert not session.sent_mids
                assert len(metrics["refusal"]) == 1
                denied = metrics["refusal"][0]
                assert denied["refused"] and denied["post_wait_cost_samples"] > 0
                assert denied["emission_delta"] == denied["offline_ptt_edge_delta"] == 0
                assert denied["before"] == denied["after"]
                assert denied["after"]["data_bytes"] > 0
                assert denied["after"]["host_pending_bytes"] > 0


def test_p4_crc_without_keyed_grant_cannot_open_receive_epoch(monkeypatch, tmp_path):
    from hfmodem.tests.shrike import test_b2f_loopback_ws8eoc as model
    from p4_b2f import run
    message, blob = model.offered()
    remote = model.Ws8eoc(model.cms_script(blob, message.mid, len(message.render())))
    try:
        host, session, gate = run(monkeypatch, remote, request_grant=False, seconds=60)
    finally:
        save_gate(run.last_gate, "p4-no-grant-negative", tmp_path)
    assert not session.done and not session.inbox
    observations = gate.bench_metrics["observations"]
    stale = [o for o in observations if o["has_frame"] and o["state"] == "IRS_ANSWER"
             and not o["receive_active"]]
    assert len(stale) >= 2
    assert {o["receive_epoch"] for o in stale} == {1}
    assert {o["acked_counter"] for o in stale} == {1}
    assert {o["frame_counter"] for o in stale} == {0}
