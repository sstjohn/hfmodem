"""Four causal CS4 scenarios, with current outgoing waveform validation.

Recorded controls remain recorded; matching ACKs are a labeled synthetic model.
"""
from pathlib import Path
from functools import partial
import json
import os
import sys
import pytest

ROOT = Path(__file__).resolve().parents[5]
BENCH = ROOT / "working/p4-plan-0930/mail-waveform-bench-1004"
if not all((BENCH / name).is_file() for name in ("waveform_gate.py", "readers.py")):
    pytest.skip("private waveform bench is absent from this distribution", allow_module_level=True)
sys.path.insert(0, str(BENCH))
from waveform_gate import WaveformGate, Field, Identity
from readers import p3_packet
from hfmodem.shrike import placement
from hfmodem.tests import evidence
from hfmodem.tests.shrike import test_p3_cs4_credit_1003 as credit
from hfmodem.tests.shrike import test_p3_ve3kpg_proposal_0924 as historical


@pytest.fixture(scope="module", params=["onair-1003-0033", "onair-1002-1137"])
def recorded_arm(request):
    # Direct immutable reads; no input directory link or writes.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evidence, "CAPTURES", Path("/Users/saul/src/radio/hfmodem/captures"))
        yield credit.arm.__wrapped__(request)


def install(monkeypatch):
    gate = WaveformGate()
    def identity(host):
        return Identity(host.arq._link_generation, host.arq.tx_seq, str(host.arq.role))
    def observe(host):
        packet = host.arq._inflight
        if packet is None:
            return None
        burst = (placement.changeover_packet(packet.payload, packet.status) if packet.breakin
                 else placement.link_packet(packet.sl, packet.payload, packet.status))
        audio = burst
        return gate.observe(audio=audio, expected=Field(packet.payload, packet.status),
                            identity=identity(host), current=lambda: identity(host),
                            decode=partial(p3_packet, level=packet.sl, changeover=packet.breakin),
                            keyed=True, protocol="P3", direction="outbound")
    original_seed = credit.seed
    def seed(arm):
        result = original_seed(arm)
        observe(result[0])
        return result
    monkeypatch.setattr(credit, "seed", seed)
    original_control = credit.control
    def control(host, *args, **kwargs):
        token = observe(host)
        if token is None:
            return original_control(host, *args, **kwargs)
        return gate.respond(token, current=lambda: identity(host),
                            response=lambda field: original_control(host, *args, **kwargs))
    monkeypatch.setattr(credit, "control", control)
    original_answer = historical.Peer.answer
    def answer(peer, host, *args, **kwargs):
        token = observe(host)
        if token is None:
            return original_answer(peer, host, *args, **kwargs)
        return gate.respond(token, current=lambda: identity(host),
                            response=lambda field: original_answer(peer, host, *args, **kwargs))
    monkeypatch.setattr(historical.Peer, "answer", answer)
    return gate


def save(gate, name, tmp_path):
    path = Path(os.environ.get("PACTOR_MAIL_EVIDENCE_DIR", str(tmp_path)))
    path.mkdir(exist_ok=True, parents=True)
    (path / (name + ".json")).write_text(json.dumps({"fields": gate.rows,
          "seam": "seeded-model-outgoing-real-waveform", "hardware_keys": 0}, indent=2) + "\n")


def test_recorded_first_field_cs4_no_credit(recorded_arm, monkeypatch, tmp_path):
    gate = install(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(evidence, "CAPTURES", Path("/Users/saul/src/radio/hfmodem/captures"))
        try:
            credit.test_operator_1003_recorded_two_cs4_answers_leave_first_normal_field_unconfirmed(recorded_arm)
        finally:
            save(gate, "cs4-first-field-" + recorded_arm["arm"], tmp_path)
    assert gate.rows


def test_historical_three_cs4_no_credit_then_validating_ack(monkeypatch, tmp_path):
    gate = install(monkeypatch)
    try:
        credit.test_operator_1003_ve3kpg_peer_model_three_cs4_credits_only_validating_cs2_complete_sid_once()
    finally:
        save(gate, "cs4-historical-three-response", tmp_path)
    assert gate.rows


def test_matching_ack_complete_recorded_b2f(recorded_arm, monkeypatch, tmp_path):
    gate = install(monkeypatch)
    try:
        credit.test_operator_1003_synthetic_matching_ack_closure_of_recorded_bytes(recorded_arm)
    finally:
        save(gate, "cs4-matching-positive-" + recorded_arm["arm"], tmp_path)
    assert gate.rows


def test_missing_late_ack_retains_unconfirmed_bytes(recorded_arm, monkeypatch, tmp_path):
    gate = install(monkeypatch)
    try:
        credit.test_operator_1003_bounded_wait_model_retains_recorded_field_without_retransmission(recorded_arm, False)
    finally:
        save(gate, "cs4-missing-late-" + recorded_arm["arm"], tmp_path)
    assert gate.rows
