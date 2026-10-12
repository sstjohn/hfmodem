# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""No-radio driver tests with a synthetic peer and the real receiver/turn machine."""

from types import SimpleNamespace
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, p4breakintx
from hfmodem.shrike.p4probe import EntryProbe
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.tests.shrike.test_p4turn import IO

FS = 48000


class Stream:
    def __init__(self, audio):
        self.audio = audio
        self.pos = 0

    def sample_now(self):
        return self.pos

    def clamp_late(self, at):
        return max(0, self.pos - at)

    def take_until(self, end):
        end = min(end, len(self.audio))
        out = self.audio[self.pos : end]
        self.pos = end
        return out


class Tx:
    def __init__(self, peer):
        self.peer = peer
        self.settle = 0.074
        self.slots_used = []
        self.refused = False
        self.sessrx = SimpleNamespace(p4=P4Receiver(FS, admitted=True))
        self.p4_probe = EntryProbe(2, 60)
        self.p4_probe.request(0, b"", 0)
        self.p4_probe.emissions = [dict(audio_start=0, audio_end=round(0.758 * FS))]
        self.p4_link_options = SimpleNamespace(
            p4_ladder="long", p4_data="none", p4_hold_cycles=6, p4_answer_offset=.920,
            p4_normal_ack="v27",
            p4_link_timeout=600, p4_text=None
        )
        self.p4_link_mail = None
        self.p4_link_host = SimpleNamespace(
            arq=arq.PactorArq(IO()), deliver=lambda b: None, supervisor=lambda *a: None
        )
        self.p4_link_host.arq.state = arq.State.CONNECTED

    def aim(self, raster, slot):
        self.raster = raster
        self.slot = slot

    def _tx(self, audio, what):
        self.tx_audio_start = self.raster.boundary(self.slot)
        self.tx_end = self.tx_audio_start + len(audio)
        self.slots_used.append(self.slot)


@pytest.mark.parametrize("offset", [-0.025, 0.0, 0.025])
def test_driver_answers_each_accepted_copy_at_its_own_instant(offset):
    audio = np.zeros(5 * FS)
    y = round((0.923 + offset) * FS)
    body = p4breakintx.changeover_packet(b"hello", 0)
    audio[y : y + len(body)] = body
    stream = Stream(audio)
    tx = Tx(stream)
    onair._run_p4_link(
        tx,
        stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    emitted = tx.p4_probe.emissions[1:]
    assert emitted and emitted[0]["word"] == "X27"
    assert abs(emitted[0]["audio_start"] - (y + round(0.92 * FS))) <= 1


def test_bad_copy_is_silent_and_valid_repeat_gets_its_own_ack():
    audio = np.zeros(6 * FS)
    y = round(0.923 * FS)
    good = p4breakintx.changeover_packet(b"hello", 0)
    bad = good.copy()
    for start in range(round(0.30 * FS), len(bad), round(0.05 * FS)):
        bad[start : start + round(0.025 * FS)] *= -1
    audio[y : y + len(bad)] = bad
    repeat = y + round(1.25 * FS)
    audio[repeat : repeat + len(good)] = good
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_link_timeout = 5
    received = []
    tx.p4_link_host.deliver = received.append
    onair._run_p4_link(
        tx,
        stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    emitted = tx.p4_probe.emissions[1:]
    assert emitted[0]["word"] == "X27"
    assert abs(emitted[0]["audio_start"] - (repeat + round(0.92 * FS))) <= 1
    assert all(e['word'] == 'X27' for e in emitted[1:])
    assert all(not e['current_slot_accepted'] for e in tx.p4_probe.link_events
               if e['event'] == 'Keyed' and e['at'] > emitted[0]['audio_start'])
    assert received == [b"hello"]


@pytest.mark.parametrize("read_ms,keys", [(26, 1), (46, 1), (76, 0)])
def test_driver_admission_accounts_for_decode_cost_and_a_stall(
    monkeypatch, read_ms, keys
):
    audio = np.zeros(3 * FS)
    y = round(0.923 * FS)
    body = p4breakintx.changeover_packet(b"hello", 0)
    audio[y : y + len(body)] = body
    stream = Stream(audio)
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.key_notice = 1652
    stream.transmit = True
    stream.clamp_late = lambda at: max(0, stream.sample_now() + stream.key_notice - at)
    tx = Tx(stream)
    tx.p4_link_options.p4_link_timeout = 1.9
    monkeypatch.setattr(onair, "_prekey_lead", lambda *a: 2131)
    read = tx.sessrx.p4.read_window

    def timed(*a, **kw):
        records = read(*a, **kw)
        if any(r.observation.kind.value == "body" for r in records):
            stream.wall = stream.pos + round(read_ms / 1000 * FS)
        return records

    monkeypatch.setattr(tx.sessrx.p4, "read_window", timed)
    onair._run_p4_link(
        tx,
        stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    assert len(tx.p4_probe.emissions) - 1 == keys
    if keys:
        assert abs(tx.p4_probe.emissions[-1]["audio_start"] - (y + round(0.92 * FS))) <= 1
    assert tx.p4_link_host.arq._buffer_raw == 0


def test_driver_retransmits_unacknowledged_data_without_changing_its_waveform():
    from hfmodem.shrike import p4robusttx

    audio = np.zeros(10 * FS)
    for onset, wave in (
        (0.923, p4breakintx.changeover_packet(b"hello", 0)),
        (2.173, p4robusttx.robust_short(13)),
        (4.013, p4breakintx.changeover_packet(b"", 0x40)),
    ):
        at = round(onset * FS)
        audio[at : at + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_link_timeout = 8
    tx.p4_link_options.p4_text = "hi"
    onair._run_p4_link(
        tx,
        stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    records = tx.p4_probe.emissions[1:]
    assert len(records) == 5
    assert records[0]["word"] == "X27" and records[1]["status"] == 0x40
    assert abs(records[2]["audio_start"] - round(4.933 * FS)) <= 1
    assert np.array_equal(
        np.diff([r["audio_start"] for r in records[2:]]), [60000, 60000]
    )
    assert len({r["waveform_sha256"] for r in records[2:]}) == 1
    assert all(r["status"] == 0 for r in records[2:])
    assert tx.p4_link_host.arq._buffer_raw == len(tx.p4_link_host.arq._outbuf) > 0


def test_source_failure_reaches_mail_verdict_on_the_same_turn(monkeypatch):
    audio = np.zeros(3 * FS)
    y = round(0.923 * FS)
    wave = p4breakintx.changeover_packet(b"unsupported", 4)
    audio[y : y + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_link_options.p4_link_timeout = 1.9
    tx.p4_link_host._txbuf = 0  # No host bytes are queued in this source-failure control.
    tx.p4_link_mail = SimpleNamespace(
        done=False, progress=None, session=SimpleNamespace(failure=None, sent_text=False)
    )
    closed = []
    turns = []
    tx.p4_link_budget = SimpleNamespace(close=lambda n: closed.append((n, len(turns))))
    monkeypatch.setattr(onair, "_mail_app_turns", lambda *a, **kw: turns.append(1))
    onair._run_p4_link(
        tx,
        stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    assert tx.p4_link_mail.session.failure == "unsupported P4 source status 0x04"
    assert tx.p4_link_mail.done is False
    assert "failed" in tx.p4_probe.stopped and closed == [(0, 1)]
    assert all(e.get("status") == 0x20 for e in tx.p4_probe.emissions[1:])


@pytest.mark.parametrize("first", ["T27", "body"])
def test_driver_preserves_peer_sample_positions_when_transmit_flushes(first):
    from hfmodem.shrike import p4robusttx

    audio = np.zeros(6 * FS)
    y = round(.923 * FS)
    wave = (p4breakintx.changeover_packet(b"hello", 0) if first == "body"
            else p4robusttx.robust_word(first))
    audio[y:y + len(wave)] = wave
    following = y + round(1.25 * FS)
    wave = p4breakintx.changeover_packet(b"next", 0)
    audio[following:following + len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    key = tx._tx

    def flushing_key(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end

    tx._tx = flushing_key
    received = []
    tx.p4_link_host.deliver = received.append
    onair._run_p4_link(
        tx, stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    assert b"next" in received
    ack = "V27" if first == "body" else "X27"
    answers = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed'
               and e.get('current_slot_accepted', True) and e.get('word') == ack]
    assert abs(answers[-1]["at"] - (following + round(.920 * FS))) <= 1


def test_ve3kpg_recording_keeps_answers_after_our_transmissions():
    from pathlib import Path
    import json

    capture = Path(__file__).resolve().parents[5] / "captures/onair-0928-0943"
    if not (capture / "stream.flac").exists():
        pytest.skip("local VE3KPG flight recording absent")
    audio = onair.session.load_wav(str(capture / "stream.flac"), FS)
    entry = json.loads((capture / "p4-probe.json").read_text())["emissions"][0]
    stream = Stream(audio)
    stream.pos = entry["audio_end"]
    tx = Tx(stream)
    tx.p4_probe.emissions = [entry]
    key = tx._tx

    def flushing_key(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end

    tx._tx = flushing_key
    onair._run_p4_link(
        tx, stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800),
    )
    words = [r for r in tx.p4_probe.heard if r["outcome"] == "new"]
    assert [r["word"] for r in words] == ["T27"] * 6 + ["X27", "Z27"]
    assert any(e.get("value") == 38 for e in tx.p4_probe.emissions[1:])


def test_robust_packet_after_three_flushed_word_keys_keeps_its_noise_floor():
    from hfmodem.shrike import p4robusttx

    audio = np.random.default_rng(92804).normal(0, .002, 7*FS)
    first = round(.923*FS)
    for k in range(3):
        wave = p4breakintx.changeover_packet(f'part {k}'.encode(), 0)
        at = first+k*60000
        audio[at:at+len(wave)] += wave
    at = first+3*60000
    wave = p4robusttx.robust_short(13, 'A')
    audio[at:at+len(wave)] += wave
    stream = Stream(audio)
    tx = Tx(stream)
    key = tx._tx
    def flushing(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end
    tx._tx = flushing
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    onair._run_p4_link(tx, stream,
        onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert delivered[:3] == [b'part 0', b'part 1', b'part 2']
    words = tx.p4_probe.emissions[1:]
    assert [e['word'] for e in words[:4]] == ['X27', 'V27', 'V27', 'X27']
    assert abs(words[3]['audio_start']-(at+round(.920*FS))) <= 2


@pytest.mark.parametrize('policy,expected', [
    ('v27', ['X27', 'V27', 'V27']),
    ('x27', ['X27', 'X27', 'X27']),
    ('parity', ['X27', 'V27', 'X27']),
    ('alternate-packet', ['X27', 'V27', 'X27']),
])
@pytest.mark.parametrize('arm,end', [('1151', 30.8), ('2200', 32.1)])
def test_normal_ack_driver_policy_and_keyed_reason(policy, expected, arm, end):
    import json
    from pathlib import Path

    capture = Path(__file__).resolve().parents[5] / f'captures/onair-0929-{arm}'
    if not (capture / 'stream.flac').exists():
        pytest.skip(f'local WS8EOC {arm} recording absent')
    audio = onair.session.load_wav(str(capture / 'stream.flac'), FS)[:round(end*FS)]
    stream = Stream(audio)
    tx = Tx(stream)
    tx.p4_probe.emissions = json.loads((capture / 'p4-probe.json').read_text())['emissions'][:1]
    tx.p4_probe.values = (7, 38)
    stream.pos = tx.p4_probe.emissions[0]['audio_end']
    tx.settle = .044
    tx.p4_link_options.p4_normal_ack = policy
    tx.p4_link_options.p4_link_timeout = 16
    key = tx._tx

    def flushing(audio, what):
        key(audio, what)
        stream.pos = tx.tx_end

    tx._tx = flushing
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    choices = [e for e in tx.p4_probe.link_events if e['event'] == 'normal-answer']
    assert {e['counter'] for e in choices} == {0, 1, 2}
    assert [e['word'] for e in keyed] == [expected[e['counter']] for e in choices]
    assert all(e['ack_policy'] == policy for e in choices)
    assert [e['ack_reason'] for e in keyed] == [e['reason'] for e in choices]
    assert any(e['counter'] == 2 and e['repeat'] for e in choices)
