# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answer to a Robust Connect, read, and what the caller does with it.

KB5LZK answered this station's first keyed robust call (onair-0923-1413) with a
100 Bd word from the modem's 20-bit CS-burst table, entry 0, 48-49 ms after
each call ended, complemented on alternate cycles; the session read every one as
"nothing decoded", counted it as somebody else transmitting, and re-aimed its
calls against it. `fixtures/kb5lzk-0923-robust-answer.wav` holds four windows
cut from that day's tapes, each starting where one of our calls ended:

  * 0.00-0.30 s  answer to TX[10], complemented, whole
  * 0.30-0.60 s  answer to TX[11], as tabulated, whole
  * 0.60-0.85 s  an answer after the grid moved, 0.10 s on from the call's end,
    cut to 13 bits by our own next key
  * 0.85-1.15 s  onair-0923-1405, a normal connect answered with P1 CS1
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, p3acquire, p3rx, pactor1, placement, session, spec
from hfmodem.shrike.arq import ISS, ArqIO, PactorArq, State

FS = p1rx.FS
SPS = FS // 100
SLOT_N = round(1.25 * FS)
FIXTURE = Path(__file__).parent / "fixtures" / "kb5lzk-0923-robust-answer.wav"
ENTRY0 = p1rx.ROBUST_CS_WORDS[0]
CALL_N = round(0.88 * FS)
"""A robust call's audio: 11 bytes at 100 Bd."""


def _burst(word: int, bits: int = 20, invert: bool = False, amp: float = 0.3,
           faded: int | None = None):
    freqs = pactor1._fsk_freqs([(word >> i) & 1 for i in range(bits)], SPS, invert)
    out = amp * np.cos(2 * np.pi * np.cumsum(freqs) / FS)
    return out * np.where(freqs == faded, 10 ** (-15 / 20), 1.0) if faded else out


def _window(burst, at: float = 0.05, size: float = 0.30, noise: float = 0.02, seed=1):
    rng = np.random.default_rng(seed)
    out = noise * rng.standard_normal(round(size * FS))
    i = round(at * FS)
    out[i:i + burst.size] += burst[:out.size - i]
    return out


@pytest.mark.parametrize("invert", [False, True])
def test_a_whole_answer_names_its_entry_and_shift(invert):
    got = p1rx.acquire_robust_answer(_window(_burst(ENTRY0, invert=invert)), 0.02, 0.1)
    assert (got.entry, got.word, got.sense, got.bits, got.errors) == (
        0, ENTRY0, int(invert), 20, 0)
    assert abs(got.at - 0.05) < 0.002
    assert str(got) == (f"robust answer: 0 (052D56) "
                        f"{'inverted' if invert else 'normal'} 20-bit 0 errs")


def test_any_entry_of_the_table_is_read():
    words = {0: ENTRY0, 7: 0x5A3C6}
    got = p1rx.acquire_robust_answer(_window(_burst(0x5A3C6, invert=True)), 0.02, 0.1,
                                     words=words)
    assert (got.entry, got.sense) == (7, 1)


def test_the_prefix_form_reads_exactly_or_not_at_all():
    got = p1rx.acquire_robust_answer(_window(_burst(ENTRY0, bits=12)), 0.02, 0.1)
    assert (got.entry, got.bits, got.errors) == (0, 12, 0)
    flipped = ENTRY0 ^ (1 << 5)
    assert p1rx.acquire_robust_answer(_window(_burst(flipped, bits=12)), 0.02, 0.1) is None


@pytest.mark.parametrize("wrong,accepted", [(2, True), (3, False)])
def test_a_whole_word_carries_two_bit_errors_and_no_more(wrong, accepted):
    word = ENTRY0 ^ sum(1 << b for b in (1, 9, 17)[:wrong])
    got = p1rx.acquire_robust_answer(_window(_burst(word)), 0.02, 0.1)
    assert (got is not None) == accepted
    if accepted:
        assert (got.entry, got.errors) == (0, wrong)


@pytest.mark.parametrize("index", range(len(pactor1.CS_WORDS)))
@pytest.mark.parametrize("invert", [False, True])
def test_a_p1_control_signal_is_not_a_robust_answer(index, invert):
    cs = pactor1.control_signal(index, invert=invert, amp=0.3)
    assert p1rx.acquire_robust_answer(_window(cs), 0.02, 0.1) is None


def test_noise_is_not_an_answer():
    assert p1rx.acquire_robust_answer(_window(np.zeros(0), noise=0.1), 0.02, 0.1) is None


WORDS = {0: ENTRY0, 3: 0x95AC5, 8: 0xF5A5, 13: 0x7A7FC}
"""Test words, not the table: entry 0 as heard, and three with long runs of one
bit, the shape a single-tone read has least to go on. Each opens with at most
three bits of either value: a word whose first four bits all sit on the faded
tone starts past the reach of `_one_tone`'s back-step."""


@pytest.mark.parametrize("faded", [p1rx.MARK, p1rx.SPACE])
@pytest.mark.parametrize("entry", sorted(WORDS))
@pytest.mark.parametrize("invert", [False, True])
def test_a_word_with_one_tone_faded_is_read_on_the_other(faded, entry, invert):
    got = p1rx.acquire_robust_answer(
        _window(_burst(WORDS[entry], invert=invert, faded=faded)), 0.02, 0.1, words=WORDS)
    assert (got.entry, got.sense, got.errors) == (entry, int(invert), 0)
    assert got.tone == ({p1rx.MARK: p1rx.SPACE, p1rx.SPACE: p1rx.MARK}[faded])
    assert str(got).endswith(f"on {got.tone} Hz alone")


def test_the_one_tone_read_takes_no_foreign_word():
    rng = np.random.default_rng(3)
    wrong = 0
    for k in range(600):
        w = int(rng.integers(0, 1 << 20))
        far = min(min(bin(w ^ v).count("1"), 20 - bin(w ^ v).count("1"))
                  for v in WORDS.values())
        if far <= 3:
            continue
        got = p1rx.acquire_robust_answer(
            _window(_burst(w, faded=(p1rx.MARK, p1rx.SPACE)[k % 2]), seed=k),
            0.02, 0.1, words=WORDS)
        wrong += got is not None
    assert wrong == 0


@pytest.mark.parametrize("faded", [p1rx.MARK, p1rx.SPACE])
@pytest.mark.parametrize("index", range(len(pactor1.CS_WORDS)))
def test_a_faded_p1_control_signal_is_not_a_robust_answer(faded, index):
    bits = [(pactor1.CS_WORDS[index] >> i) & 1 for i in range(12)]
    freqs = pactor1._fsk_freqs(bits, SPS)
    cs = pactor1.control_signal(index, amp=0.3) * np.where(freqs == faded, 0.18, 1.0)
    assert p1rx.acquire_robust_answer(_window(cs), 0.02, 0.1, words=WORDS) is None


@pytest.mark.parametrize("tone", [p1rx.MARK, p1rx.SPACE])
@pytest.mark.parametrize("length", [0.12, 0.20, 0.25])
def test_a_steady_tone_is_not_a_word_of_ones(tone, length):
    t = np.arange(round(length * FS)) / FS
    burst = 0.3 * np.cos(2 * np.pi * tone * t)
    assert p1rx.acquire_robust_answer(_window(burst), 0.02, 0.1, words=WORDS) is None


FADED = Path(__file__).parent / "fixtures" / "ve3kpg-0923-faded-answer.wav"
"""VE3KPG on 40 m, onair-0923-1513, each window 0.90-1.19 s past its slot's
boundary: slot 26 (after our v8/A, MARK 9 dB down), slot 23 (MARK 11 dB down)
and slot 11 (SPACE 9 dB down), the first answered slot, which the live reader
missed."""


@pytest.mark.parametrize("start,sense,tone", [
    (0.00, 0, p1rx.SPACE), (0.29, 1, p1rx.SPACE), (0.58, 1, p1rx.MARK)])
def test_ve3kpg_through_a_fade(start, sense, tone):
    window = session.load_wav(str(FADED), FS)[round(start * FS):round((start + 0.29) * FS)]
    got = p1rx.acquire_robust_answer(window, 0.02, 0.1)
    assert (got.entry, got.sense, got.errors, got.tone) == (0, sense, 0, tone)
    assert abs(got.at - 0.056) < 0.012


def test_only_words_heard_on_the_air_are_recognised():
    assert p1rx.ROBUST_CS_WORDS == {0: spec.CONTROL_SIGNALS[0], 3: spec.CONTROL_SIGNALS[3]}
    assert p1rx.ROBUST_P2_ENTRIES == (0, 6)
    assert all(heard for _, heard in p1rx.ROBUST_CS_HEARD.values())


def _cut(start: float, size: float):
    return session.load_wav(str(FIXTURE), FS)[round(start * FS):round((start + size) * FS)]


@pytest.mark.parametrize("start,size,sense,bits,at", [
    (0.00, 0.30, 1, 20, 0.048), (0.30, 0.30, 0, 20, 0.049), (0.60, 0.25, 1, 13, 0.085)])
def test_kb5lzk_on_the_air(start, size, sense, bits, at):
    got = p1rx.acquire_robust_answer(_cut(start, size), 0.02, 0.1)
    assert (got.entry, got.sense, got.bits, got.errors) == (0, sense, bits, 0)
    assert abs(got.at - at) < 0.003


def test_kb5lzk_answering_a_normal_connect_is_silent_here():
    window = _cut(0.85, 0.30)
    assert p1rx.acquire_robust_answer(window, 0.02, 0.1) is None
    assert p1rx.acquire_control_signal(window, 0.03, 0.12, span=0.15)[0] == pactor1.CS_ACK_A


def test_the_loop_reads_it_off_our_own_carrier():
    grid = SimpleNamespace(slot_n=60000, p1_data_n=46080)
    seg = _cut(0.30, 0.27)
    got, d = onair._robust_answer(seg, 10_000, 10_000, 10_000 + 60000 - 46080 + 46080,
                                  grid, 6240, False)
    assert got.entry == 0 and abs(d - 0.049) < 0.003


def test_a_hush_reads_it_off_the_carrier_that_really_dropped():
    # The slot's projection assumes a 0.96 s packet; the call was 0.88 s.
    grid = SimpleNamespace(slot_n=60000, p1_data_n=46080)
    boundary = 10_000 + 60000 - 46080 + 46080 + round(0.08 * FS)
    got, d = onair._robust_answer(_cut(0.30, 0.27), 10_000, 10_000, boundary, grid,
                                  6240, True)
    assert abs(d - 0.049) < 0.003


class _IO(ArqIO):
    def __init__(self):
        self.lines, self.bursts = [], 0

    def connect_burst(self, mycall, dxcall):
        self.bursts += 1

    def log(self, msg):
        self.lines.append(msg)

    def send_cs(self, *a):
        pass

    def send_packet(self, *a, **k):
        return 0


def _calling(variant="%"):
    io_ = _IO()
    arq = PactorArq(io_)
    arq.on_host_connect("W9SSJ", f"{variant}KB5LZK")
    return arq, io_


def test_a_robust_answer_brings_the_link_up_with_us_sending():
    arq, io_ = _calling()
    arq.on_rx_robust_answer(0)
    assert arq.state == State.CONNECTED and arq.role == ISS
    assert arq.robust_answer == 0 and arq.dxcall == "KB5LZK"
    assert "robust answer (entry 0) -> link up" in io_.lines
    arq.note_peer_heard()
    arq.on_cycle()
    assert io_.bursts == 1
    assert not any("still transmitting" in line for line in io_.lines)


def test_only_a_robust_call_is_answered_this_way():
    arq, _ = _calling("")
    arq.on_rx_robust_answer(0)
    assert arq.state == State.CONNECTING and arq.robust_answer is None
    arq.on_host_abort()
    arq.on_host_connect("W9SSJ", "%KB5LZK")
    assert arq.robust_answer is None


def test_after_robust_names_an_after_connect_path():
    args = SimpleNamespace(after_connect="arq", after_robust="p4-robust")
    assert onair._after_answer(args, SimpleNamespace(robust_answer=None)) == "arq"
    for after, path in (("listen", "listen"), ("p1", "arq"), ("p3", "arq"),
                        ("p4-robust", "chirp")):
        args.after_robust = after
        assert onair._after_answer(args, SimpleNamespace(robust_answer=0)) == path


# --- the whole session, against a station that answers the robust call -------

def _peer_wav(path: Path, slots: int = 16, d_ms: float = 49.0) -> None:
    out = np.zeros((slots + 2) * 60000, np.float32)
    for k in range(slots):
        rf = _burst(ENTRY0, invert=bool(k % 2), amp=0.11).astype(np.float32)
        at = k * 60000 + CALL_N + round(d_ms / 1000 * FS)
        out[at:at + rf.size] += rf
    session.write_wav(str(path), out)


def _air(tmp_path, *flags):
    wav = tmp_path / "peer.wav"
    if not wav.exists():
        _peer_wav(wav)
    save = onair._save_capture_async
    onair._save_capture_async = lambda *a, **kw: None
    argv, log = sys.argv, io.StringIO()
    sys.argv = ["shrike.onair", "--replay", str(wav), "--hold", "6",
                "--max-cycles", "6", "--mycall", "W9SSJ", "--dxcall", "K7ABC",
                "--dial", "7100000", "--outdir", str(tmp_path / "out"),
                "--connect", "robust", *flags]
    try:
        with contextlib.redirect_stdout(log):
            onair.main()
    finally:
        onair._save_capture_async = save
        sys.argv = argv
    return log.getvalue()


def _after(log):
    assert "robust answer: 0 (052D56)" in log, log
    assert "robust answer (entry 0) -> link up" in log, log
    after = log[log.index("** CONNECTED"):]
    assert "GRID PLACED" not in log, log
    assert "still transmitting" not in after, after
    return after


def test_listen_keys_nothing_after_the_answer(tmp_path):
    after = _after(_air(tmp_path, "--after-connect-cycles", "3"))
    assert "[after-connect] listening 3 cycles; keying nothing" in after, after
    assert "TX[" not in after, after


def test_p1_keys_a_data_packet_carrying_our_call(tmp_path, monkeypatch):
    fields = []
    send = onair.RadioTx.send_p1_packet
    monkeypatch.setattr(onair.RadioTx, "send_p1_packet",
                        lambda tx, payload, *a, **k: fields.append(payload) or send(tx, payload, *a, **k))
    after = _after(_air(tmp_path, "--after-robust", "p1"))
    first = after[after.index("TX["):].splitlines()[0]
    assert "connect->" not in first and "P1" in first, first
    assert fields and fields[0].startswith(b"W9SSJ\r"), fields


@pytest.mark.parametrize("cadence,slots", [
    ("spaced", [3, 8, 13]), ("consecutive", [3, 4, 5])])
def test_p4_robust_keys_the_robust_short_packets(tmp_path, cadence, slots):
    log = _air(tmp_path, "--after-robust", "p4-robust", "--p4-probe-cadence", cadence)
    after = _after(log)
    first = after[after.index("TX["):].splitlines()[0]
    assert "P4 PROBE ROBUST 1/3 v=6 hdr=A" in first, first
    probe = json.loads((tmp_path / "out" / "p4-probe.json").read_text())
    # The answer was read after the call of slot 2 went out; slot 3 is the first
    # after it.
    assert [e["slot"] for e in probe["emissions"]] == slots
    assert probe["cadence"] == cadence
    gaps = after.count("the gap to the next slot")
    assert gaps == (2 if cadence == "consecutive" else 0), after
    assert "receiving continuously for 4 seconds" in after
    reads = [w for w in probe["windows"] if w["kind"] == "post-chirp"]
    assert len(reads) == 3


KB5LZK_D_MS, VE3KPG_D_MS = 50.0, 76.0
"""Each station's answer past our call's end: 930 and 956 ms past the boundary."""


def _script_wav(path: Path, script: str, d_ms: float) -> None:
    """A peer keying the robust answer's form every slot, `script` naming the
    control signal (CS1 = 1) it keys in each."""
    out = np.zeros((len(script) + 2) * 60000, np.float32)
    for k, cs in enumerate(script):
        rf = _burst(spec.CONTROL_SIGNALS[int(cs) - 1], invert=bool(k % 2),
                    amp=0.11).astype(np.float32)
        at = k * 60000 + CALL_N + round(d_ms / 1000 * FS)
        out[at:at + rf.size] += rf
    session.write_wav(str(path), out)


def _keyed(monkeypatch):
    """(slot, label, samples) of every burst the session keys."""
    keyed, key = [], onair.RadioTx._tx

    def tx(self, audio, what, *a, **k):
        keyed.append((self.slot, what, audio.size))
        return key(self, audio, what, *a, **k)

    monkeypatch.setattr(onair.RadioTx, "_tx", tx)
    return keyed


@pytest.mark.parametrize("d_ms", [KB5LZK_D_MS, VE3KPG_D_MS])
def test_direct_p3_preserves_third_slot_entry_after_the_answered_call(tmp_path, monkeypatch,
                                                                      d_ms):
    _script_wav(tmp_path / "peer.wav", "1" * 16, d_ms)
    keyed = _keyed(monkeypatch)
    log = _air(tmp_path, "--after-robust", "p3")
    after = _after(log)
    calls = [slot for slot, what, _ in keyed if what.startswith("connect")]
    slot, what, n = next(k for k in keyed if not k[1].startswith("connect"))
    assert slot == calls[-1] + 3, keyed
    assert what.startswith("SL1 ENTRY") and "status=0x1a" in what, what
    assert n < CALL_N + round(d_ms / 1000 * FS), n
    assert "PACTOR-3 entry packet at counter 2" in log, log
    # The FSK answer repeated at the entry is the connect answer, not an ACK of it.
    assert "the peer answered the entry packet" not in after, after


K4MCK = Path(__file__).parent / "fixtures" / "k4mck-0925-kiwi-robust-p3.wav"
"""K4MCK's Robust Connect to AI4Y, from the start of its 19th and last call.

Kiwi camp, 7100.0 kHz dial, seg-0049 from 281.431 s, resampled to 12 kHz off the
receiver's 11998.908: the call, AI4Y's answer 0.99 s after the call started,
two P2 exchanges, K4MCK's P3 entry in the third slot, and AI4Y's CS4.
Both stations share the carrier offset: +82 Hz on the FSK tone plan and
+69 Hz on the P3 raster convention. The tape is compensated by 82 Hz here."""


def _k4mck():
    return p3acquire.compensate(session.load_wav(str(K4MCK), FS), 82.0)


def test_k4mck_enters_p3_three_slots_after_the_answered_call():
    x = _k4mck()
    got = p1rx.acquire_robust_answer(x, 0.90, 0.20)
    assert (got.entry, abs(got.at - 0.990) < 0.005) == (0, True), got
    assert all(p1rx.acquire_robust_answer(x[k * SLOT_N:(k + 1) * SLOT_N], 0.90, 0.30) is None
               for k in (1, 2))
    entry = 3 * SLOT_N
    scan = p3rx.decode_p3_packets(x[entry:entry + SLOT_N])
    p, = scan.packets
    assert (p.sl, p.status, p.info, p.long_cycle, p.carrier_swapped) == (
        1, 0x1a, spec.TEMPLATE[:5], False, False)
    ours = placement.link_packet(1, b"", 0x1a, flush=placement.ENTRY_FLUSH)
    ours = onair._trim_silence(ours)
    bench = np.zeros(SLOT_N)
    bench[onair.RadioTx.entry_delay_n:][:ours.size] = ours
    q, = p3rx.decode_p3_packets(bench).packets
    assert (q.status, q.info) == (p.status, p.info)
    assert abs(p.start - q.start) < SPS // 2, (p.start, q.start)


def test_direct_p3_experiment_keeps_recorded_entry_slot_without_p2(tmp_path, monkeypatch):
    peer = _k4mck()
    # K4MCK's own call and entry are where ours key, and a replay hears no TX.
    for k in range(4):
        peer[k*SLOT_N:k*SLOT_N+CALL_N] = 0
    session.write_wav(str(tmp_path / "peer.wav"), np.concatenate([peer, np.zeros(8 * SLOT_N)]))
    keyed = _keyed(monkeypatch)
    after = _after(_air(tmp_path, "--after-robust", "p3"))
    assert [(slot, what.split()[0]) for slot, what, _ in keyed[:2]] == [
        (0, "connect->K7ABC"), (3, "SL1")], keyed
    assert "status=0x1a field=0f8f87c7c31a6689" in keyed[1][1], keyed
    assert "RX P3 CS4 SPEED-UP" in after, after
    assert "the peer answered the entry packet -> SL1" in after, after


def test_after_robust_needs_a_robust_call(monkeypatch):
    monkeypatch.setattr(onair, "run", lambda args: pytest.fail("opened run"))
    monkeypatch.setattr(sys, "argv", ["onair", "--after-robust", "p1"])
    with pytest.raises(SystemExit, match="draws none"):
        onair.main()


INPUT_LEAD = 532
"""How far `sample_now()` leads the delivered count on this station's codec:
the holdback less a block (see test_late_at_key_input_latency_0923)."""


READ_COST = round(0.080 * FS)
"""What reading a gap window with the peer's answer in it costs: 80 ms, measured
on onair-0923-1624's window after v6 (0.2 ms on its empty window after v9)."""


@pytest.mark.parametrize("lead,cost", [(0, 0), (INPUT_LEAD, 0), (INPUT_LEAD, READ_COST)])
def test_consecutive_packets_key_in_consecutive_slots_on_a_live_clock(tmp_path, monkeypatch,
                                                                      lead, cost):
    """The live clocks of onair-0923-1615 and 1624. The TX flush walks the reader
    to our carrier's end ahead of the converter; once the gap has been read the
    converter leads the reader by the input latency; and reading an answered gap
    spends converter time the key needed. 1615 keyed n, n+2, n+4; 1624 n, n+2,
    n+3, late by 49.5 ms after the answered gap only."""
    from hfmodem.shrike import p4probe
    monkeypatch.setattr(onair, "_load", lambda _: np.zeros(40 * FS, np.float32))
    live = onair._ReplayInput("synthetic")
    live._blk = 128
    clock = [0]
    take = live.take_until

    def taken(until):
        out = take(until)
        clock[0] = max(clock[0], live.pos + lead)
        return out

    monkeypatch.setattr(live, "take_until", taken)
    monkeypatch.setattr(live, "sample_now", lambda: float(clock[0]))
    row = onair._p4_query_row

    def costly(*a, **k):
        clock[0] += cost
        return row(*a, **k)

    monkeypatch.setattr(onair, "_p4_query_row", costly)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live = live
    key = tx._tx

    def keyed(audio, label):
        key(audio, label)
        live.take_until(tx.tx_end)
        clock[0] = live.pos - 1200

    monkeypatch.setattr(tx, "_tx", keyed)
    tx.p4_probe = p4probe.EntryProbe(3, packet="robust", values=(6, 9, 8),
                                     headers=("A",) * 3, cadence="consecutive")
    grid = onair._MasterGrid(0, 60000, 0, packet_n=46080, cs_n=5760, d_max_n=6240)
    tx.aim(grid, 0)
    tx.sessrx = SimpleNamespace(bridge=lambda _: None)
    tx.send_p4_entry_packet(b"", 0)
    onair._run_p4_probe(tx, live, grid)
    slots = [e["slot"] for e in tx.p4_probe.emissions]
    assert slots == [slots[0], slots[0] + 1, slots[0] + 2], slots
    assert [q["index"] for q in tx.p4_probe.queries] == [1, 2, 3]
    assert [q["status_hex"] for q in tx.p4_probe.queries] == ["v6/A", "v9/A", "v8/A"]
