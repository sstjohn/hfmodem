# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The data packet's rise, and what it ends its trellis on.

Two properties of an ordinary PACTOR-III data packet that the entry packet has
had since 2026-08-31 and the data packets did not.

THE PULSE. `placement.PROTOCOL_RISE` shapes every keying with `tablegen.
symbol_pulse` -- the quasi-RRC the receiver already runs as its matched filter.
Speed level 1 always had it, because it is case 0; levels 2 to 6 went out on
`modem.ModConfig`'s generic raised cosine, and rendered side by side that
put 20 ms of energy
in front of the packet where the reference's own data packets sit on their tape's
floor, and landed the phase reference 27.8 ms past the slot boundary where the
entry's lands at 13.9. A peer that acquired on our entry then met every data
packet 1.4 symbol periods late. So the measurement here is a COMPARISON: the data
packet against our entry packet, on the instruments the report used, with the
raised-cosine render carried alongside as the negative control -- it has to fail
the same bounds, or they are not measuring the pulse.

THE FLUSH. `placement.ENTRY_FLUSH` is eight bits read off DL6MAA's entry packet
and every data packet ends its trellis on zeros instead. `placement.DATA_FLUSH`
is the switch that keys the entry's bits on the data packets too, `--p3-data-flush
entry` on the command line, and it is a trial rather than a finding: what it has
to be is separable and reversible, which is a frame whose field and CRC are
untouched and which our own receiver still reads to the same payload and status
either way. `placement.REFERENCE_FLUSH` is the third choice and the only one that
was read off data packets -- the six bits all six clean level-3 packets in
`rf-corpus/PIII_Complete_1.wav` end on (`working/pactor3-header-0913/reference-
flush.py`) -- and it carries the same burden, plus one more: the fit that read it
off the reference has to read it back off our own render.

Run:  python -m pytest hfmodem/tests/shrike/test_data_pulse_flush.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from hfmodem.shrike import (modem, onair, p3frame, p3rx, placement, rx,
                           rxfront, spec)

FS, SPS = onair.FS, rxfront.SPS
PAYLOAD = b"1234567"
"""The seven bytes the 09-13 arms keyed, so the figures here are that
transcript's `SL3 pkt 7B`."""


def _keyed(audio: np.ndarray, cfg: modem.ModConfig) -> tuple[np.ndarray, int]:
    """The waveform as the transmitter keys it, and where its phase reference is.

    `onair._tx` trims the skirt at 2 % and puts what is left on the slot
    boundary, so the offset from the first keyed sample to the pulse centre IS
    the distance from the boundary to the packet's first symbol. The lead-in is
    padded back with silence because that is what the channel carries there and
    an array's edge is not a measurement.
    """
    trim = int(np.flatnonzero(abs(audio) > .02 * np.max(abs(audio)))[0])
    lead = int(np.argmax(cfg.pulse())) - trim
    return np.concatenate([np.zeros(FS // 10), onair._trim_silence(audio)]), \
        FS // 10 + lead


def _phase_ref_ms(audio: np.ndarray, cfg: modem.ModConfig) -> float:
    return (_keyed(audio, cfg)[1] - FS // 10) / FS * 1e3


def _pre_energy_db(audio: np.ndarray, cfg: modem.ModConfig,
                   lo_ms: float, hi_ms: float) -> float:
    """Mean power in the window before the phase reference, against the packet's
    own steady level. The normalisation is per packet, so a fourteen-carrier comb
    and a two-carrier one are comparable."""
    keyed, ref = _keyed(audio, cfg)
    steady = np.mean(keyed[ref + int(.20 * FS):ref + int(.60 * FS)] ** 2)
    seg = keyed[ref - round(hi_ms * FS / 1e3):ref - round(lo_ms * FS / 1e3)]
    return 10 * np.log10(max(float(np.mean(seg ** 2)), 1e-300) / steady)


def _entry() -> np.ndarray:
    return placement.link_packet(1, b"", 0x1a, flush=placement.ENTRY_FLUSH)


def _data(sl: int, *, raised_cosine: bool = False) -> np.ndarray:
    path = placement.SPEED_PATHS[sl]
    if not raised_cosine:
        return placement.link_packet(sl, PAYLOAD, spec.status_byte(3))
    info = placement.field_info(PAYLOAD, path.crc_bytes - 3, spec.status_byte(3))
    return placement.data_packet(info, path, cfg=modem.ModConfig())


# --------------------------------------------------------------------------- #
# The pulse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sl", (3, 5))
def test_a_data_packet_puts_its_phase_reference_where_the_entry_does(sl: int):
    """13.9 ms past the boundary, which is the figure `onair.ENTRY_END_N` carries
    for the entry and nothing had carried to the data path. The raised cosine's
    own pair is in that constant's comment: 26.8 ms."""
    entry = _phase_ref_ms(_entry(), placement.protocol_config())
    data = _phase_ref_ms(_data(sl), placement.protocol_config())
    assert abs(data - entry) < 1.0, (data, entry)

    control = _phase_ref_ms(_data(sl, raised_cosine=True), modem.ModConfig())
    assert control - entry > 10.0, (control, entry)


@pytest.mark.parametrize("sl", (3, 5))
def test_a_data_packet_rises_where_the_entry_rises(sl: int):
    """Nothing on the air 40 to 10 ms before the first symbol that the entry does
    not also put there. The reference's data packets are on their tape's floor
    from -40 ms to -15 ms, exactly as its entry is; ours were at -25 dB."""
    entry = _pre_energy_db(_entry(), placement.protocol_config(), 10, 40)
    data = _pre_energy_db(_data(sl), placement.protocol_config(), 10, 40)
    assert abs(data - entry) < 3.0, (data, entry)

    control = _pre_energy_db(_data(sl, raised_cosine=True),
                             modem.ModConfig(), 10, 40)
    assert control - entry > 3.0, (control, entry)


@pytest.mark.parametrize("sl", (3, 5))
def test_nothing_is_keyed_before_the_filter_the_protocol_publishes(sl: int):
    """The sharp end of the same measurement. The protocol pulse spans three
    symbols, so 20 ms before the phase reference there is no packet at all; the
    raised cosine spans eight and is keying there."""
    assert _pre_energy_db(_data(sl), placement.protocol_config(), 20, 40) < -100
    assert _pre_energy_db(_entry(), placement.protocol_config(), 20, 40) < -100
    assert _pre_energy_db(_data(sl, raised_cosine=True),
                          modem.ModConfig(), 20, 40) > -40


def test_the_protocol_pulse_shortens_the_keying_the_transcript_prints():
    """`SL3 pkt 7B` printed `(0.9s)` and prints `(0.8s)`, which is the one line of
    an arm's transcript that says this flew. The entry is unmoved."""
    keyed = len(onair._trim_silence(_data(3))) / FS
    control = len(onair._trim_silence(_data(3, raised_cosine=True))) / FS
    assert control - keyed > .02
    assert f"{keyed:.1f}s" == "0.8s" and f"{control:.1f}s" == "0.9s"


def test_the_rise_switch_still_reaches_the_data_packet(monkeypatch):
    """Off, every keying goes back to the generic filter -- which is what
    `--no-p3-rise` is for, and it has to reach the level it was widened to."""
    monkeypatch.setattr(placement, "PROTOCOL_RISE", False)
    assert np.array_equal(_data(3), _data(3, raised_cosine=True))


# --------------------------------------------------------------------------- #
# The flush
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sl", (1, 3))
def test_the_entry_flush_moves_the_trellis_tail_and_nothing_else(sl: int):
    """The field the CRC covers is the same bytes either way, and the code bits
    that move are the last ten of the buffer. Case 0 neither punctures nor
    interleaves, so those land in named cells; case 1's are the trellis' own tail
    in code order, before the helical walk scatters them."""
    path = placement.SPEED_PATHS[sl]
    info = placement.field_info(b"", path.crc_bytes - 3, spec.status_byte(3))
    field = placement.build_field(info, path)

    zeros = placement.encode_frame(field, path, None)
    entry = placement.encode_frame(field, path, placement.ENTRY_FLUSH)
    moved = np.flatnonzero(zeros != entry)
    assert moved.size and moved[0] > path.n_buf - 16, (sl, moved)


@pytest.mark.parametrize("sl", (1, 3))
def test_the_switch_carries_the_entry_flush_onto_a_data_packet(sl: int, monkeypatch):
    """`--p3-data-flush entry`'s whole effect, at the seam an arm drives:
    `link_packet` is what `onair.RadioTx` keys through, and the entry packet
    passes its own flush explicitly so it is untouched either way."""
    keyed = placement.link_packet(sl, b"", spec.status_byte(3))
    monkeypatch.setattr(placement, "DATA_FLUSH", placement.ENTRY_FLUSH)
    flushed = placement.link_packet(sl, b"", spec.status_byte(3))
    assert not np.array_equal(keyed, flushed)
    assert np.array_equal(_entry(), placement.link_packet(
        1, b"", 0x1a, flush=placement.ENTRY_FLUSH))


@pytest.mark.parametrize("flush", (None, placement.ENTRY_FLUSH))
def test_our_receiver_reads_the_same_packet_under_either_flush(flush):
    """A speed-level-1 packet on counter 3 -- the packet VE3KPG answered CS1
    thirty-five times -- read both by the anchored decode and by the blind scan.
    The trial is only worth flying if the flush is the sole variable, and a flush
    our own receiver could not take would be a different packet."""
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)
    status = spec.status_byte(3)
    audio = np.concatenate([np.zeros(FS),
                            placement.link_packet(1, b"", status, flush=flush),
                            np.zeros(FS)])
    header = p3frame.PacketHeader(0, p3frame.variable_header(1), 1.0)

    got = p3rx.decode_at(audio, row0, 1, fs=FS, header=header)
    assert got is not None and (got.payload, got.status) == (b"", status)
    assert [(p.sl, p.payload, p.status)
            for p in p3rx.decode_p3_packets(audio).packets] == [(1, b"", status)]

    live = rxfront.SyncedRx()._level_at_lock(audio, row0, 1)
    assert live is not None and live.packet == (1, status, b"", True)


REFERENCE_FIT_LEVELS = (3, 5)
"""The reference's own level-3 packets are where `REFERENCE_FLUSH` was measured;
level 5 is a second puncture, so the tail lands in a different number of cells."""


def _code_softs(audio: np.ndarray, path: placement.Path, row0: int) -> np.ndarray:
    """Code-order softs off a rendered packet, the way `reference-flush.py` reads
    them off the air: the packet's own header block for the angle, `p3rx._cells`,
    `placement.deinterleave`."""
    Z = {cn: rx._baseband(audio, cn, FS, rx._pulse(SPS)) for cn in path.tones}
    head = p3rx.header_of(Z, [row0], path)
    return np.asarray(placement.deinterleave(
        p3rx._cells(audio, row0, path, head.rot, fs=FS, Z=Z), path), float)


def _ml_flush(softs: np.ndarray, field: bytes, path: placement.Path) -> tuple:
    """The flush those softs most nearly carry, scored over the positions any of
    the sixty-four can move -- `reference-flush.py`'s `_fit`, inlined."""
    cands = [tuple(c) for c in itertools.product((0, 1), repeat=6)]
    enc = {c: placement.encode_frame(field, path, c) for c in cands}
    moves = sorted({i for c in cands
                    for i in np.flatnonzero(enc[c] != enc[(0,) * 6])})
    return max(cands, key=lambda c: float(
        np.sum(softs[moves] * (1.0 - 2.0 * enc[c][moves]))))


@pytest.mark.parametrize("sl", REFERENCE_FIT_LEVELS)
def test_the_reference_flush_moves_the_trellis_tail_and_nothing_else(sl: int):
    """Same burden as the entry flush's: the field the CRC covers is the same
    bytes and what moves is the tail, in code order, before the helical walk
    scatters it."""
    path = placement.SPEED_PATHS[sl]
    field = placement.build_field(
        placement.field_info(PAYLOAD, path.crc_bytes - 3, spec.status_byte(3)),
        path)

    zeros = placement.encode_frame(field, path, None)
    tail = placement.encode_frame(field, path, placement.REFERENCE_FLUSH)
    moved = np.flatnonzero(zeros != tail)
    assert moved.size and moved[0] > path.n_buf - 16, (sl, moved)


@pytest.mark.parametrize("sl", REFERENCE_FIT_LEVELS)
def test_our_render_carries_the_tail_that_was_read_off_the_reference(sl: int):
    """The instrument that fitted `(0, 0, 0, 0, 1, 0)` to all six of the
    reference's clean level-3 packets, turned on our own keying: it has to return
    the tail we asked for and, with the switch off, zeros. Without this the
    constant is a claim about a stranger's modem with nothing joining it to what
    this station would key."""
    path = placement.SPEED_PATHS[sl]
    status = spec.status_byte(3)
    field = placement.build_field(
        placement.field_info(PAYLOAD, path.crc_bytes - 3, status), path)
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)

    for flush, want in ((None, (0,) * 6), (placement.REFERENCE_FLUSH,
                                          placement.REFERENCE_FLUSH)):
        audio = np.concatenate([np.zeros(FS),
                                placement.link_packet(sl, PAYLOAD, status,
                                                      flush=flush),
                                np.zeros(FS)])
        assert _ml_flush(_code_softs(audio, path, row0), field, path) == want


def test_our_receiver_reads_the_same_packet_under_the_reference_flush():
    """The level the tail was measured at, read by the anchored decode, the blind
    scan and the locked path -- the same three the entry flush answers to."""
    status = spec.status_byte(3)
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)
    audio = np.concatenate([np.zeros(FS),
                            placement.link_packet(
                                3, PAYLOAD, status,
                                flush=placement.REFERENCE_FLUSH),
                            np.zeros(FS)])
    header = p3frame.PacketHeader(0, p3frame.variable_header(3), 1.0)

    got = p3rx.decode_at(audio, row0, 3, fs=FS, header=header)
    assert got is not None and (got.payload, got.status) == (PAYLOAD, status)
    assert [(p.sl, p.payload, p.status)
            for p in p3rx.decode_p3_packets(audio).packets] == [(3, PAYLOAD,
                                                                 status)]

    live = rxfront.SyncedRx()._level_at_lock(audio, row0, 3)
    assert live is not None and live.packet == (3, status, PAYLOAD, True)


def test_the_reference_switch_stops_at_speed_level_1(monkeypatch):
    """Six bits are six bits of the K=7 trellis and case 0's is K=9, so the tail
    has no meaning there and a short one would encode a frame two steps shy of
    `n_buf`. Nothing is lost by stopping: WS8EOC's own speed-level-1 data packets
    in `captures/onair-0912-2321` fit ZEROS, 0 of 16 movable bits wrong."""
    keyed = [placement.link_packet(sl, b"", spec.status_byte(3))
             for sl in (1, 3)]
    monkeypatch.setattr(placement, "DATA_FLUSH", placement.REFERENCE_FLUSH)
    flushed = [placement.link_packet(sl, b"", spec.status_byte(3))
               for sl in (1, 3)]
    assert np.array_equal(keyed[0], flushed[0])
    assert not np.array_equal(keyed[1], flushed[1])
    assert np.array_equal(_entry(), placement.link_packet(
        1, b"", 0x1a, flush=placement.ENTRY_FLUSH))
