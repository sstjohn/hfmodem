# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The receive-side HARQ soft buffer: identity keying and the accept arbiter.

Deterministic complementary-half LLR vectors drive every case: either half
of a frame alone is undecodable, the two halves of the *same* frame combine
to a byte-exact delivery, and halves of *different* frames -- a seq
collision the identity key cannot see -- must combine to nothing, because
only the LDPC syndrome + per-codeword CRC screen ever accepts bits.
"""

import numpy as np

from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig, ArqFsm
from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy.modem import LADDER

_FULL = wire.capabilities(range(3, 7 + 1), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)
_WORKHORSE = LADDER.index("workhorse")
_ROBUST = LADDER.index("robust")


class _StubIO:
    def __init__(self):
        self.controls, self.delivered = [], b""

    def send_control(self, ctrl):
        self.controls.append(ctrl)
        return 4.5

    def send_data(self, *a):
        return 8.0

    def connected(self, *a): ...
    def disconnected(self): ...
    def deliver(self, blob):
        self.delivered += blob
    def log(self, msg): ...
    def state_changed(self, state): ...


def _irs():
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="BOB"), clock=lambda: 0.0)
    fsm.on_host_listen(True)
    tag = 0x81
    fsm.on_control(wire.Control.connect(tag, _FULL, "ALICE", destination="BOB"))
    return io, fsm, tag


def _rows(codec, payload):
    coded = np.stack([codec.encode_cw(c) for c in codec.chunk(payload)])
    return 4.0 * (1.0 - 2.0 * coded)


def _half(rows, which):
    n = rows.shape[1]
    m = np.zeros(n)
    if which == "front":
        m[: n // 2] = 1.0
    else:
        m[n // 2:] = 1.0
    return rows * m


def _hdr(tag, seq, gear, n_cw, present=None):
    present = range(n_cw) if present is None else present
    return wire.Control(wire.DATA, tag, seq=seq, gear=gear,
                        mask=wire.cw_mask(present),
                        aux=wire.data_aux(100, n_cw, (0,) * 8))


def test_two_weak_rounds_combine_byte_exact():
    io, fsm, tag = _irs()
    codec = FrameCodec(QCLDPC("r12"))
    payload = bytes(range(200)) + bytes(4 * 61 - 200)
    rows = _rows(codec, payload)
    fsm.on_data(_hdr(tag, 5, _WORKHORSE, 4), _half(rows, "front"), [10.0] * 8)
    assert io.delivered == b""                      # half a frame is nothing
    fsm.on_data(_hdr(tag, 5, _WORKHORSE, 4), _half(rows, "back"), [10.0] * 8)
    assert io.delivered == payload


def test_handover_toggle_keeps_the_store():
    io, fsm, tag = _irs()
    codec = FrameCodec(QCLDPC("r12"))
    payload = bytes(4 * 61)
    rows = _rows(codec, payload)
    fsm.on_data(_hdr(tag, 6, _WORKHORSE, 4), _half(rows, "front"), [10.0] * 8)
    fsm.on_data(_hdr(tag, 6, _WORKHORSE | wire.HANDOVER, 4),
                _half(rows, "back"), [10.0] * 8)
    assert io.delivered == payload                  # same frame, same store


def test_colliding_frames_combine_to_nothing():
    io, fsm, tag = _irs()
    codec = FrameCodec(QCLDPC("r12"))
    rng = np.random.default_rng(3)
    rows_a = _rows(codec, rng.integers(0, 256, 4 * 61, np.uint8).tobytes())
    rows_b = _rows(codec, rng.integers(0, 256, 4 * 61, np.uint8).tobytes())
    fsm.on_data(_hdr(tag, 7, _WORKHORSE, 4), _half(rows_a, "front"),
                [10.0] * 8)
    fsm.on_data(_hdr(tag, 7, _WORKHORSE, 4), _half(rows_b, "back"),
                [10.0] * 8)
    assert io.delivered == b""
    assert all(c is None for c in fsm._rx.chunks)
    acks = [c for c in io.controls if c.type == wire.ACK]
    assert all(c.mask == bytes(8) for c in acks)    # nothing claimed decoded


def test_seq_collision_across_gears_restarts_store():
    io, fsm, tag = _irs()
    fsm.on_data(_hdr(tag, 9, _WORKHORSE, 4),
                _half(_rows(FrameCodec(QCLDPC("r12")), bytes(4 * 61)),
                      "front"), [10.0] * 8)
    codec13 = FrameCodec(QCLDPC("r13"))
    payload = bytes(reversed(range(2 * codec13.data_bytes)))[: 2 * 61]
    fsm.on_data(_hdr(tag, 9, _ROBUST, 2), _rows(codec13, payload), [10.0] * 8)
    assert io.delivered == payload                  # fresh store, no crash


def test_seq_collision_across_ncw_restarts_store():
    io, fsm, tag = _irs()
    codec = FrameCodec(QCLDPC("r12"))
    fsm.on_data(_hdr(tag, 11, _WORKHORSE, 4),
                _half(_rows(codec, bytes(4 * 61)), "front"), [10.0] * 8)
    payload = bytes(6 * 61)
    fsm.on_data(_hdr(tag, 11, _WORKHORSE, 6), _rows(codec, payload),
                [10.0] * 8)
    assert io.delivered == payload                  # fresh store, no crash


def test_ncw_beyond_the_ack_mask_is_ignored():
    io, fsm, tag = _irs()
    fsm.on_data(_hdr(tag, 13, _WORKHORSE, 200, present=range(4)), None,
                [10.0] * 8)
    assert fsm._rx is None                          # no 200-codeword store
