# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent, test-only robust full-D codeword model.

No modem encoding/decoding/table-generation imports. Native data is loaded
from an independently extracted, immutable oracle-only fixture. This is an
algebra/fixture generator, not a decoder or an acquisition/acceptance API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
import numbers

import numpy as np

DATA = Path(__file__).parent / "fixtures/robust-c2-oracle-1004"
ELF_SHA = "bdf0b4a786ae344989ae98e7306a1eb4562df35ff0e4587c1aac333be2923751"
# SL3 is an algebra/order control only, never a new decoder request.
GEOMETRY = {2: (40, 348, 348, 1, 2, 1),
            3: (69, 580, 348, 1, 10, 4),
            4: (141, 1160, 696, 5, 10, 4)}
ORDER_HASHES = {
    2: "c5e38726a9247ae94b619c0c64762d706e62314f8ff9582aa9440e8a915c8b0c",
    3: "dcc57367f385712c313ff1474d1d76543c47a577cbc714fdd6b961c382eeebc7",
    4: "ab46bf3a863a8831fa5ad56ab8d57ea3b415af38139e8b3dccd1cb2355a8e621"}
SCATTER_HASHES = {
    348: "c4b4848d62531a72c066b8d11ec573803ecaa7971157993a991b8e1994b4f921",
    696: "23a551e708e65a527f2d8a4f87edfd62f2a2adb0964fc1c555dd94d217fa53c2"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " outside oracle domain")
    return value


def geometry(sl):
    sl = integer(sl, 2, 4, "speedlevel")
    return GEOMETRY[sl]


def freeze(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def bits(value):
    array = np.asarray(value)
    if array.ndim != 1 or not np.all((array == 0) | (array == 1)):
        raise ValueError("expected one-dimensional binary vector")
    return np.asarray(array, dtype=np.uint8)


def permutation(value, count):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "iu":
        raise ValueError("invalid order shape/type")
    if sorted(int(x) for x in array) != list(range(count)):
        raise ValueError("order must be a full bijection")
    return freeze(array, np.int64)


def native_array(name, count, expected):
    raw = (DATA / name).read_bytes()
    if len(raw) != 2 * count or digest(raw) != expected:
        raise ValueError("oracle native constant mismatch")
    return permutation(np.frombuffer(raw, dtype="<u2"), count)


def c2_order(sl):
    _, d, _, _, _, _ = geometry(sl)
    return native_array(f"c2-sl{sl}.u16le", d, ORDER_HASHES[sl])


def symbol_order(count):
    count = integer(count, 348, 696, "symbol count")
    if count not in SCATTER_HASHES:
        raise ValueError("unsupported scatter size")
    return native_array(f"scatter-{count}.u16le", count, SCATTER_HASHES[count])


def walk(count, stride):
    """Independent scalar form of measured in-range interleaver recurrence."""
    count = integer(count, 1, 100000, "count")
    stride = integer(stride, 1, count, "stride")
    start, current, output = 1, 0, []
    for _ in range(count):
        output.append(current)
        current += stride
        if current >= count:
            current, start = start, start + 1
    return permutation(output, count)


def crc_register(data):
    if not isinstance(data, bytes):
        raise ValueError("CRC input must be bytes")
    register = 65535
    for byte in data:
        for shift in range(8):
            feedback = (register & 1) ^ ((byte >> shift) & 1)
            register >>= 1
            if feedback:
                register ^= 33800
    return register


def crc_x25(data):
    return crc_register(data) ^ 65535


def whiten(data):
    """Bit-at-a-time PN31; full field, reset once, no padding/flush."""
    if not isinstance(data, bytes):
        raise ValueError("whitening input must be bytes")
    register, output = 12021965, bytearray()
    for byte in data:
        result = 0
        for shift in range(8):
            pn = register & 1
            result |= (((byte >> shift) & 1) ^ pn) << shift
            register >>= 1
            if pn:
                register ^= 1207959552
        output.append(result)
    return bytes(output)


def rsc_step(state, bit):
    state = integer(state, 0, 7, "state")
    bit = integer(bit, 0, 1, "bit")
    newest = (state >> 2) & 1
    middle = (state >> 1) & 1
    oldest = state & 1
    feedback = bit ^ middle ^ oldest
    return ((feedback << 2) | (state >> 1), feedback ^ newest ^ oldest)


def encode_component(inputs, initial=0):
    inputs = bits(inputs)
    state = integer(initial, 0, 7, "initial state")
    parity = []
    for bit in inputs:
        state, p = rsc_step(state, int(bit))
        parity.append(p)
    return freeze(parity, np.uint8), state


def terminate(state):
    state = integer(state, 0, 7, "state")
    suffix = []
    for _ in range(3):
        bit = ((state >> 1) & 1) ^ (state & 1)
        suffix.append(bit)
        state, _ = rsc_step(state, bit)
    if state != 0:
        raise AssertionError("independent termination recurrence")
    return freeze(suffix, np.uint8)


def dqpsk_labels(serialized):
    serialized = bits(serialized)
    if len(serialized) % 2:
        raise ValueError("DQPSK requires full bit pairs")
    labels = []
    for first, second in serialized.reshape(-1, 2):
        # The signs of Re+Im and Re-Im respectively encode the two bits.
        a, b = 1 - 2 * int(first), 1 - 2 * int(second)
        labels.append(complex((a + b) / 2, (a - b) / 2))
    return freeze(labels, np.complex128)


def integrate(labels, scatter, reference=1+0j):
    labels = np.asarray(labels, dtype=np.complex128)
    if labels.ndim != 1 or not np.all(np.isfinite(labels)):
        raise ValueError("invalid logical differentials")
    scatter = permutation(scatter, len(labels))
    reference = complex(reference)
    if not math.isfinite(reference.real) or not math.isfinite(reference.imag) or reference == 0:
        raise ValueError("finite nonzero reference required")
    physical = np.empty(len(labels), dtype=np.complex128)
    physical[scatter] = labels
    symbols, previous = [reference], reference
    for differential in physical:
        previous *= differential
        symbols.append(previous)
    return freeze(symbols, np.complex128)


@dataclass(frozen=True)
class Fixture:
    speedlevel: int
    status: int
    data: bytes = field(repr=False)
    wire: bytes = field(repr=False)
    whitened: bytes = field(repr=False)
    systematic: np.ndarray = field(repr=False)
    c1_full: np.ndarray = field(repr=False)
    c2_inputs: np.ndarray = field(repr=False)
    c2_full: np.ndarray = field(repr=False)
    c2_final: int
    serialized: np.ndarray = field(repr=False)
    logical_differentials: np.ndarray = field(repr=False)
    symbols: np.ndarray = field(repr=False)
    stage_hashes: dict


def verify_scalar_contract():
    """Check source-native RSC/CRC and separate scalar facts before encoding."""
    expected = {
        "native-parity.u8": "10b7493d27d2cd815189fa476dcc41fedc6ba0d571aadbf13869e258f0f48c9c",
        "native-transition.u8": "88c26abb7deb8625aeb7585dd94467765557193091a662ea49ba7201bf73d0c2",
        "native-flush.u8": "5d66fd45410bc1f032c3b4321a5e155e8442751c93f785a57b5d4d6463ce58e8",
        "native-crc.u16le": "638dbc5f735cced09b3d0d767a72a8800baeb1b3bb399a514af61b3ea7be8a01",
        "SCALAR-GOLDEN-FACTS.json": "79d07ac90aac1bc0ccc381962728c29c4f202ada127c3d3a019c4995d7b39d63"}
    records = {}
    for name, sha in expected.items():
        raw = (DATA / name).read_bytes()
        if digest(raw) != sha:
            raise ValueError("scalar oracle fact integrity mismatch")
        records[name] = raw
    parities, transitions, flushes = (records[name] for name in
        ("native-parity.u8", "native-transition.u8", "native-flush.u8"))
    for state in range(8):
        if int(terminate(state)[0]) != flushes[state]:
            raise ValueError("scalar flush contradicts native fact")
        for bit in range(2):
            nxt, parity = rsc_step(state, bit)
            if (nxt, parity) != (transitions[2*state+bit], parities[2*state+bit]):
                raise ValueError("scalar recurrence contradicts native fact")
    crc_table = np.frombuffer(records["native-crc.u16le"], dtype="<u2")
    for value in range(256):
        register = value
        for _ in range(8):
            register = (register >> 1) ^ (33800 if register & 1 else 0)
        if register != int(crc_table[value]):
            raise ValueError("scalar CRC contradicts native table")
    golden = json.loads(records["SCALAR-GOLDEN-FACTS.json"])
    if crc_x25(golden["crc_published_input"].encode("ascii")) != golden["crc_x25_check_decimal"]:
        raise ValueError("scalar CRC check mismatch")
    for record in golden["whitening"]:
        data = bytes((17*i+3)%256 for i in range(record["bytes"]))
        if digest(data) != record["input_sha256"] or digest(whiten(data)) != record["whitened_sha256"]:
            raise ValueError("scalar whitening contradicts separate golden fact")


def generate(sl, data, status, *, reference=1+0j):
    """Construct exact full-D fixtures, including SL3 order control only."""
    length, d, m, padding, period, phase = geometry(sl)
    if not isinstance(data, bytes) or len(data) != length:
        raise ValueError("wrong data length")
    status = integer(status, 0, 255, "status")
    verify_scalar_contract()
    protected = data + bytes([status])
    crc = crc_x25(protected)
    wire = protected + bytes([crc & 255, crc >> 8])
    white = whiten(wire)
    information = [(byte >> bit) & 1 for byte in white for bit in range(8)]
    prefix = information + [0] * padding
    _, before_flush = encode_component(prefix)
    systematic = freeze(prefix + list(terminate(before_flush)), np.uint8)
    if len(systematic) != d:
        raise AssertionError("full-D geometry")
    c1, final = encode_component(systematic)
    if final != 0:
        raise AssertionError("C1 must end at zero")
    order = c2_order(sl)
    c2_inputs = freeze([int(systematic[int(i)]) for i in order], np.uint8)
    c2, c2_final = encode_component(c2_inputs)
    serialized = freeze(list(systematic) + list(c1[0::period]) + list(c2[phase::period]), np.uint8)
    if len(serialized) != 2 * m:
        raise AssertionError("serialized geometry")
    labels = dqpsk_labels(serialized)
    symbols = integrate(labels, symbol_order(m), reference)
    stages = {"wire": digest(wire), "whitened": digest(white),
              "systematic": digest(systematic.tobytes()), "c1_full": digest(c1.tobytes()),
              "c2_inputs": digest(c2_inputs.tobytes()), "c2_full": digest(c2.tobytes()),
              "serialized": digest(serialized.tobytes()),
              "logical_differentials": digest(labels.astype("<c16").tobytes()),
              "symbols": digest(symbols.astype("<c16").tobytes())}
    return Fixture(sl, status, data, wire, white, systematic, c1, c2_inputs, c2,
                   c2_final, serialized, labels, symbols, stages)


def seeded_data(sl, seed):
    length, *_ = geometry(sl)
    seed = integer(seed, 0, 2**32-1, "seed")
    text = f"robust-c2-oracle-1004/sl={sl}/seed={seed}".encode("ascii")
    return hashlib.shake_256(text).digest(length)


def noisy_case(sl, seed, es_n0_db):
    """One draw from the predeclared grid; never redraw/select by recovery."""
    if sl not in (2, 4) or seed not in range(100300, 100316) or es_n0_db not in (24, 21, 18, 15, 12, 9):
        raise ValueError("outside frozen experiment matrix")
    fixture = generate(sl, seeded_data(sl, seed), seed % 256)
    clean = fixture.symbols / np.sqrt(np.mean(abs(fixture.symbols)**2))
    rng = np.random.Generator(np.random.PCG64(seed))
    draw = rng.standard_normal((len(clean), 2))
    n0 = 10.0 ** (-es_n0_db / 10.0)
    noise = math.sqrt(n0 / 2) * (draw[:, 0] + 1j * draw[:, 1])
    return fixture, freeze(clean + noise, np.complex128)


def logsum(values):
    values = list(values)
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


@dataclass(frozen=True)
class PathResult:
    posterior: np.ndarray = field(repr=False)
    extrinsic: np.ndarray = field(repr=False)
    admitted_paths: int
    initial_state: int | None
    final_state: int | None


def enumerate_trellis(systematic, parity, *, initial=0, final=0):
    """Exact exhaustive reference, intentionally short (0..10 positions).

    Enumerates concrete start states and input sequences. No forward/backward
    recurrence or target trellis arrays. Infinite posterior means definite
    admitted-path certainty, whereas no admitted path raises ValueError.
    """
    sys = np.asarray(systematic, dtype=np.float64)
    par = np.asarray(parity, dtype=np.float64)
    if sys.ndim != 1 or par.shape != sys.shape or len(sys) > 10 or not np.all(np.isfinite(sys)) or not np.all(np.isfinite(par)):
        raise ValueError("finite equal short score vectors required")
    for value in (initial, final):
        if value is not None:
            integer(value, 0, 7, "boundary")
    starts = range(8) if initial is None else [int(initial)]
    paths = []
    for start in starts:
        for inputs in itertools.product((0, 1), repeat=len(sys)):
            state, weight = start, 0.0
            for at, bit in enumerate(inputs):
                state, p = rsc_step(state, bit)
                weight += ((1 - 2 * bit) * float(sys[at]) + (1 - 2 * p) * float(par[at])) / 2
            if not math.isfinite(weight):
                raise ValueError("finite scores overflow path arithmetic")
            if final is None or state == final:
                paths.append((inputs, weight))
    if not paths:
        raise ValueError("no admitted constituent path")
    posterior = []
    for at in range(len(sys)):
        zero = logsum(weight for inputs, weight in paths if inputs[at] == 0)
        one = logsum(weight for inputs, weight in paths if inputs[at] == 1)
        score = zero - one
        if math.isfinite(zero) and math.isfinite(one) and not math.isfinite(score):
            raise ValueError("finite path masses overflow posterior arithmetic")
        posterior.append(score)
    posterior = freeze(posterior, np.float64)
    return PathResult(posterior, freeze(posterior - sys), len(paths), initial, final)
