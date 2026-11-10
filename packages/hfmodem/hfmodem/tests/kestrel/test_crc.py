# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CRC-16/GENIBUS: one implementation, checked against a generic CRC engine.

The modem used to carry three copies of this function — bit-serial in
``vara/vara_frames.py``, parameterised in each of the two receivers — plus
re-export shims in ``tx/``. Copies of a checksum are how a frame format quietly
forks, so the collapse is worth a guard: this asserts both that the surviving
implementation is right and that nothing has grown a second one.

The reference below is the general Rocksoft model — reflection, arbitrary width,
init and xorout — driven entirely from the ``CRCSpec`` record. It shares no code
with the specialised loop under test, which is the point of having it.
"""
from __future__ import annotations

import random

from hfmodem.kestrel.coding.crc import GENIBUS, CRCSpec, crc16_genibus

_FRAME_SIZES = (0, 1, 2, 46, 92, 899, 3257)     # empty, tiny, and real frame lengths


def _reflect(value: int, width: int) -> int:
    return int(f"{value:0{width}b}"[::-1], 2)


def _crc_reference(data: bytes, spec: CRCSpec) -> int:
    """Textbook parametric CRC: the Rocksoft model, no shortcuts."""
    mask = (1 << spec.width) - 1
    topbit = 1 << (spec.width - 1)
    reg = spec.init & mask
    for byte in data:
        reg ^= (_reflect(byte, 8) if spec.refin else byte) << (spec.width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ spec.poly) & mask if reg & topbit else (reg << 1) & mask
    if spec.refout:
        reg = _reflect(reg, spec.width)
    return reg ^ spec.xorout


def test_check_vector():
    """The catalogue's identifying vector, independent of any code in this tree."""
    assert crc16_genibus(b"123456789") == GENIBUS.check == 0xD64E


def test_matches_generic_reference():
    """The specialised loop against the parameterised engine, on random data."""
    rng = random.Random(0xD64E)
    for n in _FRAME_SIZES:
        for _ in range(25):
            data = bytes(rng.randrange(256) for _ in range(n))
            assert crc16_genibus(data) == _crc_reference(data, GENIBUS), data.hex()


def test_reference_is_general():
    """Guard the guard: the reference must also reproduce a reflected CRC, or it
    is only accidentally right about the one parameter set that matters here."""
    x25 = CRCSpec("CRC-16/X-25", 16, 0x1021, 0xFFFF, True, True, 0xFFFF, 0x906E)
    assert _crc_reference(b"123456789", x25) == x25.check


def test_one_definition():
    """Every framing module resolves to the same function object."""
    import hfmodem.kestrel.rx.varahf500 as rx500
    import hfmodem.kestrel.rx.varahf2300 as rx2300
    import hfmodem.kestrel.tx.varahf500_tx as tx500
    import hfmodem.kestrel.tx.varahf2300_tx as tx2300
    import hfmodem.kestrel.vara.vara_frames as vf

    for mod in (vf, rx500, rx2300, tx500, tx2300):
        assert mod.crc16_genibus is crc16_genibus, mod.__name__
