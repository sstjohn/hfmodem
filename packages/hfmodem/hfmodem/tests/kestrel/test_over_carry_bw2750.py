# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What `_OVER_CARRY` already buys, pinned so it cannot be given back.

THESE TESTS PASS AT THE COMMIT BEFORE THEM AND THAT IS THE POINT. The carry was
sized to hold frame + `_PROBE_COLS` columns of the longest index record of EITHER
bandwidth in `72372901`; nothing here changes it. What was missing was anything
that would notice if it were narrowed again, and the value is easy to narrow by
accident — it reads as a buffer length rather than as the thing a probe depends
on.

`_window_state` probes `_PROBE_COLS` columns behind a decoded frame to decide
whether the peer is still keying. If the carry cannot hold frame + `_PROBE_COLS`
columns of the session's longest record, the probe never fills, `_window_state`
returns `_UNKNOWN` forever, and the buffer trim walks past the onset — so NS0A's
record-101 over (19-20/24 reference columns, CRC clean, whole in an 11 s clear
window) would never be named. The last test drives that failure deliberately, by
putting the pre-`72372901` formula back.
"""
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.vara import vara_arq as VA, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora


def _every_index_record_probe_fits(family):
    cap = VA._OVER_CARRY - 1
    for lv in family:
        r = rx.RECORDS[lv]
        yield lv, cap >= r.ncols * r.dw50 + VA._PROBE_COLS * r.dw50


def test_carry_holds_frame_plus_probe_for_every_bw2750_record():
    fits = dict(_every_index_record_probe_fits(rx.INDEX_LEVELS_BW["2750"]))
    assert all(fits.values()), fits
    # record 101 is the case that failed: 228 columns of 1024 samples.
    r101 = rx.RECORDS[101]
    assert (VA._OVER_CARRY - 1 - r101.ncols * r101.dw50) // r101.dw50 >= VA._PROBE_COLS


def test_base_record_overs_are_unaffected():
    # Record 103 (dw50 512) still sits whole in the carry with room for the probe.
    r103 = rx.RECORDS[103]
    assert VA._OVER_CARRY - 1 >= (r103.ncols + VA._PROBE_COLS) * r103.dw50
    # And the BW2300 base (record 3) is untouched.
    assert all(dict(_every_index_record_probe_fits(rx.INDEX_LEVELS_BW["2300"])).values())


class _IO(VA.VaraIO):
    def __init__(self):
        self.payloads, self.keys, self.logs = [], [], []
    def key(self, on): self.keys.append(on)
    def tx(self, samples): ...
    def pending(self): ...
    def data(self, p): self.payloads.append(bytes(p))
    def connected(self, *a): ...
    def log(self, m): self.logs.append(m)


def _driven(x, carry=None):
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2750")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "NS0A"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    a, b = int(85.19 * MK.FS), int(96.06 * MK.FS)
    saved = VA._OVER_CARRY
    if carry is not None:
        VA._OVER_CARRY = carry
    try:
        for at in range(a, b, 960):
            hs._stream_over(x[at:min(at + 960, b)])
    finally:
        VA._OVER_CARRY = saved
    return io


@corpora.requires_onair_ns0a_record101
def test_ns0a_record101_over_is_named_and_delivered():
    x = corpora.wav_mono(corpora.ONAIR_NS0A_RECORD101_OVER)
    io = _driven(x)
    assert io.payloads, "the record-101 over was never named"
    assert any("record 101" in m and "DATA over" in m for m in io.logs)


@corpora.requires_onair_ns0a_record101
def test_narrowing_the_carry_to_the_old_formula_deadlocks_on_the_same_over():
    # The carry before `72372901` was the BW2300 frame max with no probe columns;
    # under it the record-101 probe can never fill and the over is lost. This is
    # what the two invariants above are guarding against.
    x = corpora.wav_mono(corpora.ONAIR_NS0A_RECORD101_OVER)
    old = max(rx.RECORDS[lv].ncols * rx.RECORDS[lv].dw50
              for lv in rx.INDEX_LEVELS)
    io = _driven(x, carry=old)
    assert not io.payloads, "the old carry unexpectedly resolved the over"
