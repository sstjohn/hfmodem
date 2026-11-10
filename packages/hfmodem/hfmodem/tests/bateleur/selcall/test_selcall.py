# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Selcall (CCIR 493-4 land mobile, UN open 6-digit): the call format
against the published layouts, and the modem through noise, fading, a
far-off carrier and a wrong clock."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample_poly

from hfmodem.bateleur import sim
from hfmodem.bateleur.selcall import code as C, fsk


def _air(call, *, snr=None, preset="awgn", freq_hz=0.0, seed=0, **kw):
    x = np.r_[np.zeros(3000), fsk.call_audio(call, **kw), np.zeros(3000)]
    if snr is None:
        return x
    # bateleur.sim runs at 9600 Hz.
    y = sim.channel(resample_poly(x, 6, 5), snr_db=snr, preset=preset, freq_hz=freq_hz,
                    seed=seed)
    return resample_poly(y, 5, 6)


def _calls(audio):
    return [h.call for h in fsk.receive(audio) if h.call is not None]


# -- the characters -------------------------------------------------------------

def test_the_check_bits_count_the_zeros_most_significant_first():
    """M.493 1.1.1: seven information bits least significant first, then the
    number of B (0) elements among them, most significant first."""
    assert C.char_bits(125) == [1, 0, 1, 1, 1, 1, 1] + [0, 0, 1]
    assert C.char_bits(0) == [0] * 7 + [1, 1, 1]
    for v in range(128):
        assert C.char_value(C.char_bits(v)) == (v, True)
    b = C.char_bits(104)
    b[3] ^= 1
    assert not C.char_value(b)[1]


def test_the_4_digit_call_is_hflinks_layout():
    """1234 calling 5678: format 120, the address in pairs, routine, the
    caller, three acknowledgment requests."""
    assert C.encode(C.Call("selcall", "5678", "1234")) == \
        [120, 56, 78, 100, 12, 34, 117, 117, 117]


def test_the_rx_position_repeats_each_character_four_later():
    dx = C.encode(C.Call("selcall", "5678", "1234"))
    air = C.air_chars(dx)
    assert air[:12] == [125, 109, 125, 108, 125, 107, 125, 106, 125, 105, 125, 104]
    body = air[12:]
    assert body[0::2] == dx and body[1::2] == [120, 120] + dx[:-2]


def test_a_6_digit_emergency_with_a_position():
    """The UN specification's example: to all (000000) from 123456, at
    32 05.714 S 115 48.036 E; the position as degrees, minutes and
    thousandths in digit pairs; its parity over the status byte and them."""
    c = C.Call("emergency", "000000", "123456", position=(-(32 + 5.714 / 60), 115 + 48.036 / 60),
               status=0)
    dx = C.encode(c)
    assert dx[:9] == [120, 0, 0, 0, 100, 12, 34, 56, 122]
    assert dx[9] == 2                                  # east, south
    assert dx[10:18] == [3, 20, 57, 14, 11, 54, 80, 36]
    assert dx[18] == 107 and dx[19:] == [117] * 3
    back = C.decode(dx)
    assert back.kind == "emergency" and back.parity_ok
    assert back.position == pytest.approx(c.position, abs=1e-5)


def test_a_6_digit_call_drawn_with_126_reads_too():
    dx = C.encode(C.Call("selcall", "567890", "123456"))
    dx[0] = 126
    c = C.decode(dx)
    assert (c.kind, c.to, c.frm) == ("selcall", "567890", "123456")


@pytest.mark.parametrize("call", [
    C.Call("selcall", "5678", "1234"),
    C.Call("beacon", "567890", "123456"),
    C.Call("hangup", "5678", "1234"),
    C.Call("telcall", "567890", "123456", number="0061732100000"),
    C.Call("page", "567890", "123456", text="RV AT GRID 7 1400Z"),
    C.Call("gps", "5678", "1234", position=(43.0731, -89.4012)),
    C.Call("gps_request", "567890", "123456"),
    C.Call("status_request", "567890", "123456"),
    C.Call("emergency", "000000", "123456"),
    C.Call("ack", "1234", "5678"),
])
def test_every_kind_is_read_as_it_was_written(call):
    back = C.decode(C.encode(call))
    assert (back.kind, back.to, back.frm) == (call.kind, call.to, call.frm)
    assert back.number == call.number and back.text == call.text and back.parity_ok
    if call.position:
        assert back.position == pytest.approx(call.position, abs=1e-5)


def test_zeros_at_the_end_of_an_address_call_more_than_one():
    """X000 all, XX00 a group, XXX0 a sub-group; six digits the same."""
    assert C.reaches("3332", "3332") and not C.reaches("3331", "3332")
    assert C.reaches("3330", "3332") and C.reaches("3300", "3332") and C.reaches("3000", "3332")
    assert not C.reaches("3100", "3332") and not C.reaches("333200", "3332")
    assert C.reaches("123000", "123456") and C.reaches("123450", "123456")
    assert C.reaches("000000", "123456") and C.reaches("0000", "3332")


# -- the modem ---------------------------------------------------------------------

def test_a_call_through_the_modem():
    assert [(c.kind, c.to, c.frm) for c in _calls(_air(C.Call("selcall", "5678", "1234")))] == \
        [("selcall", "5678", "1234")]


def test_the_tones_are_1700_for_0_and_1870_for_1():
    x = fsk.modulate([], dots_s=0.1)[:fsk.SPB * 2]
    for bit, f in ((0, 1700), (1, 1870)):
        seg = x[bit * fsk.SPB:(bit + 1) * fsk.SPB]
        spec = np.abs(np.fft.rfft(seg, 8000))
        assert abs(np.argmax(spec) - f) <= 10


def test_an_inverted_sideband_is_read_and_said():
    hs = fsk.receive(_air(C.Call("page", "567890", "123456", text="QRV"), invert=True))
    assert hs[0].inverted and hs[0].call.text == "QRV"


@pytest.mark.parametrize("ratio", [98, 102])
def test_a_clock_two_percent_out(ratio):
    x = _air(C.Call("page", "567890", "123456", text="HELLO FROM THE FIELD"))
    assert [c.text for c in _calls(resample_poly(x, ratio, 100))] == ["HELLO FROM THE FIELD"]


def test_a_far_off_carrier_and_noise():
    """35 Hz off at 0 dB in 3 kHz."""
    call = C.Call("selcall", "567890", "123456")
    got = sum(any(c.to == "567890" for c in _calls(_air(call, snr=0.0, freq_hz=35.0, seed=s)))
              for s in range(6))
    assert got == 6


def test_the_poor_channel():
    """ITU-R poor (2 ms, 1 Hz) at 15 dB: each tone levelled on its own."""
    call = C.Call("selcall", "567890", "123456")
    got = sum(any(c.to == "567890" for c in _calls(_air(call, snr=15.0, preset="poor", seed=s)))
              for s in range(8))
    assert got >= 7


def test_calls_back_to_back_streamed_in_odd_chunks():
    a = _air(C.Call("selcall", "5678", "1234"))
    b = _air(C.Call("hangup", "5678", "1234"), dots_s=0.5)
    x = np.r_[a, b]
    r = fsk.Receiver()
    got = []
    for i in range(0, x.size, 777):
        got += [h.call.kind for h in r.push(x[i:i + 777]) if h.call]
    got += [h.call.kind for h in r.flush() if h.call]
    assert got == ["selcall", "hangup"]


def test_noise_alone_makes_no_call():
    rng = np.random.default_rng(1)
    assert _calls(rng.normal(0, 0.3, 8000 * 60)) == []


# -- on a station --------------------------------------------------------------------

def _modem_pair(**kw):
    """Two selcall modems joined by a wire: what one sends, the other hears."""
    from hfmodem.bateleur.selcall.modem import SelcallModem
    wire = []

    def sender(i):
        def send(audio, rate):
            wire.append((i, audio))
            return 0
        return send
    a = SelcallModem(["1234", "123456"], send=sender(0), card_rate=8000, dots_s=1.0, **kw)
    b = SelcallModem("5678", send=sender(1), card_rate=8000, dots_s=1.0, **kw)
    return a, b, wire


def _deliver(wire, to, at=10_000):
    """What is on the wire now, to ``to``; what it sends back stays on it."""
    now = list(wire)
    del wire[:len(now)]
    for _, audio in now:
        x = np.r_[np.zeros(4000), audio, np.zeros(8000)]
        for i in range(0, x.size, 4000):
            to._audio(at + i, x[i:i + 4000])


class _Log:
    def __init__(self):
        self.events = []

    def event(self, ev):
        self.events.append(ev)


def test_a_modem_reports_a_call_for_it_and_one_for_others():
    a, b, wire = _modem_pair()
    b.observer = log = _Log()
    a.call("selcall", "5678")
    a.call("selcall", "5600")                       # a group b is in
    a.call("selcall", "9999")
    _deliver(wire, b)
    assert [(e.kind, e.call.to, e.call.frm) for e in log.events] == \
        [("CALLED", "5678", "1234"), ("CALLED", "5600", "1234"), ("HEARD", "9999", "1234")]


def test_a_call_goes_from_the_address_as_long_as_the_one_called():
    a, b, wire = _modem_pair()
    assert a.call("selcall", "567890").frm == "123456"
    assert a.call("selcall", "5678").frm == "1234"
    with pytest.raises(ValueError):
        b.call("selcall", "567890")


def test_an_acknowledgment_when_asked_for():
    a, b, wire = _modem_pair(acknowledge=True)
    a.call("selcall", "5678")
    _deliver(wire, b)
    assert [i for i, _ in wire] == [1]
    a.observer = log = _Log()
    _deliver(wire, a)
    assert [(e.kind, e.call.kind, e.call.frm) for e in log.events] == [("CALLED", "ack", "5678")]


def test_the_host_dialect():
    from hfmodem.bateleur.selcall.host import SelcallHost
    a, b, wire = _modem_pair()
    host = SelcallHost(a, port=0, emergency=True)
    a.start()
    try:
        assert host.command("PAGE 567890 meet at the bridge") == "OK PAGE 567890"
        assert host.command("GPS 5678 -32.0952 115.8006") == "OK GPS 5678"
        assert host.command("EMERGENCY") == "OK EMERGENCY 0000"
        assert host.command("CALL 12345").startswith("ERR")
        assert host.command("STATUS").startswith("OK STATUS ADDRESS 1234,123456")
    finally:
        a.stop()
    assert list(host.log)[:3] == ["SENT PAGE TO 567890", "SENT GPS TO 5678", "SENT EMERGENCY TO 0000"]
    hb = SelcallHost(b, port=0)
    _deliver(wire, b)
    assert hb.log[1] == "CALL GPS FROM 1234 TO 5678 MINE POSITION -32.09520 115.80060"
    assert hb.log[0].startswith("CALL PAGE FROM 123456 TO 567890 TEXT MEET AT THE BRIDGE")


def test_the_station_builds_it_and_checks_its_config():
    import tomllib
    from dataclasses import replace
    from hfmodem.core import config
    from hfmodem.station.link import SelcallLink
    from hfmodem.tests.core.test_link import _repo
    base = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    good = {"enabled": True, "host": "selcall", "port": 8340, "addresses": ["1234", "123456"]}
    cfg = config.parse(dict(base, protocols={"selcall": good}))
    assert cfg.protocols["selcall"].dots_s == 6.0
    for bad in (["12345"], ["1234", "5678"], []):
        with pytest.raises(config.ConfigError, match="addresses"):
            config.parse(dict(base, protocols={"selcall": dict(good, addresses=bad)}))
    with pytest.raises(config.ConfigError, match="dots_s"):
        config.parse(dict(base, protocols={"selcall": dict(good, dots_s=60)}))

    class _St:
        pass
    st = _St()
    st.cfg = replace(cfg, station=replace(cfg.station, transmit=False))
    st.rig = st.audio = st.arbiter = None
    link = SelcallLink(st)
    assert link.modem.addresses == ["1234", "123456"]
    from hfmodem.core.occupied import keyed_hz
    lo, hi = keyed_hz("selcall", "")
    assert lo < 1600 and hi > 1950
