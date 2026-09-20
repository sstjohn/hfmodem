# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Release gates for explicit DATA profiles and connectionless objects."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from hfmodem.sabir.arq import ArqConfig, LinkModem, wire
from hfmodem.sabir.arq.profiles import EXTENDED, PROFILES
from hfmodem.sabir.frame.datagram import Fragment, Reassembler, fragment_object
from hfmodem.sabir.sim.m3 import run_pair
from hfmodem.sabir.phy.rate import FS


@pytest.mark.parametrize("name", EXTENDED)
def test_explicit_profiles_bidirectional(name):
    payload = np.random.default_rng(5).bytes(38 if name.startswith("narrow") else 400)
    result = run_pair(payload, None, 40, payload_b=b"reverse", cfg_kw={
        "advertise_gears": EXTENDED, "data_profile": name, "loading": False})
    assert result.ok
    assert name in result.gears


def test_narrow_ceiling_has_real_data_and_rejects_wide():
    result = run_pair(b"narrow bytes", None, 10, cfg_kw={
        "advertise_gears": EXTENDED, "bandwidth_hz": 500})
    assert result.ok and result.gears == ["narrow"]
    modem = LinkModem(ArqConfig(advertise_gears=EXTENDED, bandwidth_hz=500))
    assert modem.fsm.local_profiles() == [18, 19, 20]
    assert modem.fsm.rx_profile(4) is None
    with pytest.raises(ValueError):
        modem.send_datagram(b"no", "wide256")


def test_unadvertised_preference_falls_back_to_legacy():
    result = run_pair(b"legacy", None, 20, cfg_kw={"data_profile": "wide256"})
    assert result.ok and "wide256" not in result.gears


def test_absolute_profile_ids_ignore_receiver_order():
    a = LinkModem(ArqConfig(advertise_gears=("wide256", "doppler")))
    a.fsm.peer_profiles = [254, 16, 21]
    assert a.fsm.tx_gear("wide256") == 21
    assert a.fsm.rx_profile(21).name == "wide256"
    assert a.fsm.rx_profile(16).name == "doppler"
    assert a.fsm.rx_profile(17) is None  # not advertised
    assert a.fsm.rx_profile(254) is None


@pytest.mark.parametrize("names", [("not-real",), ("doppler", "doppler"), ("max",)])
def test_capability_claims_are_truthful(names):
    with pytest.raises(ValueError):
        LinkModem(ArqConfig(advertise_gears=names))


@pytest.mark.parametrize("name", ["narrow", "narrow2", "narrow4", "robust", "doppler", "wide256"])
def test_connectionless_body_with_no_handshake(name):
    a, b = LinkModem(ArqConfig()), LinkModem(ArqConfig())
    raw = fragment_object(b"payload", "LONG/CALLSIGN", message_id=bytes(16))[0].pack()
    a.send_datagram(raw, name)
    # Real-valued audio, independent receiver, no shared session or GEARSET.
    from hfmodem.sabir.phy.modem import analytic
    wave = a.outbox.pop()
    b.on_air(analytic(np.sqrt(2) * wave.real))
    assert b.datagrams == [raw]
    assert not b.outbox                         # reception never emits an ACK


def test_header_known_answer_and_version_rejection():
    h = wire.DatagramHeader(21, 250, 64, 10048, bytes.fromhex("0102030405060708"))
    raw = h.pack()
    assert len(raw) == 22
    assert raw[:20].hex() == "0b01001500fa4000274001020304050607080000"
    assert wire.Control.unpack(raw) == h
    from hfmodem.sabir.frame.codec import crc16
    for i in (1, 7, 18):
        bad = bytearray(raw)
        bad[i] = 2
        bad[-2:] = crc16(bad[:-2]).to_bytes(2, "big")
        assert wire.Control.unpack(bytes(bad)) is None


def test_object_repair_out_of_order_duplicates_and_integrity():
    data = np.random.default_rng(5).bytes(2000)
    records = fragment_object(data, "SENDER", fragment_bytes=200)
    r = Reassembler()
    result = None
    for f in reversed(records[1:]):            # erase fragment 0; parity repairs it
        got = r.accept(f.pack())
        if got:
            result = got
    assert result[1] == data
    assert all(r.accept(f.pack()) is None for f in records)
    bad = replace(records[0], data=bytes(200))
    r = Reassembler()
    with pytest.raises(ValueError, match="integrity"):
        for f in [bad, *records[1:-1]]:
            r.accept(f.pack())


def test_object_does_not_deliver_with_two_erasures():
    records = fragment_object(b"abcdef" * 100, "SENDER", fragment_bytes=100)
    r = Reassembler()
    assert all(r.accept(f.pack()) is None for f in records[2:])


def test_reassembly_bounds_expiry_and_conflicting_metadata():
    now = [0.0]
    r = Reassembler(max_bytes=500, max_objects=2, ttl=10, clock=lambda: now[0])
    first = fragment_object(bytes(400), "S", fragment_bytes=100)[0]
    assert r.accept(first.pack()) is None
    with pytest.raises(ValueError, match="conflicting"):
        r.accept(replace(first, digest=sha256(b"different").digest()).pack())
    for _ in range(4):
        r.accept(fragment_object(bytes(400), "S", fragment_bytes=100)[0].pack())
    assert len(r.pending) <= 2
    now[0] = 11
    r.accept(first.pack())
    assert len(r.pending) == 1


def test_malformed_lengths_rejected_before_decoder(monkeypatch):
    m = LinkModem(ArqConfig())
    def forbidden(*args, **kwargs):
        pytest.fail("header geometry reached DSP")
    monkeypatch.setattr(m._phy("workhorse"), "receive", forbidden)
    with pytest.raises(ValueError, match="geometry"):
        m.decode_body(PROFILES["workhorse"], np.zeros(1), 65535, 1)


def test_wide256_full_frame_decodes_and_exceeds_vara_cycle_target():
    a = LinkModem(ArqConfig())
    profile = PROFILES["wide256"]
    codec = a.fsm.profile_codec(profile.name)
    payload = np.random.default_rng(123).bytes(profile.frame_cws * codec.data_bytes)
    coded = np.stack([codec.encode_cw(ch) for ch in codec.chunk(payload)])
    body, ns = a.encode_body(profile, coded)
    freq = np.fft.fftfreq(len(body), 1 / FS)
    order = np.argsort(freq)
    power = np.abs(np.fft.fft(body))[order] ** 2
    lo, hi = np.interp([.005, .995], np.cumsum(power) / power.sum(), freq[order])
    assert hi - lo < 2800
    from hfmodem.sabir.sim.m2 import add_noise_snr3k
    body = add_noise_snr3k(body, 40, np.random.default_rng(3))
    llrs, _ = a.decode_body(profile, body, ns, profile.frame_cws)
    chunks, _ = codec.decode_cws(llrs)
    assert b"".join(chunks) == payload
    cycle = len(body) / FS + 2 * a.fast.n_samples(1) / FS + 0.5
    assert len(payload) * 8 / cycle > 10000


def test_frame_size_is_not_frozen_to_transmitter_default():
    m = LinkModem(ArqConfig())
    p = PROFILES["workhorse"]
    c = m.fsm.profile_codec(p.name)
    coded = np.stack([c.encode_cw(bytes([i]) * c.data_bytes) for i in range(13)])
    body, ns = m.encode_body(p, coded)
    llr, _ = m.decode_body(p, body, ns, 13)
    chunks, _ = c.decode_cws(llr)
    assert chunks == [bytes([i]) * c.data_bytes for i in range(13)]


def test_receiver_refinements_work_through_session_endpoint():
    result = run_pair(np.random.default_rng(77).bytes(1500), "poor", 15,
                      cfg_kw={"feedback_iters": 2, "impulse_blank": 3.5})
    assert result.ok


def test_fragment_unknown_version_never_delivered():
    raw = bytearray(fragment_object(b"x", "S")[0].pack())
    raw[0] = 2
    with pytest.raises(ValueError):
        Fragment.unpack(bytes(raw))


@pytest.mark.parametrize("compressed", [False, True])
def test_hashed_stream_records_detect_corruption_before_delivery(compressed):
    from hfmodem.sabir import compress
    payload = b"record integrity " * 300
    raw = compress.pack(payload, compressed, integrity=True)
    reader = compress.Unpacker()
    assert list(reader.feed(raw[:11])) == []
    assert list(reader.feed(raw[11:])) == [payload]
    corrupt = bytearray(raw)
    corrupt[5] ^= 1
    with pytest.raises(ValueError, match="integrity"):
        list(compress.Unpacker().feed(corrupt))




def test_session_identifiers_are_not_silently_truncated():
    with pytest.raises(ValueError):
        wire.station_bytes("LONG-CALLSIGN-A")
    with pytest.raises(ValueError):
        wire.station_bytes("LONG-CALLSIGN-B")


def test_propagation_report_and_compact_power():
    from hfmodem.sabir.frame.reporting import PropagationReport, beacon_power_status, beacon_status_power
    report = PropagationReport(1788888000000, 14095600, 3700, "EN63AB")
    assert PropagationReport.unpack(report.pack()) == report
    assert beacon_status_power(beacon_power_status(37)) == 37
    with pytest.raises(ValueError):
        beacon_power_status(99)


def test_continuous_channel_does_not_restart_at_chunk_boundaries():
    from hfmodem.sabir.sim.watterson import ContinuousWatterson
    channel = ContinuousWatterson("good", FS, duration_s=10, seed=4)
    x = np.ones(48000, dtype=complex)
    # The delayed path sees silence before each supplied burst. Compare away
    # from that boundary so this tests fading continuity, not delay history.
    whole = channel(x, 1.0)
    part = channel(x[24000:], 1.5)
    np.testing.assert_allclose(part[100:], whole[24100:], atol=1e-12)


def test_host_object_delivery_and_queue_bound():
    from hfmodem.sabir.sim.air import SimulatedAir
    from hfmodem.sabir.host.modem_core import SabirModem, ModemObserver

    class Observer(ModemObserver):
        def __init__(self): self.objects = []
        def modem_connected(self, *a): pass
        def modem_disconnected(self, *a): pass
        def modem_ptt(self, *a): pass
        def modem_buffer(self, *a): pass
        def modem_data_received(self, *a): pass
        def modem_object_received(self, image): self.objects.append(image)

    air = SimulatedAir()
    a = SabirModem(air, ArqConfig(callsign="ALICE"))
    b = SabirModem(air, ArqConfig(callsign="BOB"))
    observer = Observer()
    b.start(observer)
    payload = np.random.default_rng(9).bytes(2100)
    a.send_object(payload, data_profile="wide256", repeats=2)
    with pytest.raises(ValueError, match="idle"):
        a.send_object(b"another")
    while not air._cmds.empty():
        air._cmds.get_nowait()()
    a.after_step()
    assert len(a.link.outbox) == 1               # only one rendered packet
    air._move()
    assert len(observer.objects) == 1
    assert observer.objects[0]["data"] == payload
    assert not b.link.outbox
