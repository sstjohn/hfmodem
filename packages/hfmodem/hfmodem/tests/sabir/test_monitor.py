# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The receive-only monitor: a synthetic channel of several sabir bursts
separated by noise -> segment -> decode -> a timestamped, station-named log."""

import numpy as np

from hfmodem.sabir import monitor
from hfmodem.sabir import offair
from hfmodem.sabir.arq import wire
from hfmodem.sabir.floor.mfsk import FloorModem

_TAG = 0x81
_CAP = wire.capabilities(range(3, 7 + 1), wire.FASTCTL | wire.DEFLATE)


def _burst(block_bytes):
    return offair.to_real(FloorModem().transmit(block_bytes))


def _channel(rng, snr_db=15.0):
    """beacon, CONNECT, CONNECT_ACK, ID -- keyed bursts with noise gaps."""
    bursts = [
        _burst(wire.Beacon.build(_CAP, "W1AW", profile=1).pack()),
        _burst(wire.Control.connect(_TAG, _CAP, "W1AW", destination="K6XYZ").pack()),
        _burst(wire.Control.connect(_TAG, _CAP, "K6XYZ", ack=True, destination="W1AW").pack()),
        _burst(wire.Control.ident(wire.ID, _TAG, "K6XYZ").pack()),
    ]
    ref = float(np.mean(bursts[0][np.abs(bursts[0]) > 1e-6] ** 2))
    gap = np.zeros(int(0.8 * monitor.FS))
    chan = gap.copy()
    for b in bursts:
        chan = np.concatenate([chan, b, gap])
    chan = chan + rng.standard_normal(chan.size) * np.sqrt(ref / 10 ** (snr_db / 10))
    return chan


def test_segmenter_finds_each_burst():
    chan = _channel(np.random.default_rng(0))
    segs = list(monitor.segment(chan))
    assert len(segs) == 4                              # four keyed bursts
    starts = [s0 for s0, _ in segs]
    assert starts == sorted(starts)                    # in order


def test_monitor_decodes_and_names_the_traffic():
    chan = _channel(np.random.default_rng(1))
    events = list(monitor.decode_events(chan))
    got = [(e.kind, e.text.split()[0].strip("'")) for e in events]
    assert ("BEACON", "presence") in got
    assert ("CONNECT", "W1AW") in got
    assert ("CONNECT_ACK", "K6XYZ") in got
    assert ("ID", "K6XYZ") in got
    assert [e.t for e in events] == sorted(e.t for e in events)   # timestamps rise


def test_monitor_reports_capability_image():
    chan = _channel(np.random.default_rng(2))
    conn = [e for e in monitor.decode_events(chan) if e.kind == "CONNECT"][0]
    assert "profiles [3, 4, 5, 6, 7]" in conn.text     # exact advertised permissions
    assert "fastctl" in conn.text and "deflate" in conn.text


def test_monitor_silent_on_pure_noise():
    noise = np.random.default_rng(3).standard_normal(int(5 * monitor.FS)) * 0.1
    assert list(monitor.decode_events(noise)) == []


# -- regressions for the adversarial-review findings --------------------------
def _inband_channel(rng, inband_db, call="W1AW"):
    """One floor burst in noise whose IN-BAND (700-2700 Hz) SNR is inband_db --
    the figure the decoder actually sees, not a broadband ratio."""
    b = _burst(wire.Beacon.build(_CAP, call, profile=1).pack())
    ref = float(np.mean(b[np.abs(b) > 1e-6] ** 2))
    g = np.zeros(int(0.8 * monitor.FS))
    x = np.concatenate([g, b, g])
    bw_frac = (monitor.BAND_HI - monitor.BAND_LO) / (monitor.FS / 2)
    npow = ref / 10 ** (inband_db / 10) / bw_frac
    return x + rng.standard_normal(x.size) * np.sqrt(npow)


def test_noise_floor_survives_silence_frames():
    """Squelch tails / T-R mute / dropouts (zeroed frames) must not collapse the
    floor into a runaway of 30 s junk bursts. (The min-floor bug did exactly
    that; a rolling percentile does not.)"""
    rng = np.random.default_rng(10)
    noise = rng.standard_normal(int(20 * monitor.FS)) * 0.03
    z = int(0.04 * monitor.FS)
    noise[z:z + 2 * monitor.SEG_FRAME] = 0.0
    assert list(monitor.segment(noise)) == []


def test_fast_tier_decodes_through_the_monitor():
    """A fast coherent control burst (the tier a live link runs ACKs/DATA on)
    decodes through the monitor's pre-roll offset, not just at sample 0."""
    from hfmodem.sabir.arq.fsm import ArqConfig
    from hfmodem.sabir.arq.modem import LinkModem
    m = LinkModem(ArqConfig())
    ack = wire.Control(wire.ACK, 0x42, seq=3, mask=wire.cw_mask([0, 5]),
                       aux=wire.ack_aux([10.0] * 8))
    z = offair.to_analytic(np.concatenate(
        [np.zeros(monitor.SEG_PAD * monitor.SEG_FRAME),
         offair.to_real(m.fast.transmit(ack.pack())), np.zeros(monitor.SEG_FRAME)]))
    blocks, fast = monitor._decode_burst(m, z)
    assert fast and blocks and blocks[0].type == wire.ACK


def test_gate_catches_bursts_the_decoder_decodes():
    """The gate must open near the decoder's in-band working range, not tens of
    dB above it. At +6 dB in-band the floor decodes; the gate must see it."""
    rng = np.random.default_rng(11)
    got = [(e.kind, e.text.split()[0].strip("'"))
           for e in monitor.decode_events(_inband_channel(rng, 6.0))]
    assert ("BEACON", "presence") in got


def test_raw_chunks_odd_byte_tail_no_crash():
    import io
    pcm = np.arange(9601, dtype="<i2").tobytes()          # 9601 samples' worth+1B
    list(monitor.raw_chunks(io.BytesIO(pcm)))             # must not raise


def test_timestamps_land_near_the_burst():
    """Event times track the real burst placement, within the pre-roll, not just
    monotone."""
    chan = _channel(np.random.default_rng(12))
    events = list(monitor.decode_events(chan))
    # bursts are placed at 0.8 + k*(4.5+0.8) s; the beacon opens the log ~0.8 s
    assert 0.5 < events[0].t < 1.1


def test_live_stream_matches_batch():
    """The streaming Segmenter decodes a chunked stream identically to the
    whole-capture batch path -- and with arbitrary, frame-straddling chunk
    sizes, so this is a real check on the streaming buffer bookkeeping, not a
    same-chunking tautology."""
    chan = _channel(np.random.default_rng(6))
    batch = [(e.kind, e.text) for e in monitor.decode_events(chan)]
    chunks = (chan[i:i + 997]                          # odd, != SEG_FRAME/CHUNK
              for i in range(0, len(chan), 997))
    live = [(e.kind, e.text) for e in monitor.decode_stream(chunks)]
    assert live == batch and len(batch) >= 4


def test_stdin_s16le_pipe():
    """Raw signed-16-bit mono 48 kHz on a stream (the --stdin path) decodes."""
    import io
    chan = _channel(np.random.default_rng(7))
    pcm = (np.clip(chan / np.abs(chan).max() * 0.9, -1, 1) * 32767).astype("<i2")
    events = list(monitor.decode_stream(monitor.raw_chunks(io.BytesIO(pcm.tobytes()))))
    kinds = {e.kind for e in events}
    assert "BEACON" in kinds and "CONNECT" in kinds
