# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M4 acceptance: application <-> application through the TCP host API.

Two sabir host servers (one framed CBOR TCP port each) sit on two
sample-level modems sharing a `sim.air.SimulatedAir`; host clients drive real
sessions -- connect, bidirectional multi-KB transfer, compression on/off,
disconnect -- with every payload byte crossing the impaired channel as
modulated audio. ``python -m hfmodem.sabir.sim.m4`` runs the full acceptance sweeps
-> ``tests/sabir/m4_results.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hfmodem.sabir.arq import ArqConfig
from hfmodem.sabir.host import SabirModem, HostClient
from hfmodem.sabir.host import messages as M
from hfmodem.sabir.host.hostlink import HostLinkServer
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.air import SimulatedAir
from hfmodem.sabir.sim.m2 import add_noise_snr3k
from hfmodem.sabir.sim.watterson import Watterson


def hf_channel(profile: str | None, snr_db: float, seed: int = 0):
    """Fresh Watterson realisation per transmission + AWGN at snr_db."""
    rng = np.random.default_rng(seed)

    def channel(wav, t):
        y = Watterson(profile, FS, rng)(wav) if profile else wav
        return add_noise_snr3k(y, snr_db, rng)
    return channel


@dataclass
class HostedPair:
    air: SimulatedAir
    modems: list
    servers: list

    def client(self, i: int, mycall: str) -> HostClient:
        s = self.servers[i]
        return HostClient(s.port, mycall)

    def stop(self) -> None:
        for s in self.servers:
            s.stop()
        self.air.stop()


def make_pair(channel=None, cfg_kw: dict | None = None,
              ports=(0, 0), log=None, realtime: bool = False) -> HostedPair:
    air = SimulatedAir(channel, realtime=realtime)
    kw = cfg_kw or {}
    modems = [SabirModem(air, ArqConfig(**kw)) for _ in range(2)]
    servers = [HostLinkServer(lambda m=m: m, port=ports[i], log=log)
               for i, m in enumerate(modems)]
    for s in servers:
        s.start_background()
    air.start()
    return HostedPair(air, modems, servers)


def email_text(n_bytes: int, seed: int = 0) -> bytes:
    """A plausible email-shaped text payload (headers, prose, a quote)."""
    rng = np.random.default_rng(seed)
    words = ("the antenna held through the storm and the noise floor "
             "dropped after sunset so the net moved traffic for three "
             "hours with good copy on every station that checked in "
             "please acknowledge receipt of the supply manifest and "
             "confirm the schedule for tomorrow morning").split()
    out = ["Message-ID: <2026071912345.SABIR@winlink.org>",
           "From: alice@winlink.org", "To: bob@winlink.org",
           "Subject: field report and manifest", "", ""]
    while sum(len(s) + 1 for s in out) < n_bytes:
        line = " ".join(rng.choice(words, size=9))
        out.append(line)
        out.append("> " + line)
    return "\n".join(out).encode()[:n_bytes]


def round_trip(payload_a: bytes, payload_b: bytes, profile: str | None,
               snr_db: float, compression: bool = False, seed: int = 0,
               timeout: float = 300.0) -> dict:
    """Full app-to-app session; returns byte-exactness + link metrics."""
    pair = make_pair(hf_channel(profile, snr_db, seed))
    a = b = None
    try:
        b = pair.client(1, "BOB")
        b.set_compression(compression)
        b.listen(True)
        a = pair.client(0, "ALICE")
        a.set_compression(compression)
        assert a.connect("BOB", timeout=timeout), "no CONNECTED at A"
        assert b.wait_connected(timeout=timeout), "no CONNECTED at B"
        a.send(payload_a)
        if payload_b:
            b.send(payload_b)
        got_b = b.recv(len(payload_a), timeout=timeout)
        got_a = a.recv(len(payload_b), timeout=timeout) if payload_b else b""
        a.flush(timeout=timeout)
        assert a.disconnect(timeout=timeout), "no DISCONNECTED at A"
        b.wait_disconnected(timeout=timeout)
        ma, mb = pair.modems
        return {
            "ok_ab": got_b == payload_a, "ok_ba": got_a == payload_b,
            "wall_s": pair.air.now,
            "airtime_a": ma.link.airtime_s, "airtime_b": mb.link.airtime_s,
            "link_bytes_a": ma.link_bytes_tx, "link_bytes_b": mb.link_bytes_tx,
            "frames_a": [(f["rung"], f["tx_count"])
                         for f in ma.fsm.stats["frames"]],
            "frames_b": [(f["rung"], f["tx_count"])
                         for f in mb.fsm.stats["frames"]],
            "msgs_a": list(a.messages), "msgs_b": list(b.messages),
        }
    finally:
        for c in (a, b):
            if c is not None:
                c.close()
        pair.stop()


def main():
    lines = ["sabir M4 acceptance results", "=" * 64, ""]

    # 1. bidirectional email-sized transfer over Poor Watterson, via TCP
    rng = np.random.default_rng(4)
    pa = rng.integers(0, 256, 6000, dtype=np.uint8).tobytes()
    pb = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    r = round_trip(pa, pb, "poor", 10.0, seed=7)
    net = 8 * (len(pa) + len(pb)) / r["wall_s"]
    ptt_a = sum(m["m"] == M.PHYSICAL_STATE and m.get("ptt") is True
                for m in r["msgs_a"])
    lines += [
        "App-to-app gate: TCP host client -> server -> ARQ modem -> Poor"
        " Watterson (2 ms / 1 Hz) + AWGN 10 dB -> peer modem -> server ->"
        " client",
        f"  A->B 6000 B and B->A 4000 B, both byte-exact:"
        f" {r['ok_ab'] and r['ok_ba']}",
        f"  session wall {r['wall_s']:.0f} s (virtual), net {net:.0f} bit/s;"
        f" airtime A {r['airtime_a']:.0f} s / B {r['airtime_b']:.0f} s;"
        f" {ptt_a} PTT cycles at A",
        f"  frames A {r['frames_a']}",
        f"  frames B {r['frames_b']}",
        ""]

    # 2. compression: same text, COMPRESSION OFF vs TEXT
    text = email_text(6000, seed=1)
    off = round_trip(text, b"", None, 25.0, compression=False, seed=9)
    on = round_trip(text, b"", None, 25.0, compression=True, seed=9)
    lines += [
        "Compression gate: 6000-byte email text, clean channel, DEFLATE"
        " (RFC 1951) signaled per record",
        f"  link bytes {off['link_bytes_a']} -> {on['link_bytes_a']}"
        f" (x{off['link_bytes_a'] / on['link_bytes_a']:.2f} smaller);"
        f" TX airtime {off['airtime_a']:.0f} s -> {on['airtime_a']:.0f} s",
        f"  byte-exact both modes:"
        f" {off['ok_ab'] and on['ok_ab']}",
        ""]

    # 3. station ID cadence over a long session (virtual clock)
    from hfmodem.sabir.sim.m3 import run_pair
    payload = np.random.default_rng(3).integers(
        0, 256, 60_000, dtype=np.uint8).tobytes()
    s = run_pair(payload, "moderate", 12.0, seed=4, max_exchanges=2000)
    ids = s.stats_a["ids"]
    marks = sorted([0.0] + ids + [s.wall_s])   # connect + IDs + disconnect
    gaps = np.diff(marks)
    lines += [
        "Station-ID gate: 60 kB over Moderate at 12 dB,"
        f" wall {s.wall_s:.0f} s (virtual)",
        f"  callsign-bearing transmissions at t = connect,"
        f" {[f'{t:.0f}s' for t in ids]}, disconnect;"
        f" max gap {gaps.max():.0f} s (§97.119 limit 600 s)",
        f"  byte-exact: {s.ok}",
        ""]

    text_out = "\n".join(lines) + "\n"
    print(text_out)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m4_results.txt").write_text(text_out)


if __name__ == "__main__":
    main()
