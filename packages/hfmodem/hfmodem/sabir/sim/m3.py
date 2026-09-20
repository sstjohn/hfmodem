# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M3 acceptance: two in-process modems over impaired channels, with numbers.

``run_pair`` wires two :class:`arq.LinkModem` endpoints through a
per-transmission channel (fresh Watterson realisation + AWGN at a scheduled
SNR) on a virtual clock: transmissions advance time by their real duration
plus a turnaround, and when the air is quiet the earliest FSM timer fires.

Run ``python -m hfmodem.sabir.sim.m3`` from the repo root for the full acceptance sweeps
-> ``tests/sabir/m3_results.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hfmodem.sabir.arq import ArqConfig, LinkModem, SessionState, layout
from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.m2 import add_noise_snr3k, notch
from hfmodem.sabir.sim.watterson import Watterson

# A static dead band covering carrier groups 5 and 6, with edges halfway
# between their boundary carriers and the adjacent groups.
_fast = GEARS["fast"]
NOTCH_HZ = (float(_fast.carrier_hz[35] - FS / _fast.n_fft / 2),
            float(_fast.carrier_hz[48] + FS / _fast.n_fft / 2))


@dataclass
class Session:
    ok: bool
    delivered: bytes
    wall_s: float
    airtime: dict
    stats_a: dict
    stats_b: dict
    delivered_a: bytes = b""
    terminal_states: tuple[str, str] = ("", "")
    timing: dict = field(default_factory=dict)
    byte_exact: bool = False
    log_a: list = field(repr=False, default_factory=list)
    log_b: list = field(repr=False, default_factory=list)

    @property
    def throughput_bps(self) -> float:
        return 8 * len(self.delivered) / self.wall_s if self.wall_s else 0.0

    @property
    def gears(self) -> list[str]:
        return [f["rung"] for f in self.stats_a["frames"]]


def run_pair(payload: bytes, profile: str | None, snr, seed: int = 0,
             start_rung: int = 1, dd: int = 1, cfg_kw: dict | None = None,
             notch_hz: tuple | None = None, max_exchanges: int = 600,
             payload_b: bytes = b"", channel_transform=None,
             drop_bursts: frozenset[tuple[str, int]] = frozenset()) -> Session:
    rng = np.random.default_rng(seed)
    session_rng = np.random.default_rng([seed, 0x53414249])
    session_id_factory = lambda: int.from_bytes(session_rng.bytes(8), "big") or 1
    state = {"now": 0.0}
    clock = lambda: state["now"]
    kw = cfg_kw or {}
    A = LinkModem(ArqConfig(callsign="ALICE", start_rung=start_rung, **kw),
                  clock, dd=dd, session_id_factory=session_id_factory)
    B = LinkModem(ArqConfig(callsign="BOB", **kw), clock, dd=dd)
    B.fsm.on_host_listen(True)
    if payload_b:
        B.fsm.on_host_data(payload_b)
    A.fsm.on_host_data(payload)
    A.fsm.on_host_connect("BOB")
    A.fsm.on_host_disconnect()          # queued: flush data, then hang up
    # A (forward, reverse) tuple permits asymmetric paths. SNR is relative
    # to transmitted burst power, before fading/notching; noise never follows
    # a deep fade downwards. Each direction consumes the same seeded RNG.
    def channel(x, t, direction):
        value = snr[direction] if isinstance(snr, tuple) else snr
        snr_db = value(t) if callable(value) else value
        fading = profile[direction] if isinstance(profile, tuple) else profile
        transform = channel_transform[direction] if isinstance(channel_transform, tuple) else channel_transform
        y = (transform(x, t) if transform is not None else
             Watterson(fading, FS, rng)(x) if fading else x)
        if notch_hz:
            y = notch(y, *notch_hz)
        return add_noise_snr3k(y, snr_db, rng, reference=x)

    durations = {"a": 0.0, "b": 0.0}
    counts = {"a": 0, "b": 0}
    turnaround_s = idle_s = 0.0

    turnaround = A.fsm.cfg.turnaround_s
    for _ in range(max_exchanges):
        moved = False
        for direction, (src, dst) in enumerate(((A, B), (B, A))):
            name = "a" if direction == 0 else "b"
            while src.outbox:
                wav = src.outbox.pop(0)
                t0 = state["now"]
                state["now"] = t0 + wav.size / FS
                durations[name] += wav.size / FS
                counts[name] += 1
                if (name, counts[name]) not in drop_bursts:
                    dst.on_air(channel(wav, t0, direction))
                state["now"] += turnaround
                turnaround_s += turnaround
                moved = True
        a_done = A.fsm.state in (SessionState.DISCONNECTED, SessionState.LISTENING)
        b_done = B.fsm.state in (SessionState.DISCONNECTED, SessionState.LISTENING)
        if a_done and b_done and not A.outbox and not B.outbox:
            break
        if not moved:
            ds = [d for d in (A.fsm.next_deadline(), B.fsm.next_deadline())
                  if d is not None]
            if not ds:
                break
            next_time = max(state["now"], min(ds)) + 1e-3
            idle_s += next_time - state["now"]
            state["now"] = next_time
            A.fsm.on_timer()
            B.fsm.on_timer()
    byte_exact = bytes(B.delivered) == bytes(payload) and bytes(A.delivered) == bytes(payload_b)
    terminal = all(e.fsm.state in (SessionState.DISCONNECTED, SessionState.LISTENING)
                   for e in (A, B)) and not A.outbox and not B.outbox
    return Session(ok=byte_exact and terminal, byte_exact=byte_exact,
                   terminal_states=(str(A.fsm.state), str(B.fsm.state)),
                   timing={"turnaround_s": turnaround_s, "timer_idle_s": idle_s,
                           "transmitted_bursts": counts},
                   delivered=bytes(B.delivered), wall_s=state["now"],
                   airtime=durations,
                   stats_a=A.fsm.stats, stats_b=B.fsm.stats,
                   delivered_a=bytes(A.delivered),
                   log_a=A.events, log_b=B.events)


# -- HARQ soft-combining vs blind repeat --------------------------------------
def harq_demo(snr_db: float = -2.0, n_trials: int = 20, profile: str = "poor",
              max_copies: int = 6, seed: int = 0) -> dict:
    """One workhorse frame below its single-shot threshold: transmissions
    until decode, Chase-combined vs fresh-per-copy. Airtime per copy is the
    body duration; header/ACK overhead is identical for both disciplines."""
    gear = GEARS["workhorse"]
    phy = Phy(gear)
    fc = FrameCodec(QCLDPC(gear.code))
    rng = np.random.default_rng(seed)
    n_cw = 4
    chase_tx, blind_tx = [], []
    body_s = None
    for _ in range(n_trials):
        chunks = fc.chunk(rng.integers(0, 256, n_cw * fc.data_bytes,
                                       dtype=np.uint8).tobytes())
        coded = np.stack([fc.encode_cw(c) for c in chunks])
        bits, n_syms = layout.assemble(phy, coded, grouped=False)
        tx = phy.transmit(bits)
        body_s = tx.size / FS
        store = np.zeros_like(coded, dtype=float)
        got_chase = got_blind = None
        for k in range(1, max_copies + 1):
            y = add_noise_snr3k(Watterson(profile, FS, rng)(tx), snr_db, rng)
            try:
                _, llr, _ = phy.receive(y, n_symbols=n_syms, dd=1)
                rows = layout.extract(phy, llr, n_cw, fc.code.n,
                                      grouped=False)
            except ValueError:
                rows = np.zeros_like(store)
            store += rows
            if got_chase is None and all(
                    c is not None for c in fc.decode_cws(store)[0]):
                got_chase = k
            if got_blind is None and all(
                    c is not None for c in fc.decode_cws(rows)[0]):
                got_blind = k
            if got_chase and got_blind:
                break
        chase_tx.append(got_chase or max_copies + 1)
        blind_tx.append(got_blind or max_copies + 1)
    return {"snr_db": snr_db, "profile": profile, "body_s": body_s,
            "chase_mean_tx": float(np.mean(chase_tx)),
            "blind_mean_tx": float(np.mean(blind_tx)),
            "chase_fail": sum(t > max_copies for t in chase_tx),
            "blind_fail": sum(t > max_copies for t in blind_tx),
            "n_trials": n_trials, "max_copies": max_copies}


# -- selective ACK on a notched channel ---------------------------------------
def selective_demo(snr_db: float = 20.0, seed: int = 3,
                   payload_bytes: int = 2232, selective: bool = True) -> dict:
    """Fast gear (grouped codewords) through the static dead band: only the
    dead groups' codewords should be retransmitted."""
    payload = np.random.default_rng(99).integers(
        0, 256, payload_bytes, dtype=np.uint8).tobytes()
    s = run_pair(payload, None, snr_db, seed=seed, start_rung=3,
                 notch_hz=NOTCH_HZ, cfg_kw={"selective": selective})
    sends = s.stats_a["sends"]
    return {"ok": s.ok, "sends": sends, "selective": selective,
            "airtime_a": s.airtime["a"], "wall_s": s.wall_s,
            "frames": s.stats_a["frames"]}


# -- gearshift under an SNR ramp ----------------------------------------------
def ramp_demo(payload_bytes: int = 60_000, seed: int = 5,
              lo: float = 5.0, hi: float = 27.0, period: float = 480.0
              ) -> dict:
    """SNR climbs lo -> hi over period/2 seconds, then falls back. Good
    profile so the top rungs are reachable; the link should ride the ladder
    up and back down while delivering byte-exact."""
    def snr(t):
        x = (t % period) / period
        return lo + (hi - lo) * (2 * x if x < 0.5 else 2 - 2 * x)

    payload = np.random.default_rng(7).integers(
        0, 256, payload_bytes, dtype=np.uint8).tobytes()
    s = run_pair(payload, "good", snr, seed=seed, start_rung=1,
                 max_exchanges=900)
    traj = [(f["t"], f["rung"], f["tx_count"], f["snr3k"])
            for f in s.stats_a["frames"]]
    return {"ok": s.ok, "traj": traj, "shifts": s.stats_a["shifts"],
            "wall_s": s.wall_s, "snr_fn": snr}


# -- loading loop vs uniform on a notched channel ------------------------------
def loading_demo(snr_db: float = 22.0, payload_bytes: int = 4464,
                 seed: int = 11) -> dict:
    payload = np.random.default_rng(13).integers(
        0, 256, payload_bytes, dtype=np.uint8).tobytes()
    out = {}
    for name, kw in (("loaded", {}), ("uniform", {"loading": False})):
        s = run_pair(payload, None, snr_db, seed=seed, start_rung=3,
                     notch_hz=NOTCH_HZ, cfg_kw=kw, max_exchanges=900)
        out[name] = {"ok": s.ok, "throughput_bps": s.throughput_bps,
                     "wall_s": s.wall_s, "gears": s.gears,
                     "rounds": s.stats_a["rounds"]}
    return out


# -- decision-directed channel estimation -------------------------------------
def dd_demo(gear_name: str = "fast", profile: str = "moderate",
            snr_db: float = 15.0, n_frames: int = 100, seed: int = 21) -> dict:
    """Frame error rate with and without the DD re-estimation pass."""
    gear = GEARS[gear_name]
    phy = Phy(gear)
    fc = FrameCodec(QCLDPC(gear.code))
    errs = {0: 0, 1: 0}
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        payload = rng.integers(0, 256, 2 * fc.data_bytes,
                               dtype=np.uint8).tobytes()
        bits = fc.encode(payload)
        tx = phy.transmit(bits)
        n_syms = phy.n_symbols_for(bits.size)
        y = add_noise_snr3k(Watterson(profile, FS, rng)(tx), snr_db, rng)
        for dd in (0, 1):
            try:
                _, llr, _ = phy.receive(y, n_symbols=n_syms, dd=dd)
                got, _ = fc.decode(llr)
            except ValueError:
                got = None
            errs[dd] += got != payload
    return {"gear": gear_name, "profile": profile, "snr_db": snr_db,
            "n_frames": n_frames, "fer_dd0": errs[0] / n_frames,
            "fer_dd1": errs[1] / n_frames}


def main():
    lines = ["sabir M3 acceptance results", "=" * 64, ""]

    # 1. full session over Poor Watterson
    payload = np.random.default_rng(1).integers(
        0, 256, 3000, dtype=np.uint8).tobytes()
    s = run_pair(payload, "poor", 8.0, seed=2)
    lines += [
        "Session gate: 3000-byte payload, Poor Watterson (2 ms / 1 Hz),"
        " 8 dB SNR",
        f"  byte-exact delivery: {s.ok}; wall {s.wall_s:.0f} s,"
        f" net {s.throughput_bps:.0f} bit/s",
        f"  frames: {[(f['rung'], f['tx_count']) for f in s.stats_a['frames']]}",
        f"  airtime A {s.airtime['a']:.0f} s / B {s.airtime['b']:.0f} s;"
        f" rounds {s.stats_a['rounds']}, timeouts {s.stats_a['timeouts']}",
        ""]

    # 2. HARQ
    h = harq_demo()
    save = 1 - h["chase_mean_tx"] / h["blind_mean_tx"]
    lines += [
        f"HARQ gate: workhorse frame at {h['snr_db']:+.0f} dB on"
        f" {h['profile']} (below single-shot threshold),"
        f" {h['n_trials']} trials",
        f"  transmissions to decode (cap {h['max_copies']}): Chase"
        f" {h['chase_mean_tx']:.2f} mean ({h['chase_fail']} fails),"
        f" blind {h['blind_mean_tx']:.2f} ({h['blind_fail']} fails)",
        f"  body airtime saved by soft-combining: {100 * save:.0f}%"
        f" ({h['body_s']:.1f} s/copy)",
        ""]

    # 3. selective ACK
    sel = selective_demo(selective=True)
    whole = selective_demo(selective=False)
    r1, r2 = sel["sends"][0], sel["sends"][1]
    w2 = whole["sends"][1]
    lines += [
        "Selective-ACK gate: fast gear (grouped), 2 of 8 carrier groups"
        " notched, AWGN 20 dB",
        f"  round 1: {r1['n_cw']} codewords, {r1['dur']:.1f} s;"
        f" round 2 retransmits {r2['n_cw']} (selective)"
        f" vs {w2['n_cw']} (whole-frame): {r2['dur']:.1f} s vs"
        f" {w2['dur']:.1f} s over-the-air",
        f"  over airtime saved {100 * (1 - r2['dur'] / w2['dur']):.0f}%;"
        f" session airtime {sel['airtime_a']:.0f} s vs"
        f" {whole['airtime_a']:.0f} s; both byte-exact:"
        f" {sel['ok'] and whole['ok']}",
        ""]

    # 4. gearshift ramp
    r = ramp_demo()
    lines += [
        "Gearshift gate: SNR ramp 5 -> 26 -> 5 dB over 400 s, good profile",
        "  t(s)  SNR   rung          rounds",]
    for t, rung, k, snr3k in r["traj"]:
        lines.append(f"  {t:5.0f}  {r['snr_fn'](t):4.1f}  {rung:13s} {k}")
    shifts = [(f"{x['t']:.0f}s", x["rung"]) for x in r["shifts"]]
    lines += [f"  byte-exact: {r['ok']}; shifts: {shifts}", ""]

    # 5. loading
    ld = loading_demo()
    gain = ld["loaded"]["throughput_bps"] / max(
        ld["uniform"]["throughput_bps"], 1e-9)
    lines += [
        "Loading gate: fast gear, 2 of 8 groups notched, AWGN 22 dB",
        f"  loading loop: {ld['loaded']['throughput_bps']:.0f} bit/s"
        f" (gears {ld['loaded']['gears']})",
        f"  uniform     : {ld['uniform']['throughput_bps']:.0f} bit/s"
        f" (gears {ld['uniform']['gears']})",
        f"  throughput gain x{gain:.2f}; both byte-exact:"
        f" {ld['loaded']['ok'] and ld['uniform']['ok']}",
        ""]

    # 6. decision-directed estimation
    for gear_name, profile, snr in (("fast", "moderate", 15.0),
                                    ("workhorse34", "poor", 11.0)):
        d = dd_demo(gear_name, profile, snr)
        lines.append(
            f"DD estimation: {d['gear']} on {d['profile']} at"
            f" {d['snr_db']:+.0f} dB: FER {d['fer_dd0']:.2f} -> "
            f"{d['fer_dd1']:.2f} with one DD pass ({d['n_frames']} frames)")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m3_results.txt").write_text(text)


if __name__ == "__main__":
    main()
