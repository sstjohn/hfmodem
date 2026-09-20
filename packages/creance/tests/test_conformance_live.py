# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Golden-verdict gates against the real M1 targets — the `creance selftest`
substance. Each test spawns its target on ephemeral ports, runs the suite,
and asserts zero deviation from conformance.GOLDEN.

Skipped when a target cannot be spawned here — the interpreter is asked whether
it imports the server, not the filesystem whether a path exists. Naming a target
in CREANCE_REQUIRE turns that skip into a failure: a silently skipped live gate
is worse than a red one, because it reads as coverage. Attachment never
port-probes — a bare connect would consume the one-app-at-a-time servers' accept
and desync the cmd/data pairing."""

import os
import socket
import subprocess
import time

import pytest

from creance import conformance as conf
from hfhost.client import AttachError, ModemClient
from creance.config import ModemConfig, Quirks
from hfhost.supervisor import probe
from hfhost.transcript import Transcript

# The interpreter and the module paths come from creance.selftest rather than
# being restated here: they were duplicated once, so a move fixed one copy and
# left this one pointing at something that no longer existed — which made a live
# gate skip silently instead of failing. One definition steers both.
from creance.selftest import (KESTREL_PAIR, KESTREL_SERVER, MODEM_PY,
                              SABIR_SERVER, have_kestrel, have_sabir,
                              importable)

#: Every test here spawns its target and drives a real handshake against it, so a
#: machine that is busy is a machine these measure instead of the modem. Seven of
#: twenty probes reported no session established between the pair on an evening
#: with six pytest runs up; the same test alone passes in 27 s.
#:
#: THE BOUND, SINCE IT IS A BOUND AND NOT A TOLERANCE. The pair gate needs about
#: half a minute of a core it does not have to share, for three real-time ARQ
#: threads in a spawned child. Underneath, `kestrel.arq.fsm` retransmits a
#: connect every 8 s and gives up after 4 tries, so one handshake round lost to a
#: descheduled thread eats most of a probe's 60 s budget -- and
#: `conformance.probe_preconnect_queue` is the one connecting probe that asks
#: once instead of going through `_establish`, so it is where the loss lands.
#: When it does, `buffer_cadence`, `disconnect_flush_order` and `abort_prompt`
#: each spend a budget of their own and their paired probes report no session:
#: eight deviations and a 300 s test out of a single missed rendezvous, which
#: reads like a broken modem and is a busy laptop.
#:
#: Nothing here is loosened to make that quiet. The escape is `-m 'not realtime'`,
#: and the note below is how a reader tells the two apart without re-running.
pytestmark = pytest.mark.realtime

REQUIRED = {r.strip() for r in os.environ.get("CREANCE_REQUIRE", "").split(",")
            if r.strip()}


def _guard(target: str, present: bool, reason: str):
    """Skip an absent target — unless this bench declared it required, in which
    case let the test run and fail on the missing server. Skipping a gate the
    operator said must exist reads as coverage and is how a moved path costs a
    graded target without anyone noticing."""
    return pytest.mark.skipif(not present and target not in REQUIRED,
                              reason=reason)


needs_kestrel = _guard("kestrel", have_kestrel(),
                       "kestrel host server not importable")
needs_kestrel_pair = _guard("kestrel-pair", importable(KESTREL_PAIR[1]),
                            "kestrel ARQ pair not importable")
needs_sabir = _guard("sabir", have_sabir(),
                     "sabir host server not importable")


def test_a_failure_here_says_what_else_the_machine_was_doing(request):
    """The repo-root conftest attaches "the machine this ran on" to any failing
    `realtime` test, and its own comment cites THIS gate as the reason it exists.
    It was not reaching it: `packages/creance/pyproject.toml` declares
    `[tool.pytest.ini_options]`, which makes `packages/creance` the rootdir, and
    pytest cuts conftest collection at the rootdir -- so the note was written for
    a failure it could never appear in, and this suite's eight-deviation cascade
    arrived without the one fact that separates a contended machine from a fault.
    """
    attaches = [p for p in request.config.pluginmanager.get_plugins()
                if hasattr(p, "_contention")]
    assert attaches, (
        "no loaded conftest attaches the contention note to a realtime failure, "
        "so a failure of this suite cannot be told from a busy machine without "
        "re-running it alone")


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def attach_retry(client, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.attach()
            return
        except AttachError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


@pytest.fixture
def transcript(tmp_path):
    t = Transcript(str(tmp_path / "live.jsonl"), "sid-live")
    yield t
    t.close()


def _serving(pair) -> bool:
    """Is this modem reachable — using the paired probe hfhost already provides.

    A cmd-only connect would consume the accept of these one-app-at-a-time
    servers and desync the cmd/data pairing, so readiness has to be probed as a
    pair. `supervisor.probe` is exactly that and is what `selftest` already uses;
    an earlier version of this file shelled out to `lsof` to avoid connecting at
    all, which worked but reinvented a primitive that was one import away."""
    cmd_port, data_port = pair
    return probe("127.0.0.1", cmd_port, data_port, 1.0)


def spawn(argv, pairs=(), timeout=20.0):
    """Start a modem server and wait until it is actually serving ``ports``.

    ``free_ports`` closes its probe sockets before the child binds, so a port can
    be taken in between — by another test, or by a server this suite has not
    reaped yet. Without a readiness check the test proceeds against a port nobody
    is serving and only discovers it when a 60 s connect timeout expires, which
    reads as "the modem failed to connect" rather than "the server never came up".
    Waiting here turns that into an immediate, accurate failure.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    while pairs and time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode}): "
                               f"{' '.join(map(str, argv))}")
        if all(_serving(p) for p in pairs):
            return proc
        time.sleep(0.2)
    if pairs:
        reap(proc)
        raise RuntimeError(f"server never served {list(pairs)} within "
                           f"{timeout:.0f}s: {' '.join(map(str, argv))}")
    return proc


def reap(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@needs_kestrel
def test_kestrel_loopback_matches_golden(transcript):
    cmd_p, data_p = free_ports(2)
    proc = spawn([MODEM_PY, *KESTREL_SERVER,
                  "--cmd-port", str(cmd_p), "--data-port", str(data_p),
                  "--iamalive-interval", "2", "--quiet"],
                 pairs=((cmd_p, data_p),))
    client = None
    try:
        cfg = ModemConfig(name="kestrel", cmd_port=cmd_p, data_port=data_p,
                          bandwidth="2300", compression="OFF",
                          quirks=Quirks(iamalive_s=2.0, preconnect_queue=True))
        client = ModemClient(cfg, "127.0.0.1", transcript)
        attach_retry(client)
        ctx = conf.ProbeContext(client, mycall="N0CRE", dst="K7CRE",
                                iamalive_window_s=4.0)
        findings = conf.run_single(ctx)
    finally:
        if client is not None:
            client.close()
        reap(proc)
    assert conf.diff_golden("kestrel-loopback", findings) == [], \
        conf.render(findings)


@needs_kestrel_pair
def test_kestrel_pair_matches_golden(transcript):
    """Two real KestrelModems on one AudioChannel — the ARQ core, not the
    loopback. Same shape as the sabir gate below, same timings."""
    pa, pb, pc, pd = free_ports(4)
    proc = spawn([MODEM_PY, *KESTREL_PAIR,
                  "--a-cmd", str(pa), "--a-data", str(pb),
                  "--b-cmd", str(pc), "--b-data", str(pd), "--quiet"],
                 pairs=((pa, pb), (pc, pd)))
    a = b = None
    timing = dict(connect_timeout_s=60.0, drain_timeout_s=60.0,
                  data_timeout_s=60.0, abort_timeout_s=30.0, payload_n=256)
    try:
        # 60 s IAMALIVE (run_pair has no knob), and the default preconnect_queue
        # is what actually probes §7.4 — setting it false would skip the probe
        # rather than assert the queueing this core now does
        quirks = Quirks()
        cfg_a = ModemConfig(name="kestrel-A", cmd_port=pa, data_port=pb,
                            bandwidth="500", compression="OFF", quirks=quirks)
        cfg_b = ModemConfig(name="kestrel-B", cmd_port=pc, data_port=pd,
                            bandwidth="500", compression="OFF", quirks=quirks)
        a = ModemClient(cfg_a, "127.0.0.1", transcript)
        b = ModemClient(cfg_b, "127.0.0.1", transcript)
        attach_retry(a, timeout=60.0)
        attach_retry(b, timeout=60.0)
        single = conf.run_single(
            conf.ProbeContext(a, mycall="N0CRE", dst="K7CRE", peer=b, **timing))
        paired = conf.run_paired(
            conf.ProbeContext(a, mycall="N0CRE", peer=b, **timing),
            conf.ProbeContext(b, mycall="K7CRE", **timing))
        findings = single + paired
    finally:
        for c in (a, b):
            if c is not None:
                c.close()
        reap(proc)
    assert conf.diff_golden("kestrel-pair", findings) == [], \
        conf.render(findings)


@needs_sabir
def test_sabir_pair_matches_golden(transcript):
    from hfhost.hostapi import HostApiClient
    from creance import conformance_hostapi as native

    pa, pb = free_ports(2)
    proc = spawn([MODEM_PY, *SABIR_SERVER,
                  "--ports", str(pa), str(pb),
                  "--profile", "clean", "--snr", "90", "--quiet"],
                 pairs=((pa, 0), (pb, 0)))
    a = b = None
    try:
        a = HostApiClient(ModemConfig(name="sabir-A", dialect="hostapi",
                          cmd_port=pa, data_port=0), transcript=transcript)
        b = HostApiClient(ModemConfig(name="sabir-B", dialect="hostapi",
                          cmd_port=pb, data_port=0), transcript=transcript)
        a.attach()
        b.attach()
        ctx_a = native.Ctx(a, mycall="N0CRE", dst="K7CRE")
        ctx_b = native.Ctx(b, mycall="K7CRE", dst="N0CRE")
        single = native.run(ctx_a)
        assert conf.diff_golden("sabir-hostapi", single, tables=native.GOLDEN) == [], conf.render(single)
        paired = native.run_paired(ctx_a, ctx_b)
        assert conf.diff_golden("sabir-hostapi-pair", paired, tables=native.GOLDEN) == [], conf.render(paired)
    finally:
        for c in (a, b):
            if c is not None:
                c.close()
        reap(proc)
