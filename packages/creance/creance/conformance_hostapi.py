# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Conformance probes for the structured host interface (HOST-API v1.0).

A separate catalog from `conformance.py` on purpose. Grading a modem's fidelity
to a dialect is exactly the job that must *see* the dialect, so the two are not
stretched over one abstraction — they share only the Finding vocabulary and the
golden-table machinery.

What is graded here is what the specification promises and a client depends on:
the symmetric Hello and its version rule, must-ignore in both directions, that
commands answer with events rather than silence, that errors correlate by ref,
and that the data plane preserves bytes and framing. Telemetry is *not* graded
for presence — LinkStats and CapabilitiesNegotiated are opt-in per the spec, and
a modem that has no numbers to report is not defective for withholding them.

The golden table below was derived by running this catalog against a live
sabir server, the same footing as the VARA catalog's.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable

from hfhost import cbor, hostapi
from hfhost.hostapi import HostApiClient
from hfhost.transcript import Transcript

from .conformance import ABSENT, EXTRA, FAIL, PASS, WARN, Finding  # noqa: F401

# Insertion order is execution order, and it is load-bearing:
# identity_required_is_enforced must run before any probe calls Ctx.arm(),
# because once the modem has an identity the refusal it tests cannot happen.
PROBES: dict[str, "Probe"] = {}


@dataclass(frozen=True, slots=True)
class Probe:
    id: str
    fn: Callable


def _probe(pid: str):
    def register(fn):
        PROBES[pid] = Probe(pid, fn)
        return fn
    return register


@dataclass
class Ctx:
    client: HostApiClient
    mycall: str = "N0CRE"
    dst: str = "K7CRE"
    reply_timeout_s: float = 5.0
    settle_s: float = 0.3

    def arm(self) -> None:
        """Give the modem an identity if its Hello said it needs one.

        A probe that skips this grades a *correct* refusal as a missing
        feature: sabir requires SetIdentity before Listen or Connect, so
        without it every session-shaped probe fails for the wrong reason."""
        if self.client.hello.get("identity_required"):
            self.client.set_identity(self.mycall)

    @property
    def transcript(self) -> Transcript | None:
        return self.client.transcript


def _drain(sub: queue.Queue, window: float) -> list[dict]:
    out, end = [], time.monotonic() + window
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return out
        try:
            out.append(sub.get(timeout=remaining))
        except queue.Empty:
            return out


# -- handshake ---------------------------------------------------------------

@_probe("hello_symmetric")
def probe_hello_symmetric(ctx: Ctx) -> Finding:
    """The modem answers Hello with Hello, naming itself and its proto."""
    h = ctx.client.hello
    if not h:
        return Finding("hello_symmetric", FAIL, detail="no Hello from the modem")
    missing = [k for k in ("proto", "modem") if not h.get(k)]
    if missing:
        return Finding("hello_symmetric", FAIL,
                       evidence=(str(h),),
                       detail=f"Hello omits {', '.join(missing)}")
    major = str(h["proto"]).split(".")[0]
    if major != hostapi.PROTO.split(".")[0]:
        return Finding("hello_symmetric", FAIL,
                       detail=f"proto major {h['proto']} != {hostapi.PROTO}")
    return Finding("hello_symmetric", PASS, evidence=(f"proto={h['proto']}",
                                                      f"modem={h['modem']}"))


@_probe("hello_advertises_profiles")
def probe_hello_profiles(ctx: Ctx) -> Finding:
    """Profiles are how the host learns what regulatory envelopes exist. A
    modem that advertises none is usable but leaves the host guessing."""
    profiles = ctx.client.hello.get("profiles")
    if not profiles:
        return Finding("hello_advertises_profiles", ABSENT,
                       detail="Hello carries no profiles list")
    return Finding("hello_advertises_profiles", PASS,
                   evidence=(f"profiles={profiles}",))


@_probe("identity_required_is_enforced")
def probe_identity_required(ctx: Ctx) -> Finding:
    """If Hello says identity_required, Connect before SetIdentity must fail
    with NO_IDENTITY rather than transmitting an unidentified station."""
    if not ctx.client.hello.get("identity_required"):
        return Finding("identity_required_is_enforced", ABSENT,
                       detail="modem does not require an identity")
    # actually try it: an advertisement nobody enforces is worse than none
    sub = ctx.client.subscribe(hostapi.ERROR, hostapi.STATE_CHANGED)
    try:
        ref = ctx.client.next_ref()
        ctx.client.send(hostapi.CONNECT, peer_id=ctx.dst, ref=ref)
        for msg in _drain(sub, ctx.reply_timeout_s):
            if msg["m"] == hostapi.ERROR and msg.get("code") == hostapi.ERR_NO_IDENTITY:
                return Finding("identity_required_is_enforced", PASS)
            if msg.get("state") == hostapi.ST_CONNECTING:
                return Finding("identity_required_is_enforced", FAIL,
                               detail="Connect proceeded without an identity")
        return Finding("identity_required_is_enforced", ABSENT,
                       detail="Connect without an identity drew neither an "
                              "Error nor a state change")
    finally:
        ctx.client.unsubscribe(sub)


# -- must-ignore -------------------------------------------------------------

@_probe("must_ignore_unknown_type")
def probe_unknown_type(ctx: Ctx) -> Finding:
    """An unknown message-type code must be ignored, never fatal (§10.1). This
    is the single property that lets a newer host talk to an older modem, so it
    is checked by actually sending one rather than by reading the spec."""
    ctx.client.send_raw_frame(cbor.encode({0: 250, 1: "from the future"}))
    time.sleep(ctx.settle_s)
    if not ctx.client.attached:
        return Finding("must_ignore_unknown_type", FAIL,
                       detail="modem closed the connection on an unknown type")
    if ctx.client.errors and ctx.client.errors[-1].get("code") == hostapi.ERR_MALFORMED:
        return Finding("must_ignore_unknown_type", FAIL,
                       detail="modem answered MALFORMED to a well-formed unknown type")
    return Finding("must_ignore_unknown_type", PASS)


@_probe("must_ignore_unknown_field")
def probe_unknown_field(ctx: Ctx) -> Finding:
    """An unknown map key inside a known message must be ignored, not fatal."""
    body = cbor.encode({0: hostapi.LISTEN, hostapi.KEY["on"]: False,
                        len(hostapi.NAME) + 11: "a field you do not know"})
    ctx.client.send_raw_frame(body)
    time.sleep(ctx.settle_s)
    if not ctx.client.attached:
        return Finding("must_ignore_unknown_field", FAIL,
                       detail="modem closed the connection on an unknown field")
    return Finding("must_ignore_unknown_field", PASS)


@_probe("undecodable_frame_is_reported")
def probe_undecodable(ctx: Ctx) -> Finding:
    """Garbage is not must-ignore: the spec gives it an Error code, because a
    frame that will not decode means the stream is desynchronised and silence
    would leave the host waiting forever."""
    before = len(ctx.client.errors)
    ctx.client.send_raw_frame(b"\xff\xff\xff\xff")
    time.sleep(ctx.settle_s)
    new = ctx.client.errors[before:]
    if any(e.get("code") == hostapi.ERR_MALFORMED for e in new):
        return Finding("undecodable_frame_is_reported", PASS)
    if not ctx.client.attached:
        return Finding("undecodable_frame_is_reported", WARN,
                       detail="modem closed rather than reporting MALFORMED")
    return Finding("undecodable_frame_is_reported", ABSENT,
                   detail="undecodable frame drew neither an Error nor a close")


# -- commands answer with events ---------------------------------------------

@_probe("listen_changes_state")
def probe_listen_state(ctx: Ctx) -> Finding:
    """Listen is asynchronous, so the only evidence it worked is a
    StateChanged. Silence here means a host can never know it is listening."""
    ctx.arm()
    sub = ctx.client.subscribe(hostapi.STATE_CHANGED)
    try:
        ctx.client.set_listen(True)
        for msg in _drain(sub, ctx.reply_timeout_s):
            if msg.get("state") == hostapi.ST_LISTENING:
                return Finding("listen_changes_state", PASS)
        return Finding("listen_changes_state", ABSENT,
                       detail="LISTEN drew no StateChanged{LISTENING}")
    finally:
        ctx.client.unsubscribe(sub)
        ctx.client.set_listen(False)


@_probe("errors_correlate_by_ref")
def probe_error_ref(ctx: Ctx) -> Finding:
    """A command that cannot be honored yields an Error carrying the ref the
    command supplied (§4). Without the ref a host cannot tell which of several
    in-flight commands failed — this is the defect that makes VARA's bare
    OK/WRONG useless, and the reason this dialect exists.

    The unhonorable command used here is a Connect with its required peer_id
    missing, which §4 and §10.2 make unambiguous. An earlier version used
    DISCONNECT while idle and was wrong to: the spec nowhere says that is an
    error, and treating a redundant disconnect as idempotent is a defensible
    reading. A probe must test what the document promises, not what the probe
    author assumed."""
    ctx.arm()
    sub = ctx.client.subscribe(hostapi.ERROR)
    try:
        ref = ctx.client.next_ref()
        ctx.client.send(hostapi.CONNECT, ref=ref)        # peer_id omitted
        for msg in _drain(sub, ctx.reply_timeout_s):
            if msg.get("ref") == ref:
                return Finding("errors_correlate_by_ref", PASS,
                               evidence=(f"code={msg.get('code')}",))
            if "ref" not in msg:
                return Finding("errors_correlate_by_ref", FAIL,
                               evidence=(str(msg),),
                               detail="Error carries no ref")
        return Finding("errors_correlate_by_ref", ABSENT,
                       detail="a Connect missing its required peer_id drew no Error")
    finally:
        ctx.client.unsubscribe(sub)


# -- data plane --------------------------------------------------------------

@_probe("oversize_frame_refused")
def probe_oversize(ctx: Ctx) -> Finding:
    """A length prefix beyond the 16 MiB cap is a protocol error and the modem
    closes (§2). A modem that tries to allocate it instead is a memory bomb."""
    try:
        ctx.client.send_raw_prefix((hostapi.MAX_FRAME + 1).to_bytes(4, "big"))
    except Exception as exc:
        return Finding("oversize_frame_refused", WARN, detail=f"send failed: {exc!r}")
    time.sleep(ctx.settle_s)
    closed = not ctx.client.attached
    # This probe knowingly breaks the attachment, so it puts it back: a catalog
    # that leaves the client detached cannot be followed by the paired catalog,
    # and that composition is exactly what `creance conform --pair` does.
    if closed:
        try:
            ctx.client.attach()
            ctx.arm()
        except Exception as exc:
            return Finding("oversize_frame_refused", PASS,
                           evidence=(f"reattach after the close failed: {exc!r}",),
                           detail="refused correctly, but the modem would not "
                                  "take a new attachment afterwards")
        return Finding("oversize_frame_refused", PASS)
    return Finding("oversize_frame_refused", FAIL,
                   detail="oversize length prefix did not close the connection")


# -- telemetry, graded for shape and never for presence ----------------------

@_probe("link_stats_shape")
def probe_link_stats_shape(ctx: Ctx) -> Finding:
    """LinkStats is opt-in, so absence is not a defect. But if it arrives, the
    fields must be the documented types — a string where a float belongs is
    what turns a dashboard into a crash.

    Kept rather than cut when the paired probe superseded it, for one reason:
    a single-modem site is a real deployment, not a degraded one. A remote
    station with one unit still wants its telemetry graded, and this is the only
    probe that can do it there. Against a pair it is redundant and reports
    ABSENT, which the golden table records — so it costs nothing and covers a
    case the pair catalog cannot reach."""
    stats = ctx.client.stats
    if not stats:
        return Finding("link_stats_shape", ABSENT, detail="no LinkStats observed")
    bad = []
    for key, kind in (("gear", str), ("rung", int), ("queue_bytes", int),
                      ("throughput_bps", (int, float)), ("snr3k_db", (int, float)),
                      ("harq_rounds", int), ("rebuilds", int)):
        if key in stats and not isinstance(stats[key], kind):
            bad.append(f"{key}={stats[key]!r}")
    if bad:
        return Finding("link_stats_shape", FAIL, evidence=tuple(bad),
                       detail="LinkStats fields have the wrong types")
    return Finding("link_stats_shape", PASS,
                   evidence=(f"keys={sorted(stats)}",))


# -- paired probes -----------------------------------------------------------
#
# Everything above grades one attached modem. These grade a *link*: the
# properties that only exist once two stations are talking, and that no
# single-ended probe can see. The shape mirrors the VARA paired catalog — one
# shared choreography, then per-probe judgment over what it observed — because
# running the session once and judging it many times is what keeps a pair
# suite affordable over a half-duplex link.

PAIRED: dict[str, "Probe"] = {}


def _paired(pid: str):
    def register(fn):
        PAIRED[pid] = Probe(pid, fn)
        return fn
    return register


@_paired("both_ends_see_connected")
def probe_pair_connected(ev: dict) -> Finding:
    """A link is not up until both ends say so. A modem that reports CONNECTED
    to its caller while the far end still believes it is listening will strand
    an application waiting for data that is never coming."""
    if not ev["connected"]:
        return Finding("both_ends_see_connected", FAIL,
                       detail="the pair never both reported CONNECTED")
    return Finding("both_ends_see_connected", PASS)


@_paired("connected_names_the_peer")
def probe_pair_peer_id(ev: dict) -> Finding:
    """Each end's StateChanged must name the *other* station. Getting this
    backwards is how a session gets filed against the wrong callsign."""
    if not ev["connected"]:
        return Finding("connected_names_the_peer", ABSENT,
                       detail="no link to judge")
    a_peer, b_peer = ev.get("a_peer"), ev.get("b_peer")
    if not a_peer or not b_peer:
        return Finding("connected_names_the_peer", ABSENT,
                       evidence=(f"a={a_peer!r}", f"b={b_peer!r}"),
                       detail="StateChanged carried no peer_id")
    if a_peer == ev["b_call"] and b_peer == ev["a_call"]:
        return Finding("connected_names_the_peer", PASS,
                       evidence=(f"a saw {a_peer}", f"b saw {b_peer}"))
    return Finding("connected_names_the_peer", FAIL,
                   evidence=(f"a saw {a_peer!r}, expected {ev['b_call']!r}",
                             f"b saw {b_peer!r}, expected {ev['a_call']!r}"),
                   detail="each end must name the other station")


@_paired("payload_crosses_intact")
def probe_pair_payload(ev: dict) -> Finding:
    """The link delivers what was submitted, byte for byte."""
    if not ev["connected"]:
        return Finding("payload_crosses_intact", ABSENT, detail="no link to judge")
    sent, got = ev.get("sent", b""), ev.get("received", b"")
    if got == sent:
        return Finding("payload_crosses_intact", PASS,
                       evidence=(f"{len(sent)} B round-tripped",))
    return Finding("payload_crosses_intact", FAIL,
                   evidence=(f"sent {len(sent)} B, received {len(got)} B",),
                   detail="delivered payload differs from what was submitted")


@_paired("capabilities_negotiated_surfaces")
def probe_pair_caps(ev: dict) -> Finding:
    """The point of the whole interface: the air capability intersection is
    reported to the host rather than discarded. Opt-in, so absence is not a
    defect — but a modem that negotiated and told nobody is flying an
    application blind, which is the thing this dialect exists to stop."""
    caps = ev.get("caps")
    if not ev["connected"]:
        return Finding("capabilities_negotiated_surfaces", ABSENT,
                       detail="no link to judge")
    if not caps:
        return Finding("capabilities_negotiated_surfaces", ABSENT,
                       detail="subscribed, but no CapabilitiesNegotiated arrived")
    missing = [k for k in ("peer_id", "peer_profiles", "peer_capabilities") if k not in caps]
    if missing:
        return Finding("capabilities_negotiated_surfaces", FAIL,
                       detail=f"omits {', '.join(missing)}")
    profiles = caps["peer_profiles"]
    if not isinstance(profiles, list) or any(type(p) is not int or p < 0 for p in profiles):
        return Finding("capabilities_negotiated_surfaces", FAIL,
                       detail="peer_profiles must contain unsigned profile IDs")
    return Finding("capabilities_negotiated_surfaces", PASS,
                   evidence=(f"peer_profiles={profiles}",
                             f"peer_capabilities={caps['peer_capabilities']}"))


@_paired("link_stats_during_session")
def probe_pair_stats(ev: dict) -> Finding:
    """Telemetry during a live transfer, which is the only time it means
    anything. Types are graded, presence is not."""
    stats = ev.get("stats")
    if not ev["connected"]:
        return Finding("link_stats_during_session", ABSENT, detail="no link to judge")
    if not stats:
        return Finding("link_stats_during_session", ABSENT,
                       detail="subscribed, but no LinkStats arrived during transfer")
    bad = [f"{k}={stats[k]!r}" for k, kind in
           (("gear", str), ("rung", int), ("queue_bytes", int),
            ("throughput_bps", (int, float)))
           if k in stats and not isinstance(stats[k], kind)]
    if bad:
        return Finding("link_stats_during_session", FAIL, evidence=tuple(bad),
                       detail="LinkStats fields have the wrong types")
    return Finding("link_stats_during_session", PASS,
                   evidence=(f"keys={sorted(stats)}",))


@_paired("both_ends_see_disconnected")
def probe_pair_disconnected(ev: dict) -> Finding:
    """A teardown one end never learns about leaves it holding a dead link."""
    if not ev["connected"]:
        return Finding("both_ends_see_disconnected", ABSENT, detail="no link to judge")
    if ev.get("a_disc") and ev.get("b_disc"):
        return Finding("both_ends_see_disconnected", PASS)
    return Finding("both_ends_see_disconnected", FAIL,
                   evidence=(f"a_disconnected={ev.get('a_disc')}",
                             f"b_disconnected={ev.get('b_disc')}"),
                   detail="both ends must observe the teardown")


def run_paired(ctx_a: Ctx, ctx_b: Ctx) -> list[Finding]:
    """One choreography — B listens, A connects, A sends, A disconnects — then
    each probe judges what it saw. Failures inside the choreography degrade the
    verdicts to ABSENT rather than raising: a pair that would not link is a
    finding, not a crash."""
    a, b = ctx_a.client, ctx_b.client
    ev: dict = {"connected": False, "a_call": ctx_a.mycall, "b_call": ctx_b.mycall}
    stats_seen: list[dict] = []
    caps_seen: list[dict] = []
    sub_a = a.subscribe(hostapi.LINK_STATS, hostapi.CAPABILITIES)
    try:
        # Start from a known state. The single catalog deliberately pokes the
        # modem with commands it must refuse, so a pair run that follows it
        # cannot assume idle — and `creance conform --pair` runs them in exactly
        # that order.
        for cl in (a, b):
            for reset in (cl.abort, lambda c=cl: c.set_listen(False)):
                try:
                    reset()
                except Exception:
                    pass
        _wait(lambda: not (a.connected or b.connected), 2.0)
        time.sleep(0.3)          # let the resets land before arming
        for side in (ctx_a, ctx_b):
            side.arm()
        for cl in (a, b):
            cl.subscribe_events([hostapi.LINK_STATS, hostapi.CAPABILITIES],
                                stats_period=0.5)
        b.set_listen(True)
        a.bump_epoch()
        b.bump_epoch()

        a.connect(ctx_b.mycall)
        if _wait(lambda: a.connected and b.connected, ctx_a.reply_timeout_s * 6):
            ev["connected"] = True
            ev["a_peer"], ev["b_peer"] = a.peer, b.peer

            payload = bytes(range(256)) * 4
            ev["sent"] = payload
            a.send_data(payload, label="paired")
            ev["received"] = b.read_data(len(payload),
                                         timeout=ctx_a.reply_timeout_s * 6)

            for msg in _drain(sub_a, 0.5):
                (caps_seen if msg["m"] == hostapi.CAPABILITIES
                 else stats_seen).append(msg)
            ev["caps"] = a.caps or (caps_seen[-1] if caps_seen else None)
            ev["stats"] = a.stats or (stats_seen[-1] if stats_seen else None)

            a.disconnect()
            ev["a_disc"] = _wait(lambda: not a.connected, ctx_a.reply_timeout_s * 4)
            ev["b_disc"] = _wait(lambda: not b.connected, ctx_a.reply_timeout_s * 4)
        if a.connected or b.connected:      # never leave a wedged session
            # Best-effort, like the resets above. A client whose socket has gone
            # away still reads as connected — it never heard a disconnect — so
            # this is exactly the path a detached modem takes, and raising here
            # would throw away every finding the run had already made. A modem
            # that drops mid-choreography is a finding, not a crash.
            for cl in (a, b):
                try:
                    cl.abort()
                except Exception:
                    pass
            _wait(lambda: not (a.connected or b.connected), ctx_a.reply_timeout_s)
    finally:
        a.unsubscribe(sub_a)
        a.bump_epoch()
        b.bump_epoch()
        try:
            b.set_listen(False)
        except Exception:
            pass

    findings = []
    for probe in PAIRED.values():
        try:
            findings.append(probe.fn(ev))
        except Exception as exc:
            findings.append(Finding(probe.id, FAIL, detail=f"probe raised {exc!r}"))
    return findings


def _wait(pred, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


GOLDEN: dict[str, dict[str, str]] = {
    # Sabir exposes a single CBOR connection per station. A disconnected
    # modem has no LinkStats to report; connected telemetry is checked below.
    "sabir-hostapi": {
        "hello_symmetric": PASS,
        "hello_advertises_profiles": PASS,
        "identity_required_is_enforced": PASS,
        "must_ignore_unknown_type": PASS,
        "must_ignore_unknown_field": PASS,
        "undecodable_frame_is_reported": PASS,
        "listen_changes_state": PASS,
        "errors_correlate_by_ref": PASS,
        "oversize_frame_refused": PASS,
        "link_stats_shape": ABSENT,
    },
    "sabir-hostapi-pair": {
        "both_ends_see_connected": PASS,
        "connected_names_the_peer": PASS,
        "payload_crosses_intact": PASS,
        "capabilities_negotiated_surfaces": PASS,
        "link_stats_during_session": PASS,
        "both_ends_see_disconnected": PASS,
    },
}


def run(ctx: Ctx) -> list[Finding]:
    findings = []
    for probe in PROBES.values():
        try:
            findings.append(probe.fn(ctx))
        except Exception as exc:      # a probe that dies is a finding, not a crash
            findings.append(Finding(probe.id, FAIL, detail=f"probe raised {exc!r}"))
    return findings
