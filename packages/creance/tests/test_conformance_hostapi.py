# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from creance import conformance_hostapi as ch
from creance.conformance import ABSENT, FAIL, PASS
from hfhost import hostapi
from hfhost.config import ModemConfig
from hfhost.hostapi import HostApiClient
from hfhost.testing.fakehostapi import FakeHostApiModem


def _ctx(modem, **kw):
    cfg = ModemConfig(name="m", dialect="hostapi", cmd_port=modem.port, data_port=0)
    c = HostApiClient(cfg)
    c.attach()
    return ch.Ctx(c, settle_s=0.15, reply_timeout_s=1.5, **kw)


@pytest.fixture
def modem():
    m = FakeHostApiModem(profiles=[1, 2])
    yield m
    m.close()


def _verdicts(findings):
    return {f.probe: f.verdict for f in findings}


def test_a_conformant_modem_passes_the_catalog(modem):
    ctx = _ctx(modem)
    try:
        v = _verdicts(ch.run(ctx))
        assert v["hello_symmetric"] == PASS
        assert v["hello_advertises_profiles"] == PASS
        assert v["must_ignore_unknown_type"] == PASS
        assert v["must_ignore_unknown_field"] == PASS
        assert v["listen_changes_state"] == PASS
        assert FAIL not in v.values()
    finally:
        ctx.client.close()


def test_every_probe_reports_something(modem):
    """A probe that returns nothing is worse than one that fails: it silently
    drops a property from the grade."""
    ctx = _ctx(modem)
    try:
        findings = ch.run(ctx)
        assert {f.probe for f in findings} == set(ch.PROBES)
    finally:
        ctx.client.close()


def test_a_probe_that_raises_becomes_a_finding(modem, monkeypatch):
    ctx = _ctx(modem)
    try:
        def boom(_ctx):
            raise RuntimeError("kaboom")
        monkeypatch.setitem(ch.PROBES, "hello_symmetric",
                            ch.Probe("hello_symmetric", boom))
        v = _verdicts(ch.run(ctx))
        assert v["hello_symmetric"] == FAIL
    finally:
        ctx.client.close()


def test_wrong_proto_major_fails_hello():
    m = FakeHostApiModem(proto="9.0")
    try:
        cfg = ModemConfig(name="m", dialect="hostapi", cmd_port=m.port, data_port=0)
        c = HostApiClient(cfg)
        with pytest.raises(hostapi.Incompatible):
            c.attach()
        # the client refuses to attach at all, which is the stronger outcome:
        # the catalog never runs against a modem it cannot speak to
    finally:
        m.close()


def test_missing_profiles_is_absent_not_fail():
    m = FakeHostApiModem(profiles=[])
    try:
        ctx = _ctx(m)
        v = _verdicts(ch.run(ctx))
        assert v["hello_advertises_profiles"] == ABSENT
        ctx.client.close()
    finally:
        m.close()


def test_a_modem_that_dies_on_an_unknown_type_fails(modem):
    """The must-ignore rule is the one that lets old and new interoperate, so
    breaking it is a FAIL rather than a WARN."""
    def on_command(msg, mm):
        pass

    m = FakeHostApiModem(on_command=on_command)
    try:
        ctx = _ctx(m)
        # a modem that closes the socket when it meets something unfamiliar
        original = ctx.client.send_raw_frame

        def hostile(body):
            original(body)
            m.drop()
        ctx.client.send_raw_frame = hostile
        v = _verdicts(ch.run(ctx))
        assert v["must_ignore_unknown_type"] == FAIL
        ctx.client.close()
    finally:
        m.close()


def test_link_stats_types_are_graded_when_present(modem):
    ctx = _ctx(modem)
    try:
        modem.emit(hostapi.LINK_STATS, gear="floor", rung="two",   # wrong type
                   queue_bytes=0, throughput_bps=1.0)
        import time
        time.sleep(0.2)
        v = _verdicts(ch.run(ctx))
        assert v["link_stats_shape"] == FAIL
    finally:
        ctx.client.close()


def test_absent_telemetry_is_not_a_defect(modem):
    ctx = _ctx(modem)
    try:
        v = _verdicts(ch.run(ctx))
        assert v["link_stats_shape"] == ABSENT
    finally:
        ctx.client.close()


def test_golden_table_covers_every_probe():
    """A probe missing from the golden table would silently escape the gate."""
    for target, table in ch.GOLDEN.items():
        expected = ch.PAIRED if target.endswith("-pair") else ch.PROBES
        assert set(table) == set(expected), f"{target} is out of step"


# -- paired catalog ------------------------------------------------------


def test_paired_probes_run_over_a_bridged_pair(tmp_path):
    """The paired catalog against two bridged fakes: proves the choreography
    and every probe's judgment path, without needing a live modem."""
    from hfhost.testing.hostapi_bridge import HostApiBridge
    br = HostApiBridge(tmp_path).start()
    ca = HostApiClient(ModemConfig(name="a", dialect="hostapi",
                                   cmd_port=br.a.port, data_port=0))
    cb = HostApiClient(ModemConfig(name="b", dialect="hostapi",
                                   cmd_port=br.b.port, data_port=0))
    ca.attach()
    cb.attach()
    try:
        # The bridge links both sides when either issues a Connect, so the
        # choreography drives itself. A test-side timer would race the reset the
        # choreography does first.
        findings = ch.run_paired(ch.Ctx(ca, mycall="N0AAA", reply_timeout_s=2.0),
                                 ch.Ctx(cb, mycall="N0BBB", reply_timeout_s=2.0))
        v = _verdicts(findings)
        assert set(v) == set(ch.PAIRED)
        assert v["both_ends_see_connected"] == PASS
        assert v["payload_crosses_intact"] == PASS
    finally:
        ca.close()
        cb.close()
        br.stop()


def test_paired_probes_degrade_rather_than_raise_when_no_link_forms():
    """A pair that will not link is a finding, not a crash: every probe must
    still report, and everything downstream of the link as ABSENT.

    Two *unbridged* fakes, deliberately: a Connect reaches one modem and the
    other never hears of it, which is exactly what a pair that cannot link
    looks like."""
    ma, mb = FakeHostApiModem(), FakeHostApiModem()
    ca = HostApiClient(ModemConfig(name="a", dialect="hostapi",
                                   cmd_port=ma.port, data_port=0))
    cb = HostApiClient(ModemConfig(name="b", dialect="hostapi",
                                   cmd_port=mb.port, data_port=0))
    ca.attach()
    cb.attach()
    try:
        findings = ch.run_paired(ch.Ctx(ca, mycall="N0AAA", reply_timeout_s=0.4),
                                 ch.Ctx(cb, mycall="N0BBB", reply_timeout_s=0.4))
        v = _verdicts(findings)
        assert set(v) == set(ch.PAIRED)
        assert v["both_ends_see_connected"] == FAIL
        assert all(x in (ABSENT, FAIL) for x in v.values())
    finally:
        ca.close()
        cb.close()
        ma.close()
        mb.close()


def test_paired_golden_table_covers_every_paired_probe():
    assert set(ch.GOLDEN["sabir-hostapi-pair"]) == set(ch.PAIRED)


def test_the_single_catalog_leaves_the_client_usable(modem):
    """oversize_frame_refused deliberately provokes a close. It must put the
    attachment back, or a caller cannot run anything after it."""
    ctx = _ctx(modem)
    try:
        v = _verdicts(ch.run(ctx))
        assert v["oversize_frame_refused"] == PASS
        assert ctx.client.attached, "the catalog left the client detached"
    finally:
        ctx.client.close()


def test_a_pair_run_is_graded_against_both_tables():
    """conform --pair runs the single and paired catalogs, so the expected set
    is the union. Grading against the pair table alone reports every
    single-ended probe as an unexpected extra."""
    single, pair = ch.GOLDEN["sabir-hostapi"], ch.GOLDEN["sabir-hostapi-pair"]
    assert not (set(single) & set(pair)), "the two catalogs must not overlap"
    union = {**single, **pair}
    assert set(union) == set(ch.PROBES) | set(ch.PAIRED)
    assert len(union) == len(ch.PROBES) + len(ch.PAIRED)
