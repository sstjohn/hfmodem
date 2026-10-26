# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Authenticated post-first-grant replay; no fabricated peer replies."""
import contextlib
import hashlib
import io
import importlib.util
import json
import re
import time
from unittest.mock import patch

import numpy as np

from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, Role
from hfmodem.tests.evidence import CAPTURES, WORKING, require_input
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

FIRST = 2928503
GRANT = 2972663
GRANT_END = 3010949
CLEAR = (3288500, 3348503, 3468503, 3528503, 3648503,
         3708503, 3828503, 3888503, 4068503)
FILL_HASH = '412153b93609307dee339415db04279f3579a6aeb1d5f42cb36eebd7da76bc50'


def replay(*, queued=False, breakin=True, legacy=False, silent=False, stall_ms=0, clock="actual", revoke_after_partial=False,
           empty_partial_work_ms=0, revoke_at_peer_capture=False, peer_capture_lag_ms=0):
    assert clock in ("actual", "declared")
    host_clock = time.perf_counter
    cap = CAPTURES / 'onair-1003-1116'
    path = require_input(cap / 'stream.flac')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        '4dfc346722aab08c5e3d644d3bcc1a8ac2419f750d5d384565fc4a4d9153e597')
    for name, digest in (
            ('p4-probe.json', '708872174e7b60ae4526668009b9eeb7aa9f48cf4843a8a2a605d557dd4f1798'),
            ('stream.json', 'd3a6d6e67cf2942e45327fd2d43a076519c6ecb0efd75b4142c4e213eda450ea')):
        assert hashlib.sha256((cap / name).read_bytes()).hexdigest() == digest
    recorded = json.loads((cap / 'p4-probe.json').read_text())
    metadata = json.loads((cap / 'stream.json').read_text())
    audio = onair.session.load_wav(str(path), FS).astype(np.float32)
    assert len(audio) == metadata['samples']
    intervals = [(e['audio_start'], e['audio_end']) for e in recorded['emissions']]
    for left, right in intervals:
        audio[left:right] = 0
    if silent:
        audio[GRANT_END:] = 0
    stream = Stream(audio)
    stream.pos = stream.wall = GRANT_END
    wall_mark = time.perf_counter()

    def sample_now():
        nonlocal wall_mark
        now = time.perf_counter()
        stream.wall = max(stream.pos, stream.wall + round((now - wall_mark) * FS))
        wall_mark = now
        return stream.wall

    # Charge all host work, including feed/render/bookkeeping, once. Capture
    # waits advance the sample clock; they do not erase already spent work.
    stream.sample_now = (sample_now if clock == "actual" else
                         lambda: max(stream.pos, stream.wall))
    stream.transmit = True
    stream.key_notice = round(.041 * FS)
    stream._blk = 128
    stream.tx_latency_n = 336
    stream._callback_timing = [(0, 128, 0., 0), (128, 128, .0001, 0), (256, 128, .0002, 0)]
    stream.clamp_late = lambda at: max(0, stream.sample_now() + stream.key_notice - at)
    tx = Tx(stream)
    tx.settle = .041
    tx.p4_probe.emissions.clear()
    tx.p4_probe.values = (7, 38)
    tx.p4_probe.peer_breakin = True
    args = tx.p4_link_options
    for name, value in dict(data='body', ladder='none', normal_ack='parity',
                            hold_cycles=16, breakin='body', chirp_ack='alternate',
                            train_policy='repeat', irs_silent_cycles=0,
                            rx_scan='legacy', rx_timing_trace=True).items():
        setattr(args, 'p4_' + name, value)
    args.p4_link_timeout = (len(audio) - stream.pos) / FS
    if queued:
        log_path = (WORKING / 'onair-1003-1116' /
                    'pactor-p4-bench-ws8eoc-40-force-20261003T161619Z-p4-mail.log')
        assert hashlib.sha256(log_path.read_bytes()).hexdigest() == (
            '0247569849b2f11467da321d04da9ac36cdeab19cd12ee4e0362438254dea420')
        log = log_path.read_text()
        lines = re.findall(r'^mail: we queued: (.*)$', log, re.M)
        proposal = ('\r'.join(lines) + '\r').encode('latin-1')
        assert len(lines) == 2 and len(proposal) == 35
        tx.p4_link_host.arq.on_host_data(proposal)
    rx = tx.sessrx.p4
    rx.cfo_hz = -1.48
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, FIRST, .330, 1.25, -1.48)
    rx.peer_onset = FIRST
    rx._peer_heads = [FIRST - 60000, FIRST]
    take_until = stream.take_until

    captures = []
    def take(end):
        machine = machines[0] if machines else None
        captures.append(dict(requested=end, available=stream.sample_now(),
            state=None if machine is None else machine.state,
            last_state=None if machine is None or machine.last is None else machine.last.state,
            last_key=None if machine is None or machine.last is None else machine.last.stream_sample,
            emission=None if rx.last_emission is None else rx.last_emission.stream_sample,
            grant=rx.peer_normal_grant, generation=rx._train_generation))
        stream.sample_now()
        start = stream.pos
        part = take_until(end)
        stream.sample_now()
        if end == FIRST + 6 * 60000 + round(.810 * FS):
            if revoke_at_peer_capture:
                rx._train_generation += 1
            if peer_capture_lag_ms:
                stream.wall = stream.sample_now() + round(peer_capture_lag_ms * FS / 1000)
        for left, right in intervals:
            if start < right and stream.pos > left:
                rx.note_gap(max(left, start), min(right, stream.pos))
        return part

    stream.take_until = take
    lead = onair._prekey_lead(stream, round(tx.settle * FS))
    key_lead = max(round(tx.settle * FS) + stream.tx_latency_n, stream.key_notice)
    original_key = tx._tx
    keys, reads, cycles = [], [], []

    def key(wave, what):
        at = tx.raster.boundary(tx.slot)
        now = stream.sample_now()
        if now > at - key_lead:
            raise onair._MissedTxSlot('modeled enqueue/settle deadline')
        keys.append(dict(at=at, now=now, lead=lead, key_lead=key_lead,
                         read_margin=at - lead - now, margin=at - key_lead - now))
        original_key(wave, what)
        stream.pos = max(stream.pos, tx.tx_end)
        stream.wall = max(stream.wall, stream.pos)

    tx._tx = key
    # Reader measurements are diagnostics; sample_now charges nested work too.
    # Declared protocol controls charge each enclosing reader; inner CPU time is
    # not an actual deadline measurement. The separate actual-clock test retains
    # the unmodified host-work clock and strict first-answer assertion.
    read_depth = 0
    revoked = False
    empty_partial_charged = False
    declared_ms = dict(read_partial=20, read_window=8, read_chirp=8,
                       scan_idle=5, refine=35)
    for name in ('read_partial', 'read_window', 'read_chirp', 'scan_idle', 'refine'):
        original = getattr(rx, name)

        def timed(*a, _read=original, _name=name, **kw):
            nonlocal read_depth, revoked, empty_partial_charged
            start, support = stream.sample_now(), stream.pos
            began = host_clock()
            outer = read_depth == 0
            read_depth += 1
            try:
                result = _read(*a, **kw)
            finally:
                read_depth -= 1
            spent = host_clock() - began
            if (revoke_after_partial and not revoked and _name == 'read_partial'
                    and result and any(r.observation.stream_sample == CLEAR[0] for r in result)):
                # A callback invalidates the retained reader owner after early
                # capture; later pending reads must check generation again.
                rx._train_generation += 1
                revoked = True
            charge_ms = declared_ms[_name] if outer else 0
            if clock == 'declared':
                stream.wall = stream.sample_now() + round(charge_ms * FS / 1000)
            stream.sample_now()
            if stall_ms and result and any(r.observation.stream_sample == CLEAR[0]
                                          for r in result):
                stream.wall += round(stall_ms * FS / 1000)
            host_work_ms = 0
            if (empty_partial_work_ms and not empty_partial_charged and outer
                    and _name == 'read_partial' and not result
                    and support == FIRST + 5 * 60000 + round(.810 * FS)):
                host_work_ms = empty_partial_work_ms
                stream.wall += round(host_work_ms * FS / 1000)
                empty_partial_charged = True
            reads.append(dict(reader=_name, wall_start=start, wall_end=stream.wall,
                              capture_end=support, elapsed_ms=spent * 1000,
                              declared_ms=charge_ms if clock == 'declared' else None,
                              registered=[dict(onset=r.observation.stream_sample,
                                  kind=r.observation.kind.value,
                                  crc=r.observation.frame is not None)
                                  for r in result] if result else []))
            if host_work_ms:
                reads[-1]['host_work_ms'] = host_work_ms
            return result

        setattr(rx, name, timed)
    finish_cycle = rx.finish_cycle

    def finish(at):
        value = finish_cycle(at)
        cycles.append((at, value, tuple(rx.live_onsets)))
        return value

    rx.finish_cycle = finish
    ctor, machines = p4turn.P4Turn, []

    def seeded(*a, **kw):
        m = ctor(*a, **dict(kw, role='irs'))
        m.first_body = False
        m.allow_breakin = breakin
        m._body(GRANT, 0x40, p4turn.State.ISS_GRANT)
        m.keyed()
        rx.note_emission(Emission(Kind.BODY, Role.ISS, GRANT,
                                 (GRANT_END - GRANT) / FS, packet_reply=True))
        tx.p4_probe.emissions.append(next(e.copy() for e in recorded['emissions']
                                          if e['audio_start'] == GRANT))
        machines.append(m)
        return m

    arm = onair._p4_peer_normal_reader

    def selected_arm(machine, receiver):
        arm(machine, receiver)
        if legacy and machine.state == p4turn.State.ISS_GRANT:
            receiver.peer_normal_pending = False
            receiver.peer_normal_grant = None

    with contextlib.ExitStack() as stack:
        if clock == 'declared':
            from hfmodem.tests.shrike.r2_replay import ROOT
            helper = ROOT / 'working/p4-plan-0930/designs-1001/receive_forms_replay.py'
            spec = importlib.util.spec_from_file_location('grant_declared_clock', helper)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class ClockView:
                @property
                def wall(self):
                    return stream.sample_now()
            stack.enter_context(module.declared_time(ClockView()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(p4turn, 'P4Turn', seeded))
        stack.enter_context(patch.object(onair, '_p4_peer_normal_reader', selected_arm))
        onair._run_p4_link(tx, stream, onair._MasterGrid(
            GRANT % 60000, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    m = machines[0]
    heard = tx.p4_probe.heard
    for row in heard:
        field = row.pop('data', None)
        if field is not None:
            row['field_sha256'] = hashlib.sha256(bytes.fromhex(field)).hexdigest()
            row['field_bytes'] = len(bytes.fromhex(field))
    return dict(clock=clock, revoked=revoked, captures=captures, heard=heard, emissions=tx.p4_probe.emissions, events=tx.p4_probe.link_events,
                keys=keys, reads=reads, cycles=cycles, gaps=intervals,
                reason=tx.p4_probe.reason, grants=m.grants, breakins=m.breakins,
                queued=m.outbox.queued, inflight=m.outbox.inflight,
                buffered=tx.p4_link_host.arq._buffer_raw, lead=lead)
