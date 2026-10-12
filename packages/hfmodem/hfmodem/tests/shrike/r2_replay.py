# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline evidence boundary for R2 pins; no hardware or production patching.

A tape can justify one changed proposal. After divergence, only audio rendered
by a reactive branch triggered by a successful fake key may enter the receiver.
Historical queries/traces are evidence, never receiver registrations.
"""
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.core import wav

from hfmodem.shrike import compress, p4breakintx, p4robusttx
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Emission, P4Receiver, Role
from hfmodem.tests.shrike.test_p4_link_0927 import Stream

ROOT = Path(__file__).resolve().parents[5]
FIXTURES = Path(__file__).parent / 'fixtures/turn-and-probe-r2'
FS = 48000


class BoundaryError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads(require_input(FIXTURES / 'manifest.json').read_text())


def canonical(value):
    if isinstance(value, bytes):
        return {'hex': value.hex()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        row = asdict(value)
        if row.get('purpose') is None:
            row.pop('purpose', None)
        if row.get('train') is False:
            row.pop('train', None)
        if row.get('train_root') is None:
            row.pop('train_root', None)
        if row.get('train_words') is None:
            row.pop('train_words', None)
        if row.get('response_generation') is None:
            row.pop('response_generation', None)
        if row.get('response_slot') is None:
            row.pop('response_slot', None)
        return canonical(row)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set)):
        return [canonical(v) for v in (sorted(value, key=repr) if isinstance(value, set) else value)]
    return value


def snapshot(turn, receiver):
    """Reviewable cut state, including source ownership and outstanding work."""
    names = ('state', 'counter', 'ladder', 'index', 'last', '_next', 'pending',
             '_deferred', '_hold_until', '_rollback_until', '_refused_due',
             '_waiting_ack', '_waiting_until', '_deferred_ack', '_deferred_slot',
             '_irs_silent_until', '_data_packet', '_speculative')
    return canonical(dict(role=receiver.last_emission.role if receiver.last_emission else None,
                          turn={n: getattr(turn, n, None) for n in names},
                          queued=bytes(turn.outbox.arq._outbuf),
                          inflight=turn.outbox.inflight,
                          receiver_clock=receiver.clock,
                          pending_body=receiver.pending_body,
                          heads=receiver._body_heads,
                          emissions=receiver.emissions,
                          gaps=receiver.capture_gaps))


class Tape:
    def __init__(self, arm, *, contract=None, root=RECORD):
        m = manifest()
        if arm not in m['fixtures']:
            raise BoundaryError('unmanifested fixture')
        self.contract = contract if contract is not None else m['fixtures'][arm]
        self.root = Path(root)
        self.arm = arm
        self.divergence = None
        self.registered = set()
        self.validate()
        fs, raw = wavfile.read(wav.source(str(self.root / self.contract['stream'])))
        self.audio = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            self.audio /= -np.iinfo(raw.dtype).min
        if fs != FS or self.audio.ndim != 1 or len(self.audio) != self.contract['metadata']['samples']:
            raise BoundaryError('stream shape/rate mismatch')

    def input_path(self, name):
        local = self.root / name
        if local.exists():
            return local
        # Capture links locate the corresponding immutable historical log.
        main = (self.root / self.contract['stream']).resolve().parents[2]
        return require_input(main / name)

    def validate(self):
        if self.contract != manifest()['fixtures'][self.arm]:
            raise BoundaryError('contract differs from frozen manifest')
        meta = self.contract['metadata']
        if meta.get('first_stream_sample', 0) != 0 or meta['samplerate'] != FS:
            raise BoundaryError('rebased stream origin or rate')
        for name, expected in self.contract['files'].items():
            path = self.input_path(name)
            if digest(path) != expected:
                raise BoundaryError(f'mutated fixture: {name}')
        actual = json.loads(self.input_path(f'captures/onair-{self.arm}/stream.json').read_text())
        if actual != meta:
            raise BoundaryError('metadata differs from frozen contract')
        masks = [[e['audio_start_sample'], e['audio_end_sample']] for e in self.contract['tx_events']]
        if masks != self.contract['own_tx_masks']:
            raise BoundaryError('own TX masks differ from journal')

    def diverge(self, at):
        if self.divergence is not None:
            raise BoundaryError('divergence already fixed')
        self.divergence = at

    def intervals(self, start, end):
        return [(max(start, a), min(end, b + self.contract['settle_samples']))
                for a, b in self.contract['own_tx_masks']
                if a < end and b + self.contract['settle_samples'] > start]

    def read(self, receiver, start, end, *, needed=None, mask=True):
        if not 0 <= start <= end <= len(self.audio):
            raise BoundaryError('unavailable capture interval')
        if self.divergence is not None and end > self.divergence:
            raise BoundaryError('fixed tape after divergence requires reactive continuation')
        gaps = self.intervals(start, end)
        if gaps and not mask:
            raise BoundaryError('original TX admitted as RX')
        if needed and any(a < needed[1] and b > needed[0] for a, b in gaps):
            raise BoundaryError('needed RX overlaps original TX; use scripted continuation')
        if not self.contract['journal_available'] and not any(
                a <= start and end <= b for a, b in self.contract['trusted_rx_intervals']):
            raise BoundaryError('TX journal unavailable outside saved RX windows')
        audio = self.audio[start:end].copy()
        for a, b in gaps:
            receiver.note_gap(a, b)
            audio[a-start:b-start] = 0
        return receiver.read_window(audio, start, whole=True)

    def register_recorded(self, receiver, emission, *, available_at):
        at = emission.stream_sample
        if self.divergence is not None and at >= self.divergence:
            raise BoundaryError('invented post-divergence emission')
        if at in self.registered or at > available_at:
            raise BoundaryError('duplicate or future registration')
        row = next((e for e in self.contract['emissions'] if e['audio_start'] == at), None)
        if row is None or row.get('refused') or row['audio_end'] != at + round(emission.duration_s * FS):
            raise BoundaryError('emission has no recorded keyed provenance')
        kinds = {e['at']: e['kind'] for e in self.contract['control_trace']
                 if e.get('event') == 'Keyed'}
        expected_kind = kinds.get(at, 'word' if row.get('word') else 'short')
        expected_role = Role.IRS if expected_kind == 'word' else Role.ISS
        if emission.kind.value != expected_kind or emission.role != expected_role:
            raise BoundaryError('recorded emission kind/role mismatch')
        receiver.note_emission(emission)
        self.registered.add(at)

    def prefix(self, turn, receiver, *, end, emission_kind=Kind.SHORT):
        """Reconstruct opening from the real opening API, then feed real RX.

        This bounded prefix ends before a second local emission; longer prefixes
        must drive proposals/key outcomes, rather than import a later state.
        """
        rows = [e for e in self.contract['emissions'] if e['audio_start'] < end]
        if len(rows) != 1 or emission_kind != Kind.SHORT or rows[0].get('value') is None:
            raise BoundaryError('prefix needs an explicit longer driver replay')
        row = rows[0]
        self.register_recorded(receiver, Emission(emission_kind, Role.ISS,
                               row['audio_start'], (row['audio_end']-row['audio_start'])/FS),
                               available_at=end)
        turn.seed_opening(rows, [], values=[row['value'], 38])
        records = self.read(receiver, row['audio_end'], end)
        for reg in records:
            turn.observe(reg, now=end)
        return snapshot(turn, receiver), records


@dataclass
class Clocks:
    sample: int = 0
    wall: int = 0  # monotonic in sample ticks; independent of capture position
    lag: int = 0
    lead: int = 5280  # 110 ms TOTAL render/settle/capture reserve

    def work(self, *, read=0, refinement=0, build=0, render=0):
        self.wall += read + refinement + build + render

    def feasible_due(self, due, cycle, *, outstanding=0, wall_due=None):
        if cycle <= 0:
            raise ValueError('nonpositive peer cycle')
        wall_due = due if wall_due is None else wall_due
        while min(due-self.sample-self.lag, wall_due-self.wall) < self.lead + outstanding:
            due += cycle
            wall_due += cycle
        return due


class ReactivePeer:
    """Named branches render real waveform audio only after actual Keyed."""
    def __init__(self, branch, *, payload=b'peer', word='X27'):
        if branch not in ('ACK', 'refusal', 'body', 'silence', 'P1 continuation'):
            raise ValueError('unknown reactive branch')
        self.branch, self.payload, self.word = branch, payload, word
        self.triggers = []

    def keyed(self, proposal):
        self.triggers.append(proposal.stream_sample)
        onset = proposal.stream_sample + round((3.403 if proposal.kind in (Kind.LONG, Kind.CHIRP) else .935) * FS)
        if self.branch == 'silence':
            return []
        if self.branch == 'body':
            wave = p4breakintx.changeover_packet(self.payload, 0)
        elif self.branch == 'P1 continuation':
            from hfmodem.shrike import pactor1
            wave = pactor1.control_signal(1)
        else:
            wave = p4robusttx.robust_word('Z27' if self.branch == 'refusal' else self.word)
        return [(onset, wave)]


class ReplayStream(Stream):
    def __init__(self, length, clocks):
        super().__init__(np.zeros(length, np.float32))
        self.clocks = clocks

    def take_until(self, end):
        data = super().take_until(end)
        self.clocks.sample = self.pos
        return data

    def read_into(self, receiver, turn, end):
        origin = self.pos
        audio = self.take_until(end)
        records = receiver.read_window(audio, origin, whole=True)
        for record in records:
            turn.observe(record, now=self.clocks.sample)
        return records

    def append_peer(self, events):
        for at, wave in events:
            if at < self.pos or at + len(wave) > len(self.audio):
                raise BoundaryError('reactive audio outside available future')
            self.audio[at:at+len(wave)] += wave


class KeyOutcomes:
    def __init__(self, stream, receiver, turn, peer, *, sample_limit=60*FS, wall_limit=60*FS, attempts=64):
        self.sample_limit, self.wall_limit, self.attempt_limit = sample_limit, wall_limit, attempts
        self.stream, self.receiver, self.turn, self.peer = stream, receiver, turn, peer
        self.trace = []

    def transmit(self, *, keyed=True, reason='waveform not prepared', duration=.758, role=Role.ISS):
        if (self.stream.clocks.sample >= self.sample_limit
                or self.stream.clocks.wall >= self.wall_limit or len(self.trace) >= self.attempt_limit):
            self.turn.link_down('replay terminal limit')
            raise BoundaryError('terminal replay limit reached')
        proposal = self.turn._next
        if proposal is None:
            raise BoundaryError('no proposed emission')
        scheduled_wall = self.stream.clocks.wall + max(0, proposal.stream_sample-self.stream.clocks.sample)
        if proposal.stream_sample >= self.sample_limit or scheduled_wall >= self.wall_limit:
            self.turn.link_down('replay terminal limit')
            raise BoundaryError('terminal replay limit reached at scheduled emission')
        if not keyed:
            self.turn.refused(reason)
            self.trace.append(('Refused', canonical(proposal)))
            return None
        if self.stream.clocks.feasible_due(proposal.stream_sample, 60000) != proposal.stream_sample:
            raise BoundaryError('key is infeasible after outstanding work/lead')
        emitted = self.turn.keyed()
        self.receiver.note_emission(Emission(emitted.kind, role, emitted.stream_sample, duration))
        self.stream.append_peer(self.peer.keyed(emitted))
        self.trace.append(('Keyed', canonical(emitted)))
        return emitted


class ByteLedger:
    def __init__(self, turn, original):
        self.turn, self.original = turn, original
        self.committed = b''

    def check(self):
        queued = logical_bytes(bytes(self.turn.outbox.arq._outbuf))
        inflight = logical_bytes(self.turn.outbox.inflight)
        if self.original != self.committed + inflight + queued:
            raise AssertionError('logical source bytes lost, split, duplicated or reordered')


def logical_bytes(wire):
    out = bytearray()
    i = 0
    while i < len(wire):
        if wire[i] == compress.SB:
            if i + 1 == len(wire) or wire[i+1] not in (0x7c, 0x7d, 0x7e):
                raise AssertionError('split or invalid source escape')
            out.append(wire[i+1] - compress.SIC_CONTROL)
            i += 2
        else:
            out.append(wire[i])
            i += 1
    return bytes(out)


def recorded_receiver_trace(arm):
    """Default receiver control over original schedule, retaining real registry.

    This is a conditional fixed-tape mechanism trace, never a prediction of
    responses after keys have been omitted. It does not seed a later turn state.
    """
    tape = Tape(arm)
    rx = P4Receiver(FS, admitted=True, chirp=True, breakin=True)
    rx.enable_hold()
    rows = tape.contract['emissions']
    keyed_kinds = {e['at']: e['kind'] for e in tape.contract['control_trace']
                   if e.get('event') == 'Keyed'}
    records = []
    for i, row in enumerate(rows):
        at, end = row['audio_start'], row['audio_end']
        kind = Kind(keyed_kinds.get(at, 'word' if row.get('word') else 'short'))
        tape.register_recorded(rx, Emission(kind, Role.IRS if kind == Kind.WORD else Role.ISS,
                                           at, (end-at)/FS), available_at=end)
        stop = min(rows[i+1]['audio_start'] if i+1 < len(rows) else end+FS, len(tape.audio))
        if stop > end:
            records.extend(tape.read(rx, end, stop))
    return [dict(kind=r.observation.kind.value, word=r.observation.word,
                 at=r.observation.stream_sample, outcome=r.outcome.value,
                 responds_to=r.responds_to,
                 status=r.observation.frame.status.byte if r.observation.frame else None,
                 field=r.observation.frame.data.hex() if r.observation.frame else None)
            for r in records]


def held_window(arm, n):
    from hfmodem.shrike import rxfront
    tape = Tape(arm)
    row = tape.contract['held_windows'][str(n)]
    return rxfront.load_wav(str(tape.root / row['audio'])), row['origin']


def p3_control_trace(arm, tmp_path):
    """Receiver/driver control in a declared synthetic linked entry context.

    The initial P3 context is the existing test_entry_answer fixture, not a
    reconstructed P1 prefix. Every subsequent event comes from captured audio.
    RadioTx uses only test_grid's fake rig and fake audio bench.
    """
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_entry_answer import _Session
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    s = _Session(entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    grid = irs_grid()
    tx = driver(grid, tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    s.host.peer = tx
    tx.p3_follow_offset = 'all'
    tx.defer_p3_cs = True
    nums = range(15, 23) if arm == '1001-0215' else (14, 15)
    reads = []
    for n in nums:
        audio, origin = held_window(arm, n)
        end = origin + len(audio)
        tx.live.now = tx.live.pos = end
        slot = (end-grid.anchor)//60000+1
        tx.aim(grid, slot)
        s.rx.new_cycle()
        if n == nums[0] or arm == '1001-0321':
            onset = 1368552 if arm == '1001-0215' else 1308708+(n-14)*60000
            s.rx.control_signal(audio, origin, onset)
        else:
            onair._scan_frame(s.rx, audio, origin, tracked_only=True)
        slot = onair._p3_place_reply(grid, tx, slot)
        tx.emit_pending_cs()
        reads.append(dict(hold=n, packets=canonical([e.packet for e in s.packets]),
                          keys=tx.n, offset_hz=round(s.rx.p3_receive_offset_hz, 3),
                          fixed=s.rx._p3_offset_fixed))
    return reads


def p1_control_trace(arm):
    from hfmodem.shrike import p1rx
    tape = Tape(arm)
    rx = P4Receiver(FS, admitted=True)
    out = []
    for row in tape.contract['p1_control_windows']:
        a, b = row['start'], row['end']
        # Validate availability and own-TX exclusion before using P1 audio.
        tape.read(rx, a, b, needed=(a, b))
        hit = p1rx.cs_anchored(tape.audio[a:b], (row['logged_onset_sample']-a)/FS, FS)
        out.append(canonical(hit))
    return out
