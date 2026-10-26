# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 session receiver: one physical-emission registry and a role-aware raster."""

from dataclasses import InitVar, dataclass, replace
from contextlib import contextmanager
from enum import StrEnum
from time import monotonic
import numpy as np
from . import p4oracle, p4words, p4robusttx, p4rx, p4shorttx, p4chirprx, p4short, p4chirp, p4robustrx, p4sf8rx, p4long, p4longtx, p4frame
from . import p4shortlevels
from .p4observation import Kind, Observation, Registered, RegisteredObservation, RESPONSE_S, WORD_HORIZON_S

WORD_SLACK_S = .035
SCAN_WINDOW_S = 1.25
SCAN_STEP_S = .400
SCAN_ALLOWANCE_S = .080


class Role(StrEnum):
    ISS = "iss"
    IRS = "irs"


class Expect(StrEnum):
    WORD = "word"
    PACKET = "packet"


@dataclass(frozen=True)
class Emission:
    kind: Kind
    role: Role
    stream_sample: int
    duration_s: float
    packet_reply: bool = False
    train: bool = False
    train_root: int | None = None
    train_words: frozenset[str] | None = None
    accepted_target: int | None = None
    current_slot: int | None = None
    current_slot_accepted: bool = True
    close_grant: InitVar[bool] = False

    def __post_init__(self, close_grant):
        object.__setattr__(self, '_close_grant', close_grant)


@dataclass
class Clock:
    expects: Expect
    next_due: int
    after_s: float
    cycle_s: float
    cfo_hz: float
    misses: int = 0
    slack_s: float = 0.035


@dataclass(frozen=True)
class GrantEmissionContext:
    """Physical pre-key facts, never receive/source/credit permission."""
    emission: Emission
    registry: list
    generation: int
    predecessor: Emission | None
    predecessor_generation: int
    packet_due: int | None
    short_packet: bool
    peer_onset: int | None
    heads: tuple[int, ...]


@dataclass
class PeerPacketWatch:
    generation: int
    onset: int
    clock: Clock
    attempts: int
    last_attempt: int | None


@dataclass
class CloseLease:
    emission: int
    first: int
    expiry: int
    first_request: int | None = None


@dataclass
class TrainLease:
    generation: int
    root: int
    active: int
    first: int
    expiry: int
    kind: Kind
    words: frozenset[str]
    max_slot: int = 0


@dataclass(frozen=True)
class PreparationFailure:
    reason: str


class P4Receiver:
    def __init__(self, fs=48000, *, admitted=False, cfo_hz=None, chirp=False, breakin=False,
                 scan='legacy'):
        if scan not in ('legacy', 'incremental'):
            raise ValueError('unknown P4 receive scan mode')
        self.scan = scan
        self.fs = fs
        self.admitted = admitted
        self.cfo_hz = cfo_hz
        self.chirp = chirp
        self.breakin = breakin
        self.new_call()

    def new_call(self):
        self.grant_emission_context = None
        self._body_carrier_known = False
        self._word_cfo_hz = None
        self.expected = False
        self.clock = None
        self.peer_watch = None
        self._watch_read_generation = None
        self._peer_watch_evidence = {}
        self.last_emission = None
        self._close_grant_word = False
        self.observations = []
        self.registrations = []
        self.consumed = None
        self.pending_body = None
        self._answered = {}
        self._body_attempted = set()
        self._missed_due = None
        self.emissions = []
        self._responds_to = {}
        self.expected_words = None
        self.hold = None
        self._body_heads = {}
        self._body_carrier_check = False
        self.live_onsets = []
        self.peer_onset = None
        self.turnarounds = {}
        self._tau_seen = set()
        self.live_events = []
        self._peer_heads = []
        self.silent_cycles = 0
        self._counted_cycle = None
        self._counted_live_onset = -1
        self.capture_gaps = []
        self.acked_counter = None
        self.normal_deadline = None
        self.sf8_deadline = None
        self._normal_cache = {}
        self._normal_level = None
        self._normal_costs = {width: [] for width in (1, 2, 3, 4)}
        self._chirp_cache = {}
        self._chirp_headers = {}
        self._chirp_prior_cycles = {}
        self._long_cache = {}
        # A robust hypothesis must never become a normal placement (or vice versa).
        self._robust_long_cache = {}
        self._sf8_cache = {}
        self._sf8_prepared = {}
        self._sf8_noise = {}
        self._long_prepared = {}
        self._long_hint = None
        self._long_phase = None
        self._long_generation = getattr(self, '_long_generation', 0)+1
        self._sf8_suppressed = set()
        self._sf8_recoveries = {}
        self._long_frame = None
        self._long_scores = None
        # An eight-iteration pass measured about 52 ms on the receive host.
        self._long_refine_cost_s = .052
        self._normal_baseband = None
        self.normal_coherence = 0.
        self.normal_waiting = False
        self.peer_normal_pending = False
        self.peer_normal_grant = None
        self.packet_read = None
        self.close_lease = None
        self.train_lease = None
        # Keep generations distinct if this receiver is reused for another call.
        self._train_generation = getattr(self, '_train_generation', 0)
        self._response_evidence = {}
        self._answered_generations = {}
        self._scan_audio = np.empty(0, np.float32)
        self._scan_origin = None
        self._scan_cursor = None
        self._scan_after = None
        self._scan_generation = None
        self._scan_generation_number = 0
        self._scan_last_block = None
        self._scan_rescan = None
        self._scan_pending = []
        self._scan_seen = []
        self._scan_eligible = False
        self._scan_misses = 0
        self._scan_missed_due = None
        self._scan_upper_s = SCAN_ALLOWANCE_S

    def _retain_scan(self, audio, origin, carrier_end=None):
        """Queue support only; acquisition is dispatched outside decision work."""
        end = origin + len(audio)
        if not len(audio):
            return
        retained = round(6*self.fs)
        if self._scan_origin is None:
            self._scan_origin = max(origin, end-retained)
            self._scan_audio = audio[self._scan_origin-origin:].copy()
            self._scan_cursor = max(origin, carrier_end or origin)
        else:
            old_end = self._scan_origin + len(self._scan_audio)
            if end > old_end:
                new_origin = max(self._scan_origin, end-retained)
                prior = self._scan_audio[min(len(self._scan_audio),
                                             new_origin-self._scan_origin):]
                if origin > old_end:
                    self.note_gap(old_end, origin)
                gap = np.zeros(max(0, origin-max(old_end, new_origin)), np.float32)
                fresh = audio[max(origin, old_end, new_origin)-origin:]
                self._scan_audio = np.concatenate((prior, gap, fresh))
                self._scan_origin = new_origin
        if self._scan_cursor < self._scan_origin:
            if self._scan_eligible:
                self.live_events.append(dict(event='acquisition coverage lost',
                    begin=self._scan_cursor, end=self._scan_origin))
            self._scan_cursor = self._scan_origin
        if self._scan_after is None:
            self._scan_after = origin
        if carrier_end is not None:
            self._scan_after = max(self._scan_after, carrier_end)
        if self.last_emission:
            self._scan_after = max(self._scan_after, self.last_emission.stream_sample
                                   + round(self.last_emission.duration_s*self.fs))
        self._scan_cursor = max(self._scan_cursor, self._scan_after)
        generation = (self.cfo_hz, tuple(self.capture_gaps),
                      tuple((e.stream_sample, e.duration_s) for e in self.emissions))
        if generation != self._scan_generation:
            self._scan_generation_number += 1
            if (self._scan_generation is not None and self._scan_last_block is not None
                    and generation[0] != self._scan_generation[0]):
                self._scan_rescan = self._scan_last_block
            self._scan_generation = generation
        self._scan_seen = [s for s in self._scan_seen if s[0] >= self._scan_origin]

    def scan_idle(self, *, now, next_event, critical=False):
        """At most one complete acquisition block, with reserved wall slack.

        The caller supplies the next mandatory capture/read/decision event in
        stream wall samples. CPU time never supplies missing audio.
        """
        if (self.scan != 'incremental' or critical
                or not (self._scan_eligible or self._scan_pending)
                or self._scan_origin is None
                or next_event-now < self._scan_upper_s*self.fs):
            return []
        support_end = self._scan_origin + len(self._scan_audio)
        start = max(self._scan_cursor, self._scan_origin, self._scan_after)
        window = round(SCAN_WINDOW_S*self.fs)
        # Fresh chronological work already uses the new carrier and overlaps
        # the previous block. Repeated carrier refinements must not starve it.
        rescan = self._scan_rescan is not None and start+window > support_end
        if rescan:
            start = max(self._scan_rescan, self._scan_origin, self._scan_after)
        end = start + window
        if end > support_end or not self._scan_eligible:
            before = len(self._scan_pending)
            started = monotonic()
            result = self._resolve_scan(support_end)
            if len(self._scan_pending) < before:
                elapsed = monotonic()-started
                self.live_events.append(dict(event='word scan extent',
                    support_end=support_end, wall_start=now, next_event=next_event,
                    elapsed_ms=1000*elapsed))
                self._scan_upper_s = max(self._scan_upper_s, elapsed)
            return result
        started = monotonic()
        crop = self._scan_audio[start-self._scan_origin:end-self._scan_origin].copy()
        gaps = self._scan_gaps(crop, start)
        for a, b in gaps:
            crop[a:b] = 0
        allowance = self._scan_upper_s
        hits = p4oracle.word_scan(crop, self.fs, self.cfo_hz, origin=start,
                                 whole=False, gaps=gaps)
        if not rescan:
            self._scan_cursor = start + round(SCAN_STEP_S*self.fs)
        self._scan_rescan = None
        self._scan_last_block = start
        self._scan_pending.extend(hits)
        result = self._resolve_scan(support_end)
        elapsed = monotonic()-started
        self._scan_upper_s = max(self._scan_upper_s, elapsed)
        self.live_events.append(dict(event='word scan', begin=start, end=end,
            support_end=support_end, wall_start=now, next_event=next_event,
            generation=self._scan_generation_number,
            allowance_s=allowance, elapsed_ms=1000*elapsed))
        return result

    def _resolve_scan(self, support_end):
        result, pending = [], []
        for obs in self._scan_pending:
            at = obs.stream_sample
            # Retained context resolves a Y27/body extent at the crop edge.
            horizon_s = .805+WORD_SLACK_S if obs.word in ('Y27', 'BREAKIN') else self._word_horizon(obs)
            horizon = at + round(horizon_s*self.fs)
            if at < max(self._scan_origin, self._scan_after) or any(a < horizon and b > at
                    for a, b in self._scan_absolute_gaps()):
                continue
            if horizon > support_end:
                pending.append(obs)
                continue
            extent = p4oracle._extent(self._scan_audio, self.fs,
                (at-self._scan_origin)/self.fs, word=obs.word,
                gaps=self._scan_gaps(self._scan_audio, self._scan_origin))
            obs = replace(obs, extent_s=extent)
            qualification = (obs.word, obs.weak, obs.unexpected,
                             extent is not None and extent > .50)
            if any(abs(at-s) <= round(.010*self.fs) and q == qualification
                   for s, q in self._scan_seen):
                continue
            self._scan_seen.append((at, qualification))
            self._emit(self._ladder_word(obs, support_end), result)
        self._scan_pending = pending
        return result

    def _scan_absolute_gaps(self):
        margin = round(p4rx.MATCHED_DELAY*self.fs/p4rx.SAMPLE_RATE)
        return [(a-margin, b+margin) for a, b in self.capture_gaps +
            [(e.stream_sample, e.stream_sample+round(e.duration_s*self.fs))
             for e in self.emissions]]

    def _scan_gaps(self, audio, origin):
        return [(max(0, a-origin), min(len(audio), b-origin))
                for a, b in self._scan_absolute_gaps()
                if a < origin+len(audio) and b > origin]

    def enable_hold(self):
        from .p4hold import BreakinHold

        if self.hold is None:
            self.hold = BreakinHold(self.fs)
            self._body_heads.update((o.stream_sample, o) for o in self.observations
                                    if o.word == 'Y27')
        if self.expected_words is None:
            self.expected_words = set(('X27', 'Y27', 'Z27', 'V27', 'T27', 'U27', 'W27'))

    def _placed_words(self):
        own = self.clock and any(
            em.role == Role.ISS and abs(self.clock.next_due-em.stream_sample-round(
                (3.403 if em.kind in (Kind.LONG, Kind.CHIRP) else RESPONSE_S)*self.fs)) <= round(.035*self.fs)
            for em in self.emissions)
        lease = self.close_lease
        close_slot = (lease is not None and self.clock is not None
                      and lease.first - round(.035*self.fs) <= self.clock.next_due <= lease.expiry)
        train_slot = (self.train_lease is not None and self.clock is not None
                      and self.train_lease.first - round(.035*self.fs) <= self.clock.next_due
                      <= self.train_lease.first
                      + self.train_lease.max_slot*round(1.25*self.fs)
                      + round(.035*self.fs))
        return self.expected_words if own or close_slot or train_slot or len(self._peer_heads) >= 2 else None

    def _train_target(self, obs):
        lease = self.train_lease
        if (lease is None or obs.kind != Kind.WORD or obs.word not in lease.words
                or obs.stream_sample > lease.expiry or obs.stream_sample < lease.first-round(.035*self.fs)):
            return None
        cycle = round(1.25*self.fs)
        slot = round((obs.stream_sample-lease.first)/cycle)
        if (not 0 <= slot <= lease.max_slot
                or abs(obs.stream_sample-lease.first-slot*cycle) > round(.035*self.fs)
                or any(a < obs.stream_sample+round(.275*self.fs) and b > obs.stream_sample
                       for a, b in self.capture_gaps)):
            return None
        return lease.active, lease.generation, slot

    def arm_close_lease(self, emission, *, end):
        first = emission + round(RESPONSE_S*self.fs)
        self.close_lease = CloseLease(emission, first,
            min(end, first + round((2*1.25+.810)*self.fs)))

    def revoke_train(self):
        if self.train_lease is not None:
            self.train_lease = None
            self._train_generation += 1

    def _close_target(self, obs):
        lease = self.close_lease
        if (lease is None or obs.kind not in (Kind.WORD, Kind.BODY)
                or obs.kind == Kind.WORD and obs.word not in (None, 'BREAKIN', 'Y27')
                or obs.stream_sample > lease.expiry):
            return None
        cycle = round(1.25*self.fs)
        slot = round((obs.stream_sample-lease.first)/cycle)
        if (not 0 <= slot <= 2 or slot and lease.first_request is None
                or abs(obs.stream_sample-lease.first-slot*cycle) > round(.035*self.fs)
                or any(a < obs.stream_sample+round(.500*self.fs) and b > obs.stream_sample
                       for a, b in self.capture_gaps)):
            return None
        if slot == 0 and lease.first_request is None:
            lease.first_request = obs.stream_sample
        return lease.emission

    def _response_target(self, obs):
        for em in reversed(self.emissions):
            if not em.current_slot_accepted:
                continue
            after = .330 if em.role == Role.IRS else (3.403 if em.kind in (Kind.LONG, Kind.CHIRP) else RESPONSE_S)
            if (self._close_grant_word and em is self.last_emission
                    and obs.kind == Kind.WORD and obs.word == 'V27'
                    and abs(obs.stream_sample-em.stream_sample-round(RESPONSE_S*self.fs))
                    <= round(.035*self.fs)):
                return em.stream_sample
            if abs(obs.stream_sample - em.stream_sample - round(after * self.fs)) <= round(.035 * self.fs):
                if em.train and obs.kind == Kind.WORD and self._train_target(obs) is None:
                    head = (obs.provisional and obs.word in (None, 'BREAKIN')
                            and not any(a < obs.stream_sample+round(.275*self.fs)
                                        and b > obs.stream_sample
                                        for a, b in self.capture_gaps))
                    if not head:
                        continue
                if self.close_lease is not None and self.close_lease.emission == em.stream_sample:
                    if obs.kind == Kind.WORD and obs.word not in (None, 'BREAKIN', 'Y27'):
                        return (None if any(a < obs.stream_sample+round(.500*self.fs)
                                            and b > obs.stream_sample
                                            for a, b in self.capture_gaps) else em.stream_sample)
                    return self._close_target(obs)
                return em.stream_sample
        train = self._train_target(obs)
        return train[0] if train else self._close_target(obs)

    def _live(self, reg):
        obs = reg.observation
        if reg.outcome in (Registered.DUPLICATE, Registered.EXPIRED, Registered.CONTRADICTED):
            if (reg.outcome != Registered.DUPLICATE and reg.replaced and reg.replaced.provisional
                    and not (reg.replaced.kind == Kind.LONG and reg.replaced.placed)):
                self.live_onsets = [s for s in self.live_onsets
                                    if abs(s-reg.replaced.stream_sample) > self.fs//100]
            if reg.outcome != Registered.CONTRADICTED:
                return
        # A header or partial envelope does not prove a complete long body,
        # even on an established peer raster or measured turnaround.
        if obs.kind == Kind.LONG and obs.provisional and not obs.placed:
            return
        structure = (obs.kind == Kind.WORD or obs.frame is not None
                     or obs.kind == Kind.LONG or obs.kind == Kind.SHORT and (
                         obs.header in ('A', 'B') or obs.placed and obs.value is not None
                         and obs.walsh_peak is not None and obs.walsh_peak >= p4oracle.ROBUST_BLOCK_KNEE)
                     or obs.kind == Kind.NORMAL and obs.lattice >= (
                         (.35 if self.peer_onset is None else .30) if obs.placed else .33))
        if not structure:
            return
        at = obs.stream_sample
        word_live = (obs.kind == Kind.WORD and not obs.provisional and not obs.weak
                and not obs.unexpected and obs.word is not None
                and (self.expected_words is None or obs.word in self.expected_words))
        live = False
        if (self.peer_watch is not None and reg.peer_watch_generation == self.peer_watch.generation
                and obs.kind in (Kind.LONG, Kind.CHIRP) and obs.frame is not None
                and not obs.provisional and not obs.weak and not obs.unexpected):
            live |= abs(at-self.peer_watch.clock.next_due) <= round(.035*self.fs)
        if reg.responds_to is not None:
            em = next(e for e in self.emissions if e.stream_sample == reg.responds_to)
            key = 'word' if em.role == Role.IRS else 'long' if em.kind in (Kind.LONG, Kind.CHIRP) else 'short'
            samples = self.turnarounds.setdefault(key, [])
            tau = (at-em.stream_sample)/self.fs
            lease = self.close_lease
            if lease is not None and lease.emission == em.stream_sample and lease.first_request is not None:
                slot = round((at-lease.first)/round(1.25*self.fs))
                if 0 <= slot <= 2:
                    tau -= slot*1.25
            if reg.response_slot is not None and (lease is None or lease.emission != em.stream_sample):
                tau -= reg.response_slot*1.25
            timely = not samples or abs(tau-np.median(samples)) <= .010
            eligible = (obs.kind == Kind.WORD and (
                self.expected_words is None or obs.word in self.expected_words)
                or obs.kind in (Kind.NORMAL, Kind.LONG, Kind.CHIRP) and obs.frame is not None)
            live |= timely or eligible
            if eligible and not obs.provisional:
                identity = (key, reg.responds_to)
                if not timely:
                    self.live_events.append(dict(event='off-turnaround', at=at, tau=tau))
                if not obs.weak and identity not in self._tau_seen:
                    self._tau_seen.add(identity)
                    samples.append(tau)
                    del samples[:-5]
        if self.peer_onset is not None:
            cycle = self.clock.cycle_s if self.clock else 1.25
            k = round((at-self.peer_onset)/(cycle*self.fs))
            live |= abs(at-self.peer_onset-round(k*cycle*self.fs)) <= round(.010*self.fs)
        if (obs.kind == Kind.LONG and obs.placed and self.clock is not None
                and self.clock.expects == Expect.PACKET and self.clock.cycle_s == 3.75):
            # The preceding short remains peer_onset until a long CRC passes.
            # A placed long on the adopted packet clock is still live carrier.
            live |= abs(at-self.clock.next_due) <= round(self.clock.slack_s*self.fs)
        if self.peer_onset is None and not self.emissions and obs.word == 'Y27':
            live = True
        if live or word_live:
            if not any(abs(at-s) < self.fs//10 for s in self.live_onsets):
                self.live_onsets.append(at)
            if live and not obs.provisional and not (obs.kind == Kind.NORMAL and obs.frame is None
                                            and self.peer_onset is None):
                if self._peer_heads and abs(at-self._peer_heads[-1]) >= self.fs//10:
                    delta = (at-self._peer_heads[-1])/self.fs
                    if abs(delta-round(delta/1.25)*1.25) > .010:
                        self._peer_heads.clear()
                if not self._peer_heads or abs(at-self._peer_heads[-1]) >= self.fs//10:
                    self._peer_heads.append(at)
                    del self._peer_heads[:-2]
                self.peer_onset = at

    def finish_cycle(self, onset):
        if self._counted_cycle is not None and onset <= self._counted_cycle + self.fs//10:
            return self.silent_cycles
        self._counted_cycle = onset
        live = any(at > self._counted_live_onset for at in self.live_onsets)
        self._counted_live_onset = max(self._counted_live_onset,
                                       max(self.live_onsets, default=-1))
        if any(o.kind == Kind.LONG and o.provisional and o.placed
               and abs(o.stream_sample-onset) < self.fs//10
               for o in self.observations):
            self.silent_cycles = 0
            return self.silent_cycles
        # A peer can restart words off our packet raster; spend each onset once.
        if not live and self._masked(onset):
            return self.silent_cycles
        self.silent_cycles = 0 if live else self.silent_cycles + 1
        return self.silent_cycles

    def _masked(self, onset):
        # 0923 105.994-109.744: the peer kept its raster under our 0.8 s bodies.
        if self.peer_onset is None:
            return False
        cycle = round((self.clock.cycle_s if self.clock else 1.25)*self.fs)
        at = self.peer_onset + round((onset-self.peer_onset)/cycle)*cycle
        return any(em.stream_sample <= at < em.stream_sample+round(em.duration_s*self.fs)
                   for em in self.emissions)

    def note_emission(self, em, *, close_grant=False):
        potential_grant = (em.kind == Kind.BODY and em.role == Role.ISS
                           and em.packet_reply and not em.train and not em._close_grant)
        if potential_grant:
            clock = self.clock
            predecessor, generation = self.last_emission, self._train_generation
            short_packet = (self.admitted and self.expected and clock is not None
                            and clock.expects == Expect.PACKET and clock.cycle_s == 1.25
                            and self._watch_read_generation is None)
            due = clock.next_due if clock is not None else None
            onset, heads = self.peer_onset, tuple(self._peer_heads)
        close_grant = close_grant or em._close_grant
        if em.role == Role.ISS or close_grant:
            self.invalidate_long_phase()
        self.close_lease = None
        if em.train and em.role == Role.ISS and em.kind in (Kind.SHORT, Kind.BODY):
            previous = self.train_lease
            cycle = round(1.25*self.fs)
            phase = (em.stream_sample-(previous.root if previous else 0)) % cycle
            same = (previous is not None and em.train_root == previous.root
                    and em.kind == previous.kind and previous.active < em.stream_sample <= previous.expiry
                    and min(phase, cycle-phase) <= round(.035*self.fs))
            if not same:
                self._train_generation += 1
                root = em.stream_sample
                expiry = root + round(10*1.25*self.fs)
            else:
                root, expiry = previous.root, previous.expiry
            self.train_lease = TrainLease(
                self._train_generation, root, em.stream_sample,
                em.stream_sample+round(RESPONSE_S*self.fs), expiry,
                em.kind,
                em.train_words or frozenset(('X27', 'T27', 'W27', 'Z27', 'Y27')))
        else:
            self.train_lease = None
            self._train_generation += 1
        self._close_grant_word = close_grant
        if not any(e.stream_sample == em.stream_sample for e in self.emissions):
            self.emissions.append(em)
        if self.hold and em.current_slot_accepted:
            self.hold.reset('our key')
            self._body_heads.clear()
        self.last_emission = em
        self.grant_emission_context = (GrantEmissionContext(
            em, self.observations, self._train_generation, predecessor,
            generation, due, bool(short_packet), onset, heads)
            if potential_grant else None)
        self.expected = True
        if em.current_slot_accepted:
            for obs in self.observations:
                if (obs.provisional and (em.current_slot is None
                        or abs(obs.stream_sample-em.current_slot) < self.fs//10)):
                    if obs.stream_sample not in self._answered:
                        evidence = self._response_evidence.get(obs.stream_sample)
                        if evidence is not None:
                            self._answered_generations[obs.stream_sample] = (
                                evidence[0], self._train_generation)
                    self._answered.setdefault(obs.stream_sample, em.stream_sample)
        long = em.kind in (Kind.LONG, Kind.CHIRP)
        peer_long = (
            em.role == Role.IRS
            and (self.peer_watch is not None
                 or self.clock is not None and self.clock.cycle_s == 3.75)
        )
        after = (.330 if close_grant else
                 float(np.median((self.turnarounds.get("word") or [.330])))
                 if em.role == Role.IRS else (3.403 if long else RESPONSE_S))
        next_due = em.stream_sample + round(after * self.fs)
        if em.role == Role.IRS and not close_grant:
            packets = [o for o in self.observations if o.kind != Kind.WORD
                       and not o.provisional and o.stream_sample <= em.stream_sample
                       and (o.frame is not None or o.kind == Kind.SHORT)]
            if packets:
                peer = max(packets, key=lambda o: o.stream_sample)
                cycle = round((3.75 if peer.kind in (Kind.LONG, Kind.CHIRP) else 1.25)*self.fs)
                elapsed = em.stream_sample-peer.stream_sample
                next_due = peer.stream_sample+(elapsed//cycle+1)*cycle
        self.clock = Clock(
            Expect.PACKET if em.role == Role.IRS else Expect.WORD,
            next_due,
            after,
            3.75 if long or peer_long else 1.25,
            self.cfo_hz or 0.0,
            slack_s=.035 if em.role == Role.IRS else WORD_SLACK_S,
        )

    def sync_peer_watch(self, watch, *, receiving=False):
        """Turn admission owns authority; unread slots never create authority."""
        old = self.peer_watch
        if watch is None:
            self.peer_watch = None
            if receiving and self._watch_read_generation is not None and self.clock is not None:
                self.clock = replace(self.clock)
            return
        if old is None or old.generation != watch.generation:
            clock = Clock(Expect.PACKET, watch.onset+watch.cycle, .330,
                          watch.cycle/self.fs, watch.cfo_hz)
            self.peer_watch = PeerPacketWatch(watch.generation, watch.onset, clock,
                                             watch.attempts, watch.last_attempt)
            if receiving and self.clock is not None and (
                    self.clock.expects == Expect.WORD or self._watch_read_generation is not None):
                self.clock = replace(clock, next_due=watch.onset)
        else:
            old.attempts, old.last_attempt = watch.attempts, watch.last_attempt

    @contextmanager
    def scoped_clock(self, clock, *, watch_generation=None):
        """A feed may replace/revoke the owner. Such a transition wins restoration."""
        primary = self.clock
        old_generation = self._watch_read_generation
        self.clock = clock
        self._watch_read_generation = watch_generation
        try:
            yield
        finally:
            if self.clock is clock:
                self.clock = primary
            self._watch_read_generation = old_generation

    def finish_peer_watch(self, watch, onset):
        # Account the nominal consumed slot, even when admission replaced phase.
        silent = self.finish_cycle(onset)
        current = self.peer_watch
        if current is not None and abs(current.clock.next_due-onset) <= round(.035*self.fs):
            current.clock.next_due += round(current.clock.cycle_s*self.fs)
        return silent

    def note_gap(self, begin, end):
        self.capture_gaps = [(a, b) for a, b in self.capture_gaps if b > end-6*self.fs]
        self.capture_gaps.append((begin, end))
        overlaps = False
        for cache, support_s in ((self._sf8_prepared, 3.230), (self._sf8_noise, 3.230),
                                (self._sf8_cache, 3.280), (self._long_prepared, 3.230),
                                (self._robust_long_cache, 3.355), (self._sf8_recoveries, 3.280)):
            for key in list(cache):
                support_end = key[0]+round(support_s*self.fs)
                if cache is self._robust_long_cache:
                    support_end = min(support_end, cache[key]['end'])
                if begin < support_end and end > key[0]-round(.100*self.fs):
                    del cache[key]
                    overlaps = True
        self._sf8_suppressed = {key for key in self._sf8_suppressed
            if not (begin < key[0]+round(3.306*self.fs)
                    and end > key[0]-round(.100*self.fs))}
        if (overlaps or self.clock is not None and self.clock.expects == Expect.PACKET
                and begin < self.clock.next_due+round(3.306*self.fs)
                and end > self.clock.next_due-round(.100*self.fs)):
            self._long_hint = None

    def invalidate_long_phase(self):
        self._long_generation += 1
        self._long_hint = None
        self._sf8_prepared.clear()
        self._sf8_noise.clear()
        self._sf8_cache.clear()
        self._long_prepared.clear()
        self._robust_long_cache.clear()
        self._sf8_suppressed.clear()
        self._sf8_recoveries.clear()

    def set_receive_phase(self, phase):
        if phase != self._long_phase:
            self.invalidate_long_phase()
        self._long_phase = phase

    def _prune_long_preparation(self, predicted):
        for cache in (self._sf8_prepared, self._sf8_noise, self._sf8_cache,
                      self._long_prepared, self._sf8_recoveries):
            for key in list(cache):
                if key[0] < predicted-6*self.fs:
                    del cache[key]
        self._sf8_suppressed = {key for key in self._sf8_suppressed
                                if key[0] >= predicted-6*self.fs}

    def sf8_prepared(self):
        if self.clock is None or not self._sf8_allowed():
            return False
        entry = self._sf8_prepared.get((self.clock.next_due, self.clock.cfo_hz))
        return entry is not None and not isinstance(entry, PreparationFailure)

    def has_normal_long(self, onset):
        return any(v.get('kind') == 'normal' and v.get('obs') is not None
                   and abs(k[0]-onset) < self.fs//20
                   for k, v in self._long_cache.items())

    def _sf8_allowed(self):
        if self.clock is None:
            return False
        key = (self.clock.next_due, self.clock.cfo_hz)
        if self._long_hint == 'sf16':
            self._sf8_suppressed.add(key)
        return key not in self._sf8_suppressed

    def _recover_sf8(self, *a, **kw):
        key = (self.clock.next_due, self.clock.cfo_hz)
        self._prune_long_preparation(key[0])
        self._sf8_recoveries[key] = self._sf8_recoveries.get(key, 0)+1
        return p4rx.recover_robust(*a, **kw)

    def _preparation_status(self, format, result, reason, at, **fields):
        self.live_events.append(dict(event=f'{format} prepare', at=at,
            result=result, reason=reason, format_hint=self._long_hint,
            generation=self._long_generation, **fields))

    def _gaps(self, audio, origin):
        return [(max(0, a-origin), min(len(audio), b-origin)) for a, b in self.capture_gaps
                if a < origin+len(audio) and b > origin]

    def _bursts(self, audio, origin, **kwargs):
        return p4oracle.bursts(audio, self.fs, gaps=self._gaps(audio, origin), **kwargs)

    def note_refusal(self, kind, stream_sample):
        # No emitted waveform: the peer's raster is unchanged.
        pass

    def new_cycle(self, at=None):
        self.consumed = None
        if self.clock:
            if at is None:
                self.clock.next_due += round(self.clock.cycle_s * self.fs)
            else:
                while self.clock.next_due < at:
                    self.clock.next_due += round(self.clock.cycle_s * self.fs)

    def _classification(self, sample):
        return next(
            (
                o
                for o in self.observations
                if o.kind != Kind.BODY and abs(o.stream_sample - sample) < self.fs // 10
            ),
            None,
        )

    def _answered_response(self, reg, generation):
        """Resolve an acted-on provisional head across its actual next key."""
        emission = self.last_emission
        proof = self._answered_generations.get(reg.observation.stream_sample)
        return (emission is not None and proof == (generation, self._train_generation)
                and reg.replaced is not None and reg.replaced.provisional
                and reg.outcome in (Registered.CONFIRMED, Registered.CONTRADICTED,
                                    Registered.EXPIRED)
                and reg.answered == emission.stream_sample
                and self._answered.get(reg.observation.stream_sample) == reg.answered
                and any(current is reg.observation for current in self.observations))

    def register(self, obs):
        reg = self._register(obs)
        generation = self._watch_read_generation
        independent = generation is not None
        watched = generation is not None and obs.kind in (Kind.LONG, Kind.CHIRP)
        previous_watch = reg.replaced.stream_sample if reg.replaced is not None else obs.stream_sample
        if not watched and obs.kind in (Kind.LONG, Kind.CHIRP):
            generation = self._peer_watch_evidence.get(previous_watch)
            watched = generation is not None
        if (obs.kind == Kind.SHORT and obs.value is not None and not obs.provisional
                and not obs.weak and not obs.unexpected and reg.outcome != Registered.EXPIRED):
            self.invalidate_long_phase()
        if (not obs.provisional and obs.frame is not None
                and reg.outcome != Registered.EXPIRED):
            if obs.kind == Kind.LONG and obs.sf8:
                self._long_hint = 'sf8'
            elif obs.kind == Kind.LONG and any(
                    entry['obs'] is obs for entry in self._robust_long_cache.values()):
                self._long_hint = 'sf16'
            elif obs.kind != Kind.WORD:
                self.invalidate_long_phase()
        if (self.close_lease is not None and reg.replaced is not None
                and reg.outcome in (Registered.CONTRADICTED, Registered.EXPIRED)
                and reg.replaced.stream_sample == self.close_lease.first_request):
            self.close_lease.first_request = None
        previous = reg.replaced.stream_sample if reg.replaced else obs.after if obs.kind == Kind.BODY else obs.stream_sample
        target = self._responds_to.get(previous)
        evidence = self._response_evidence.get(previous)
        if (previous not in self._responds_to or target is None
                and (obs.kind in (Kind.LONG, Kind.CHIRP)
                     or self._train_target(reg.observation) is not None)):
            target = self._response_target(reg.observation)
            lease = self._train_target(reg.observation)
            evidence = lease[1:] if lease and lease[0] == target else None
        elif evidence is None and obs.kind == Kind.WORD:
            lease = self._train_target(reg.observation)
            evidence = lease[1:] if lease and lease[0] == target else None
        elif (evidence is not None and (self.train_lease is None
                or evidence[0] != self.train_lease.generation)
                and not self._answered_response(reg, evidence[0])):
            target, evidence = None, None
        self._responds_to[reg.observation.stream_sample] = target
        self._response_evidence[reg.observation.stream_sample] = evidence
        reg = replace(reg, responds_to=target,
                      response_generation=evidence[0] if evidence else None,
                      response_slot=evidence[1] if evidence else None,
                      response_owner=self if evidence else None)
        if watched:
            self._peer_watch_evidence[reg.observation.stream_sample] = generation
            reg = replace(reg, peer_watch_generation=generation,
                          primary_packet_due=(self.clock.next_due if not independent
                              and self.clock is not None and self.clock.expects == Expect.PACKET
                              and abs(obs.stream_sample-self.clock.next_due) <= round(.035*self.fs)
                              else None))
            if independent:
                reg = replace(reg, responds_to=None, response_generation=None,
                              response_slot=None, response_owner=None)
                self._responds_to[reg.observation.stream_sample] = None
                self._response_evidence[reg.observation.stream_sample] = None
        if (self.train_lease is not None and reg.response_generation == self.train_lease.generation
                and reg.response_slot == 0 and reg.outcome in (Registered.NEW, Registered.CONFIRMED)
                and obs.kind == Kind.WORD and not obs.provisional and not obs.weak
                and not obs.unexpected and (obs.word in ('T27', 'W27', 'Z27')
                    or self.train_lease.words == frozenset(('X27', 'Z27', 'Y27'))
                    and obs.word == 'X27')):
            self.train_lease.max_slot = (
                9 if self.train_lease.words == frozenset(('X27', 'Z27', 'Y27')) else 8)
        if self.hold is not None:
            self._live(reg)
            if reg.outcome != Registered.DUPLICATE:
                self.hold.peer(reg.observation)
        if self.breakin and reg.outcome != Registered.DUPLICATE:
            self.registrations.append(reg)
        if (obs.kind == Kind.BODY and obs.frame is not None
                and reg.outcome not in (Registered.DUPLICATE, Registered.EXPIRED)):
            self.revoke_train()
        return reg

    def _register(self, obs):
        if obs.fs != self.fs:
            raise ValueError("observation sample rate differs from the session")
        old = self._classification(obs.stream_sample)
        answer = self._answered.get(old.stream_sample) if old else None
        if obs.kind == Kind.BODY:
            if old is None or old.provisional or old.word != "Y27":
                return RegisteredObservation(obs, Registered.DUPLICATE)
            if any(
                o.kind == Kind.BODY
                and abs(o.stream_sample - old.stream_sample) < self.fs // 10
                for o in self.observations
            ):
                return RegisteredObservation(obs, Registered.DUPLICATE)
            obs = replace(
                obs,
                stream_sample=old.stream_sample,
                first_chip=old.stream_sample - obs.origin,
                after=old.stream_sample,
            )
            self.observations.append(obs)
            return RegisteredObservation(obs, Registered.ENRICHED)
        if old is None:
            self.observations.append(obs)
            if obs.kind in (Kind.WORD, Kind.SHORT):
                self.expected = True
            return RegisteredObservation(obs, Registered.NEW)
        outcome = Registered.DUPLICATE
        watch = self.peer_watch
        if (watch is not None and self._watch_read_generation == watch.generation
                and watch.clock.expects == Expect.PACKET
                and abs(obs.stream_sample-watch.clock.next_due) <= round(.035*self.fs)
                and obs.kind in (Kind.LONG, Kind.CHIRP) and obs.frame is not None
                and not obs.provisional and not obs.weak and not obs.unexpected
                and old.kind == Kind.SHORT and old.provisional and not old.placed
                and old.frame is None and old.header is None
                and old.value is None and old.word is None):
            # A current watched CRC packet resolves an empty short hypothesis.
            # Keep its measured coordinates and the ordinary replacement journal.
            outcome = Registered.CONFIRMED
        elif (old.kind == obs.kind == Kind.LONG and old.provisional
                and obs.provisional and not old.placed and obs.placed):
            outcome = Registered.ENRICHED
        elif (old.kind == obs.kind and (obs.kind == Kind.NORMAL or
                not old.provisional and obs.kind in (Kind.LONG, Kind.CHIRP))
                and old.frame is None and obs.frame is not None):
            if obs.kind == Kind.NORMAL:
                obs = replace(obs, stream_sample=old.stream_sample,
                              first_chip=old.stream_sample-obs.origin)
            outcome = Registered.ENRICHED
        elif (old.kind == obs.kind == Kind.NORMAL and old.frame is None
              and not old.peer_raster_tail and obs.peer_raster_tail):
            obs = replace(obs, stream_sample=old.stream_sample,
                          first_chip=old.stream_sample-obs.origin)
            outcome = Registered.ENRICHED
        elif (old.kind == obs.kind == Kind.LONG and old.frame is not None
              and obs.frame is not None and old.frame != obs.frame):
            outcome = Registered.CONTRADICTED
            self.live_events.append(dict(event='long-repeat-contradicted', at=obs.stream_sample))
        elif old.provisional and (not obs.provisional
                or old.kind == Kind.NORMAL and obs.kind == Kind.CHIRP):
            # An acquired chirp head must extend the read before its payload CRC.
            agrees = old.kind == obs.kind and all(
                getattr(old, f) is None or getattr(old, f) == getattr(obs, f)
                for f in ("word", "header", "value")
            )
            outcome = Registered.CONFIRMED if agrees else Registered.CONTRADICTED
        elif (old.kind == obs.kind == Kind.WORD and old.word == obs.word
              and not obs.provisional
              and abs(old.cfo_hz-obs.cfo_hz) > p4words.SYMBOL_RATE/2):
            outcome = Registered.ENRICHED
        elif (
            old.kind == obs.kind == Kind.WORD
            and not obs.provisional
            and old.word == obs.word
            and old.extent_s is None
            and obs.extent_s is not None
        ):
            outcome = Registered.EXTENDED
        if outcome != Registered.DUPLICATE:
            if old.stream_sample in self._body_attempted:
                self._body_attempted.add(obs.stream_sample)
            self.observations[self.observations.index(old)] = obs
            if answer is not None:
                self._answered[obs.stream_sample] = answer
                proof = self._answered_generations.get(old.stream_sample)
                if proof is not None:
                    self._answered_generations[obs.stream_sample] = proof
            if (
                self.pending_body
                and abs(self.pending_body.stream_sample - old.stream_sample)
                < self.fs // 10
            ):
                self.pending_body = obs if obs.word == "Y27" else None
            return RegisteredObservation(obs, outcome, old, answer)
        return RegisteredObservation(obs, outcome)

    def _emit(self, obs, result):
        reg = self.register(obs)
        if reg.outcome != Registered.DUPLICATE:
            result.append(reg)
            self.consumed = self.consumed or obs
        if obs.word == "Y27":
            self.pending_body = self._classification(obs.stream_sample)
            if self.hold is not None:
                for at in list(self._body_heads):
                    if abs(at-obs.stream_sample) < self.fs//10:
                        del self._body_heads[at]
                if obs.stream_sample not in self._body_attempted:
                    self._body_heads[obs.stream_sample] = self.pending_body
        if obs.kind == Kind.NORMAL and obs.frame is None:
            if self.clock and abs(obs.stream_sample-self.clock.next_due) < self.fs//10:
                self.clock.misses = 0
            return
        if (not obs.provisional or self.breakin and obs.word == "Y27"
                or self.hold is not None and obs.kind == Kind.WORD
                and not (obs.word is None and self._word_horizon(obs) == .300)
                and not obs.weak):
            if obs.kind == Kind.WORD:
                self._word_cfo_hz = obs.cfo_hz
                if obs.word in p4words.CARRIER_HZ:
                    self.cfo_hz = obs.cfo_hz-p4words.CARRIER_HZ[obs.word]
                    self._body_carrier_known = True
                elif not self._body_carrier_known:
                    self.cfo_hz = obs.cfo_hz
            else:
                self.cfo_hz = obs.cfo_hz
                self._body_carrier_known |= obs.kind in (Kind.BODY, Kind.NORMAL)
            if self.clock is None and obs.kind != Kind.BODY:
                cycle = 3.75 if obs.kind in (Kind.LONG, Kind.CHIRP) else 1.25
                self.clock = Clock(
                    Expect.WORD if obs.kind == Kind.WORD else Expect.PACKET,
                    obs.stream_sample,
                    0.923 if obs.kind == Kind.WORD else 0.330,
                    cycle,
                    self.cfo_hz,
                )
            if self.clock:
                self.clock.cfo_hz = self.cfo_hz
                self.clock.misses = 0
                if (
                    obs.kind != Kind.BODY
                    and obs.stream_sample >= self.clock.next_due - self.fs // 10
                ):
                    self.clock.next_due = obs.stream_sample
                elif obs.kind == Kind.LONG and obs.stream_sample < self.clock.next_due:
                    cycle = round(3.75*self.fs)
                    self.clock.next_due = obs.stream_sample + max(
                        0, (self.clock.next_due-obs.stream_sample+cycle-1)//cycle)*cycle
                if obs.kind in (Kind.LONG, Kind.CHIRP):
                    self.clock.cycle_s = 3.75
                elif (
                    obs.kind in (Kind.SHORT, Kind.NORMAL, Kind.BODY)
                    or obs.word == "Y27"
                ):
                    self.clock.cycle_s = 1.25

    def read_breakin(self, audio, origin, *, carrier_end, work=None, words_only=False):
        self.breakin = True
        result = []
        cut = max(0, carrier_end - origin)
        for obs in p4oracle.word_tail(audio[cut:], self.fs, origin=origin + cut,
                                      refine=True, **({'work': work} if work else {})):
            if obs.word in (('X27', 'Y27') if words_only else ('Y27',)):
                if words_only and any(a < obs.stream_sample+round(.240*self.fs)
                                      and b > max(origin+cut, obs.stream_sample)
                                      for a, b in self.capture_gaps):
                    continue
                available = origin + len(audio) - obs.stream_sample
                if available < round(0.240 * self.fs):
                    obs = replace(obs, provisional=True,
                                  symbols_read=min(20, round(available * 112.5 / self.fs)))
                self._emit(obs, result)
        if not words_only:
            result.extend(self.read_window(audio, origin, whole=True,
                                           **({'work': work} if work else {})))
        return result

    def read_hold_word(self, audio, origin, *, due, carrier_end, work):
        """One P1 answer opportunity; no packet search behind a word head."""
        result = []
        self.breakin = True
        if len(audio) > round(.680*self.fs):
            lo = max(0, due-origin-round(.405*self.fs))
            audio = np.asarray(audio)[lo:lo+round(.680*self.fs)]
            origin += lo
        cut = max(0, carrier_end-origin)
        audio, origin = np.asarray(audio)[cut:], origin+cut
        if not len(audio) or not np.any(audio):
            return result
        # P1 tones need no wide word-bank search. Nominate SF16 coherence on
        # coarse carriers first; half-symbol-rate spacing preserves offset
        # words while using the existing robust despreading threshold.
        if not work.admits('head'):
            return result
        for offset in (0., -56.25, 56.25, -112.5, 112.5):
            if not work.checkpoint('head', reserve_s=.003):
                return result
            fraction, _, _ = p4oracle.placed_despread(audio, self.fs,
                max(0., (due-origin)/self.fs), (self.cfo_hz or 0.)+offset,
                count=12, slack_s=.130)
            if fraction >= p4oracle.ROBUST_FRACTION_KNEE:
                break
        else:
            return result
        if len(audio) >= round(.275*self.fs):
            for obs in p4oracle.word_scan(audio, self.fs, self.cfo_hz,
                    origin=origin, gaps=self._gaps(audio, origin), work=work):
                intact = not any(a < obs.stream_sample+round(.240*self.fs)
                                 and b > obs.stream_sample for a, b in self.capture_gaps)
                if obs.word in ('X27', 'Y27') and intact:
                    self._emit(obs, result)
            if result or (not cut and not work.deferred
                          and origin+len(audio) >= due+round(.275*self.fs)):
                return result
        obs = p4oracle.word_at(audio, self.fs, origin, (due-origin)/self.fs,
            cfo_hz=self.cfo_hz or 0., slack_s=.130, provisional=True,
            reacquire=True, work=work)
        if obs is not None and obs.word in ('X27', 'Y27') and not obs.weak:
            intact = not any(a < obs.stream_sample+round(20/112.5*self.fs)
                             and b > max(origin, obs.stream_sample)
                             for a, b in self.capture_gaps)
            if intact:
                self._emit(obs, result)
                if (origin+len(audio) >= obs.stream_sample+round(.240*self.fs)
                        and work.admits('refine')):
                    result.extend(self.read_breakin(audio, origin, carrier_end=origin,
                                                    work=work, words_only=True))
            return result
        if work.deferred:
            return result
        fraction, _, _ = p4oracle.placed_despread(audio, self.fs,
            max(0., (due-origin)/self.fs), self.cfo_hz or 0., count=12, slack_s=.130)
        if fraction < p4oracle.ROBUST_FRACTION_KNEE:
            return result
        if not work.admits('refine'):
            return result
        return self.read_breakin(audio, origin, carrier_end=origin,
                                  work=work, words_only=True)

    def expire_hold_words(self, onsets):
        for obs in list(self.observations):
            if (not obs.provisional or obs.word not in ('X27', 'Y27')
                    or obs.stream_sample not in onsets):
                continue
            self.observations.remove(obs)
            self._body_heads.pop(obs.stream_sample, None)
            if self.pending_body == obs:
                self.pending_body = None
            reg = RegisteredObservation(obs, Registered.EXPIRED,
                obs, self._answered.get(obs.stream_sample),
                self._responds_to.get(obs.stream_sample))
            self.registrations.append(reg)
            self._live(reg)

    def _word(self, audio, origin, *, provisional=False, whole=False):
        onset = (self.clock.next_due-origin)/self.fs
        words = self._placed_words()
        slack = self.clock.slack_s
        cfo = self.clock.cfo_hz
        offsets = None
        if self._word_cfo_hz is not None:
            # Z27 has no measured body offset; preserve its legacy word reference.
            offsets = dict(p4words.CARRIER_HZ) if self._body_carrier_known else {}
            for name in p4words.BANK.keys()-p4words.CARRIER_HZ.keys():
                offsets[name] = self._word_cfo_hz-cfo
        # An unnamed head has only an absolute carrier, not a body reference.
        for head in reversed(self.observations):
            if (head.word is None and head.kind == Kind.WORD and head.provisional
                    and not head.weak and self._word_horizon(head) == .300
                    and abs(head.stream_sample-self.clock.next_due) <= round(slack*self.fs)):
                cfo = head.cfo_hz
                offsets = {}
                break
        obs = p4oracle.word_at(audio, self.fs, origin, onset,
            cfo_hz=cfo, slack_s=slack, provisional=provisional, whole=whole,
            expected_words=words, diagnostics=self.live_events, gaps=self._gaps(audio, origin),
            carrier_offsets=offsets, reacquire=True)
        if obs is None and provisional and self.hold is not None and self.cfo_hz is None:
            obs = p4oracle.word_at(audio, self.fs, origin, onset,
                cfo_hz=None, slack_s=slack, provisional=True, whole=whole, expected_words=words,
                diagnostics=self.live_events, gaps=self._gaps(audio, origin))
        if (obs is None and self.hold is not None and self.cfo_hz is None
                and p4oracle.placed_despread(audio, self.fs, onset, 0., count=20)[0]
                >= p4oracle.ROBUST_FRACTION_KNEE):
            for candidate in p4oracle.word_tail(audio, self.fs, origin=origin, refine=True):
                if abs(candidate.stream_sample-self.clock.next_due) > round(slack*self.fs):
                    continue
                obs = p4oracle.word_at(audio, self.fs, origin, onset,
                    cfo_hz=candidate.cfo_hz-p4words.CARRIER_HZ.get(candidate.word, 0.), slack_s=slack, provisional=provisional, whole=whole,
                    expected_words=words, diagnostics=self.live_events, gaps=self._gaps(audio, origin))
                if obs is not None:
                    break
        return self._ladder_word(obs, origin + len(audio))

    def read_turn_word(self, audio, origin):
        """The IRS can hear a held V27 at word + .935 instead of a packet at + .330."""
        em = self.last_emission
        if (self.clock is None or self.clock.expects != Expect.PACKET
                or self.clock.cycle_s != 1.25 or em is None
                or (em.kind != Kind.WORD and not self._close_grant_word)
                or em.role != Role.IRS):
            return []
        cycle = round(1.25*self.fs)
        at = em.stream_sample + round(RESPONSE_S*self.fs)
        at += max(0, round((self.clock.next_due-at)/cycle))*cycle
        args = (audio, self.fs, origin, (at-origin)/self.fs)
        kwargs = dict(slack_s=.035, expected_words=self.expected_words,
                      gaps=self._gaps(audio, origin))
        obs = p4oracle.word_at(*args, **kwargs,
            cfo_hz=self.clock.cfo_hz, whole=True, diagnostics=self.live_events)
        if obs is None:
            head = p4oracle.word_at(*args, **kwargs, cfo_hz=None, provisional=True)
            if head is not None and head.word == 'V27':
                obs = p4oracle.word_at(*args, **kwargs,
                    cfo_hz=head.cfo_hz, whole=True, diagnostics=self.live_events)
        result = []
        if (obs is not None and obs.word == 'V27' and not obs.provisional
                and not obs.weak and not obs.unexpected
                and (obs.extent_s is None or obs.extent_s <= .50)):
            packet_due = self.clock.next_due
            self._emit(obs, result)
            if self._close_grant_word:
                # The grant's word answer is evidence beside its packet raster.
                self.clock.next_due = packet_due
            else:
                self.clock.expects = Expect.WORD
        return result

    def _ladder_word(self, obs, end):
        if obs is not None and self._word_horizon(obs) == .300:
            if end < obs.stream_sample + round(.300 * self.fs):
                return replace(obs, word=None, provisional=True, unexpected=False)
        return obs

    def _word_horizon(self, obs):
        target = self._response_target(obs)
        if (obs.word in (None, 'BREAKIN', 'Y27')
                and any(e.stream_sample == target and e.role == Role.ISS and e.kind == Kind.SHORT
                        for e in self.emissions)):
            return .300
        return WORD_HORIZON_S

    def _reply_carrier(self, audio, origin):
        if self.last_emission is None or not self.last_emission.packet_reply:
            return False
        begin = max(0, self.clock.next_due-round(.15*self.fs)-origin)
        if begin >= len(audio):
            return False
        for start, duration in self._bursts(audio[begin:], origin+begin, bridge_s=.035):
            at = origin+begin+round(start*self.fs)
            if (abs(at-self.clock.next_due) <= round(.05*self.fs)
                    and at+round(duration*self.fs) >= self.clock.next_due+round(.10*self.fs)):
                return True
        # A filled window can hide the energy edge. A coherent normal header
        # also reserves the packet window, without accepting any source bytes.
        z = p4rx.audio_to_baseband(np.asarray(audio)[begin:], self.fs)
        nominal = round((self.clock.next_due-origin-begin)*28800/self.fs)+p4short.GEOM
        positions = nominal+128+16*(np.arange(p4shorttx.HEADER_CHIPS)-p4shorttx.HEADER_CHIPS)
        indices = p4short.WIDE[:, None]+positions
        indices = indices[(indices[:, 0] >= 0) & (indices[:, -1] < len(z))]
        if not len(indices):
            return False
        matched = p4rx.matched_filter(z*np.exp(-2j*np.pi*self.clock.cfo_hz*np.arange(len(z))/28800))
        chips = matched[indices]
        energy = np.sqrt(np.sum(abs(chips)**2, axis=1)*p4shorttx.HEADER_CHIPS)
        scores = abs(chips @ np.array([p4shorttx.HEADER, p4shorttx.HEADER_B]).conj().T)
        return bool(np.max(scores/np.maximum(energy[:, None], 1e-30)) >= .35)

    def _grant_take(self, audio, origin, result):
        em = self.last_emission
        if (em is None or not em.packet_reply or em.role != Role.ISS
                or self.peer_onset is None or self.clock is None
                or self.clock.expects != Expect.WORD):
            return
        cycle = round(1.25*self.fs)
        at = self.peer_onset + ((em.stream_sample-self.peer_onset)//cycle+1)*cycle
        if any(o.kind == Kind.NORMAL and o.peer_raster_tail
               and abs(o.stream_sample-at) < self.fs//10 for o in self.observations):
            return
        end = at+round(.805*self.fs)
        carrier_end = em.stream_sample+round(em.duration_s*self.fs)
        latest_end = end+round(.030*self.fs)
        if origin+len(audio) < latest_end:
            return
        cut = max(0, carrier_end-origin)
        frame = round(p4oracle.rxfront.OCCUPANCY_FRAME_S*self.fs)
        # Place a frame edge on the last permitted tail sample. Read past it
        # when available so an active run cannot borrow the capture boundary
        # as its end.
        shift = (latest_end-origin-cut) % frame
        start = cut+shift
        scan_end = min(len(audio), latest_end+2*frame-origin)
        if start >= scan_end:
            return
        # A delayed burst leaves quiet at the opening. A continuous burst
        # leaves quiet after its observed end. Both floors come from audio.
        scan_levels, _ = p4oracle._frames(audio[start:scan_end], self.fs)
        if len(scan_levels) < 8:
            return
        opening = scan_levels[:8]
        peak = float(np.percentile(scan_levels, 90))
        first = float(np.median(opening[:4]))
        if first < peak-20:
            floor = first
        elif scan_levels[-1] < peak-20:
            floor = float(scan_levels[-1])
        else:
            return
        for burst_start_s, duration in self._bursts(audio[start:scan_end], origin+start,
                                                    bridge_s=.035, knee_db=20,
                                                    floor_db=floor):
            burst_start = origin+start+round(burst_start_s*self.fs)
            burst_end = burst_start+round(duration*self.fs)
            # _bursts closes an active run at its input boundary too. Only a
            # following, measured quiet frame proves this is a burst end.
            end_frame = round((burst_end-origin-start)/frame)
            if (end_frame >= len(scan_levels)
                    or scan_levels[end_frame] > floor+20):
                continue
            if (burst_start <= carrier_end+shift
                    and abs(burst_end-end) <= round(.030*self.fs)):
                obs = Observation(Kind.NORMAL, self.fs, origin, at-origin, at,
                                  .805, self.cfo_hz, 0., placed=True,
                                  peer_raster_tail=True)
                self._emit(obs, result)
                self.clock = replace(self.clock, expects=Expect.PACKET,
                                     next_due=at+cycle, after_s=.330,
                                     cycle_s=1.25, slack_s=.035, misses=0)
                self._normal(audio, origin, predicted=at)
                return

    def restore_grant_clock(self, onset):
        """Resume the grant's word-slot decision before its next key deadline."""
        if self.clock is not None:
            self.invalidate_long_phase()
            self.clock = replace(self.clock, expects=Expect.WORD,
                                 next_due=onset, after_s=RESPONSE_S,
                                 cycle_s=1.25, slack_s=WORD_SLACK_S, misses=0)

    def read_partial(self, audio, origin, *, deadline):
        result = []
        if not (self.admitted or self.expected) or not self.clock:
            return result
        audio = np.asarray(audio)[: max(0, deadline - origin)]
        word = self._word(audio, origin, provisional=True)
        if word:
            self._emit(word, result)
        elif (self.clock.expects == Expect.PACKET
              or self._reply_carrier(audio, origin)):
            self.clock.expects = Expect.PACKET
            obs = self._packet(audio, origin, partial=True)
            if obs:
                self._emit(obs, result)
        if word is None:
            self._grant_take(audio, origin, result)
        self._read_peer_normal(audio, origin, result)
        return result

    def _normal_eligible(self):
        return (self.hold is not None and self.clock is not None
                and self.clock.expects == Expect.PACKET and self.clock.cycle_s in (1.25, 3.75)
                and (len(self._peer_heads) >= 2 or any(e.role == Role.IRS for e in self.emissions)
                     or self.last_emission is not None and self.last_emission.packet_reply))

    def _qam_work_eligible(self, entry):
        owner = entry.get('qam_owner')
        if owner is None:
            return True
        generation, expects, due, at = owner
        allowed = (self.admitted and self.clock is not None
                   and generation == self._train_generation and self.clock.expects == expects
                   and abs(self.clock.next_due-due) < self.fs//20
                   and (self._normal_eligible() or self.peer_normal_pending
                        and self.clock.expects == Expect.WORD)
                   and not entry['decoder']._expired())
        end = at+round(.805*self.fs)
        allowed = allowed and not any(a < end and b > at for a,b in self._scan_absolute_gaps())
        if not allowed:
            entry['decoder'].discard()
        return allowed

    def _normal(self, audio, origin, *, predicted=None):
        predicted = self.clock.next_due if predicted is None else predicted
        for old in list(self._normal_cache):
            if old < predicted-6*self.fs:
                del self._normal_cache[old]
        key = next((k for k in self._normal_cache if abs(k-predicted) < self.fs//20), predicted)
        entry = self._normal_cache.get(key)
        self.hold.acked_counter = self.acked_counter
        if entry is not None and entry.get('complete'):
            obs = entry.get('obs')
            if obs is None:
                return None
            if not self._qam_work_eligible(entry):
                return replace(obs, frame=None, speedlevel=None)
            if (entry.get('legacy') and self.hold.kind == 'normal'
                    and self.hold.last == obs.stream_sample):
                deadline = self.normal_deadline(obs.stream_sample) if self.normal_deadline else None
                frame = self.hold._normal_decode(deadline)
                if frame is not None:
                    entry['decoder'] = None
                    self._normal_level = 1
                    obs = entry['obs'] = replace(obs, frame=frame)
            if entry.get('decoder') is not None and not obs.frame:
                deadline = self.normal_deadline(obs.stream_sample) if self.normal_deadline else None
                frame = entry['decoder'].decode(deadline)
                if frame is not None:
                    self._normal_level = entry['decoder'].width
                    self.hold.reset('single-copy normal')
                    obs = entry['obs'] = replace(obs, frame=frame,
                        speedlevel='SL9/10-short' if self._normal_level == 4 else None)
            return obs
        lo = max(0, predicted-origin-round(.050*self.fs))
        if lo >= len(audio):
            self.normal_waiting = True
            return None
        z = p4rx.audio_to_baseband(np.asarray(audio)[lo:], self.fs)
        nominal = round((predicted-origin-lo)*28800/self.fs)+p4short.GEOM
        self._normal_baseband = (z, origin+lo, nominal)
        acquisition = (self.peer_onset is None
                       or self.last_emission is not None and self.last_emission.packet_reply)
        offsets = None
        if self.last_emission is not None and self.last_emission.packet_reply:
            # A reply can start before the predicted word slot. Read complete
            # placements now without waiting for the latest possible onset.
            offsets = [o for o in p4short.WIDE if p4short.complete(z, nominal+int(o))]
            if not offsets:
                self.normal_waiting = True
                return None
        result = p4short.place_short(z, nominal, self.clock.cfo_hz,
                                    acquisition=acquisition, offsets=offsets)
        if (not acquisition and result is not None
                and result[0] < p4short.NORMAL_ACQUIRE_KNEE):
            offsets = [o for o in p4short.WIDE if p4short.complete(z, nominal+o)]
            recovered = p4short.place_short(z, nominal, self.clock.cfo_hz,
                                            acquisition=True, offsets=offsets)
            # A marginal tracked placement remains usable if acquisition fails.
            # SL7's 2045 c5 needs this retry even above the tracking knee.
            if (result[0] < p4short.NORMAL_COHERENCE_KNEE
                    or recovered[0] >= p4short.NORMAL_ACQUIRE_KNEE):
                acquisition = True
                result = recovered
        self.normal_waiting = result is None
        if result is None:
            self._normal_cache[key] = dict(complete=False)
            return None
        coherence, training, cfo = result
        self.normal_coherence = coherence
        self._normal_cache[key] = entry = dict(complete=True, obs=None)
        knee = p4short.NORMAL_ACQUIRE_KNEE if acquisition else p4short.NORMAL_COHERENCE_KNEE
        if coherence < knee:
            if offsets is not None and len(offsets) < len(p4short.WIDE):
                entry['complete'] = False
                self.normal_waiting = True
            return None
        equalized = []
        scored = p4short.short_scores(z, first_training_start=training, frequency_offset=cfo, refine=0,
                                     on_equalized=lambda symbols, noise: equalized.extend((symbols, noise)))
        if scored is None:
            entry['complete'] = False
            self.normal_waiting = True
            return None
        at = predicted+round((training-nominal)*self.fs/28800)
        heard = p4shorttx.read_header(z, training, cfo)
        if heard is not None:
            self.live_events.append(dict(event='normal-header', at=at, header=heard[0],
                                         a=round(heard[1], 3), b=round(heard[2], 3)))
        obs = Observation(Kind.NORMAL, self.fs, origin, at-origin, at, .805, cfo,
                          coherence, lattice=coherence, placed=True)
        deadline = self.normal_deadline(at) if self.normal_deadline else None
        ranked = p4shortlevels.candidates(*equalized, scored[2], self._normal_level, qam=True)
        qam = any(width == 4 and p4shortlevels.agreement(fields)
                  >= p4shortlevels.LEVEL_CHANGE_AGREEMENT for width, fields in ranked)
        if not qam:
            ranked = [pair for pair in ranked if pair[0] != 4]
        frame = None
        if not qam and (ranked[0][0] == 1 or p4shortlevels.agreement(ranked[0][1])
                       < p4shortlevels.LEVEL_CHANGE_AGREEMENT):
            entry['legacy'] = True
            frame = self.hold.normal(at, training-nominal, cfo, coherence, scored[2],
                                     (z, nominal, self.clock.cfo_hz, at), deadline=deadline)
            if frame is not None:
                self._normal_level = 1
            ranked = [pair for pair in ranked if pair[0] != 1]
        if frame is None:
            decoder_type = p4shortlevels.QamShortDecode if qam else p4shortlevels.ShortDecode
            entry['decoder'] = decoder_type(
                ranked, costs=self._normal_costs, events=self.hold.events, at=at,
                status_ok=self.hold.status_ok, now=self.hold.now)
            if qam:
                entry['qam_owner'] = (self._train_generation, self.clock.expects,
                                      self.clock.next_due, at)
                entry['decoder'].status_ok = lambda frame, **kwargs: (
                    self._qam_work_eligible(entry)
                    and self._normal_status_current(frame, **kwargs))
            frame = entry['decoder'].decode(deadline)
            if frame is not None:
                self._normal_level = entry['decoder'].width
                self.hold.reset('single-copy normal')
        entry['obs'] = replace(obs, frame=frame,
            speedlevel='SL9/10-short' if frame is not None and self._normal_level == 4 else None)
        return entry['obs']

    def _normal_status_current(self, frame, **kwargs):
        self.hold.acked_counter = self.acked_counter
        return self.hold.status_ok(frame, **kwargs)

    def refine(self, *, deadline=None):
        body_work = self.hold.kind == 'body' and self.hold.work is not None and self.hold.frame is None
        started = monotonic()
        frame = self.hold.refine(deadline=deadline)
        if body_work:
            self.live_events.append(dict(event='body decode', stage='refine', at=self.hold.last,
                elapsed_ms=1000*(monotonic()-started), accepted=frame is not None,
                budget_ms=None if deadline is None else 1000*(deadline-started),
                copies=len(self.hold.members)))
        result = []
        if frame is not None and self.hold.kind == 'normal':
            for entry in self._normal_cache.values():
                obs = entry.get('obs')
                if obs is not None and obs.stream_sample == self.hold.last:
                    entry['obs'] = replace(obs, frame=frame)
                    self._emit(entry['obs'], result)
        elif frame is not None and self.hold.kind == 'body':
            head = self._classification(self.hold.last)
            if head is not None and head.word == 'Y27' and not head.provisional:
                self._body(head, frame, head.origin, result)
        latest = self._normal_cache.get(max(self._normal_cache, default=0))
        for entry in (() if latest is None else (latest,)):
            decoder = entry.get('decoder')
            obs = entry.get('obs')
            if obs is not None and decoder is not None and not decoder.done and obs.frame is None:
                if not self._qam_work_eligible(entry):
                    continue
                limit = monotonic()+.250 if deadline is None else deadline
                if isinstance(decoder, p4shortlevels.QamShortDecode) and self.normal_deadline:
                    limit = min(limit, self.normal_deadline(obs.stream_sample))
                frame = decoder.decode(limit)
                if frame is not None:
                    self._normal_level = decoder.width
                    self.hold.reset('single-copy normal')
                    entry['obs'] = replace(obs, frame=frame,
                        speedlevel='SL9/10-short' if decoder.width == 4 else None)
                    self._emit(entry['obs'], result)
        for entry in self._long_cache.values():
            if (entry.get('kind') != 'normal' or entry.get('scores') is None
                    or entry.get('refined') or entry.get('obs') is None):
                continue
            if deadline is not None and monotonic() >= deadline:
                break
            scores = entry.get('score_sets', (entry['scores'],))
            attempts = [(field, copies, polarity)
                        for copies, field in enumerate(scores, start=1)
                        for polarity in (1, -1)]
            while entry.get('refine_index', 0) < len(attempts):
                if (deadline is not None and
                        deadline-monotonic() < self._long_refine_cost_s):
                    break
                index = entry.get('refine_index', 0)
                field, copies, polarity = attempts[index]
                started = monotonic()
                frame = p4long.decode_scores(field, iterations=8, polarity=polarity)
                elapsed = monotonic()-started
                self._long_refine_cost_s = max(.001, elapsed)
                entry['refine_index'] = index+1
                self.live_events.append(dict(event='long-decode', at=entry['obs'].stream_sample,
                    iterations=8, polarity=polarity, elapsed_ms=1000*elapsed,
                    crc_valid=frame is not None, copies=copies))
                if frame is not None:
                    entry['obs'] = replace(entry['obs'], frame=frame, provisional=False, extent_s=3.282)
                    self._remember_long(frame)
                    self._long_scores = None
                    self._emit(entry['obs'], result)
                    break
            entry['refined'] = entry.get('refine_index', 0) == len(attempts) or entry['obs'].frame is not None
        return result

    def _body(self, head, frame, origin, result):
        if head.extent_s is None:
            self._emit(replace(head, extent_s=.796), result)
        self._emit(Observation(
            Kind.BODY, self.fs, origin, head.stream_sample-origin,
            head.stream_sample, .796, head.cfo_hz-p4words.CARRIER_HZ["Y27"], head.coherence,
            after=head.stream_sample, frame=frame), result)

    def _remember_long(self, frame):
        if self._long_frame is None or self._long_frame[0] != frame:
            field = p4frame.encode_field(frame.data, frame.status)
            self._long_frame = frame, p4longtx.body_symbols(p4longtx.coded_bits(field))

    def _long_agreement(self, z, start, cfo):
        if self._long_frame is None:
            return 0.
        matched = p4rx.matched_filter(z * np.exp(
            -2j*np.pi*cfo*np.arange(len(z))/28800))
        expected = self._long_frame[1]
        numerator = denominator = 0.
        for block in range(5):
            training = p4short._LONG_TRAIN[block]
            idx = start + 128 + 16*(239*block+np.arange(4, 28))
            if idx[-1] >= len(matched):
                return 0.
            gain = np.vdot(training[4:28], matched[idx]) / np.vdot(training[4:28], training[4:28])
            data = start + 128 + 16*(p4long.LAYOUT.data_offsets[block]
                                     + np.arange(p4long.LAYOUT.block_symbols))
            if data[-1] >= len(matched) or abs(gain) == 0:
                return 0.
            symbols = (matched[data]/gain).real
            want = expected[p4long.LAYOUT.data_offsets[block]:
                            p4long.LAYOUT.data_offsets[block]+p4long.LAYOUT.block_symbols].real
            numerator += float(symbols @ want)
            denominator += float(np.sum(abs(symbols)))
        return numerator/denominator if denominator else 0.

    def _normal_long(self, audio, origin, *, partial=False, alias=False):
        predicted, fs = self.clock.next_due, self.fs
        self._long_cache = {k: v for k, v in self._long_cache.items()
                            if k[0] >= predicted-6*fs}
        key = next((k for k, v in self._long_cache.items()
                    if v.get('kind') == 'normal' and abs(k[0]-predicted) < fs//20),
                   (predicted, self.clock.cfo_hz))
        entry = self._long_cache.get(key)
        end = origin+len(audio)
        if entry is not None and (entry.get('obs') is None
                                  or entry.get('complete') or end <= entry['end']):
            return entry.get('obs'), entry['coherence']
        if self._normal_baseband is not None:
            z, base_origin, nominal = self._normal_baseband
        else:
            lo = max(0, predicted-origin-round(.050*fs))
            available = np.asarray(audio)[lo:]
            if len(available) == 0:
                return None, 0.
            z = p4rx.audio_to_baseband(available, fs)
            base_origin = origin+lo
            nominal = round((predicted-base_origin)*28800/fs)+p4short.GEOM
        if entry is not None and alias:
            entry['alias'] = True
        if entry is None:
            offsets = p4short.WIDE
            bases = (self.clock.cfo_hz,)
            placed = p4short.place_long(z, nominal, bases[0], offsets=offsets, trainings=5)
            if placed[0] < .45 and alias:
                alternatives = [p4short.place_long(z, nominal, self.clock.cfo_hz+step,
                                    offsets=offsets, trainings=5) for step in (-1800/239, 1800/239)]
                placed = max((placed, *alternatives), key=lambda x: x[0])
                if placed[0] < .60:
                    return None, placed[0]
            elif placed[0] < .45:
                return None, placed[0]
            coherence, start, cfo = placed
            at = predicted+round((start-nominal)*fs/28800)
            obs = Observation(Kind.LONG, fs, origin, at-origin, at,
                              (end-at)/fs, cfo, coherence, provisional=True,
                              lattice=coherence, placed=True)
            agreement = self._long_agreement(z, start, cfo)
            if agreement >= .70:
                obs = replace(obs, frame=self._long_frame[0], provisional=False, extent_s=3.282)
                self.live_events.append(dict(event='long-repeat', at=at, agreement=agreement))
            entry = dict(kind='normal', obs=obs, coherence=coherence, end=end,
                         start=start, base_origin=base_origin, cfo=cfo, scores=None,
                         iterations=0, complete=False, alias=alias)
            self._long_cache[key] = entry
            self.clock.cycle_s = 3.75
            self.clock.misses = 0
            self.live_events.append(dict(event='long-candidate', at=at,
                                         coherence=coherence, trainings=5))
        if partial or end < entry['obs'].stream_sample+round(3.300*fs):
            entry['end'] = end
            return entry['obs'], entry['coherence']
        start = entry['start'] + round((entry['base_origin']-base_origin)*28800/fs)
        placed = p4short.place_long(z, start, entry['cfo'], offsets=p4short.NARROW, trainings=25)
        entry['end'] = end
        entry['complete'] = True
        if placed[0] < .45:
            entry['obs'] = None
            return None, placed[0]
        coherence, start, cfo = placed
        header = start+128+16*(np.arange(p4longtx.HEADER_CHIPS)-p4longtx.HEADER_CHIPS)
        if header[0] >= 0:
            matched = p4rx.matched_filter(z[:header[-1]+1] * np.exp(
                -2j*np.pi*cfo*np.arange(header[-1]+1)/28800))
            chips = matched[header]
            energy = np.sqrt(np.vdot(chips, chips).real*p4longtx.HEADER_CHIPS)
            if energy:
                self.live_events.append(dict(event='long-header', at=entry['obs'].stream_sample,
                    a=float(abs(np.vdot(p4longtx.HEADER, chips))/energy)))
        fields = p4long.body_scores(z, first_training_start=start, frequency_offset=cfo)
        if fields is None:
            entry['complete'] = False
            return entry['obs'], coherence
        entry.update(start=start, base_origin=base_origin, cfo=cfo,
                     coherence=coherence, scores=fields, iterations=1)
        started = monotonic()
        frame = p4long.decode_scores(fields, iterations=1, polarity=1)
        self.live_events.append(dict(event='long-decode', at=entry['obs'].stream_sample,
            iterations=1, polarity=1, elapsed_ms=1000*(monotonic()-started),
            crc_valid=frame is not None, copies=1))
        score_sets = [fields]
        if frame is None and entry.get('alias'):
            # The five-training lattice can prefer a 7.53 Hz alias even at
            # coherence .679. A full-packet CRC decides the carrier.
            for step in (-1800/239, 1800/239):
                alternative = p4short.place_long(z, start, self.clock.cfo_hz+step,
                                                  offsets=p4short.NARROW, trainings=25)
                if alternative[0] < .45 or abs(alternative[2]-cfo) < 1.:
                    continue
                alt_fields = p4long.body_scores(z, first_training_start=alternative[1],
                                                frequency_offset=alternative[2])
                if alt_fields is None:
                    continue
                started = monotonic()
                alt_frame = p4long.decode_scores(alt_fields, iterations=1, polarity=1)
                self.live_events.append(dict(event='long-decode', at=entry['obs'].stream_sample,
                    iterations=1, polarity=1, elapsed_ms=1000*(monotonic()-started),
                    crc_valid=alt_frame is not None, copies=1))
                if alt_frame is not None:
                    frame, fields = alt_frame, alt_fields
                    coherence, start, cfo = alternative
                    entry.update(start=start, cfo=cfo, coherence=coherence)
                    break
        if frame is None and self._long_scores is not None:
            previous_at, previous_scores = self._long_scores
            if abs(entry['obs'].stream_sample-previous_at-round(3.75*fs)) < fs//20:
                combined = fields+previous_scores
                started = monotonic()
                frame = p4long.decode_scores(combined, iterations=1, polarity=1)
                self.live_events.append(dict(event='long-decode', at=entry['obs'].stream_sample,
                    iterations=1, polarity=1, elapsed_ms=1000*(monotonic()-started),
                    crc_valid=frame is not None, copies=2))
                if frame is None:
                    score_sets.append(combined)
        if frame is None:
            entry['score_sets'] = score_sets
            self._long_scores = entry['obs'].stream_sample, fields
            return entry['obs'], coherence
        self._long_scores = None
        self._remember_long(frame)
        entry['refined'] = True
        entry['obs'] = replace(entry['obs'], frame=frame, provisional=False,
                               extent_s=3.282, cfo_hz=cfo, coherence=coherence,
                               lattice=coherence)
        return entry['obs'], coherence

    def prepare_long(self, audio, origin):
        if self.clock is None or self.clock.expects != Expect.PACKET:
            return None
        predicted, cfo, fs = self.clock.next_due, self.clock.cfo_hz, self.fs
        self._prune_long_preparation(predicted)
        key = (predicted, cfo)
        if key in self._long_prepared:
            return self._long_prepared[key]
        if origin+len(audio) < predicted+round(3.230*fs):
            self._preparation_status('long', 'retryable', 'incomplete', predicted)
            return None
        if predicted-origin < round(.035*fs):
            self._preparation_status('long', 'retryable', 'pre-roll', predicted)
            return None
        if any(a < predicted+round(3.230*fs) and b > predicted-round(.035*fs)
               for a, b in self.capture_gaps):
            self._preparation_status('long', 'retryable', 'capture-gap', predicted)
            return None
        started = monotonic()
        prepared = p4oracle.prepare_despread_long(audio, fs, (predicted-origin)/fs, cfo)
        if prepared is None:
            return None
        prepared.update(lo=prepared['lo']+origin, onset=predicted,
                        generation=self._long_generation)
        self._long_prepared[key] = prepared
        self._preparation_status('long', 'prepared', 'prefix', predicted,
            prefix_ms=1000*(monotonic()-started))
        return prepared

    def _long(self, audio, origin, *, placed=False, partial=False, alias=False,
              fallback=False):
        if placed:
            return self._normal_long(audio, origin, partial=partial, alias=alias)
        predicted, cfo, fs = self.clock.next_due, self.clock.cfo_hz, self.fs
        self._robust_long_cache = {k: v for k, v in self._robust_long_cache.items() if k[0] >= predicted-6*fs}
        key = (predicted, cfo)
        if key not in self._robust_long_cache:
            key = next((k for k, v in self._robust_long_cache.items() if v['obs'] is None
                and k[1] == cfo and abs(k[0]-predicted) < round(.035*fs)), key)
        entry = self._robust_long_cache.get(key)
        end = origin+len(audio)
        if entry is not None and (entry['obs'] is not None
                or not (fallback and entry.get('splice_failed'))
                and (entry['end'] >= key[0]+round(3.355*fs) or end <= entry['end'])):
            return entry['obs'], entry['coherence']
        started = monotonic()
        prepared = self._long_prepared.get((predicted, cfo))
        reason = 'unprepared'
        diagnostics = {}
        split = prepared is not None and not (fallback and entry is not None
                                              and entry.get('splice_failed'))
        if split:
            coherence, symbols, first, reason = p4oracle.finish_despread_long(
                audio, fs, dict(prepared, lo=prepared['lo']-origin, onset=predicted-origin),
                diagnostics=diagnostics)
        else:
            coherence, symbols, first = p4oracle.placed_despread(
                audio, fs, (predicted-origin)/fs, cfo, count=373)
        placed_at = monotonic()
        obs = None
        reception = None
        if coherence >= p4oracle.ROBUST_FRACTION_KNEE:
            reception = p4robustrx.decode_symbols(symbols)
            if reception is not None:
                first += round(reception.packet_symbol*fs/112.5)
                obs = Observation(Kind.LONG, fs, origin, first, origin+first,
                                  369/112.5, cfo, coherence, frame=reception.frame, placed=True,
                                  speedlevel=str(reception.speedlevel))
        self.live_events.append(dict(event='long decode', at=predicted, coherence=coherence,
            symbols=len(symbols), crc_valid=reception is not None,
            elapsed_ms=1000*(monotonic()-started),
            speedlevel=None if reception is None else reception.speedlevel,
            corrected_bits=None if reception is None else reception.corrected_bits,
            preparation=reason, format_hint=self._long_hint,
            sf8_recoveries=self._sf8_recoveries.get(key, 0),
            tail_ms=1000*(placed_at-started) if split else None,
            overlap_correlation=diagnostics.get('overlap_correlation'),
            decode_ms=1000*(monotonic()-placed_at)))
        self._robust_long_cache[key] = dict(obs=obs, coherence=coherence, end=end,
            splice_failed=split and (reason in ('overlap', 'incomplete') or reception is None))
        if obs is not None:
            self._robust_long_cache[(obs.stream_sample, cfo)] = self._robust_long_cache[key]
        if obs is None and end >= predicted+round(3.306*fs) and key in self._sf8_suppressed:
            self._long_hint = None
        return obs, coherence

    def prepare_sf8(self, audio, origin):
        if self.clock is None or self.clock.expects != Expect.PACKET:
            return None
        predicted, fs, cfo = self.clock.next_due, self.fs, self.clock.cfo_hz
        self._prune_long_preparation(predicted)
        key = (predicted, cfo)
        if not self._sf8_allowed():
            self._preparation_status('sf8', 'skipped', 'sf16-hint', predicted)
            return None
        if key in self._sf8_prepared:
            entry = self._sf8_prepared[key]
            return None if isinstance(entry, PreparationFailure) else entry
        if origin + len(audio) < predicted + round(3.230*fs):
            self._preparation_status('sf8', 'retryable', 'incomplete', predicted)
            return None
        begin = predicted-origin-round(.100*fs)
        if begin < 0:
            self._preparation_status('sf8', 'retryable', 'pre-roll', predicted)
            return None
        if any(a < predicted+round(3.230*fs) and b > origin+begin
               for a, b in self.capture_gaps):
            self._preparation_status('sf8', 'retryable', 'capture-gap', predicted)
            return None
        chunk = np.asarray(audio)[begin:predicted-origin+round(3.230*fs)]
        started = monotonic()
        z = p4rx.audio_to_baseband(chunk, fs)
        reason = None
        try:
            observed = self._recover_sf8(
                z, reference_start=8000, spread_factor=8, data_symbols=680,
                timing_offsets=range(-48, 49), cfo_hz=(cfo,))
        except ValueError:
            reason = 'recovery-error'
        else:
            if observed.coherence < .40:
                reason = 'below-coherence'
            else:
                noise = self._sf8_noise_floor(z)
                if not self._sf8_energy_present(z, observed, noise):
                    reason = 'energy'
        self._preparation_status('sf8', 'negative' if reason else 'prepared',
            reason or 'prefix', predicted, sf8_recoveries=self._sf8_recoveries[key],
            coherence=None if reason == 'recovery-error' else observed.coherence,
            elapsed_ms=1000*(monotonic()-started))
        if reason:
            self._sf8_prepared[key] = PreparationFailure(reason)
            return None
        self._sf8_prepared[key] = observed
        self._sf8_noise[key] = noise
        return observed

    @staticmethod
    def _sf8_noise_floor(z):
        blocks = np.abs(z[:1440])**2
        return float(np.percentile(blocks[:blocks.size//144*144].reshape(-1, 144).mean(axis=1), 10))

    @staticmethod
    def _sf8_energy_present(z, observed, noise):
        start = observed.reference_start + 8*16
        end = observed.reference_start + (len(observed.symbols)*8-1)*16 + len(p4rx.RRC_TAPS)
        if end > len(z):
            return False
        samples = np.abs(z[start:end])**2
        width = 29
        blocks = samples[:samples.size//width*width].reshape(-1, width).mean(axis=1)
        if not blocks.size:
            return False
        floor = max(.01*float(np.median(blocks)), .1*noise)
        return bool(np.all(blocks >= floor) and np.min(samples[-2:]) >= floor)

    @staticmethod
    def _sf8_present(body, tail):
        reference_power = np.mean(abs(body.symbols[256:512])**2)
        if reference_power <= 1e-30:
            return False
        power = np.abs(np.concatenate((body.symbols[:672], tail.symbols[-25:])))**2
        coherence = np.concatenate((body.symbol_coherence[:672],
                                    tail.symbol_coherence[-25:]))
        window = np.ones(4)/4
        return (np.min(np.convolve(power[1:], window, 'valid')) >= .15*reference_power
                and np.min(np.convolve(coherence[1:], window, 'valid')) >= .12
                and np.mean(power[-2:]) >= .15*reference_power
                and np.mean(coherence[-2:]) >= .15
                and power[-1] >= .10*reference_power
                and coherence[-1] >= .10)

    def _sf8_long(self, audio, origin, *, work=None):
        predicted, fs, cfo = self.clock.next_due, self.fs, self.clock.cfo_hz
        if not self._sf8_allowed():
            return None
        if origin + len(audio) < predicted + round(3.280*fs):
            return None
        key = (predicted, cfo)
        if key in self._sf8_cache:
            return self._sf8_cache[key]
        if work is not None and not work.admits('long'):
            return None
        started = monotonic()
        prepared = self._sf8_prepared.get(key)
        if isinstance(prepared, PreparationFailure):
            return None
        if prepared is not None:
            begin = predicted-origin+round(3.000*fs)
            tail = p4rx.audio_to_baseband(np.asarray(audio)[begin:], fs)
            try:
                last = self._recover_sf8(
                    tail, reference_start=4736+prepared.reference_start-8000,
                    spread_factor=8, data_symbols=24, timing_offsets=(0,),
                    cfo_hz=(cfo,))
            except ValueError:
                return None
            phase = np.vdot(last.symbols[:9], prepared.symbols[672:681])
            if abs(phase) < 1e-30:
                return None
            symbols = np.concatenate((prepared.symbols[:672],
                                      last.symbols*phase/abs(phase)))
            coherence = prepared.coherence
            present = (self._sf8_present(prepared, last) and
                       self._sf8_energy_present(tail, last, self._sf8_noise[key]))
        else:
            begin = predicted-origin-round(.100*fs)
            if begin < 0:
                return None
            z = p4rx.audio_to_baseband(np.asarray(audio)[begin:], fs)
            try:
                observed = self._recover_sf8(
                    z, reference_start=8000, spread_factor=8, data_symbols=696,
                    timing_offsets=range(-48, 49), cfo_hz=(cfo,))
            except ValueError:
                return None
            coherence, symbols = observed.coherence, observed.symbols
            present = (self._sf8_present(observed, observed) and
                       self._sf8_energy_present(z, observed, self._sf8_noise_floor(z)))
        if coherence < .40 or not present:
            return None
        placed_at = monotonic()
        deadline = self.sf8_deadline(predicted) if self.sf8_deadline else None
        reception = p4sf8rx.decode_symbols(symbols, c1=False)
        if reception is None and (deadline is None or deadline-monotonic() >= .035):
            reception = p4sf8rx.decode_symbols(symbols)
        timely = deadline is None or monotonic()+.005 <= deadline
        frame = reception.frame if reception is not None and timely else None
        obs = Observation(Kind.LONG, fs, origin, predicted-origin, predicted,
                          3.280, cfo, coherence, frame=frame,
                          provisional=frame is None, placed=True, extent_s=3.280, sf8=True)
        self.live_events.append(dict(event='sf8-long', at=predicted,
            coherence=coherence, crc_valid=reception is not None,
            timely=timely, placement_ms=1000*(placed_at-started),
            decode_ms=1000*(monotonic()-placed_at)))
        self._sf8_cache = {k: v for k, v in self._sf8_cache.items()
                           if k[0] >= predicted-6*fs}
        self._sf8_cache[key] = obs
        return obs

    def has_chirp_header(self, onset):
        """Whether a retained marker places this onset at the current carrier."""
        return self.clock is not None and any(
            header is not None and header[0] == onset and key[1] == self.clock.cfo_hz
            for key, header in self._chirp_headers.items())

    def _reject_chirp_header(self, onset):
        """An unread provisional marker cannot retain a new peer cadence."""
        if self.clock is None:
            return
        for key, header in list(self._chirp_headers.items()):
            if header is None or header[0] != onset or key[1] != self.clock.cfo_hz:
                continue
            previous = self._chirp_prior_cycles.pop(key, None)
            self._chirp_headers[key] = None
            if (previous is not None and previous[0] is self.clock
                    and previous[1] == self._train_generation
                    and abs(self.clock.next_due-onset) < self.fs//10
                    and not any(o.frame is not None and not o.provisional
                                and abs(o.stream_sample-onset) < self.fs//10
                                for o in self.observations)):
                self.clock.cycle_s = previous[2]
                self.live_events.append(dict(event='chirp header unread', at=onset,
                                             restored_cycle_s=previous[2]))

    def _chirp(self, audio, origin, onset):
        fs = self.fs
        first = onset-origin
        if first < 0 or len(audio)-first < round(p4chirp.PACKET_S*fs):
            return None
        key = (onset, self.clock.cfo_hz)
        self._chirp_cache = {k: v for k, v in self._chirp_cache.items()
                             if k[0] >= onset-6*fs}
        if key not in self._chirp_cache:
            self._chirp_cache[key] = p4chirprx.decode_packet(
                audio, fs=fs, start_sample=first, frequency_offset=key[1])
        reception = self._chirp_cache[key]
        if reception is not None:
            return Observation(Kind.CHIRP, fs, origin, first, onset,
                               p4chirp.PACKET_S, key[1], 0., frame=reception.frame)
        return None

    def read_chirp(self, audio, origin, *, onset):
        """Register an acquired chirp; None means unread, [] an already registered frame."""
        if onset < origin or not self.has_chirp_header(onset):
            return None
        started = monotonic()
        obs = self._chirp(audio, origin, onset)
        self.packet_read = dict(at=onset, decoder='chirp',
            elapsed_ms=round(1000*(monotonic()-started), 3),
            reason='decoded' if obs is not None else 'CRC/header unread')
        if obs is None:
            if origin+len(audio) >= onset+round(p4chirp.PACKET_S*self.fs):
                self._reject_chirp_header(onset)
            return None
        result = []
        self._emit(obs, result)
        return result

    def _packet(self, audio, origin, partial=False, *, work=None, long_fallback=False):
        if work is not None and not work.admits('packet'):
            return None
        fs = self.fs
        self._normal_baseband = None
        context, context_origin = audio, origin
        long_coherence = None
        sf8_candidate, sf8_tried = None, False
        chirp_prefix_available = True
        if self.clock and self.clock.expects == Expect.PACKET:
            begin = max(0, self.clock.next_due-round(.5*fs)-origin)
            end = max(0, self.clock.next_due+round(3.355*fs)-origin)
            audio = np.asarray(audio)[begin:end]
            origin += begin
        onset = (self.clock.next_due - origin) / fs if self.clock else None
        if self.clock and self.clock.expects == Expect.PACKET:
            sf8_allowed = self._sf8_allowed()
            if (not partial and
                    sf8_allowed and self.sf8_prepared() and
                    origin+len(audio) >= self.clock.next_due+round(3.280*fs)):
                sf8_candidate = self._sf8_long(audio, origin, work=work)
                sf8_tried = True
                if sf8_candidate is not None and sf8_candidate.frame is not None:
                    return sf8_candidate
            field = p4oracle.robust_at(audio, fs, onset, self.clock.cfo_hz)
            knee = p4oracle.coherence_knee(field['p4_walsh_peak'], p4oracle.ROBUST_FRACTION_KNEE)
            self.packet_read = dict(
                at=self.clock.next_due, partial=partial,
                header=field['p4_hdr'], value=field['p4_value'],
                walsh_peak=field['p4_walsh_peak'], coherence=field['coherence'],
                reason=('incomplete' if field['first_chip'] is None else
                        'below-coherence' if field['coherence'] < knee else
                        'no-walsh' if field['p4_value'] is None else 'decoded'))
            if (field['coherence'] >= knee
                    and field['p4_value'] is not None
                    and field['p4_walsh_peak'] >= p4oracle.ROBUST_BLOCK_KNEE):
                first = field['first_chip']
                return Observation(
                    Kind.SHORT, fs, origin, first, origin+first,
                    p4robusttx.SHORT_SYMBOLS/112.5, self.clock.cfo_hz,
                    field['coherence'], provisional=partial, header=field['p4_hdr'],
                    value=field['p4_value'], walsh_peak=field['p4_walsh_peak'], placed=True)
            if (not partial
                    and origin+len(audio)-self.clock.next_due >= round(3.28*fs)
                    and p4oracle.placed_despread(audio, fs, onset, self.clock.cfo_hz)[0]
                    >= p4oracle.ROBUST_FRACTION_KNEE):
                if not sf8_tried and sf8_allowed and (self.clock.cycle_s == 3.75 or
                        self.sf8_prepared()):
                    sf8_candidate = self._sf8_long(audio, origin, work=work)
                    if sf8_candidate is not None and sf8_candidate.frame is not None:
                        return sf8_candidate
                long, long_coherence = self._long(audio, origin, fallback=long_fallback)
                if long is not None:
                    return long
            if (not partial and not sf8_allowed
                    and origin+len(audio) >= self.clock.next_due+round(3.306*fs)):
                self._long_hint = None
        if (self.chirp and self.expected and self.clock
                and self.clock.expects == Expect.PACKET):
            nominal = self.clock.next_due-origin
            header_key = (self.clock.next_due, self.clock.cfo_hz)
            if nominal >= 0 and len(audio)-nominal >= round(.235*fs):
                self._chirp_headers = {k: v for k, v in self._chirp_headers.items() if k[0] >= header_key[0]-6*fs}
                self._chirp_prior_cycles = {k: v for k, v in self._chirp_prior_cycles.items()
                                            if k in self._chirp_headers}
                if header_key not in self._chirp_headers:
                    header = p4chirprx.acquire_header(audio, fs=fs, nominal_sample=nominal,
                                                     frequency_offset=self.clock.cfo_hz)
                    self._chirp_headers[header_key] = None if header is None else (origin+header.start_sample, header.score)
                    if header is not None:
                        self._chirp_prior_cycles[header_key] = (
                            self.clock, self._train_generation, self.clock.cycle_s)
            header = self._chirp_headers.get(header_key)
            first = None if header is None else header[0]-origin
            # Prediction and acquired onset have different retention bounds.
            # A cached later onset can survive even when nominal is negative.
            chirp_prefix_available = (nominal >= 0 if first is None else
                                      first >= 0 and len(audio)-first >= round(.200*fs))
            if not chirp_prefix_available:
                first = None
            if first is not None:
                self.clock.cycle_s = 3.75
            if partial and first is not None and len(audio)-first > round(.80*fs):
                return Observation(Kind.CHIRP, fs, origin, first, origin+first,
                                   (len(audio)-first)/fs, self.clock.cfo_hz, 0., provisional=True)
            if not partial and first is not None:
                chirp = self._chirp(audio, origin, origin+first)
                if chirp is not None:
                    return chirp
                if origin+len(audio) >= origin+first+round(p4chirp.PACKET_S*fs):
                    self._reject_chirp_header(origin+first)
        if self._normal_eligible():
            head = p4oracle.placed_despread(audio, fs, onset, self.clock.cfo_hz)[0]
            if head < p4oracle.ROBUST_FRACTION_KNEE:
                normal = self._normal(audio, origin)
                if normal is not None:
                    return normal
                if self.normal_coherence < p4short.NORMAL_ACQUIRE_KNEE or self.clock.cycle_s == 3.75:
                    if work is None or work.admits('long'):
                        long, _ = self._long(audio, origin, placed=True, partial=partial,
                                             alias=self.normal_coherence < p4short.NORMAL_ACQUIRE_KNEE)
                        if long is not None:
                            return long
        # SF8 and SF16 projections overlap. An unread SF8 placement must not
        # suppress the SF16/normal/chirp readers for this same complete body.
        if sf8_candidate is not None:
            return sf8_candidate
        candidates = self._bursts(np.asarray(audio), origin)
        if self.clock:
            head, _, _ = p4oracle.placed_despread(audio, fs, onset, self.clock.cfo_hz)
            if head >= p4oracle.ROBUST_FRACTION_KNEE:
                # A coherent placed head admits a faded envelope. A 6 dB
                # acquisition envelope fragments the recorded k0003 long.
                candidates = self._bursts(
                    np.asarray(audio), origin, knee_db=3.0, bridge_s=0.035
                )
                begin = self.clock.next_due-fs
                if self.clock.expects == Expect.PACKET and context_origin <= begin < origin:
                    end = self.clock.next_due+round(3.355*fs)-context_origin
                    wider = np.asarray(context)[begin-context_origin:end]
                    shift = (begin-origin)/fs
                    candidates += [(start+shift, duration) for start, duration in
                                   self._bursts(wider, begin, knee_db=3., bridge_s=.035)]
        for start, duration in candidates:
            if onset is not None and abs(start - onset) > 0.05:
                if (head >= p4oracle.ROBUST_FRACTION_KNEE
                        and start < onset < start+duration-.80
                        and (partial or start+duration >= len(audio)/fs-.005)):
                    duration -= onset-start
                    start = onset
                else:
                    continue
            piece = np.asarray(audio)[
                max(0, round(start * fs)) : round((start + duration + 0.025) * fs)
            ]
            if len(piece) < 0.15 * fs:
                continue
            if self.clock:
                cfo = self.clock.cfo_hz
                fraction, _, located = p4oracle.placed_despread(audio, fs, start, cfo)
            else:
                fraction, _, cfo = p4oracle.despread(piece[: round(0.24 * fs)], fs)
                located = None
            robust = fraction >= p4oracle.ROBUST_FRACTION_KNEE
            closed = start + duration < len(audio) / fs - 0.005
            kind = None
            header = value = peak = None
            lattice = 0.0
            first = round(start * fs)
            frame = None
            if robust:
                if (self.clock and not closed
                        and start+duration > onset+.85):
                    # A provisional long opens a window before it owns the cadence.
                    self.clock.cycle_s = 3.75
                if partial:
                    kind = (
                        Kind.SHORT
                        if closed and duration <= 0.78
                        else (Kind.LONG if duration > 0.80 else None)
                    )
                elif closed and 0.65 <= duration <= 0.85:
                    field = (
                        p4oracle.robust_at(audio, fs, onset, cfo)
                        if self.clock
                        else p4oracle.robust_field(piece, fs, p4robusttx.SHORT_SYMBOLS)
                    )
                    if field.get("first_chip") is not None:
                        first = field["first_chip"]
                    header, value, peak = (
                        field["p4_hdr"],
                        field["p4_value"],
                        field["p4_walsh_peak"],
                    )
                    if header in ("A", "B"):
                        kind = Kind.SHORT
                elif closed and 3.20 <= duration <= 3.35:
                    if (
                        long_coherence if long_coherence is not None else
                        p4oracle.placed_despread(audio, fs, start, cfo, count=369)[0]
                        if self.clock
                        else p4oracle.despread_fraction(piece, fs)
                    ) >= p4oracle.ROBUST_FRACTION_KNEE:
                        kind = Kind.LONG
            elif duration >= 0.69 and p4oracle.occupied_bw(piece, fs) > 1500:
                if partial:
                    kind = (Kind.CHIRP if self.chirp and self.expected and self.clock
                            and self.clock.expects == Expect.PACKET and self.clock.cycle_s == 3.75
                            and duration > .80 else Kind.NORMAL)
                    if kind == Kind.CHIRP and not chirp_prefix_available:
                        continue
                elif closed:
                    if kind is None and duration <= 0.9:
                        lattice, training, cfo = p4oracle.training_origin(piece, fs)
                        if lattice >= 0.33:
                            first += round(
                                (
                                    training
                                    - p4shorttx.HEADER_CHIPS * p4rx.SAMPLES_PER_CHIP
                                )
                                * fs
                                / p4rx.SAMPLE_RATE
                            )
                            kind = Kind.NORMAL
            if kind:
                if partial and located is not None and kind != Kind.LONG:
                    first = located
                elif partial and kind == Kind.LONG and self.clock:
                    # The open envelope can start in noise ahead of the placed head.
                    first = self.clock.next_due-origin
                return Observation(
                    kind,
                    fs,
                    origin,
                    first,
                    origin + first,
                    duration,
                    cfo,
                    fraction,
                    provisional=partial,
                    header=header,
                    value=value,
                    walsh_peak=peak,
                    lattice=lattice,
                    frame=frame,
                )
        return None

    def _read_peer_normal(self, audio, origin, result):
        grant = self.peer_normal_grant
        def current_grant():
            return (grant is None or self.last_emission is not None
                    and grant == (self.last_emission.stream_sample, self._train_generation))
        if (self.peer_normal_pending and self.hold is not None
                and self.peer_onset is not None and self.clock is not None
                and self.clock.expects == Expect.WORD and current_grant()):
            cycle = round(1.25*self.fs)
            end = origin + len(audio)
            at = self.peer_onset + ((end-round(.810*self.fs)-self.peer_onset)//cycle)*cycle
            cached = any(abs(key-at) < self.fs//20 for key in self._normal_cache)
            if ((at > self.peer_onset or cached) and at >= origin
                    and (grant is None or at > grant[0])
                    and not any(left < at+round(.805*self.fs) and right > at
                                for left, right in self.capture_gaps)
                    and not any(em.stream_sample < at+round(.805*self.fs)
                                and em.stream_sample+round(em.duration_s*self.fs) > at
                                for em in self.emissions)):
                # Side-raster progress must not delay the independent answer slot.
                deadline = self.normal_deadline(at) if self.normal_deadline else None
                if grant is not None and deadline is not None and monotonic() >= deadline:
                    return
                waiting, coherence = self.normal_waiting, self.normal_coherence
                try:
                    normal = self._normal(audio, origin, predicted=at)
                finally:
                    self.normal_waiting, self.normal_coherence = waiting, coherence
                if normal is not None and normal.frame is not None and current_grant():
                    # Reading the continuing ISS raster must not move the
                    # independent grant window anchored on our request.
                    grant_clock = replace(self.clock)
                    self._emit(normal, result)
                    self.clock = grant_clock

    def read_window(self, audio, origin, *, carrier_end=None, whole=False, work=None,
                    long_fallback=False):
        if work is not None:
            return self._read_hold_body(audio, origin, work=work) if whole else []
        self.consumed = None
        if not (self.admitted or self.expected):
            return []
        audio = np.asarray(audio)
        if self.scan == 'incremental':
            self._retain_scan(audio, origin, carrier_end)
        result = []
        hits = []
        if self.clock:
            obs = self._word(audio, origin, whole=whole)
            if obs:
                hits = [obs]
        if not hits and carrier_end is not None:
            cut = max(0, carrier_end - origin)
            hits = p4oracle.word_tail(audio[cut:], self.fs, origin=origin + cut)
        placed_packet = self.clock and self.clock.expects == Expect.PACKET
        packet = self._packet(audio, origin, long_fallback=long_fallback) if whole and not hits and placed_packet else None
        pending = self._classification(self.clock.next_due) if self.clock else None
        pending_packet = (pending is not None and self.clock.expects == Expect.PACKET
                          and (pending.kind == Kind.NORMAL or pending.kind in (Kind.LONG, Kind.CHIRP)
                               and origin+len(audio) < pending.stream_sample+round(3.35*self.fs)))
        if self.clock and (packet is not None or pending_packet):
            self.clock.misses = 0
        elif self.clock and not hits:
            horizon = (WORD_HORIZON_S if self.clock.expects == Expect.WORD else
                       3.355 if self.peer_watch is not None and self.clock.cycle_s == 3.75 else .85)
            if (whole or self.clock.expects == Expect.WORD) and origin+len(audio) >= self.clock.next_due+round(horizon*self.fs):
                if self._missed_due != self.clock.next_due:
                    self.clock.misses += 1
                    self._missed_due = self.clock.next_due
        if self.scan == 'incremental':
            # Provisional packet/energy evidence cannot suppress acquisition.
            if hits or packet is not None and not packet.provisional:
                self._scan_misses = 0
            elif (self.clock and whole and origin+len(audio) >= self.clock.next_due
                    + round(.85*self.fs) and self._scan_missed_due != self.clock.next_due):
                self._scan_misses += 1
                self._scan_missed_due = self.clock.next_due
        if (self.scan == 'legacy' and not hits and packet is None and not pending_packet
                and (not self.clock or self.clock.misses >= 3)):
            end = carrier_end or origin
            if self.last_emission:
                end = max(
                    end,
                    self.last_emission.stream_sample
                    + round(self.last_emission.duration_s * self.fs),
                )
            cut = max(0, end - origin)
            hits = p4oracle.word_scan(
                audio[cut:], self.fs, self.cfo_hz, origin=origin + cut, whole=whole,
                gaps=self._gaps(audio[cut:], origin + cut)
            )
        for obs in hits:
            obs = self._ladder_word(obs, origin + len(audio))
            available = origin + len(audio) - obs.stream_sample
            if self.breakin and available < round(0.240 * self.fs):
                obs = replace(obs, provisional=True,
                              symbols_read=min(20, round(available * 112.5 / self.fs)))
            self._emit(obs, result)
        if not hits:
            self._grant_take(audio, origin, result)
        if whole and not hits and not placed_packet:
            packet = self._packet(audio, origin, long_fallback=long_fallback)
        if packet:
            self._emit(packet, result)
        if self.scan == 'incremental':
            # Successful placement resets clock misses in _emit. Re-evaluate
            # fallback eligibility afterwards; pending edges resolve separately.
            self._scan_eligible = (not self.clock or self.clock.misses >= 3
                                   or self._scan_misses >= 3)
        if whole:
            self._read_peer_normal(audio, origin, result)
        if whole:
            heads = list(self._body_heads.values()) if self.hold is not None else [self.pending_body]
            for head in heads:
                if head is None or head.provisional or head.stream_sample in self._body_attempted:
                    continue
                end = head.stream_sample + round(.805 * self.fs)
                if origin + len(audio) < end:
                    continue
                started = monotonic()
                corrected = (p4oracle.body_carrier(audio, self.fs, origin, head)
                             if self._body_carrier_check else head)
                self._body_attempted.add(head.stream_sample)
                if self.hold is not None:
                    self.hold.acked_counter = self.acked_counter
                if corrected is not head:
                    # Spend one trellis on the better training hypothesis,
                    # preserving the old carrier and sums unless its CRC passes.
                    frame = p4oracle.body_after(audio, self.fs, origin, corrected)
                    if frame is not None and (self.hold is None
                            or self.hold.status_ok(frame, combined=False)):
                        self.live_events.append(dict(event='body carrier alias', at=head.stream_sample,
                            previous_cfo_hz=head.cfo_hz-p4words.CARRIER_HZ['Y27'],
                            cfo_hz=corrected.cfo_hz-p4words.CARRIER_HZ['Y27']))
                        if self.hold is not None:
                            self.hold.reset('carrier alias')
                        self._emit(corrected, result)
                        self._body_heads.pop(head.stream_sample, None)
                        self._body_attempted.add(corrected.stream_sample)
                        head = corrected
                    else:
                        frame = None
                else:
                    frame = (self.hold.inline(audio, origin, head) if self.hold is not None and head.placed
                             else p4oracle.body_after(audio, self.fs, origin, head))
                self._body_carrier_check = frame is None
                self.live_events.append(dict(event='body decode', stage='inline', at=head.stream_sample,
                    elapsed_ms=1000*(monotonic()-started), accepted=frame is not None,
                    extent_s=head.extent_s, available_s=(origin+len(audio)-head.stream_sample)/self.fs,
                    copies=len(self.hold.members) if self.hold is not None else 1))
                if frame:
                    self._body(head, frame, origin, result)
                self._body_heads.pop(head.stream_sample, None)
                if self.pending_body and self.pending_body.stream_sample == head.stream_sample:
                    self.pending_body = None
        for old in list(self.observations):
            if not old.provisional:
                continue
            if (old.kind == Kind.CHIRP and old.stream_sample < origin
                    and self.clock is not None and self.clock.expects == Expect.PACKET
                    and abs(self.clock.next_due-old.stream_sample) <= round(.035*self.fs)
                    and self.has_chirp_header(old.stream_sample)):
                # Missing prefix bytes are not a failed CRC or cadence evidence.
                continue
            if old.kind == Kind.WORD:
                horizon = old.stream_sample + round(self._word_horizon(old) * self.fs)
            else:
                if not whole:
                    continue
                horizon = old.stream_sample + round(3.35 * self.fs)
                for start, duration in (() if old.kind in (Kind.LONG, Kind.CHIRP) else self._bursts(audio, origin)):
                    if (
                        abs(origin + round(start * self.fs) - old.stream_sample)
                        < self.fs // 10
                        and start + duration < len(audio) / self.fs - 0.005
                    ):
                        horizon = origin + round((start + duration + 0.005) * self.fs)
            if origin + len(audio) >= horizon:
                if old.kind == Kind.CHIRP:
                    self._reject_chirp_header(old.stream_sample)
                self.observations.remove(old)
                if (self.close_lease is not None
                        and old.stream_sample == self.close_lease.first_request):
                    self.close_lease.first_request = None
                self._body_heads.pop(old.stream_sample, None)
                result.append(
                    RegisteredObservation(
                        old,
                        Registered.EXPIRED,
                        old,
                        self._answered.get(old.stream_sample),
                        self._responds_to.get(old.stream_sample),
                    )
                )
                if self.hold is not None:
                    self._live(result[-1])
                if self.breakin:
                    self.registrations.append(result[-1])
                if self.pending_body == old:
                    self.pending_body = None
        return result

    def _read_hold_body(self, audio, origin, *, work):
        """Deferred P1 takeover bodies have no blind packet-search authority."""
        if not (self.admitted or self.expected) or not work.admits('window'):
            return []
        result = []
        heads = list(self._body_heads.values()) if self.hold else [self.pending_body]
        for head in heads:
            if head is None or head.stream_sample in self._body_attempted:
                continue
            end = head.stream_sample+round(.810*self.fs)
            if head.provisional or origin+len(audio) < end:
                continue
            if not work.admits('body'):
                break
            lo = max(0, head.stream_sample-origin-round(.035*self.fs))
            hi = min(len(audio), end-origin+round(.025*self.fs))
            if any(a < end and b > head.stream_sample+round(.240*self.fs)
                   for a, b in self.capture_gaps):
                continue
            deferred = len(work.deferred)
            frame = p4oracle.body_after(np.asarray(audio)[lo:hi], self.fs,
                                         origin+lo, head, work=work)
            if len(work.deferred) > deferred:
                break
            self._body_attempted.add(head.stream_sample)
            if frame is not None:
                self._body(head, frame, origin, result)
            self._body_heads.pop(head.stream_sample, None)
        return result

    def read_long_answer(self, audio, origin):
        return self.read_window(audio, origin, whole=True)
