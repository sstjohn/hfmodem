# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pure experimental P4 turn machine. Proposals are not emissions.

No radio, audio, wall clock or decoder lives here. Only a Keyed outcome moves
an outgoing step; a full accepted ack or associated data reply settles bytes.
"""

from hfmodem.core.recording import contact_ended

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count
from time import monotonic
from .p4observation import Kind, Registered, RegisteredObservation, RESPONSE_S, WORD_HORIZON_S
from .p4frame import Status
from .p4host import P4LateFrame, P4StreamFailure, observed_extension, source_payload
from .p4credit import ReceiptContext, SourceReceipt
from . import p4words, spec

DATA_KINDS = (Kind.BODY, Kind.NORMAL, Kind.CHIRP, Kind.LONG)

CYCLE = {Kind.WORD: 1.25, Kind.NORMAL: 1.25, Kind.SHORT: 1.25, Kind.BODY: 1.25,
         Kind.LONG: 3.75, Kind.CHIRP: 3.75}
T = {Kind.SHORT: 0.920, Kind.BODY: 0.920,
     Kind.NORMAL: 0.920, Kind.LONG: 3.400, Kind.CHIRP: 3.400}
ACK = {7: "X27", 8: "X27", 11: "X27", 12: "T27", 13: "X27",
       38: "X27", 49: "X27", 59: "X27", 60: "U27", 61: "U27"}
# Trimode's opening v61 progresses on X27 (0451), not N9VC's U27.
PEER_ACK = {**{value: word for value, word in ACK.items() if value not in (8, 38, 59)},
            49: "X27", 53: "U27", 61: "X27"}
# What follows a robust-short value on tape.
# Transitions are SEEN; the value as a level is INFERRED, and v13 also opens
# walks that end elsewhere.
LADDER_MEANING = {
    **dict.fromkeys((7, 38, 6, 9, 8), 'opening'),
    **dict.fromkeys((53, 55, 5), 'SL1 chirp next'),
    49: 'SL2 next',
    **dict.fromkeys((13, 60, 61), 'SL3 next'),
    **dict.fromkeys((58, 59), 'SL4 next'),
    10: 'SL5 short next',
    37: 'close',
}
LADDER_LONG = (13, 60)
LONG_LADDERS = {2: (49,), 3: LADDER_LONG, 4: (8, 59)}
LONG_BYTES = {2: 40, 3: 69, 4: 141}
LADDER_CHIRP = (13, 49, 53)
# k0003 1482.730-1488.060: N9VC's idle body drew V27, then 13 -> 12 -> 11 -> 10
# and an SL5 normal short offering the turn back.
LADDER_SL5 = (13, 12, 11, 10)
HANDBACK_REOFFER_SECONDS = 20.
CLOSE_TIMEOUT_S = 30.
# A late decode can owe a word or changeover reply one full peer cycle later.
REPLY_DRAIN_S = max(T[kind] + CYCLE[kind] for kind in T) + CYCLE[Kind.BODY]
_peer_watch_generations = count(1)


def counter_ack(counter):
    return 'V27' if counter & 1 else 'X27'


def chain_resets(value):
    members = value.split(',')
    if any(member not in ('turn', 'form', 'ladder') for member in members):
        raise ValueError('P4 chain reset must be a comma list of turn, form, ladder')
    return value


class State(StrEnum):
    ISS_LADDER = "ISS_LADDER"
    ISS_DATA = "ISS_DATA"
    ISS_HOLD = "ISS_HOLD"
    ISS_HANDBACK = "ISS_HANDBACK"
    ISS_GRANT = "ISS_GRANT"
    ISS_CLOSE = "ISS_CLOSE"
    IRS_ANSWER = "IRS_ANSWER"
    IRS_YIELD = "IRS_YIELD"
    IRS_BREAKIN = "IRS_BREAKIN"
    ISS_DATA_ACK = "ISS_DATA_ACK"
    IRS_CLOSE = "IRS_CLOSE"
    DOWN = "DOWN"


ISS_TURN = (State.ISS_LADDER, State.ISS_GRANT, State.ISS_DATA, State.ISS_DATA_ACK,
            State.ISS_HOLD, State.ISS_HANDBACK)


@dataclass(frozen=True)
class Proposal:
    kind: Kind
    stream_sample: int
    word: str | None = None
    value: int | None = None
    header: str = "A"
    payload: bytes = b""
    status: int = 0
    state: State | None = None
    index: int = 0
    repeat: bool = False
    ack_identity: tuple | None = None
    ack_reason: str | None = None
    retry_of: int | None = None
    peer_control: int | None = None
    responds_to: int | None = None
    chirp_slot: int = 0
    receive_epoch: int | None = None
    source_root: tuple | None = None
    purpose: str | None = None
    accepted_target: int | None = None
    current_slot: int | None = None
    current_slot_accepted: bool = True
    peer_watch_generation: int | None = None
    level: int | None = None
    final: bool = False


@dataclass(frozen=True)
class HeldAnswer:
    kind: Kind
    onset: int
    answer: Proposal
    context: dict
    generation: int
    acknowledged: bool = False


@dataclass(frozen=True)
class GrantReceiveOrigin:
    """History of an actual empty-grant receive exchange, not read authority."""
    context: object
    actual: Proposal
    receiver_generation: int
    turn_generation: int
    receive_epoch: int


@dataclass(frozen=True)
class PeerPhase:
    onset: int
    kind: Kind
    cycle: int
    generation: int
    identity: tuple


@dataclass(frozen=True)
class PeerWatch:
    onset: int
    cycle: int
    cfo_hz: float
    generation: int
    attempts: int = 0
    last_attempt: int | None = None


@dataclass(frozen=True)
class DataEpisode:
    serial: int
    first_key: int
    kind: Kind
    status: int
    field: bytes

    def matches(self, proposal):
        return (proposal.kind == self.kind and proposal.status == self.status
                and proposal.payload == self.field)


@dataclass(frozen=True)
class HeldWordYield:
    episode: DataEpisode
    response_at: int
    responds_to: int
    response_generation: int | None


@dataclass
class CloseContext:
    reason: str | None
    terminal: bool
    queued: bytes
    inflight: bytes
    started: int
    wall_started: float
    end: int
    grants: int = 0
    resumed: bool = False
    generation: int | None = None
    emission: int | None = None
    first_request: int | None = None
    last_request: int | None = None
    grant_at: int | None = None
    receive_at: int | None = None
    progress_at: int | None = None
    held_word_yield: HeldWordYield | None = None
    expired_at: int | None = None


class P4Turn:
    def __init__(
        self,
        outbox,
        adapter,
        *,
        fs=48000,
        data="none",
        long_level=3,
        ladder_values=None,
        ladder_admit="u27,x27",
        spug="queued",
        iss_u27="repeat",
        hold_cycles=6,
        prekey_lead=0.074,
        role="iss",
        app_done=False,
        managed_application=False,
        answer_offset=T[Kind.SHORT],
        breakin="body",
        normal_ack="toggle",
        long_ack="toggle",
        chirp_ack="toggle",
        first_ack="x27",
        chain_reset="turn,form",
        v53_hold=24,
        y27_first_body="c0",
        turn_entry="current",
        peer_ack=None,
        train_policy="repeat",
        wall_clock=monotonic,
        session_end=None,
        session_wall_end=None,
    ):
        if data not in ('none', 'body', 'chirp', 'long') or breakin not in ('body', 'y27'):
            raise ValueError("unknown P4 data or break-in policy")
        if ladder_admit not in ('u27', 'u27,x27') or iss_u27 not in ('repeat', 'ack'):
            raise ValueError('unknown P4 ladder admission or ISS U27 policy')
        if long_level not in LONG_BYTES or spug not in ('queued', 'off'):
            raise ValueError('unknown P4 long level or SPUG policy')
        self.long_level = long_level
        self.spug_policy = spug
        self.iss_u27 = iss_u27
        self.ladder_admit = tuple(word.upper() for word in ladder_admit.split(','))
        if ladder_values is not None and (not ladder_values or any(
                not isinstance(value, int) or not 0 <= value <= 63 for value in ladder_values)):
            raise ValueError('P4 ladder values must be a nonempty list of values 0..63')
        for name, policy, choices in (
            ('normal ack', normal_ack, ('toggle', 'parity', 'x27')),
            ('long ack', long_ack, ('toggle', 'toggle-t27', 'counter-even-t27')),
            ('chirp ack', chirp_ack, ('toggle', 'parity')),
            ('first ack', first_ack, ('x27', 'v27')),
        ):
            if policy not in choices:
                raise ValueError(f"unknown P4 {name} policy")
        if y27_first_body not in ('c0', 'c1'):
            raise ValueError('unknown first body counter after a bare Y27')
        if turn_entry not in ('current', 'ladder'):
            raise ValueError('unknown P4 turn entry')
        self.turn_entry = turn_entry
        self._turn_body = self._turn_values = None
        if v53_hold < 0:
            raise ValueError('P4 v53 hold bound must be nonnegative')
        if train_policy not in ("repeat", "passive"):
            raise ValueError("unknown P4 train policy")
        self.train_policy = train_policy
        self.normal_ack = normal_ack
        self.long_ack = long_ack
        self.chirp_ack = chirp_ack
        self.first_ack = first_ack.upper()
        self.chain_reset = chain_resets(chain_reset)
        self._ack_chain = None
        self._ack_form = None
        self._ack_epoch = None
        self._ack_boundary = True
        self._chain_epoch = 0
        self._chain_reason = 'opening'
        self.keyed_ack = {}
        self._answer_kinds = {}
        self.v53_hold = v53_hold
        self.y27_first_body = y27_first_body
        self.peer_ack = {**PEER_ACK, **(peer_ack or {})}
        self.fs = fs
        self.t = {**T, Kind.SHORT: answer_offset, Kind.BODY: answer_offset,
                  Kind.NORMAL: answer_offset}
        self.outbox = outbox
        self.adapter = adapter
        self.sf8_eligible = None
        self._sf8_acks = {}
        self.ladder = ()
        if data in ('long', 'chirp'):
            self.ladder = (tuple(ladder_values) if ladder_values is not None else
                           LONG_LADDERS[long_level] if data == 'long' else LADDER_CHIRP)
        self.sends_data = data != 'none'
        self.data_kind = {'long': Kind.LONG, 'chirp': Kind.CHIRP}.get(data, Kind.BODY)
        self._restart_packet = None
        self._awaiting_peer_turn = False
        self.require_long_crc = self.sends_data
        self.hold_cycles = hold_cycles
        self.breakin = breakin
        self.lead = round(prekey_lead * fs)
        self.state = State.ISS_LADDER if role == "iss" else State.IRS_ANSWER
        self.index = 0
        self.counter = 0
        self._first_data = True
        self._train_initial_data = False
        self._word_yield = False
        self._data_refusals = {}
        self._ladder_returned = False
        self._chirp_origin = None
        self._last_data_ack = None
        self._last_data_form = None
        self._chirp_ack_word = 'X27'
        self._last_chirp = None
        self.repeats = 0
        self.grants = 0
        self.breakins = 0
        self._peer_cycle = 1.25
        self.peer_phase = None
        self.peer_watch = None
        self._peer_generation = 0
        self._peer_consumed = None
        self.unread = 0
        self._unread_data_due = None
        self._withdrawn = None
        self.first_body = True
        self.app_done = app_done
        self.managed_application = managed_application
        self.application_deadline = None
        self.application_failure = None
        self.allow_breakin = True
        self.reason = None
        self.events = []
        self.pending = {}
        self._decided = set()
        self._dropped = set()
        self._deferred = None
        self._seen = set()
        self._next = None
        self.last = None
        self._data_packet = None
        self._data_episode_serial = 0
        self._data_episode = None
        self._held_word_yield = None
        self.data_repeats = 0
        self._speculative = {}
        self._hold_until = None
        self._hold_started = None
        self._offer_at = None
        self.next_offer_at = None
        self._handback_attempted = False
        self._handback_pending = None
        self._handback_refusals = set()
        self._acked_idle = None
        self._blocked_redue = False
        self._rollback_until = None
        self._closing_failure = False
        self._close_resumed_in_peer_turn = False
        self.identification_closing = False
        self._refused_due = None
        self._close_refusals = 0
        self._peer_word = None
        self._peer_word_at = -1
        self._close_held_v27 = False
        self.close_context = None
        self._grant_receive_origin = None
        self._wall_clock = wall_clock
        self.session_end = session_end
        self.session_wall_end = session_wall_end
        self._close_candidates = {}
        self._clear_ack_wait()
        self._expired_acks = set()
        self._emissions = {}
        self.weak_ack = None
        self.weak_pairs = True
        self._weak_seen = set()
        self._last_body_ack = None
        self._held_answer = None
        self._accepted_proposal = None
        self._last_ack = None
        self.acked_counter = None
        self.link_events = []
        self._opening = False
        self._opening_unread_head = None
        self._opening_wait = False
        self._opening_end = None
        self._opening_wall_end = None
        self._train_root = None
        self._train_wall_end = None
        self._train_active = False
        self._train_fallback = False
        self._train_fallback_keyed = False
        self._train_elapsed = -1
        self._train_keyed = {}
        self._train_heads = {}
        self._payload_ladder = self.ladder
        self._unsupported_copies = []
        self._unsupported_control = None
        self._opening_source = True
        self._opening_prefix = None
        self._opening_negotiated = False
        self._ladder_head_onset = None
        self._ladder_answered = None
        self._receive_epoch = 0
        self._receive_active = False
        self._receive_boundary = None
        self._root_epoch = {}
        self._observation_roots = {}
        self._accepted_roots = set()
        self._last_source_onset = None
        self._emission_epoch = {}
        self._handoff_ticket = None
        self._granted = None
        self._receipt = None
        self._negotiation = 0
        self._control_roots = {}
        self._v49_root = None
        self._handoff_root = None

    def _clear_opening_credit(self):
        self._opening_source = False
        self._opening_prefix = None
        self._opening_negotiated = False

    def _cancel_credit(self, reason):
        if self._receipt is not None:
            self.adapter.cancel(self._receipt, reason)
            self._receipt = None
        self._opening_prefix = None
        self._opening_negotiated = False
        self._negotiation = 0
        self._v49_root = None

    def _close_receive_epoch(self, reason, *, terminal=False):
        self.adapter.end_epoch(self._receive_epoch, reason, terminal=terminal)
        self._receive_active = False
        self._receipt = None
        self._opening_prefix = None
        self._opening_negotiated = False
        self._negotiation = 0
        self._control_roots.clear()
        self._v49_root = None
        self._handoff_root = None
        self.pending.clear()
        self._deferred = None
        self._held_answer = self._accepted_proposal = None
        self._unread_data_due = None
        if self._next is not None and self._next.receive_epoch is not None:
            self._next = None
        self._last_source_onset = None
        self._handoff_ticket = self._granted = None
        self._receive_boundary = (self.last.stream_sample if self.last else self._receive_boundary)
        self._last_ack = self._last_body_ack = None
        self.acked_counter = None

    def _open_receive_epoch(self, onset):
        if self._receive_epoch == 0 or 'turn' in self.chain_reset.split(','):
            self._ack_boundary = True
            self._chain_reason = 'opening' if self._chain_epoch == 0 else 'turn'
        self._held_answer = self._accepted_proposal = None
        self._unread_data_due = None
        self._receive_epoch += 1
        self._receive_active = True
        self._receive_boundary = onset
        self._last_source_onset = None
        self._opening_source = self._opening_source and self._receive_epoch == 1
        if self._next is not None and self._next.receive_epoch is not None:
            self._next = None

    def _source_root(self, reg):
        obs = reg.observation
        prior = self._observation_roots.get(id(reg.replaced)) if reg.replaced else None
        root = prior[1] if prior is not None and prior[0] is reg.replaced else None
        if root is None and reg.replaced:
            root = (reg.replaced.kind, reg.replaced.stream_sample)
        if root is None:
            root = (obs.kind, obs.stream_sample)
        self._observation_roots[id(obs)] = obs, root
        return root

    def _accept_source(self, reg, *, handoff=False):
        obs = reg.observation
        root = self._source_root(reg)
        target = self._emissions.get(reg.responds_to)
        ordinary_reply = (obs.kind == Kind.BODY and target is not None
                          and target.state in ISS_TURN and target.kind != Kind.WORD
                          and self._emission_epoch.get(reg.responds_to) == self._receive_epoch)
        close_reply = (self.close_context is not None and self.state == State.ISS_CLOSE
                       and obs.kind == Kind.BODY and target is not None
                       and target.state == State.ISS_CLOSE
                       and reg.responds_to == self.close_context.emission)
        epoch = self._root_epoch.get(root)
        fresh = epoch is None
        if epoch is None:
            epoch = self._receive_epoch
            self._root_epoch[root] = epoch
        if (epoch != self._receive_epoch or root in self._accepted_roots
                or not self._receive_active and self._receive_epoch > 0
                and not (handoff and fresh or ordinary_reply or close_reply)
                or reg.outcome not in (Registered.NEW, Registered.ENRICHED, Registered.CONFIRMED)
                or obs.frame is None or obs.provisional or obs.weak or obs.unexpected
                or self._receive_boundary is not None and obs.stream_sample < self._receive_boundary
                or self._last_source_onset is not None
                and obs.stream_sample < self._last_source_onset
                or reg.responds_to in self._emission_epoch
                and self._emission_epoch[reg.responds_to] != self._receive_epoch
                and reg.responds_to != self._handoff_ticket
                and not (self._awaiting_peer_turn and target is not None and target.final)
                and not (self._receive_epoch == 0 and obs.kind == Kind.BODY
                         and self._emission_epoch[reg.responds_to] == 1)):
            return False, root
        return True, root

    def expected_words(self):
        words = {'Y27'}
        if self.state == State.ISS_LADDER and self.index < len(self.ladder):
            words.update(('Z27', ACK.get(self.ladder[self.index], 'Y27')))
            if not self._opening and self.data_kind == Kind.LONG:
                words.update(self.ladder_admit)
            if self._opening:
                words.update(('T27', 'W27', 'X27'))
            if self.ladder[self.index] in (49, 53) or self._sl5_turn():
                words.update(('X27', 'V27', 'T27', 'U27'))
        elif (self.state in (State.ISS_DATA, State.ISS_DATA_ACK)
              and self._data_packet is not None):
            words.update(('X27', 'V27', 'T27', 'Z27', 'W27'))
            if self._word_yield or self._data_packet.kind == Kind.LONG:
                words.add('U27')
        else:
            words.update(('V27', 'X27', 'Z27'))
        return words

    def seed_opening(self, emissions, observations, *, values=None):
        if not emissions:
            return
        self._opening = True
        self.ladder = tuple(values or [e["value"] for e in emissions if e.get("value") is not None] or (7, 38))
        self.state = State.ISS_LADDER
        self.index = 0
        for entry in emissions:
            proposal = Proposal(Kind.SHORT, entry['audio_start'], value=entry.get('value', 38),
                                header=entry.get('header', 'A'), state=State.ISS_LADDER)
            self._emissions[proposal.stream_sample] = proposal
            self._emission_epoch[proposal.stream_sample] = self._receive_epoch + 1
        self.last = proposal
        self._train_root = next(
            e['audio_start'] for e in emissions if e.get('value', 38) == proposal.value)
        self._opening_end = (min(self._train_root + round(10*1.25*self.fs),
                                 self.session_end)
                             if proposal.value == 38 and self.session_end is not None else
                             self._train_root + round(10*1.25*self.fs)
                             if proposal.value == 38 else None)
        first = next((e for e in emissions if e.get('value') == self.ladder[0]), None)
        if self.last.value != self.ladder[0] and first and any(o.word == ACK.get(self.ladder[0]) and not o.provisional
                         and self._in_answer_slot(o.stream_sample, self._emissions[first['audio_start']])
                         for o in observations):
            self.index = 1

    def _in_answer_slot(self, at, target, *, repeated=False):
        offset = T[target.kind] if target.kind in (Kind.LONG, Kind.CHIRP) else RESPONSE_S
        delta = at-target.stream_sample-round(offset*self.fs)
        tolerance = round(.035*self.fs)
        if repeated:
            if delta < -tolerance:
                return False
            cycle = round(CYCLE[target.kind]*self.fs)
            delta -= round(delta/cycle)*cycle
        return abs(delta) <= tolerance

    def reply_pending_before(self, boundary, now, incoming=None):
        if now >= boundary + round(REPLY_DRAIN_S*self.fs):
            return False
        packets = [(p.get('body') or p.get('word'), p.get('accepted'))
                   for s, p in self.pending.items() if s not in self._decided]
        if incoming is not None:
            packets.append((incoming, incoming.frame is not None))
        for packet, accepted in packets:
            if (packet is None or packet.stream_sample >= boundary
                    or packet.kind == Kind.WORD and packet.word not in ('Y27', 'BREAKIN')):
                continue
            kind = Kind.BODY if packet.kind == Kind.WORD else packet.kind
            span = self.t[kind] + WORD_HORIZON_S + (CYCLE[kind] if accepted else 0)
            if now < packet.stream_sample + round(span*self.fs):
                return True
        if self._deferred is not None:
            sample, pending = self._deferred
            if sample < boundary and now < pending['due'] + round(WORD_HORIZON_S*self.fs):
                return True
        for reply in (self._next, self.last):
            if reply is None:
                continue
            source = reply.accepted_target if reply.accepted_target is not None else reply.current_slot
            if source is None and self._accepted_proposal is not None and reply is self._next:
                source = self._accepted_proposal[1]
            span = WORD_HORIZON_S if reply.kind == Kind.WORD else CYCLE[reply.kind]
            if (source is not None and source < boundary
                    and now < reply.stream_sample + round(span*self.fs)):
                return True
        return False

    def _close_expired(self, now, incoming=None):
        ctx = self.close_context
        if ctx is None:
            return False
        if ctx.expired_at is None:
            if now < ctx.end and self._wall_clock() - ctx.wall_started < CLOSE_TIMEOUT_S:
                return False
            ctx.expired_at = min(now, ctx.end)
        if self.reply_pending_before(ctx.expired_at, now, incoming):
            return False
        self.link_down('close deadline')
        return True

    def _resume_close(self, now, *, exact=False, reply_to=None):
        ctx = self.close_context
        if (ctx is None or ctx.resumed or self.state == State.DOWN
                or ctx.expired_at is not None
                and (reply_to is None or reply_to >= ctx.expired_at)):
            return
        ctx.resumed = True
        self._next = self._deferred = None
        self.pending.clear()
        self._blocked_redue = False
        due = now if exact else (max(now + self.lead, self.next_raster(now))
                                 if self.last else now + self.lead)
        if not exact and self._peer_holds_turn(now):
            # K4MSU 05:05Z 88.781: a 0x20 keyed into the peer's own ISS slots.
            # The close asks for the turn first, and keys 0x20 once granted.
            self._closing_failure = self._close_resumed_in_peer_turn = True
            self.state = State.IRS_CLOSE
            self._break_in(due)
            return
        self._body(due, 0x20, State.ISS_CLOSE)
        self._next = replace(self._next, current_slot=reply_to)

    def _close_request(self, reg):
        ctx = self.close_context
        obs = reg.observation
        if (ctx is None or self.state != State.ISS_CLOSE or ctx.grants
                or ctx.emission is None or reg.responds_to != ctx.emission
                or ctx.generation != self._peer_generation
                or reg.outcome in (Registered.DUPLICATE, Registered.EXPIRED,
                                   Registered.CONTRADICTED)
                or obs.weak or obs.unexpected):
            return False
        target = self._emissions.get(ctx.emission)
        if target is None or target.state != State.ISS_CLOSE:
            return False
        if (self._last_source_onset is not None
                and ctx.emission < self._last_source_onset < obs.stream_sample):
            return False
        first = ctx.emission + round(RESPONSE_S * self.fs)
        cycle = round(1.25 * self.fs)
        slot = round((obs.stream_sample - first) / cycle)
        if (not 0 <= slot <= 2
                or abs(obs.stream_sample - first - slot * cycle) > round(.035 * self.fs)
                or obs.stream_sample > min(ctx.end, first + 2 * cycle + round(.810 * self.fs))):
            return False
        if slot and ctx.first_request is None:
            return False
        if obs.kind == Kind.WORD:
            if obs.word not in (None, 'BREAKIN', 'Y27'):
                return False
        elif (obs.kind != Kind.BODY or obs.frame is None
              or obs.frame.status.byte != 0x40
              or source_payload(obs.frame.status, obs.frame.data)):
            return False
        if slot == 0 and ctx.first_request is None:
            ctx.first_request = obs.stream_sample
        ctx.last_request = obs.stream_sample
        self._close_candidates[obs.stream_sample] = ctx.emission
        return True

    @staticmethod
    def _bare_word(obs):
        return obs.kind == Kind.WORD and (obs.extent_s is None or obs.extent_s <= .50)

    def _retire_data_episode(self):
        self._data_episode = None
        self._held_word_yield = None

    def _clear_held_word_yield(self):
        self._held_word_yield = None
        if self.close_context is not None:
            self.close_context.held_word_yield = None

    def _requeue(self):
        self.weak_ack = None
        self.outbox.requeue()

    def _body_ack(self, reg):
        if reg.peer_watch_generation is not None:
            return False
        if not self._current_response(reg):
            return False
        target = self._emissions.get(reg.responds_to)
        return (target is not None and self._data_packet is not None
                and target.kind == self._data_packet.kind
                and target.stream_sample >= self._data_packet.stream_sample
                and target.status == self._data_packet.status
                and target.payload == self.outbox.inflight)

    @staticmethod
    def _current_response(reg):
        if reg.response_generation is None:
            return True
        owner = reg.response_owner
        lease = owner.train_lease if owner is not None else None
        if lease is not None:
            return ((reg.response_generation == lease.generation
                     and reg.responds_to == lease.active)
                    or owner._answered_response(reg, reg.response_generation))
        if owner is not None and owner._answered_response(reg, reg.response_generation):
            return True
        obs = reg.observation
        # A decoded BODY replaces its Y27 head and revokes the lease during
        # registration. That revocation must not revoke this replacement too.
        head = owner._classification(obs.stream_sample) if owner is not None else None
        return (owner is not None and obs.kind == Kind.BODY and obs.frame is not None
                and reg.outcome == Registered.ENRICHED and head is not None
                and head.word == 'Y27' and not head.provisional
                and owner._train_generation == reg.response_generation + 1
                and owner._responds_to.get(obs.stream_sample) == reg.responds_to
                and any(current is obs for current in owner.observations))

    def offers_turn(self):
        return (self.state in (State.ISS_GRANT, State.IRS_BREAKIN) and self.last is not None
                and self.last.kind == Kind.BODY and self.last.state == self.state
                and not self.last.payload)

    def _grant_ack(self, reg):
        target = self._emissions.get(reg.responds_to)
        return (target is not None and target is self.last and self.offers_turn()
                and not (self.close_context is not None
                         and target.stream_sample < self.close_context.started))

    def _bytes_waiting(self):
        return bool(self.outbox.inflight) or self.outbox.queued and self.allow_breakin

    def update_application_progress(self, deadline, expired=False, failure_reason=None):
        if self.app_done:
            if expired and failure_reason:
                self.application_failure = self.reason = failure_reason
                if self.close_context is not None:
                    self.close_context.reason = failure_reason
            return
        self.application_deadline = deadline
        if expired:
            self.application_failure = failure_reason or 'application progress timeout'

    def _application_limits(self, now):
        if not self.managed_application:
            return False
        if (self.session_end is not None and now >= self.session_end
                or self.session_wall_end is not None and self._wall_clock() >= self.session_wall_end):
            if self.state != State.DOWN:
                self.link_down('session ceiling')
            return True
        if (not self.app_done
                and (self.application_failure is not None or self.application_deadline is not None
                     and self._wall_clock() >= self.application_deadline)):
            self.finish_application(now, failure=self.application_failure or 'application progress timeout')
        return self.state == State.DOWN

    @property
    def _handback_offer_open(self):
        return self._handback_attempted and (self.managed_application or self._hold_until is not None)

    def _clear_offer(self):
        self._offer_at = None
        if self.peer_watch is None:
            self.next_offer_at = None
        self._handback_pending = None

    def revoke_peer_watch(self):
        if self.peer_watch is not None:
            self.peer_watch = None
            self.next_offer_at = None

    def _long_peer_open(self):
        return self.peer_phase is not None and self.peer_phase.kind in (Kind.LONG, Kind.CHIRP)

    def peer_span_fits(self, proposal, length, allowance):
        phase = self.peer_phase
        if phase is None or phase.kind not in (Kind.LONG, Kind.CHIRP):
            return True
        following = phase.onset + ((proposal.stream_sample-phase.onset)//phase.cycle+1)*phase.cycle
        return proposal.stream_sample + length + allowance <= following

    def watch_acquisition(self, proposal):
        return (proposal is self._next and self.peer_watch is not None
                and self._long_peer_open() and self.peer_phase.generation == self._peer_generation
                and not proposal.payload and proposal.accepted_target is None
                and (proposal.kind == Kind.WORD and proposal.word == 'Y27'
                     or proposal.kind == Kind.BODY and proposal.status == 0x40)
                and proposal.state in (State.ISS_GRANT, State.IRS_BREAKIN)
                and proposal.peer_watch_generation in (None, self.peer_watch.generation))

    def _watch_qualifies(self, reg):
        watch, obs = self.peer_watch, reg.observation
        return (watch is not None and reg.peer_watch_generation == watch.generation
                and reg.outcome in (Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED)
                and obs.kind in (Kind.LONG, Kind.CHIRP) and obs.frame is not None
                and not obs.weak and not obs.provisional and not obs.unexpected
                and obs.stream_sample >= watch.onset
                and abs(obs.stream_sample-watch.onset
                        - round((obs.stream_sample-watch.onset)/watch.cycle)*watch.cycle)
                <= round(.035*self.fs))

    def _recover_peer_turn(self, due=None):
        watch = self.peer_watch
        self.state = State.IRS_ANSWER
        if self._next is not None and self._next.accepted_target is None:
            self._next = None
        self._clear_ack_wait()
        self._handback_pending = None
        self._offer_at = None
        if watch.last_attempt is not None:
            self.next_offer_at = watch.last_attempt + round(HANDBACK_REOFFER_SECONDS*self.fs)
        if due is not None and self._next is None:
            self._next = self._held_proposal(due)

    def _handback_exit(self, at):
        self._clear_ack_wait()
        if self.managed_application and not self.app_done:
            self.state = State.ISS_HOLD
            self._idle(at)
        else:
            self._requeue()
            self._body(at, 0x20, State.ISS_CLOSE)

    def waiting_status(self, onset):
        result = []
        if self._offer_at is not None:
            elapsed = max(0, min(4, (onset-self._offer_at)//round(1.25*self.fs)))
            result.append(f'handback={elapsed}/4')
            if self.next_offer_at is not None:
                result.append(f'reoffer_in={max(0., (self.next_offer_at-onset)/self.fs):.3f}')
        elif (not self.managed_application and self.state == State.ISS_HOLD
              and not self._handback_attempted and self._hold_started is not None
              and self._hold_until is not None and onset < self._hold_until):
            elapsed = max(0, (onset-self._hold_started)//round(1.25*self.fs))
            result.append(f'hold={min(self.hold_cycles, elapsed)}/{self.hold_cycles}')
        if self.managed_application:
            remaining = ('unarmed' if self.application_deadline is None else
                         f'{max(0., self.application_deadline-self._wall_clock()):.3f}')
            result.append(f'application_remaining={remaining}')
        return result

    def finish_application(self, now, failure=None):
        self._grant_receive_origin = None
        if self.app_done and (failure is None or self._closing_failure):
            return
        self._cancel_credit('application finished')
        self._clear_opening_credit()
        self.revoke_peer_watch()
        self.app_done = True
        if failure:
            self.application_failure = self.reason = failure
            self._closing_failure = True
        terminal = self._closing_failure or not self.outbox.inflight and not self._bytes_waiting()
        # A successful IRS finish can be caused synchronously by accepted FQ.
        # Its pending/selected/deferred answer must retain its normal key turn.
        if (self.managed_application and terminal
                and (self._closing_failure or self.state in ISS_TURN)):
            self._clear_offer()
            self._clear_ack_wait()
            self._cancel_deferred('application finished')
            self._accepted_proposal = None
            self.pending.clear()
            self._speculative.clear()
            if self.state in ISS_TURN:
                self._next = None
                self._requeue()
                self._body(self.next_raster(now), 0x20, State.ISS_CLOSE)
                self.state = State.ISS_CLOSE
            elif self.state in (State.IRS_ANSWER, State.IRS_YIELD, State.IRS_BREAKIN):
                self._next = None
                self._break_in(self.next_raster(now))
                self.state = State.IRS_CLOSE
        elif self.managed_application and self.state in (State.ISS_HOLD, State.ISS_HANDBACK,
                                                        State.ISS_GRANT):
            self._next = None
            self._begin_data_turn(self.next_raster(now), word_yield=True)
        if (
            self.state == State.ISS_HOLD
            and not self.outbox.queued
            and not self.outbox.inflight
        ):
            self._hold_until = now
            if self._next is not None and self._next.state == State.ISS_HOLD:
                self._body(self._next.stream_sample, 0x20, State.ISS_CLOSE)

    def close_for_identification(self, now):
        """Retire data at a packet boundary, retaining every selected reply."""
        if self.state in (State.DOWN, State.ISS_CLOSE, State.IRS_CLOSE):
            return
        self.revoke_peer_watch()
        if self._next is not None and self._next.peer_watch_generation is not None:
            self._next = replace(self._next, peer_watch_generation=None)
        self.app_done = self._closing_failure = True
        self.identification_closing = True
        self.reason = 'identification deadline'
        if (self.state not in ISS_TURN or self._accepted_proposal is not None
                or self._deferred is not None
                or any(s not in self._decided for s in self.pending)
                or self._next is not None and (self._next.kind == Kind.WORD
                                              or self._next.accepted_target is not None)):
            return
        due = self._next.stream_sample if self._next is not None else self.next_raster(now)
        self._cancel_credit('identification deadline')
        self._clear_opening_credit()
        self._clear_ack_wait()
        self._requeue()
        self._body(due, 0x20, State.ISS_CLOSE)
        self.state = State.ISS_CLOSE

    def start(self, at):
        self._advance(at)

    def _short(self, at, index, header="A", repeat=False):
        self._next = Proposal(
            Kind.SHORT,
            at,
            value=self.ladder[index],
            header=header,
            state=State.ISS_LADDER,
            index=index,
            repeat=repeat,
        )

    def _body(self, at, status, state, payload=b""):
        if state == State.ISS_CLOSE:
            self._grant_receive_origin = None
            self.revoke_peer_watch()
            self._clear_offer()
        if state == State.ISS_CLOSE and status == 0x20 and self.close_context is None:
            self.close_context = CloseContext(
                self.reason, self.app_done, bytes(self.outbox.arq._outbuf),
                self.outbox.inflight, at, self._wall_clock(),
                min(at + round(CLOSE_TIMEOUT_S * self.fs),
                    self.session_end or at + round(CLOSE_TIMEOUT_S * self.fs)),
                held_word_yield=self._held_word_yield)
            # Keep the admitted evidence in this one close context even if its
            # first proposal is refused. Its data episode may now release bytes.
            self._retire_data_episode()
        self._next = Proposal(
            Kind.BODY, at, payload=payload, status=status, state=state
        )

    def _idle(self, at):
        self._next = Proposal(Kind.NORMAL, at, status=self.counter, state=State.ISS_HOLD,
                              header=self._normal_header(self.counter))

    def _normal_header(self, status):
        """B for another copy of the normal we last keyed, as stock keys every
        copy after a packet's first; A opens a packet."""
        last = self.last
        return "B" if last is not None and last.kind == Kind.NORMAL and last.status == status else "A"

    def _begin_data_turn(self, at, state=State.ISS_DATA, *, word_yield=False,
                         counted_continuation=False):
        if self.peer_watch is not None or self.peer_phase is not None and self.peer_phase.kind in (Kind.LONG, Kind.CHIRP):
            self.revoke_peer_watch()
            self.peer_phase = None
            self._peer_consumed = None
        if self._closing_failure:
            self._requeue()
            self._body(at, 0x20, State.ISS_CLOSE)
            return
        if self._handback_offer_open:
            # Every route out of a hand-back offer, including a counted-idle
            # ACK or an empty checked BODY, must use the same mail gate.
            if not self._bytes_waiting():
                self.state = State.ISS_HOLD
                self._idle(at)
                return
            if counted_continuation and not self.ladder and self.data_kind != Kind.CHIRP:
                # Body mode continues the offer's counted cycle.
                self._handback_attempted = False
                if self.managed_application:
                    self._hold_until = None
                self._clear_offer()
                self._data(at)
                return
            word_yield = True
        self._clear_held_word_yield()
        self._retire_data_episode()
        self._requeue()
        self._data_packet = None
        # H1 of K4MSU 05:05Z: the peer may count our bare Y27 as c0.
        self.counter = int(word_yield and self.y27_first_body == 'c1' and self.last is not None
                           and self.last.kind == Kind.WORD and self.last.word == 'Y27')
        self._first_data = True
        self._train_initial_data = not word_yield
        self._word_yield = word_yield
        self._ladder_returned = False
        self._chirp_origin = None
        self._last_data_ack = None
        self._last_data_form = None
        self._chirp_ack_word = 'X27'
        self._last_chirp = None
        self._handback_attempted = False
        if self.managed_application:
            self._hold_until = None
        self._clear_offer()
        self._data_refusals.clear()
        self._clear_ack_wait()
        self._restart_packet = None
        self._awaiting_peer_turn = False
        if word_yield and self.ladder and self.data_kind != Kind.LONG:
            self._short(at, 0)
        else:
            self._data(at, state, Kind.BODY)

    def _data_ack(self, obs):
        return (self._data_packet is not None
                and (obs.word == self._expected_data_ack()
                     or self._data_packet.kind == Kind.LONG
                     and obs.word == "U27" and self.iss_u27 == "ack"))

    def _expected_data_ack(self, *, idle=False):
        packet = self.last if idle else self._data_packet
        policy = self._ack_policy(packet.kind)
        # x27 shapes only our answers; the peer's acks still follow the toggle.
        if policy.startswith("toggle") or policy == 'x27':
            if self._last_data_ack is None or self._packet_form(packet) != self._last_data_form:
                return 'X27'
            return (self._ack_partner(packet.kind, self._packet_form(packet))
                    if self._last_data_ack == "X27" else "X27")
        if packet.kind == Kind.LONG:
            t27 = (packet.status & 1) == 0
            return "T27" if t27 else "X27"
        if packet.kind == Kind.CHIRP:
            return self._chirp_ack_word
        return counter_ack(packet.status)

    def _handback_seed(self, reg, now):
        obs = reg.observation
        if (not self._bare_word(obs) or obs.word not in ('V27', 'X27', 'Z27')
                or obs.weak or obs.unexpected
                or reg.outcome in (Registered.EXPIRED, Registered.DUPLICATE)
                or self.last is None):
            return False
        if self.state == State.ISS_HANDBACK:
            if (self.last.state != State.ISS_HANDBACK
                    or reg.responds_to != self.last.stream_sample):
                return False
            if obs.word == 'Z27':
                if obs.provisional:
                    self._wait_for_ack(obs)
                else:
                    self._clear_ack_wait()
                    self._handback_refusals.add(reg.responds_to)
                    if len(self._handback_refusals) >= 2:
                        self._handback_exit(self.next_raster(now))
                        if not self.managed_application or self.app_done:
                            self.reason = 'handback refused twice by Z27'
                return True
            if obs.provisional:
                self._wait_for_ack(obs)
                if obs.word == 'V27':
                    # Reserve the imminent raster before the word's tail resolves.
                    self._handback_pending = (obs.stream_sample, self.last, self.repeats)
                    self._seeded(self.next_raster(now), self.last)
                return True
            self._clear_ack_wait()
            if obs.word == 'X27':
                self._accept_handback(self.next_raster(now))
                event = 'handback-accepted'
            else:
                self._seeded(self.next_raster(now), self.last)
                event = 'handback-seed'
            self.link_events.append(dict(event=event, at=obs.stream_sample))
            return True
        if (obs.word != 'V27' or obs.provisional or not self.sends_data
                or self._handback_attempted or self.app_done
                or self._bytes_waiting() or self._data_packet is not None):
            return False
        if self.state == State.IRS_ANSWER and self.last.kind == Kind.WORD:
            if self.last.state != State.IRS_ANSWER or self.last.word == 'Y27':
                return False
            if not self._in_answer_slot(obs.stream_sample, self.last, repeated=True):
                return False
        elif self.state == State.ISS_HOLD and self.last.kind == Kind.NORMAL:
            if (reg.responds_to != self.last.stream_sample
                    or obs.word == self._expected_data_ack(idle=True)):
                return False
        else:
            return False
        at = self.next_raster(now)
        self._clear_ack_wait()
        self._body(at, 0x40, State.ISS_HANDBACK)
        self._hold_until = at + round(4*1.25*self.fs)
        return True

    def _seeded(self, at, body):
        """The slot after V27 answers our hand-back body.

        Every stock new ISS in k0003 (1460.934, 1473.708, 1481.812, 1488.988,
        1524.906) keyed a robust-short ladder after its first body, never a
        normal: 13 -> 60/61 before the SL3 long, 13 -> 12 -> 11 -> 10 before an
        SL5 short."""
        if self.turn_entry != 'ladder':
            self._idle(at)
            return
        self._turn_body = body
        self._turn_values = (self._payload_ladder if self._payload_ladder and self._bytes_waiting()
                             else LADDER_SL5)
        self.index = 0
        self.state = State.ISS_LADDER
        self._short(at, 0)

    @property
    def ladder(self):
        if self._turn_body is not None and self.state == State.ISS_LADDER:
            return self._turn_values
        return self._ladder

    @ladder.setter
    def ladder(self, values):
        self._ladder = values

    def _sl5_turn(self):
        return self._turn_body is not None and self.ladder == LADDER_SL5

    def _after_turn_ladder(self, at):
        body, values = self._turn_body, self.ladder
        self._turn_body, self.index = None, 0
        # INFERRED, k0003: each acked ladder short takes a counter. Bodies c0
        # led long c2 after 13, 60 (1477.461, 1529.906), long c1 after 61
        # (1491.503) and normal c0 after 13, 12, 11, 10 (1488.060).
        self.counter = (body.status + len(values)) % 4
        if not self._bytes_waiting():
            self.state = State.ISS_HOLD
            self._hold_until = None
            self._next = Proposal(Kind.NORMAL, at, status=0x40 | self.counter, state=State.ISS_HOLD)
            return
        self._handback_attempted = False
        if self.managed_application:
            self._hold_until = None
        self._clear_offer()
        self._first_data = self._word_yield = False
        self._after_ladder(at)

    def _accept_handback(self, at):
        """Continue a confirmed X27 through the common hand-back mail gate."""
        self.counter = 1
        self._last_data_ack = 'X27'
        self._last_data_form = 'normal short'
        self._first_data = self._word_yield = False
        self._begin_data_turn(at, counted_continuation=True)

    def _commit_data(self, word=None):
        self._clear_held_word_yield()
        self._retire_data_episode()
        self._train_active = False
        self._train_initial_data = False
        self._train_heads.clear()
        self.weak_ack = None
        self._last_data_ack = word or self._expected_data_ack()
        self._last_data_form = self._packet_form(self._data_packet)
        if self._data_packet is not None and self._data_packet.kind == Kind.CHIRP:
            self._chirp_ack_word = 'V27' if self._chirp_ack_word == 'X27' else 'X27'
        self.outbox.commit()
        self._handback_attempted = False
        self._first_data = False
        self._word_yield = False
        self.counter = (self.counter + 1) % 4
        self._data_packet = None
        self.data_repeats = 0
        self._data_refusals.clear()

    def _data(self, at, state=State.ISS_DATA, kind=None):
        if self._handback_offer_open:
            self._begin_data_turn(at, state, counted_continuation=True)
            return
        kind = kind or (Kind.NORMAL if self.data_kind == Kind.BODY and not self._first_data
                        else self.data_kind)
        if self._closing_failure:
            self._requeue()
            self._body(at, 0x20, State.ISS_CLOSE)
            return
        level = self.long_level if kind == Kind.LONG else None
        size = LONG_BYTES[level] if level is not None else {Kind.CHIRP: 22, Kind.NORMAL: 40, Kind.BODY: 31}[kind]
        payload = self.outbox.transfer(size)
        if not payload:
            if self.app_done:
                self._body(at, 0x20, State.ISS_CLOSE)
            else:
                self.state = State.ISS_HOLD
                if self._hold_until is None:
                    self._hold_until = at + round(self.hold_cycles * 1.25 * self.fs)
                    self._hold_started = at
                self._idle(at)
            return
        status = self.counter | (0x60 if kind == Kind.CHIRP else
                                0x40 if self._first_data and self._word_yield else 0)
        final = kind == Kind.LONG and not self.outbox.queued
        if kind == Kind.LONG:
            queued = len(self.outbox.arq._outbuf)
            spug = bool(queued) and self.spug_policy == 'queued'
            status = self.counter | (0x40 if final else 0x60 if spug else 0)
        elif kind == Kind.BODY and self.data_kind == Kind.LONG:
            status = self.counter
        if kind == Kind.CHIRP and self._chirp_origin is None:
            self._chirp_origin = at
        self._next = Proposal(kind, at, payload=payload, status=status, state=state,
                              level=level, final=final,
                              header=self._normal_header(status) if kind == Kind.NORMAL else "A",
                              repeat=kind == Kind.CHIRP and self._last_chirp == (status, payload),
                              chirp_slot=self._chirp_slot(at) if kind == Kind.CHIRP else 0)

    def _chirp_slot(self, at):
        if self._chirp_origin is None:
            return 0
        return round((at - self._chirp_origin) / (CYCLE[Kind.CHIRP] * self.fs))

    def _advance(self, at):
        if self._closing_failure:
            self._requeue()
            self._body(at, 0x20, State.ISS_CLOSE)
        elif not self.sends_data:
            self.state = State.ISS_HOLD
            self._idle(at)
        elif self.ladder:
            self._short(at, 0)
        else:
            self._data(at)

    def _after_ladder(self, at):
        if self.data_kind == Kind.LONG:
            if self._restart_packet is not None:
                packet, self._restart_packet = self._restart_packet, None
                self.outbox.transfer(len(packet.payload))
                self._next = replace(packet, stream_sample=at, repeat=True)
                return
            self._last_data_ack = self._last_data_form = None
        if self.data_kind == Kind.CHIRP:
            # A failed c0 still precedes the negotiated c1; its loan is not committed.
            if self._first_data and not self._word_yield:
                self.counter = (self.counter + 1) % 4
                self._first_data = False
            self._requeue()
            self._data_packet = None
            self._chirp_origin = None
        self._data(at)

    def _finish_data_turn(self, at):
        self._next = None
        self._clear_ack_wait()
        self._hold_until = None
        self._clear_offer()
        self.state = State.IRS_ANSWER
        self._open_receive_epoch(at)
        self._awaiting_peer_turn = True
        self.link_events.append(dict(event='data-turn-ended', at=at))

    def _clear_ack_wait(self):
        self._waiting_ack = self._waiting_until = self._deferred_ack = None
        self._waiting_observation = None
        self._deferred_slot = None

    def _wait_for_ack(self, obs):
        self._waiting_observation = obs
        self._waiting_ack = obs.stream_sample
        kind = Kind.SHORT if self.state == State.ISS_LADDER else self.data_kind
        self._waiting_until = obs.stream_sample + round(
            (WORD_HORIZON_S + CYCLE[kind]) * self.fs
        )

    def armed_repeat(self):
        """The next slot of an unacknowledged body or normal, which its repeat (or
        the close its exhausted budget owes) holds until an ACK read in time
        swaps it; None when nothing is owed."""
        if (self.state not in (State.ISS_DATA, State.ISS_DATA_ACK) or self._next is not None
                or self._data_packet is None or self._data_packet.kind not in (Kind.BODY, Kind.NORMAL)
                or self.last is None or self.last.payload != self._data_packet.payload):
            return None
        return self.last.stream_sample + round(CYCLE[self.last.kind] * self.fs)

    def next_raster(self, now):
        if ((self._breaking_in() or self.peer_watch is not None)
                and self.peer_phase is not None
                and self.peer_phase.generation == self._peer_generation):
            return self.breakin_due(now)
        if self.last is None:
            return now + self.lead
        if (self.state == State.ISS_LADDER and self._ladder_head_onset is not None
                and self._ladder_head_onset > self.last.stream_sample):
            # An unnamed/Y27 head may carry a body through the next short slot.
            due = self._ladder_head_onset + round(.940 * self.fs)
            cycle = round(1.25 * self.fs)
            floor = max(now + self.lead, self._refused_due or 0)
            return due + max(0, (floor - due + cycle - 1) // cycle) * cycle
        cycle = round((self._peer_cycle if self._requesting() or self.state == State.IRS_ANSWER
                       else CYCLE[self.last.kind]) * self.fs)
        floor = max(now + self.lead, self._refused_due or 0)
        k = max(1, (floor - self.last.stream_sample + cycle - 1) // cycle)
        return self.last.stream_sample + k * cycle

    def _turn_raster(self, now):
        self.revoke_peer_watch()
        self.peer_phase = None
        self._peer_consumed = None
        self._peer_cycle = 1.25
        return self.next_raster(now)

    def breakin_due(self, now, *, lead=None):
        phase = self.peer_phase
        if phase is None or phase.generation != self._peer_generation:
            return self.next_raster(now) if self.last is not None else now + (self.lead if lead is None else lead)
        lead = self.lead if lead is None else lead
        first = phase.onset + round(self.t[phase.kind] * self.fs)
        floor = max(now + lead, self._refused_due or 0)
        if self._peer_consumed is not None:
            floor = max(floor, self._peer_consumed + 1)
        k = max(0, (floor - first + phase.cycle - 1) // phase.cycle)
        return first + k * phase.cycle

    def _breaking_in(self):
        return (self.state == State.IRS_BREAKIN and self.last is not None
                and self.last.state == State.IRS_BREAKIN)

    def reschedule_breakin(self, now, *, lead=None):
        candidate = self.breakin_due(now, lead=lead) if self.peer_phase is not None else None
        if (self._next is None and self.state == State.IRS_ANSWER
                and self._wants_turn(candidate) and self.peer_phase is not None
                and (not self.pending or self.peer_phase.kind in (Kind.LONG, Kind.CHIRP)
                     and all(s in self._decided for s in self.pending))
                and self._deferred is None):
            self._break_in(candidate)
        if (self._next is not None and self._next.state == State.IRS_BREAKIN
                and self.peer_phase is not None):
            due = self.breakin_due(now, lead=lead)
            if due > self._next.stream_sample:
                self._next = replace(self._next, stream_sample=due)

    def _record_peer_phase(self, reg, now):
        obs = reg.observation
        if (self.state not in (State.IRS_ANSWER, State.IRS_YIELD,
                               State.IRS_BREAKIN, State.IRS_CLOSE)
                or obs.provisional or obs.weak or obs.unexpected
                or reg.outcome not in (Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED)):
            return
        if obs.kind in (Kind.NORMAL, Kind.LONG, Kind.CHIRP):
            if obs.frame is None:
                return
            identity = (obs.kind, obs.stream_sample, obs.frame.status.byte, obs.frame.data)
        elif obs.kind == Kind.SHORT:
            if (not obs.placed or obs.header not in ('A', 'B')
                    or obs.value is None or not 0 <= obs.value < 64):
                return
            identity = (obs.kind, obs.stream_sample, obs.header, obs.value)
        else:
            return
        if self.peer_phase is not None and self.peer_phase.generation == self._peer_generation:
            replacing = (reg.replaced is not None and obs.kind in (Kind.NORMAL, Kind.LONG, Kind.CHIRP)
                         and reg.replaced.stream_sample == self.peer_phase.onset
                         and reg.replaced.kind == self.peer_phase.kind)
            if (obs.stream_sample < self.peer_phase.onset and not replacing
                    or identity == self.peer_phase.identity):
                return
        self._grant_receive_origin = None
        self.peer_phase = PeerPhase(obs.stream_sample, obs.kind,
                                    round(CYCLE[obs.kind] * self.fs), self._peer_generation,
                                    identity)
        self._peer_cycle = CYCLE[obs.kind]
        if obs.kind in (Kind.LONG, Kind.CHIRP) and not self.app_done:
            old = self.peer_watch
            self.peer_watch = PeerWatch(obs.stream_sample, self.peer_phase.cycle, obs.cfo_hz,
                next(_peer_watch_generations), old.attempts if old else 0,
                old.last_attempt if old else None)
            if old is None:
                self._offer_at = self.next_offer_at = None
        elif obs.kind not in (Kind.LONG, Kind.CHIRP):
            self.revoke_peer_watch()
        if self._next is not None and self._next.state == State.IRS_BREAKIN:
            self._next = replace(self._next,
                                 stream_sample=self.breakin_due(max(obs.stream_sample, now)))

    def observe(self, reg, *, now=None):
        obs = reg.observation
        if (self._next is not None and obs.kind in DATA_KINDS
                and reg.outcome not in (Registered.EXPIRED, Registered.CONTRADICTED,
                                        Registered.DUPLICATE)
                and self._overtaken(self._next, obs)):
            # Review 1107: the proposal was placed before this packet was seen.
            self._withdrawn, self._next = self._next, None
        watched = reg.peer_watch_generation is not None
        qualified = watched and self._watch_qualifies(reg)
        phase = self.peer_phase
        ordinary_phase = (self.state in (State.IRS_ANSWER, State.IRS_YIELD, State.IRS_CLOSE)
                   and (self._receive_active or self._receive_epoch == 0)
                   and phase is not None and phase.generation == self._peer_generation
                   and phase.kind in (Kind.LONG, Kind.CHIRP)
                   and obs.stream_sample >= phase.onset
                   and abs(obs.stream_sample-phase.onset
                           -round((obs.stream_sample-phase.onset)/phase.cycle)*phase.cycle)
                   <= round(.035*self.fs))
        primary = (ordinary_phase and reg.primary_packet_due is not None
                   and abs(obs.stream_sample-reg.primary_packet_due) <= round(.035*self.fs))
        journal = self.adapter._delivery_journal
        completing_source = (ordinary_phase and self.app_done and not self._closing_failure
                             and journal is not None and journal['done'] and obs.frame is not None
                             and obs.kind in (Kind.LONG, Kind.CHIRP)
                             and reg.outcome in (Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED)
                             and journal['token'] == ('accept', obs.frame.status.byte, obs.frame.data)
                             and not obs.provisional and not obs.weak and not obs.unexpected)
        if watched and not qualified and not primary and not completing_source:
            return
        saved = None
        self._watch_recovery = False
        if qualified and not primary:
            # Stage IRS only around existing source admission. The adapter owns
            # external callback journaling; never restore or replace that journal.
            # Snapshot turn-owned values, never the host/outbox or callback journal.
            # Containers restore in place so accepted context/proposal identities survive.
            saved = {name: value for name, value in vars(self).items()
                     if name not in ('adapter', 'outbox')}
            containers = {}
            def retain(value):
                if isinstance(value, (dict, set, list)) and id(value) not in containers:
                    containers[id(value)] = value, value.copy()
                    for item in (value.values() if isinstance(value, dict) else value):
                        retain(item)
                elif isinstance(value, tuple):
                    for item in value:
                        retain(item)
                elif isinstance(value, HeldAnswer):
                    retain(value.context)
            for value in saved.values():
                retain(value)
            close = self.close_context
            close_values = vars(close).copy() if close is not None else None
            self._watch_admitted = False
            self._watch_recovery = (self.state in (State.ISS_GRANT, State.IRS_BREAKIN)
                or self._next is not None and self._next.state in (State.ISS_GRANT, State.IRS_BREAKIN))
            self._recover_peer_turn()
        admit = ((lambda: self.sf8_eligible(obs))
                 if obs.kind == Kind.LONG and self.sf8_eligible is not None else None)
        try:
            with self.adapter.admission(admit):
                return self._observe(reg, now=now)
        except P4LateFrame:
            return
        finally:
            if (saved is not None and not self._watch_admitted and not self._closing_failure
                    and self.state != State.DOWN):
                completed = self.app_done and not saved['app_done']
                for value, original in containers.values():
                    if isinstance(value, list):
                        value[:] = original
                    else:
                        value.clear()
                        value.update(original)
                for name, value in saved.items():
                    setattr(self, name, value)
                if close is not None:
                    vars(close).update(close_values)
                if completed:
                    # Completion happened in staged IRS. Keep the exact source
                    # inputs under the existing journal, retire restored retry
                    # work, and never run finish_application twice against it.
                    self._recover_peer_turn()
                    self.app_done = True
                    self.revoke_peer_watch()
            self._watch_recovery = False

    def _observe(self, reg, *, now=None):
        obs = reg.observation
        now = obs.stream_sample if now is None else now
        if self.state == State.DOWN:
            return
        if self._close_expired(now, obs):
            return
        if (self.close_context is not None and self.close_context.expired_at is not None
                and obs.stream_sample >= self.close_context.expired_at):
            return
        if self._train_expired(now, decision=False):
            return
        if reg.peer_watch_generation is None and not self._current_response(reg):
            return
        if (self._long_peer_open() and obs.kind == Kind.BODY and obs.frame is not None
                and obs.frame.status.protocol == 2 and not source_payload(obs.frame.status, obs.frame.data)
                and (obs.provisional or obs.weak or obs.unexpected
                     or reg.outcome not in (Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED))):
            return
        if (self.state == State.IRS_ANSWER
                and obs.kind in DATA_KINDS
                and not obs.weak and not obs.unexpected
                and reg.outcome in (Registered.NEW, Registered.ENRICHED,
                                    Registered.CONFIRMED, Registered.CONTRADICTED)
                and self._next is not None
                and not self._next.current_slot_accepted
                and self._next.stream_sample != obs.stream_sample
                + round(self.t[obs.kind] * self.fs)):
            self._next = None
        source = (obs.kind == Kind.CHIRP or obs.kind == Kind.BODY
                  and (obs.frame is None or bool(source_payload(obs.frame.status, obs.frame.data))))
        if source:
            status = obs.frame.status if obs.frame is not None else None
            supported = (status is not None and not status.source_mode and
                         (not status.extended or observed_extension(obs.frame)))
            handoff = (self._handoff_ticket is not None and obs.kind == Kind.BODY
                       and supported and status.protocol in (0, 2)
                       and bool(source_payload(obs.frame.status, obs.frame.data))
                       and reg.responds_to == self._handoff_ticket)
            accepted, root = self._accept_source(reg, handoff=handoff)
            if not accepted:
                if (obs.kind in (Kind.BODY, Kind.CHIRP) and obs.frame is None
                        and not obs.weak and not obs.unexpected
                        and reg.outcome in (Registered.NEW, Registered.ENRICHED,
                                            Registered.CONFIRMED, Registered.CONTRADICTED)):
                    self.pending.setdefault(obs.stream_sample,
                        {'word': None, 'accepted': False, 'text': False})['body'] = obs
                    if not obs.provisional:
                        self.link_events.append(dict(event=f'{obs.kind}-data-unread', at=obs.stream_sample))
                return
            initial = (obs.kind == Kind.BODY and supported and not self._receive_active
                       and self._receive_epoch == 0)
            if handoff:
                self._close_receive_epoch('peer handoff')
                self._open_receive_epoch(obs.stream_sample)
                self._root_epoch[root] = self._receive_epoch
                self._handoff_root = root
            elif initial:
                self._open_receive_epoch(obs.stream_sample)
                self._root_epoch[root] = self._receive_epoch
            elif (self.close_context is not None and self.state == State.ISS_CLOSE
                  and supported and obs.frame is not None
                  and source_payload(obs.frame.status, obs.frame.data) and not self._receive_active):
                self._open_receive_epoch(obs.stream_sample)
                self._root_epoch[root] = self._receive_epoch
            if supported:
                self._accepted_roots.add(root)
                self._last_source_onset = obs.stream_sample
        elif (obs.kind in (Kind.NORMAL, Kind.LONG) and obs.frame is not None
              and not obs.provisional and not obs.weak and not obs.unexpected):
            self._last_source_onset = max(obs.stream_sample, self._last_source_onset or obs.stream_sample)
        key = (obs.stream_sample, obs.kind, reg.outcome)
        if not source and key in self._seen:
            return
        if not source and reg.replaced and reg.replaced.stream_sample in self._expired_acks:
            return
        self._seen.add(key)
        if reg.replaced is not None:
            self._train_heads.pop(reg.replaced.stream_sample, None)
        if (self.train_policy == 'passive' and obs.kind == Kind.WORD
                and obs.provisional and not obs.weak and not obs.unexpected
                and obs.word in ('T27', 'W27', 'Z27', 'Y27')
                and reg.responds_to == self._train_root):
            self._train_heads[obs.stream_sample] = obs.stream_sample + round(.300*self.fs)
        if (self._opening_wait and obs.kind == Kind.BODY and obs.frame is not None
                and not obs.provisional and not obs.weak and not obs.unexpected
                and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)):
            self._opening_wait = False
        if (self._train_active and obs.kind == Kind.BODY and obs.frame is not None
                and not obs.provisional and not obs.weak and not obs.unexpected
                and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)):
            self._train_active = False
        if obs.frame is None or obs.kind not in DATA_KINDS:
            self._record_peer_phase(reg, now)
        self.events.append(reg)
        self._observe_registered(reg, now)
        if (obs.kind == Kind.SHORT and self._next is not None
                and self._next.kind == Kind.WORD
                and self._next.responds_to == obs.stream_sample):
            if (reg.outcome in (Registered.NEW, Registered.ENRICHED, Registered.CONFIRMED)
                    and not obs.provisional and not obs.weak and not obs.unexpected):
                self._next = self._hold_control(reg, self._next)

    def _observe_registered(self, reg, now):
        obs = reg.observation
        root = self._observation_roots.get(id(obs), (obs, (obs.kind, obs.stream_sample)))[1]
        if (self.close_context is not None and reg.replaced is not None
                and reg.outcome in (Registered.EXPIRED, Registered.CONTRADICTED)
                and reg.replaced.stream_sample == self.close_context.first_request):
            self.close_context.first_request = None
            self.close_context.last_request = None
            self._close_candidates.pop(reg.replaced.stream_sample, None)
        close_request = self._close_request(reg)
        if close_request and self._next is not None and self._next.state == State.ISS_CLOSE:
            self._next = None
        if self.state == State.ISS_LADDER and obs.kind == Kind.WORD:
            target = self._emissions.get(reg.responds_to)
            if target is not None and target.kind == Kind.SHORT:
                if not obs.provisional:
                    expected = (ACK.get(target.value, 'any') if self._sl5_turn()
                                else ','.join(self.ladder_admit)
                                if not self._opening and self.data_kind == Kind.LONG
                                and self.index == len(self.ladder)-1
                                else 'X27' if not self._opening and self.data_kind == Kind.LONG
                                else ACK.get(target.value))
                    reliable = (not obs.weak and not obs.unexpected
                                and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)
                                and target.value == self.ladder[self.index])
                    verdict = ('ambiguous' if not reliable or obs.word is None
                               else 'ack' if self._ack(obs, self.index)
                               else 'refused' if obs.word == 'Z27' else 'repeat')
                    self.link_events.append(dict(event='IssAck', at=obs.stream_sample,
                                                 responds_to=reg.responds_to, word=obs.word,
                                                 expected=expected, verdict=verdict))
                if reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE):
                    if target is self.last:
                        if obs.word in (None, 'BREAKIN', 'Y27'):
                            self._ladder_head_onset = obs.stream_sample
                        else:
                            self._ladder_head_onset = None
                    if (not obs.provisional and obs.word in p4words.STEPS and not obs.unexpected
                            and self.last is not None and target.value == self.last.value
                            and (self._opening or self.data_kind != Kind.LONG
                                 or not obs.weak and self._ack(obs, self.index))
                            and (self._ladder_answered is None
                                 or target.stream_sample > self._ladder_answered)):
                        self._ladder_answered = target.stream_sample
                        later = sum(e.kind == Kind.SHORT and e.stream_sample > target.stream_sample
                                    for e in self._emissions.values())
                        self.repeats = min(self.repeats, max(0, later - 1))
                if (obs.provisional and obs.word in (None, 'BREAKIN')
                        and reg.outcome != Registered.EXPIRED):
                    return
        if (not obs.provisional and not obs.weak and not obs.unexpected
                and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)
                and obs.stream_sample > self._peer_word_at):
            target = self._emissions.get(reg.responds_to)
            answer = (obs.kind == Kind.WORD and obs.word in p4words.STEPS
                      and target is not None
                      and self._in_answer_slot(obs.stream_sample, target))
            if answer:
                if obs.word != 'V27':
                    self._clear_held_word_yield()
                self._peer_word, self._peer_word_at = obs.word, obs.stream_sample
                if self.state == State.ISS_CLOSE:
                    if obs.word != 'V27':
                        self._close_held_v27 = False
                    # 2028: the gateway held V27 through 0x20 and both v37s,
                    # then resumed with Y27 bodies after our teardown.
                    if (obs.word == 'V27' and not self._close_held_v27
                            and target.kind == Kind.SHORT and target.value == 37):
                        self.link_down('close acknowledged')
                        return
                    if obs.word != 'Y27':
                        self._close_refusals += 1
            elif obs.frame is not None:
                self._clear_held_word_yield()
                self._retire_data_episode()
                self._peer_word, self._peer_word_at = None, obs.stream_sample
                self._close_held_v27 = False
            elif (self.state == State.ISS_CLOSE and obs.kind == Kind.WORD
                  and obs.word in p4words.STEPS and obs.word not in ('V27', 'Y27')):
                self._close_refusals += 1
        if not obs.provisional and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE):
            if obs.frame is not None or (obs.kind == Kind.SHORT
                    and obs.value in self.peer_ack and obs.value != 53):
                self._unsupported_copies.clear()
                self._unsupported_control = None
            elif (obs.kind == Kind.SHORT
                  and (obs.value == 53 or obs.value not in self.peer_ack)
                  and obs.value is not None and not obs.weak and not obs.unexpected
                  and not any(abs(obs.stream_sample-at) < self.fs//10
                              for at in self._unsupported_copies)):
                if self._unsupported_control != obs.value:
                    self._unsupported_copies.clear()
                    self._unsupported_control = obs.value
                self._unsupported_copies.append(obs.stream_sample)
                self.link_events.append(dict(event='peer-control-no-progress',
                                             at=obs.stream_sample, value=obs.value,
                                             header=obs.header, copies=len(self._unsupported_copies)))
        for sample, pending in list(self.pending.items()):
            incoming = pending.get('body')
            if (pending.get('accepted') and incoming is not None
                    and sample not in self._decided
                    and self._past(now, sample+round(self.t[incoming.kind]*self.fs))):
                self._defer(sample, pending, sample+round(self.t[incoming.kind]*self.fs))
        if self._repeat_of_deferred(reg):
            if obs.frame is not None:
                # Its held answer already owns this committed source. Do not
                # re-enter parsing or replace the accepted SourceResult.
                eligible = (obs.kind != Kind.LONG
                            or self.sf8_eligible is None or self.sf8_eligible(obs))
                if reg.peer_watch_generation is not None:
                    if (self.adapter._delivery_journal is not None
                            or self.adapter._admit is not None and not self.adapter._admit()):
                        raise P4LateFrame
                if eligible:
                    self._watch_admitted = True
                    self._record_peer_phase(reg, now)
            for sample in (obs.stream_sample, reg.replaced and reg.replaced.stream_sample):
                if sample is not None:
                    self._decided.add(sample)
                    self.pending.pop(sample, None)
            return
        qualifying = self._body_ack(reg)
        # Wrong policy words leave the ordinary data retry budget in force.
        # An admitted held seed is close provenance, never delivery credit or
        # an extra refusal. Own identical retry keys change peer generation but
        # not this immutable data/outbox episode. Fresh responses still pass
        # _current_response and _body_ack independently on every observation.
        if (qualifying and self._word_yield and self._data_episode is not None
                and self._data_episode.matches(self._data_packet)
                and reg.responds_to >= self._data_episode.first_key
                and self._data_packet.kind != Kind.CHIRP
                and obs.word == 'V27' and not self._data_ack(obs)
                and not obs.provisional and not obs.weak and not obs.unexpected
                and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)):
            self._held_word_yield = HeldWordYield(
                self._data_episode, obs.stream_sample, reg.responds_to,
                reg.response_generation)
        if qualifying and obs.kind == Kind.WORD and not obs.provisional:
            expected = self._expected_data_ack()
            reliable = (not obs.weak and not obs.unexpected
                        and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE))
            verdict = ('ambiguous' if not reliable or obs.word is None else 'ack' if self._data_ack(obs)
                       else 'refused' if obs.word == 'Z27' else 'repeat'
                       if obs.word in (self._last_data_ack, 'U27', 'X27',
                                        self._ack_partner(self._data_packet.kind,
                                                          self._packet_form(self._data_packet)))
                       else 'ambiguous')
            self.link_events.append(dict(event='IssAck', at=obs.stream_sample,
                                         responds_to=reg.responds_to, word=obs.word,
                                         expected=expected, verdict=verdict))
        refused = obs.word == 'Z27'
        if (qualifying and refused and not obs.provisional and not obs.weak
                and not obs.unexpected and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)):
            self._data_refusals.setdefault(reg.responds_to, obs.word)
            if len(self._data_refusals) >= 2:
                self._train_active = False
                self._clear_ack_wait()
                self._speculative.clear()
                self._rollback_until = None
                if self.data_kind == Kind.LONG:
                    self._restart_packet = self._data_packet
                self._requeue()
                self._data_packet = None
                self.data_repeats = 0
                if (self.ladder and not self._ladder_returned
                        and all(w == 'Z27' for w in self._data_refusals.values())):
                    self._ladder_returned = True
                    self._short(self.next_raster(now), 0)
                    return
                self.reason = ('word-yield data refused twice' if self._word_yield
                               else 'data refused twice by Z27')
                self._body(self.next_raster(now), 0x20, State.ISS_CLOSE)
                return
        if obs.word == 'Y27':
            self.weak_ack = None
        if obs.kind == Kind.WORD and not obs.provisional and self._data_packet is not None:
            accepted_words = (self._expected_data_ack(),)
            if self._data_packet.kind == Kind.LONG and self.iss_u27 == 'ack':
                accepted_words += ('U27',)
            if obs.word not in accepted_words or reg.outcome == Registered.EXPIRED:
                self.weak_ack = None
            elif obs.weak and qualifying:
                if obs.stream_sample in self._weak_seen:
                    return
                self._weak_seen.add(obs.stream_sample)
                self._clear_ack_wait()
                if reg.answered is not None:
                    self.weak_ack = None
                    self._speculative.pop(reg.replaced.stream_sample, None)
                    self.state = State.ISS_DATA_ACK
                    self._rollback_until = reg.answered + round(1.25*self.fs)
                    self._retry_data(self._rollback_until)
                    return
                pair = (self.weak_pairs and self._data_packet.kind == Kind.BODY
                        and self.weak_ack is not None
                        and self._emissions[reg.responds_to].retry_of == self.weak_ack)
                if not pair:
                    self.weak_ack = reg.responds_to if self._data_packet.kind == Kind.BODY else None
                    self._retry_data(self.next_raster(now))
                    return
                obs = replace(obs, weak=False)
            elif obs.word in accepted_words and not qualifying and reg.outcome != Registered.EXPIRED:
                return
        if (obs.kind == Kind.WORD and not obs.provisional and not obs.weak
                and not obs.unexpected and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)
                and reg.responds_to == self._train_root and obs.word in ('T27', 'W27', 'Z27')
                and (reg.response_slot == 0 or self._train_active)):
            if (self.train_policy == 'passive' and self._train_root is not None
                    and self.last is not None and self.last.stream_sample == self._train_root
                    and self.state in (State.ISS_LADDER, State.ISS_DATA, State.ISS_DATA_ACK)):
                if (self._opening and self.last.value == 38 and obs.word == 'Z27'):
                    self._opening_wait = True
                elif (self._opening and self.last.value == 7
                      or self.state in (State.ISS_DATA, State.ISS_DATA_ACK)
                      and self._data_packet is not None and self._data_packet.kind == Kind.BODY
                      and self._data_packet.status == 0 and self._first_data
                      and self._train_initial_data and not self._word_yield):
                    self._train_active = True
                if self._train_active or self._opening_wait:
                    self._next = None
        # A refinement can move the lock by a few samples. Carry the pending
        # decision across that refinement instead of inventing another packet.
        if reg.replaced and reg.replaced.stream_sample in self.pending:
            pending = self.pending.pop(reg.replaced.stream_sample)
            self.pending[obs.stream_sample] = pending
            if reg.replaced.stream_sample in self._decided:
                self._decided.remove(reg.replaced.stream_sample)
                self._decided.add(obs.stream_sample)
            if reg.replaced.stream_sample in self._dropped:
                self._dropped.remove(reg.replaced.stream_sample)
                self._dropped.add(obs.stream_sample)
        if reg.outcome == Registered.EXTENDED:
            # WS8EOC 15:38Z: a chirp peer's V27 grant is measured only once the
            # confirmed word is extended, 0.945 s after our Y27.
            if (self.state == State.IRS_BREAKIN and not reg.observation.unexpected
                    and self._breakin_granted(reg)):
                self._take_breakin_turn(reg, now)
                return
            if (self._opening_unread_head is not None and obs.word == 'Y27'
                    and obs.extent_s is not None
                    and abs(obs.stream_sample - self._opening_unread_head) < self.fs//10):
                self._opening_unread_head = None
            if (
                obs.stream_sample in self.pending
                and obs.stream_sample not in self._decided
            ):
                self.pending[obs.stream_sample]["word"] = obs
            return
        resolution = reg.outcome in (
            Registered.CONFIRMED,
            Registered.CONTRADICTED,
            Registered.EXPIRED,
        )
        if (resolution and reg.replaced and self._handback_pending is not None
                and reg.replaced.stream_sample == self._handback_pending[0]):
            _, offer, repeats = self._handback_pending
            self._handback_pending = None
            self._clear_ack_wait()
            if (reg.outcome != Registered.EXPIRED and not obs.provisional
                    and not obs.weak and not obs.unexpected and obs.kind == Kind.WORD
                    and obs.word in ('V27', 'X27') and reg.responds_to == offer.stream_sample
                    and (obs.extent_s is None or obs.extent_s <= .50)):
                if obs.word == 'X27':
                    self._accept_handback(self.next_raster(now))
                self.link_events.append(dict(event='handback-seed' if obs.word == 'V27'
                                             else 'handback-accepted', at=obs.stream_sample))
                return
            # The speculative idle conveyed no bytes or counter acceptance.
            self.state = State.ISS_HANDBACK
            self.repeats = repeats
            if repeats >= 3:
                self._handback_exit(self.next_raster(now))
            else:
                self._next = replace(offer, stream_sample=self.next_raster(now), repeat=True)
            if reg.outcome == Registered.EXPIRED:
                return
        if resolution and reg.replaced and reg.replaced.stream_sample == self._waiting_ack:
            if self._deferred_ack == self._waiting_ack:
                now = max(now, self._deferred_slot - self.lead + 1)
            self._clear_ack_wait()
        if (
            resolution
            and self.state == State.ISS_DATA_ACK
            and reg.answered is not None
            and self._data_ack(obs)
            and qualifying
            and not obs.weak
            and reg.outcome != Registered.EXPIRED
            and not obs.provisional
            and reg.replaced.stream_sample not in self._speculative
        ):
            self._commit_data()
            self._advance(reg.answered + round(1.25 * self.fs))
            return
        if (
            resolution
            and reg.replaced
            and reg.replaced.stream_sample in self._speculative
        ):
            prior, index, prior_repeats = self._speculative.pop(
                reg.replaced.stream_sample
            )
            ack = (
                self._data_ack(obs)
                if prior == State.ISS_DATA_ACK
                else self._ack(obs, index)
            )
            ack = ack and reg.outcome != Registered.EXPIRED and not obs.provisional
            if prior == State.ISS_DATA_ACK:
                ack = ack and qualifying and not obs.weak
            if ack:
                if prior == State.ISS_DATA_ACK and reg.answered is not None:
                    self._commit_data()
                if reg.answered is not None:
                    return
            elif reg.answered is not None:
                self.index = index
                self.repeats = prior_repeats
                if obs.word == "Y27" and reg.outcome != Registered.EXPIRED:
                    self._blocked_redue = True
                    self._next = None
                    return
                at = reg.answered + round(1.25 * self.fs)
                if prior == State.ISS_DATA_ACK:
                    self.state = prior
                    self._rollback_until = at
                    self._retry_data(at)
                else:
                    self._short(at, index, "B", True)
                return
            elif obs.word != "Y27" or reg.outcome == Registered.EXPIRED:
                at = (
                    self._next.stream_sample
                    if self._next
                    else self.next_raster(now)
                )
                if prior == State.ISS_DATA_ACK:
                    self._retry_data(at)
                else:
                    self._short(at, index, "B", True)
                return
        if resolution and reg.answered is not None and obs.kind != Kind.NORMAL:
            return
        if resolution and reg.replaced and reg.replaced.kind != Kind.WORD:
            self._next = None
            if reg.replaced.kind == Kind.NORMAL and obs.kind in (Kind.LONG, Kind.CHIRP):
                self.pending.pop(obs.stream_sample, None)
                self._decided.discard(obs.stream_sample)
        if reg.outcome == Registered.EXPIRED:
            if self.state == State.ISS_DATA_ACK and self._data_packet:
                at = self.next_raster(now)
                self._retry_data(at)
            self.pending.pop(obs.stream_sample, None)
            if reg.replaced and reg.replaced.word == "Y27":
                self._blocked_redue = False
            return
        if obs.unexpected and reg.outcome != Registered.EXPIRED:
            self.weak_ack = None
            return
        if (self.state == State.ISS_GRANT and obs.kind == Kind.NORMAL
                and obs.frame is None and obs.peer_raster_tail and not obs.provisional
                and not obs.weak and not self._bytes_waiting()):
            self.state = State.IRS_YIELD
            self._next = None
            return
        if (self.state == State.IRS_YIELD and obs.kind in (Kind.NORMAL, Kind.LONG, Kind.CHIRP)
                and obs.frame is None):
            return
        if self._handback_seed(reg, now):
            return
        if obs.kind == Kind.NORMAL and obs.stream_sample not in self._decided:
            self._next = None
        if obs.kind == Kind.NORMAL and obs.frame is None:
            self.pending.setdefault(obs.stream_sample,
                                    {"word": None, "accepted": False, "text": False})['body'] = obs
        if (obs.kind in (Kind.LONG, Kind.CHIRP) and obs.frame is None
                and (self.require_long_crc or obs.kind == Kind.CHIRP
                     or self.long_ack.startswith('toggle'))):
            self.pending.setdefault(obs.stream_sample,
                {'word': None, 'accepted': False, 'text': False})['body'] = obs
            if not obs.provisional:
                self.link_events.append(dict(event=f'{obs.kind}-data-unread', at=obs.stream_sample))
            return
        if obs.kind in (Kind.BODY, Kind.CHIRP) or obs.kind in (Kind.NORMAL, Kind.LONG) and obs.frame is not None:
            if obs.frame is None:
                return
            if obs.kind == Kind.BODY and not obs.provisional and not obs.weak:
                self._opening_unread_head = None
            if obs.kind == Kind.LONG and self.sf8_eligible is not None and not self.sf8_eligible(obs):
                return
            if (self.close_context is not None and self.state in (State.ISS_CLOSE, State.IRS_ANSWER)
                    and source_payload(obs.frame.status, obs.frame.data)):
                self.close_context.progress_at = obs.stream_sample
            data_reply = False
            idle_reply = (self.state == State.ISS_HOLD and self.last is not None
                          and self.last.kind == Kind.NORMAL
                          and reg.responds_to == self.last.stream_sample
                          and reg.responds_to != self._acked_idle)
            if (idle_reply and not obs.provisional and reg.outcome != Registered.DUPLICATE
                    and not obs.frame.status.source_mode and not obs.frame.status.extended
                    and obs.frame.status.protocol in (0, 2)):
                self._acked_idle = reg.responds_to
                self._last_data_ack = obs.word
                self._last_data_form = self._packet_form(self.last)
                self.counter = (self.counter + 1) % 4
                self._first_data = self._word_yield = False
                data_reply = True
                self._clear_ack_wait()
                self._next = None
            if (obs.kind in (Kind.BODY, Kind.NORMAL) and not obs.provisional
                    and reg.outcome != Registered.DUPLICATE
                    and obs.frame.status.protocol in (0, 2)
                    and not obs.frame.status.source_mode and not obs.frame.status.extended
                    and (source_payload(obs.frame.status, obs.frame.data)
                         or obs.kind == Kind.BODY and obs.frame.status.protocol == 2)
                    and qualifying
                    and (self._data_packet.kind in (Kind.BODY, Kind.NORMAL)
                         or self._data_packet.kind == Kind.LONG and self._data_packet.final)):
                # The checked reply, including an empty turn request, accepts
                # our field before a synchronous client can queue more bytes.
                data_reply = True
                final = self._data_packet.final
                self._commit_data()
                if final:
                    self._finish_data_turn(obs.stream_sample)
                self._clear_ack_wait()
                self._next = None
                self.link_events.append(dict(event='data-reply-ack', at=obs.stream_sample,
                                             responds_to=reg.responds_to))
            if self._closing_failure and self.state == State.IRS_CLOSE:
                due = obs.stream_sample + round(self.t[obs.kind] * self.fs)
                if now <= due - self.lead:
                    self._break_in(due)
                return
            try:
                identity = (obs.frame.status.byte, obs.frame.data)
                control_only = (
                    self._closing_failure
                    and obs.frame.status.protocol == 2
                    and not obs.frame.status.source_mode
                    and not obs.frame.status.extended
                    or self._repeats_granted(obs.frame)
                )
                prefix = (self._opening_prefix if self._opening_negotiated
                          and obs.kind == Kind.LONG and self.state == State.IRS_ANSWER else None)
                if obs.frame.status.protocol == 1 and obs.kind not in (Kind.NORMAL, Kind.LONG):
                    self._cancel_credit('peer close')
                own_receipt = SourceReceipt(self._receive_epoch,
                    root if obs.kind in (Kind.BODY, Kind.CHIRP) else (obs.kind, obs.stream_sample),
                    identity)
                receipt = (own_receipt if obs.kind == Kind.BODY and obs.frame.status.protocol == 2
                           else self._receipt)
                text = (None if control_only else self.adapter.accept(obs.frame, opening_prefix=prefix,
                    changeover=obs.kind == Kind.BODY and obs.frame.status.protocol == 2,
                    counted=obs.kind in (Kind.NORMAL, Kind.LONG),
                    receipt_context=ReceiptContext(receipt, obs.kind),
                    admit=(lambda: self.sf8_eligible(obs)) if obs.kind == Kind.LONG and self.sf8_eligible else None))
            except P4LateFrame:
                raise
            except P4StreamFailure as exc:
                self._cancel_credit('unsupported source')
                self._clear_opening_credit()
                if not self.adapter.terminal:
                    return
                self._closing_failure = True
                self.reason = str(exc)
                self._requeue()
                self.app_done = True
                self.revoke_peer_watch()
                self._next = None
                self.state = (
                    State.IRS_CLOSE
                    if self.state.name.startswith("IRS")
                    else State.ISS_CLOSE
                )
                if self.state == State.ISS_CLOSE:
                    at = obs.stream_sample + round(self.t[obs.kind] * self.fs)
                    while now > at - self.lead:
                        at += round(1.25 * self.fs)
                    self._body(at, 0x20, State.ISS_CLOSE)
                return
            if not control_only and source_payload(obs.frame.status, obs.frame.data):
                # New source has moved past the packet the grant answered.
                self._granted = None
            self._watch_admitted = True
            self._record_peer_phase(reg, now)
            if (obs.kind in (Kind.BODY, Kind.NORMAL, Kind.LONG) and obs.frame.status.protocol in (0, 2)
                    and source_payload(obs.frame.status, obs.frame.data)):
                self._hold_until = None
            if self.managed_application:
                # This checked idle reply grants a turn that, without eligible
                # mail, returns to waiting. Keep the actual offer's timestamps,
                # but supersede any speculative control transaction.
                self._handback_pending = None
                if not (idle_reply and data_reply and self._handback_offer_open
                        and not self.app_done and not self._bytes_waiting()
                        and obs.kind == Kind.BODY and obs.frame.status.protocol == 2
                        and not obs.frame.status.source_mode and not obs.frame.status.extended
                        and not source_payload(obs.frame.status, obs.frame.data)):
                    self._clear_offer()
            if obs.kind == Kind.LONG:
                self._sf8_acks[(obs.frame.status.byte, obs.frame.data)] = obs
            if not control_only and obs.kind in (Kind.NORMAL, Kind.LONG):
                result = self.adapter.result
                event = ('source-unclassified' if result.reason in (
                    'unclassified same-counter source', 'unclassified counter gap') else
                    'source-replay' if result.duplicate or result.replayed_source_bytes
                    or result.resegmented else 'source-accept')
                self.link_events.append(dict(event=event, at=obs.stream_sample,
                    kind=obs.kind, counter=obs.frame.status.counter, duplicate=result.duplicate,
                    replayed_source_bytes=result.replayed_source_bytes,
                    new_source_bytes=result.new_source_bytes, resegmented=result.resegmented,
                    reason=result.reason))
            if text is not None:
                if self.adapter.last_credit:
                    self.link_events.append(dict(
                        event='opening-prefix-credit' if prefix else 'changeover-prefix-credit',
                        at=obs.stream_sample,
                        reason='opening negotiated long' if prefix else 'changeover body in counted packet',
                        prefix=(prefix or source_payload(obs.frame.status, obs.frame.data)[:self.adapter.last_credit]).hex(),
                        credited_source_bytes=self.adapter.last_credit,
                        remaining_source_bytes=len(source_payload(obs.frame.status, obs.frame.data))-self.adapter.last_credit,
                        delivered_bytes=len(text)))
                if (self._opening_source and obs.kind == Kind.BODY
                        and obs.frame.status.byte == 0 and len(obs.frame.data) == 31
                        and len(source_payload(obs.frame.status, obs.frame.data)) == 31):
                    self._opening_source = False
                    self._opening_prefix = obs.frame.data
                    self._receipt = own_receipt
                    self.adapter.candidate(own_receipt, obs.frame.data, 'opening')
                    self._negotiation = 0
                    self._v49_root = None
                elif (obs.kind == Kind.BODY and obs.frame.status.protocol == 2
                      and source_payload(obs.frame.status, obs.frame.data)):
                    self._receipt = own_receipt
                    self._negotiation = 0 if root == self._handoff_root else -1
                    self._v49_root = None
                    self._clear_opening_credit()
                elif (obs.kind == Kind.BODY and root == self._handoff_root
                      and obs.frame.status.protocol == 0
                      and 1 <= len(source_payload(obs.frame.status, obs.frame.data)) <= 31):
                    self._receipt = own_receipt
                    self.adapter.candidate(own_receipt, source_payload(obs.frame.status, obs.frame.data), 'changeover')
                    self._negotiation = 0
                    self._v49_root = None
                    self._clear_opening_credit()
                else:
                    self._clear_opening_credit()
                if (self._receipt is not None and self.adapter.result.receipt == self._receipt
                        and self.adapter.result.state in ('spent', 'committed', 'cancelled')):
                    self._receipt = None
            if text is not None and obs.frame.status.extended:
                self.link_events.append(dict(event='source-extension', at=obs.stream_sample,
                    status=obs.frame.status.byte, prefix=obs.frame.data[:3].hex()))
            if obs.frame.status.protocol == 1:
                self.revoke_peer_watch()
                self._close_receive_epoch('peer close', terminal=True)
            due = obs.stream_sample + round(self.t[obs.kind]*self.fs)
            if (obs.stream_sample in self._decided
                    and obs.stream_sample not in self._dropped):
                return
            if self._held_answer is not None and obs.stream_sample < self._held_answer.onset:
                return
            pending = dict(self.pending.get(obs.stream_sample,
                           dict(word=None, accepted=False, text=False)))
            if obs.stream_sample in self._dropped:
                pending.update(current_slot=obs.stream_sample, current_slot_accepted=True)
            self._unread_data_due = None
            pending.update(accepted=True, text=bool(text), body=obs,
                           responds_to=reg.responds_to, data_reply=data_reply,
                           receive_epoch=self._receive_epoch,
                           source_root=root if obs.kind in (Kind.BODY, Kind.CHIRP) else None,
                           watch_recovery=self._watch_recovery,
                           peer_handover=self._awaiting_peer_turn)
            if self._awaiting_peer_turn and source_payload(obs.frame.status, obs.frame.data):
                self._awaiting_peer_turn = False
            answer = self._accepted_answer(obs, pending)
            if answer is not None and not obs.provisional and not obs.weak:
                pending['answer'] = self._hold_answer(obs, answer, pending)
            self._decided.discard(obs.stream_sample)
            self._dropped.discard(obs.stream_sample)
            if self._next is not None and not self._next.current_slot_accepted:
                self._next = None
            if self._past(now, due):
                self._defer(obs.stream_sample, pending, due)
            else:
                self.pending[obs.stream_sample] = pending
            return
        if obs.word == "Y27":
            if self.state == State.ISS_CLOSE and not close_request:
                return
            self._blocked_redue = True
            if obs.stream_sample not in self._decided:
                pending = self.pending.setdefault(
                    obs.stream_sample, {"accepted": False, "text": False}
                )
                pending["word"] = obs
                self._next = None
            return
        if reg.replaced and reg.replaced.word == "Y27":
            self.pending.pop(obs.stream_sample, None)
            self._blocked_redue = False
        if self._rollback_until is not None:
            return
        if (self.state == State.IRS_YIELD or
                self.state == State.IRS_BREAKIN and obs.frame is not None) and obs.kind in (
            Kind.SHORT,
            Kind.LONG,
            Kind.NORMAL,
        ):
            self.state = State.IRS_ANSWER
        if self.state in (State.IRS_ANSWER, State.IRS_CLOSE):
            if (self.state == State.IRS_ANSWER and self._bare_word(obs)
                    and obs.word == 'V27' and not obs.provisional and not obs.weak
                    and not obs.unexpected
                    and self.close_context is None
                    and reg.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)
                    and self.outbox.queued and self.allow_breakin
                    and self.last is not None and self.last.kind == Kind.WORD
                    and obs.stream_sample > self.last.stream_sample
                    and reg.responds_to in (None, self.last.stream_sample)):
                if self._in_answer_slot(obs.stream_sample, self.last, repeated=True):
                    self._break_in(self.next_raster(now))
                return
            if obs.kind not in (Kind.SHORT, Kind.LONG, Kind.NORMAL):
                return
            due = obs.stream_sample + round(self.t[obs.kind] * self.fs)
            if now > due - self.lead:
                if (self.state == State.IRS_ANSWER and obs.kind == Kind.SHORT
                        and not obs.provisional and not obs.weak and not obs.unexpected
                        and reg.outcome in (Registered.NEW, Registered.ENRICHED,
                                            Registered.CONFIRMED)):
                    word = (self._answer_53(obs, due) if obs.value == 53
                            else self.peer_ack.get(obs.value) or 'X27')
                    if word is not None:
                        self._hold_control(reg, Proposal(Kind.WORD, due, word=word, state=self.state,
                            peer_control=obs.value, responds_to=obs.stream_sample))
                return
            if self.state == State.IRS_CLOSE and obs.value == 37:
                self._next = Proposal(Kind.WORD, due, word="V27", state=State.IRS_CLOSE)
            elif (self.state == State.IRS_ANSWER and obs.kind == Kind.SHORT
                  and obs.value == 53):
                if obs.provisional:
                    return
                word = self._answer_53(obs, due)
                if word is not None:
                    self._next = Proposal(Kind.WORD, due, word=word, state=self.state,
                                          peer_control=53, responds_to=obs.stream_sample)
            elif (self.app_done or self.state == State.IRS_CLOSE) and self._wants_turn():
                # The observed onset can differ by a few samples from the
                # accepted peer clock. Snap from readiness, not that onset's
                # slightly later answer timestamp, to keep a feasible slot.
                # An incomplete long packet still owns its full receive span.
                phase_ready = (self.peer_phase is not None and
                               (obs.kind != Kind.LONG or
                                obs.frame is not None and not obs.provisional))
                self._break_in(self.breakin_due(now) if phase_ready else due)
            elif obs.kind != Kind.NORMAL and not obs.provisional:
                word = "X27" if obs.kind == Kind.LONG else self.peer_ack.get(obs.value)
                if obs.value == 59 and word == "W27":
                    word = PEER_ACK.get(59)
                if (self.peer_ack.get(59) == 'W27' and self.state == State.IRS_ANSWER
                        and obs.kind == Kind.SHORT and obs.value == 59
                        and obs.frame is None and obs.word is None
                        and obs.placed and obs.header in ('A', 'B')
                        and not obs.weak and not obs.unexpected
                        and reg.outcome in (Registered.NEW, Registered.ENRICHED,
                                            Registered.CONFIRMED)
                        and self.last is not None and self.last.kind == Kind.WORD
                        and self.last.state == State.IRS_ANSWER
                        and reg.responds_to == self.last.stream_sample
                        and self.peer_phase is not None
                        and self.peer_phase.generation == self._peer_generation
                        and self.peer_phase.identity == (
                            Kind.SHORT, obs.stream_sample, obs.header, obs.value)):
                    word = 'W27'
                    self.link_events.append(dict(event='experimental-v59-answer',
                                                 at=obs.stream_sample, header=obs.header,
                                                 word=word, reason='experimental v59 answer'))
                elif word is None and obs.value != 53:
                    word = 'X27'
                    self.link_events.append(dict(event='unmeasured-peer-ack',
                                                 at=obs.stream_sample, value=obs.value,
                                                 header=obs.header, word=word))
                if word is not None:
                    self._next = Proposal(Kind.WORD, due, word=word, state=self.state,
                                          peer_control=obs.value, responds_to=obs.stream_sample)
            return
        if obs.kind != Kind.WORD:
            return
        if (self.state == State.ISS_HOLD and self.last is not None
                and self.last.kind == Kind.NORMAL and reg.responds_to == self.last.stream_sample
                and obs.word == self._expected_data_ack(idle=True)
                and reg.responds_to != self._acked_idle):
            if obs.provisional:
                self._wait_for_ack(obs)
            elif not obs.weak:
                self._acked_idle = reg.responds_to
                self._last_data_ack = obs.word
                self._last_data_form = self._packet_form(self.last)
                self._clear_ack_wait()
                self.counter = (self.counter + 1) % 4
                self._first_data = self._word_yield = False
                if self.sends_data:
                    self._data(self.next_raster(now))
                else:
                    self._idle(self.next_raster(now))
            return
        if obs.word == "V27" and self._grant_ack(reg):
            if (self._long_peer_open() and (obs.weak or obs.unexpected
                    or reg.outcome not in (Registered.NEW, Registered.CONFIRMED, Registered.ENRICHED)
                    or not (obs.extent_s is not None and obs.extent_s >= .240
                            or obs.extent_s is None and obs.word_span_supported))):
                return
            if obs.provisional:
                self._wait_for_ack(obs)
                return
            self.link_events.append(dict(event='turn', at=obs.stream_sample,
                                         word=obs.word, responds_to=reg.responds_to))
            due = self._turn_raster(now)
            if self.sends_data:
                self._begin_data_turn(due, word_yield=True)
            else:
                self._advance(due)
            return
        if self.state == State.IRS_BREAKIN:
            if self._breakin_granted(reg):
                self._take_breakin_turn(reg, now)
            elif not obs.provisional and obs.word != "V27":
                self.state = State.IRS_ANSWER
            return
        if self.state == State.ISS_CLOSE:
            return
        if self.state == State.ISS_GRANT:
            if not obs.provisional and self.ladder:
                self._requeue()
                self._short(self.next_raster(now), 0)
            return
        if self.state == State.ISS_LADDER:
            target = self._emissions.get(reg.responds_to)
            if target is None or target.kind != Kind.SHORT or self.index >= len(self.ladder):
                return
            if target is not self.last:
                return
            if self._opening_wait:
                return
            if (self._opening and target.value == 38
                    and (obs.weak or obs.unexpected
                         or reg.outcome not in (Registered.NEW, Registered.CONFIRMED,
                                                Registered.ENRICHED, Registered.EXTENDED))):
                return
            if self._ack(obs, self.index):
                if target.value != self.ladder[self.index]:
                    return
                if not obs.provisional:
                    self._train_active = False
                next_index = self.index + 1
                if next_index < len(self.ladder):
                    if obs.provisional:
                        self._speculative[obs.stream_sample] = (
                            self.state,
                            self.index,
                            self.repeats,
                        )
                    self._short(self.next_raster(now), next_index)
                elif obs.provisional:
                    self._wait_for_ack(obs)
                elif self._opening and target.value == 38:
                    self._opening_wait = True
                    if self._opening_end is None:
                        self._opening_end = min(
                            self._train_root + round(10*1.25*self.fs),
                            self.session_end or self._train_root + round(10*1.25*self.fs))
                    self._next = None
                    self._clear_ack_wait()
                elif self._turn_body is not None:
                    self._after_turn_ladder(self.next_raster(now))
                else:
                    self._train_active = False
                    self._after_ladder(self.next_raster(now))
            elif not obs.provisional and not self._train_active:
                self._repeat(now)
        elif self.state in (State.ISS_DATA, State.ISS_DATA_ACK):
            if self._data_packet is None or obs.stream_sample <= self._data_packet.stream_sample:
                return
            if not qualifying:
                return
            if self._data_ack(obs):
                if obs.provisional:
                    if obs.weak:
                        self._wait_for_ack(obs)
                    elif (self.state == State.ISS_DATA_ACK and self.ladder
                          and self._data_packet.kind != Kind.LONG):
                        self._speculative[obs.stream_sample] = (
                            self.state,
                            0,
                            self.repeats,
                        )
                        self._short(self.next_raster(now), 0)
                    else:
                        self._wait_for_ack(obs)
                    return
                final = self._data_packet.final
                first_body = self._data_packet.kind == Kind.BODY and self._first_data
                self._commit_data(obs.word)
                self._train_active = False
                if final:
                    self._finish_data_turn(obs.stream_sample)
                elif first_body and self.data_kind == Kind.LONG:
                    self._advance(self.next_raster(now))
                elif self.state == State.ISS_DATA_ACK:
                    self._advance(self.next_raster(now))
                else:
                    self._data(self.next_raster(now))
            elif not obs.provisional and not self._train_active:
                self._repeat(now)

    def _answer_53(self, obs, due):
        """The word for this v53 copy, or None once an unchanged hold reaches its bound."""
        copies = max(1, len(self._unsupported_copies))
        if self.v53_hold and copies >= self.v53_hold:
            # WS8EOC 1107 held v53 41 times under U27 until the operator
            # interrupted, which cost the closing ID. Chirps have followed
            # 1, 3, 7 and 17 copies; holds ran 41 and 99.
            leave = self._wants_turn(due)
            self.link_events.append(dict(event='v53-hold-exit', at=obs.stream_sample, copies=copies,
                                         action='break-in' if leave else 'close'))
            if leave:
                self._break_in(due)
            else:
                self._closing_failure = self.app_done = True
                self.reason = f'no progress after {copies} peer control 53 copies'
                self._requeue()
                self.revoke_peer_watch()
                self.state = State.IRS_CLOSE
                self._break_in(due)
            return None
        key = '53/B' if obs.header == 'B' and '53/B' in self.peer_ack else 53
        return self.peer_ack.get(key, 'U27')

    def _wants_turn(self, candidate=None):
        if self.peer_watch is not None:
            if self.next_offer_at is not None:
                if not self.managed_application or self.app_done:
                    return self.app_done
                if candidate is not None and candidate < self.next_offer_at:
                    return False
            if self._deferred is not None or self._accepted_proposal is not None:
                return False
        if self.close_context is not None and self.state == State.IRS_ANSWER:
            return False
        if self.close_context is not None and self.close_context.grants:
            return self.state == State.IRS_CLOSE
        eligible = (self.sends_data and self._bytes_waiting() if self._long_peer_open()
                    else self.outbox.queued and self.allow_breakin)
        return (eligible or self.app_done
                or self.state == State.IRS_CLOSE)

    def _break_in(self, due):
        if self.peer_phase is not None:
            due = self.breakin_due(due - self.lead)
        if self.peer_watch is not None:
            if not self._wants_turn(due):
                self._recover_peer_turn(due)
                return
            if self.next_offer_at is not None or self.peer_watch.attempts >= 4:
                if (self.managed_application and not self.app_done
                        and self.next_offer_at is not None and due >= self.next_offer_at):
                    pass  # Only an admitted actual key starts the renewed group.
                else:
                    self._recover_peer_turn(due)
                    return
            self._next = Proposal(Kind.WORD, due, word='Y27', state=State.IRS_BREAKIN,
                                  peer_watch_generation=self.peer_watch.generation)
        elif self.breakins >= 4:
            self.link_down("break-in unanswered")
        elif self._closing_turn(due):
            # Closing asks with a word in the peer's answer slot, and keeps
            # reading the peer's raster: a body request sat over K4MSU's own
            # shorts at 05:05Z 88.781.
            self._next = Proposal(Kind.WORD, due, word='Y27', state=State.IRS_BREAKIN,
                                  purpose='close-request')
        elif self.peer_phase is not None and self.peer_phase.kind in (Kind.LONG, Kind.CHIRP):
            self._next = Proposal(Kind.WORD, due, word='Y27', state=State.IRS_BREAKIN)
        elif self.breakin == "y27":
            self._next = Proposal(Kind.WORD, due, word="Y27", state=State.IRS_BREAKIN)
        else:
            self._body(due, 0x40, State.IRS_BREAKIN)

    def _closing_turn(self, now):
        """Closing while the peer holds the ISS raster (a close resumed into its
        turn, K4MSU 05:05Z) or keys its long slots (K4SPB 05:11Z): our 0x20 waits
        for its grant. Any other close asks with the 0x40 body gateways grant."""
        return (self._closing_failure and self.state in (State.IRS_CLOSE, State.IRS_BREAKIN)
                and (self._close_resumed_in_peer_turn
                     or self._peer_holds_turn(now) and self.peer_phase.kind in (Kind.LONG, Kind.CHIRP)))

    def _peer_holds_turn(self, now):
        phase = self.peer_phase
        return (phase is not None and phase.generation == self._peer_generation
                and now - phase.onset <= 2 * phase.cycle)

    def _breakin_granted(self, reg):
        obs = reg.observation
        return (obs.kind == Kind.WORD and obs.word == "V27" and not obs.provisional and not obs.weak
                and self._requesting() and reg.responds_to == self.last.stream_sample
                and (not self._long_peer_open() or not obs.unexpected
                     and reg.outcome in (Registered.NEW, Registered.CONFIRMED,
                                         Registered.ENRICHED, Registered.EXTENDED)
                     and (obs.extent_s is not None and obs.extent_s >= .240
                          or obs.extent_s is None and obs.word_span_supported)))

    def _take_breakin_turn(self, reg, now):
        self.link_events.append(dict(event='turn', at=reg.observation.stream_sample,
                                     word=reg.observation.word, responds_to=reg.responds_to))
        due = self._turn_raster(now)
        if self.sends_data:
            self._begin_data_turn(due, word_yield=True)
        else:
            self._advance(due)

    def _requesting(self):
        return (self.state == State.IRS_BREAKIN and self.last is not None
                and self.last.kind == Kind.WORD and self.last.word == "Y27")

    def _ack(self, obs, index):
        value = self.ladder[index]
        if self._sl5_turn():
            return (obs.word in p4words.BANK and obs.word not in ('Z27', 'Y27')
                    if index == len(self.ladder) - 1 else obs.word == ACK[value])
        if self.data_kind == Kind.LONG and not self._opening:
            return (obs.word in self.ladder_admit if index == len(self.ladder) - 1
                    else obs.word == 'X27')
        if self.ladder == LADDER_CHIRP and not self._opening:
            return (obs.word in ('X27', 'V27', 'U27') if value in (49, 53)
                    else obs.word == ACK[value])
        return (
            obs.word in p4words.BANK and obs.word not in ("Z27", "Y27")
            if value == 49
            else obs.word is not None and obs.word == ACK.get(value)
        )

    def train_root_for(self, proposal):
        if proposal.stream_sample in self._train_keyed:
            return self._train_keyed[proposal.stream_sample]
        if proposal.kind == Kind.SHORT and proposal.state == State.ISS_LADDER:
            if self._opening and proposal.value in (7, 38):
                return (self._train_root if proposal.repeat and self.last is not None
                        and self.last.value == proposal.value else proposal.stream_sample)
        if (proposal.kind == Kind.BODY and
                ((proposal.state in (State.ISS_DATA, State.ISS_DATA_ACK)
                  and proposal.status == 0 and self._first_data and self._train_initial_data
                  and not self._word_yield
                  and self.counter == 0 and self._data_packet is None)
                 or (proposal.repeat and self._train_root is not None
                     and self._train_active and proposal.payload == self.outbox.inflight))):
            return self._train_root if proposal.repeat else proposal.stream_sample
        return None

    def _train_expired(self, now, *, decision=True):
        if self._opening_wait:
            if (self._opening_end is not None and now >= self._opening_end
                    or self._opening_wall_end is not None
                    and self._wall_clock() >= self._opening_wall_end
                    or self.session_wall_end is not None
                    and self._wall_clock() >= self.session_wall_end):
                self.link_down('opening wait expired')
                return True
            return False
        if not self._train_active or self._train_root is None:
            return False
        cycle = round(1.25*self.fs)
        self._train_elapsed = max(
            self._train_elapsed, (now-self._train_root)//cycle)
        if (self._train_wall_end is not None and self._wall_clock() >= self._train_wall_end
                or self.session_wall_end is not None
                and self._wall_clock() >= self.session_wall_end):
            if self.state in (State.ISS_DATA, State.ISS_DATA_ACK):
                self._exhaust_data(self.next_raster(now))
            else:
                self.link_down('passive train expired')
            self._train_active = False
            return True
        if now + (self.lead if decision else 0) >= self._train_root + 9*cycle:
            if self.state in (State.ISS_DATA, State.ISS_DATA_ACK):
                self._exhaust_data(self.next_raster(now))
            else:
                self.link_down('passive train expired')
            self._train_active = False
            return True
        if now + self.lead >= self._train_root + 8*cycle and not self._train_fallback:
            due = self._train_root + 8*cycle
            if (now + self.lead <= due and self._next is None
                    and self._waiting_ack is None and not self.pending
                    and not any(end > now for end in self._train_heads.values())):
                self._next = replace(self.last, stream_sample=due, repeat=True,
                                     header='B' if self.last.kind in (Kind.SHORT, Kind.NORMAL)
                                     else self.last.header)
            self._train_fallback = True
        return False

    def _repeats_granted(self, frame):
        """Keying the grant ended the receive epoch and its duplicate ledger.

        A peer that missed the grant repeats the packet it answered, under the
        same counter, whatever its status bits or admitted extension; that
        repeat is control, not source delivered again. Anything the host would
        refuse still goes to the host.
        """
        status = frame.status
        if (self._granted is None or status.source_mode
                or status.extended and not observed_extension(frame)):
            return False
        byte, data = self._granted
        granted = Status.from_byte(byte)
        source = source_payload(status, frame.data)
        return (bool(source) and status.counter == granted.counter
                and source == source_payload(granted, data))

    def _accepted_answer(self, obs, pending):
        status = obs.frame.status
        identity = status.byte, obs.frame.data
        if not (status.protocol in (0, 3) or status.protocol == 2 and (
                obs.kind in (Kind.LONG, Kind.CHIRP)
                or obs.kind == Kind.BODY and (source_payload(status, obs.frame.data)
                    or pending.get('peer_handover')))):
            return None
        word, reason = self._counted_ack(obs)
        return Proposal(Kind.WORD, obs.stream_sample+round(self.t[obs.kind]*self.fs),
            word=word, ack_reason=reason, ack_identity=identity, state=State.IRS_ANSWER,
            receive_epoch=pending['receive_epoch'], source_root=pending['source_root'],
            accepted_target=obs.stream_sample, current_slot=obs.stream_sample)

    def _hold_answer(self, obs, answer, pending):
        held = self._held_answer
        if (held is not None and held.answer.receive_epoch == self._receive_epoch
                and obs.stream_sample <= held.onset):
            return answer
        self._cancel_deferred('new accepted context')
        self._accepted_proposal = None
        if (self._next is not None and self._next.accepted_target is not None
                and self._next.accepted_target < obs.stream_sample):
            self._next = None
        self._held_answer = HeldAnswer(obs.kind, obs.stream_sample, answer,
                                      dict(pending), self._peer_generation)
        return answer

    def _hold_control(self, reg, proposal):
        obs = reg.observation
        root = self._source_root(reg)
        self._control_roots[obs.stream_sample] = root
        proposal = replace(proposal, receive_epoch=self._receive_epoch, source_root=root,
                           accepted_target=obs.stream_sample, current_slot=obs.stream_sample)
        self._hold_answer(obs, proposal, dict(body=obs, accepted=True,
            receive_epoch=self._receive_epoch, source_root=root, responds_to=reg.responds_to))
        return proposal

    @staticmethod
    def _acks_unread(word, kind):
        """Whether a word could acknowledge a packet of this kind (U27 is the 1002 NAK)."""
        return word in ('X27', 'V27') or kind == Kind.LONG and word == 'T27'

    def _overtaken(self, proposal, obs):
        if proposal.purpose == 'repeat-request':
            return (obs.kind in (Kind.LONG, Kind.CHIRP) and obs.stream_sample <= proposal.stream_sample
                    < obs.stream_sample + round(self.t[Kind.LONG]*self.fs))
        return (proposal.ack_identity is None and proposal.peer_control is not None
                and not proposal.current_slot_accepted and obs.frame is None
                and abs(proposal.stream_sample-obs.stream_sample-round(self.t[obs.kind]*self.fs))
                <= self.fs//10
                and self._acks_unread(proposal.word, obs.kind))

    def _held_proposal(self, due, incoming=None, *, slot=None, log=True):
        held = self._held_answer
        if slot is None and incoming is not None:
            slot = incoming.stream_sample
        kind = incoming.kind if incoming is not None else held.kind if held else Kind.BODY
        unread = self._unread_data_due
        form = self._packet_form(incoming) if incoming is not None else self._ack_form
        if incoming is None and unread is not None and due >= unread[0]:
            kind, form = unread[1:3]
        x27 = self._ack_policy(kind) == 'x27'
        toggle = self._ack_policy(kind).startswith('toggle')
        partner = self._ack_partner(kind, form)
        repeat_word = ('X27' if self.first_ack == 'V27' else partner) if toggle else 'U27'
        if not self._chain_starts(form):
            repeat_word = self._ack_chain[1]
        if x27:
            repeat_word = 'V27'
        if unread is not None and (kind, form) == unread[1:3]:
            repeat_word = unread[3]
        self._unread_data_due = due, kind, form, repeat_word
        if (held is None or held.answer.receive_epoch != self._receive_epoch
                or held.generation != self._peer_generation):
            answer = Proposal(Kind.WORD, due, word=repeat_word, state=State.IRS_ANSWER,
                              purpose='repeat-request', current_slot=slot, current_slot_accepted=False)
            if log:
                self._log_answer('repeat-request', slot, answer, kind, repeat=True)
            return answer
        answer = replace(held.answer, stream_sample=due, current_slot=slot,
                         current_slot_accepted=False)
        source = held.context['body']
        # Control replies have no read identity, but cannot own an unread data slot.
        # Under x27 an X27 to an unread slot would advance whatever it held.
        retain = (not x27 and self._read_unacknowledged()
            or answer.peer_control is not None and kind not in DATA_KINDS and (
                toggle and not self._chain_starts(form)
                or not self._acks_unread(answer.word, kind)
                and (toggle or not self._chain_starts(form)))
            or source.frame is not None and (
            held.kind == kind == Kind.LONG and self._ack_policy(kind) == 'counter-even-t27'
            or source.frame.status.protocol == 2 and source.frame.data
            or self.close_context is not None or held.context.get('watch_recovery')))
        if not retain:
            answer = replace(answer, word=repeat_word, ack_identity=None,
                             purpose='repeat-request', peer_control=None)
        if answer.purpose == 'repeat-request':
            answer = replace(answer, ack_reason=None)
        self._answer_kinds[due] = (held.kind, answer.ack_identity,
                                  answer.receive_epoch, self._peer_generation,
                                  self._packet_form(held.context['body']))
        if log and answer.ack_identity is not None:
            self._log_answer('acked-repeat', slot if slot is not None else held.onset,
                             answer, held.kind, repeat=True)
        return answer

    def _read_unacknowledged(self):
        held = self._held_answer
        return (held is not None and held.answer.receive_epoch == self._receive_epoch
                and held.generation == self._peer_generation and not held.acknowledged
                and held.answer.ack_identity is not None)

    def _past(self, now, due):
        # A zero lead decides at the due itself, which is still keyable.
        return now > due - (self.lead > 0)

    def _defer(self, sample, pending, due):
        """Answer a CRC-valid frame decoded after its instant one peer cycle later."""
        self._decided.add(sample)
        self._dropped.discard(sample)
        later = due + round(CYCLE[pending['body'].kind] * self.fs)
        self._deferred = sample, dict(pending, due=later, fallback=True)
        self.link_events.append(dict(event='answer-deferred', at=sample, due=later,
                                     kind=pending['body'].kind))

    def _long_owns_answer(self, due):
        """An undecided long packet owns its receive span until its answer."""
        return any(sample not in self._decided
                   and (incoming := pending.get('body')) is not None
                   and incoming.kind in (Kind.LONG, Kind.CHIRP)
                   and sample <= due < sample + round(self.t[Kind.LONG] * self.fs)
                   for sample, pending in self.pending.items())

    def _unread_opening_breakin(self):
        """A requested opening repeat whose packet form is still unknown."""
        return (self.state == State.ISS_LADDER and self._opening and self.first_body
                and self.normal_ack in ('toggle', 'x27')
                and self._opening_unread_head is not None
                and self.last is not None and self.last.purpose == 'repeat-request')

    def _answer_unresolved(self, due, incoming, now, head=None):
        if (self._next is not None and self._next.current_slot_accepted
                or self._past(now, due) or self._long_owns_answer(due)):
            return
        # A Y27 head can repeat an established BODY answer; it cannot create
        # the first answer or turn a held NORMAL/control into a BODY context.
        if (self.state == State.IRS_ANSWER
                and (incoming is None and head is not None and head.word == 'Y27'
                     and not head.provisional and not head.weak
                     and self._held_answer is not None
                     and self._held_answer.kind == Kind.BODY
                     and not self._long_peer_open()
                     or incoming is not None and incoming.kind in DATA_KINDS
                     and (incoming.kind in (Kind.LONG, Kind.CHIRP) or not self._long_peer_open()))):
            self._blocked_redue = False
            self._next = self._held_proposal(due, incoming,
                slot=head.stream_sample if incoming is None else incoming.stream_sample)
        elif (self.state == State.ISS_LADDER and self._opening and self.first_body
              and head is not None and head.word == 'Y27'
              and not head.provisional and not head.weak and not self._long_peer_open()):
            # WS8EOC 1107 17.242: the gateway's Y27 + body broke into our opening
            # and three unread copies drew silence. The opposite chain-start
            # word asks for a repeat; a 0x40 request's answer is our grant. The
            # role stays until the body is read, which decides it.
            self._blocked_redue = False
            # The 1114 break-in interrupted the first opening short, before
            # the v38 train. A completed opening retains its existing WORD
            # receive path, including when it later repeats the first short.
            first_stage = all(e.value == self.ladder[0] for e in self._emissions.values()
                              if e.kind == Kind.SHORT)
            self._opening_unread_head = (head.stream_sample
                if first_stage and head.extent_s is None else None)
            word = (('X27' if self.first_ack == 'V27' else 'V27') if self.normal_ack == 'toggle'
                    else 'V27' if self.normal_ack == 'x27' else 'U27')
            self._next = Proposal(Kind.WORD, due, word=word, state=None,
                                  purpose='repeat-request', current_slot_accepted=False)
        elif (self._unread_opening_breakin() and incoming is not None
              and incoming.kind == Kind.NORMAL):
            self._blocked_redue = False
            self._next = replace(self._held_proposal(due, incoming), state=None)

    def _cancel_deferred(self, reason):
        if self._deferred is not None:
            self.link_events.append(dict(event='deferred-answer-cancelled',
                                         at=self._deferred[0], reason=reason))
            self._deferred = None

    def _repeat_of_deferred(self, reg):
        obs = reg.observation
        if (self._deferred is None or obs.provisional or obs.weak or obs.unexpected
                or reg.outcome in (Registered.DUPLICATE, Registered.EXPIRED)):
            return False
        sample, pending = self._deferred
        body = pending['body']
        if obs.stream_sample <= sample + self.fs // 10:
            return False
        repeat = (abs(obs.stream_sample - sample - round(CYCLE[body.kind] * self.fs)) <= self.fs // 10
                  and (obs.kind == body.kind or body.kind == Kind.BODY and obs.word == 'Y27'))
        if repeat and obs.frame is None and obs.kind != Kind.SHORT:
            pending.update(current_slot=obs.stream_sample, current_slot_accepted=False)
            return False
        same = (obs.frame is not None and body.frame is not None
                and (obs.frame.status.byte, obs.frame.data) == (body.frame.status.byte, body.frame.data)
                or obs.kind == body.kind == Kind.SHORT and obs.value == body.value)
        if repeat and same:
            self._unread_data_due = None
            pending.update(current_slot=obs.stream_sample, current_slot_accepted=True)
            if obs.frame is not None:
                identity = obs.frame.status.byte, obs.frame.data
                if obs.kind != Kind.LONG:
                    self._sf8_acks.pop(identity, None)
                elif self.sf8_eligible is None or self.sf8_eligible(obs):
                    self._sf8_acks[identity] = obs
            return True
        self._cancel_deferred('different packet' if repeat else f'peer {obs.kind}')
        return False

    def decide(self, now):
        if self._turn_body is not None and self.state != State.ISS_LADDER:
            self._turn_body, self.index = None, 0
        if self._application_limits(now):
            return None
        context = self._accepted_proposal
        if (context is not None and self._next is context[0]
                and self._past(now, self._next.stream_sample)):
            proposal, sample, pending, generation = context
            self._next = self._accepted_proposal = None
            if (not pending.get('fallback') and generation == self._peer_generation
                    and pending.get('receive_epoch') == self._receive_epoch):
                self._defer(sample, pending, sample+round(self.t[pending['body'].kind]*self.fs))
        if self.state == State.DOWN:
            return None
        if self._close_expired(now):
            return None
        ctx = self.close_context
        receive_at = None if ctx is None else ctx.grant_at or ctx.receive_at
        if (ctx is not None and not ctx.resumed
                and self.state == State.IRS_ANSWER and receive_at is not None
                and now >= min(ctx.end, (ctx.progress_at or receive_at) + 5*self.fs)):
            self._resume_close(now)
        # A current-slot decision must supersede an earlier slot's fallback.
        entries = list(self.pending.items())
        if self._deferred is not None:
            entries.insert(0, self._deferred)
        for sample, pending in entries:
            selected_repeat_answer = False
            deferred = pending.get('due')
            if sample in self._decided and deferred is None:
                continue
            incoming = pending.get("body")
            due = deferred or sample + round(
                (self.t[incoming.kind] if incoming else self.t[Kind.BODY]) * self.fs
            )
            if deferred:
                # The driver decides on the repeat's own onset, which may sit a
                # few milliseconds either side of the original raster.
                slack = round(CYCLE[incoming.kind] * self.fs) // 2
                if now < due - self.lead - slack:
                    continue
                self._deferred = None
                if now > due - self.lead + slack:
                    self.unread += 1
                    continue
            elif now < due - self.lead:
                continue
            late = not deferred and now > due - self.lead
            close_candidate = (ctx is not None and self.state == State.ISS_CLOSE
                               and sample in self._close_candidates)
            incoming = pending.get('body')
            empty_request = (incoming is not None and incoming.frame is not None
                             and incoming.frame.status.byte == 0x40
                             and not source_payload(incoming.frame.status, incoming.frame.data))
            if close_candidate and (incoming is None or incoming.frame is None or empty_request):
                word = pending.get('word')
                bare = (word is not None and word.word == 'Y27' and not word.provisional
                        and not word.weak and word.extent_s is not None
                        and 0 < word.extent_s <= .50 and incoming is None)
                if not (bare or empty_request) and now < sample + round(.810*self.fs):
                    return None
                self._decided.add(sample)
                self._blocked_redue = False
                if empty_request and not late and not ctx.grants:
                    # K4MSU 05:05Z 81.611: Y27 + idle 0x40 is the peer taking the
                    # turn; it keys on 1.25 s from there, so its answer is a word.
                    self._next = Proposal(Kind.WORD, due, word='V27',
                                          state=State.IRS_ANSWER, purpose='close-grant',
                                          current_slot=sample)
                elif bare and not late and not ctx.grants:
                    self._next = Proposal(Kind.BODY, due, status=0x40,
                                          state=State.IRS_ANSWER, purpose='close-grant',
                                          current_slot=sample)
                else:
                    self._dropped.add(sample)
                    self.unread += 1
                continue
            if close_candidate:
                self._close_candidates.pop(sample, None)
                if ctx.first_request == sample:
                    ctx.first_request = None
                if ctx.last_request == sample:
                    ctx.last_request = None
            self._decided.add(sample)
            if self._requesting():
                self._blocked_redue = False
            word = pending.get("word")
            bare = (
                word is not None
                and not word.provisional
                and not word.weak
                and word.extent_s is not None
                and word.extent_s <= 0.50
                and "body" not in pending
            )
            accepted = pending["accepted"]
            grant = (accepted and incoming is not None and incoming.kind == Kind.BODY
                     and incoming.frame.status.protocol == 2
                     and not source_payload(incoming.frame.status, incoming.frame.data))
            if self._closing_turn(due) and not grant:
                # K4SPB 05:11Z: the gateway held its fill longs for 26 s while
                # we keyed nothing. Every slot, read or not, asks for the turn.
                if not late:
                    self._break_in(due)
                continue
            if accepted and self._past(now, due):
                if not deferred:
                    self._defer(sample, pending, due)
                continue
            if (late and not accepted) or not (bare or accepted):
                self._dropped.add(sample)
                self.unread += 1
                self._answer_unresolved(due, incoming, now, word)
                # k0003 1534.257: an unread grant draws the request again.
                if (now <= due - self.lead and self._requesting() and word is not None
                        and not word.provisional):
                    self._break_in(due)
                continue
            if self._opening:
                self._opening = False
                self._opening_wait = False
                self.ladder = self._payload_ladder
                self.index = 0
            identity = ((incoming.frame.status.byte, incoming.frame.data)
                        if incoming is not None else None)
            if (ctx is not None and ctx.grants and not ctx.resumed
                    and self.state == State.IRS_ANSWER and accepted
                    and incoming.frame.status.byte == 0x40
                    and not source_payload(incoming.frame.status, incoming.frame.data)):
                self._resume_close(due, exact=True, reply_to=sample)
                continue
            if (ctx is not None and ctx.grants and self.state == State.IRS_ANSWER
                    and bare):
                if ctx.expired_at is not None:
                    self._resume_close(due, exact=True, reply_to=sample)
                continue
            repeat = False
            status = incoming.frame.status if accepted else None
            queued_turn = (
                self.state == State.IRS_ANSWER
                and (self.identification_closing or (self.sends_data and self._bytes_waiting()
                     if self._long_peer_open()
                     else self.outbox.queued and self.allow_breakin))
                and self._wants_turn(due)
                and self.close_context is None
                and not pending.get('data_reply')
                and not pending.get('peer_handover')
            )
            field = incoming.frame.data if status is not None else b''
            link_idle = (field == spec.field_fill(len(field))
                         or field == bytes([spec.IDLE])*len(field))
            idle = (status is not None and status.protocol == 0
                    and not status.extended and not status.source_mode and link_idle)
            invitation = (status is not None and status.protocol == 2
                          and not status.extended and not status.source_mode and link_idle)
            long_invitation = (accepted and incoming.kind in (Kind.LONG, Kind.CHIRP)
                               and status.protocol == 2 and pending.get('answer') is not None)
            if pending.get('peer_handover') and accepted and pending.get('answer') is not None:
                self._next = replace(pending['answer'], stream_sample=due)
            elif long_invitation:
                if (queued_turn and not pending.get('watch_recovery')
                        and (self.next_offer_at is None or due >= self.next_offer_at)):
                    self._break_in(due)
                else:
                    self._next = replace(pending['answer'], stream_sample=due)
            # Not on a repeat of a packet we acked: that gets the ack again.
            # WS8EOC 02:06Z and 04:48Z kept sending under every uninvited request.
            elif queued_turn and (idle or invitation or self.identification_closing):
                self._break_in(due)
            elif (self.state == State.IRS_BREAKIN and idle
                  and self.outbox.queued and self.allow_breakin):
                self._break_in(due)
            elif (accepted and self.state == State.IRS_BREAKIN
                  and status.protocol != 2
                  and self._last_ack is not None and identity == self._last_ack[0]
                  and self._last_ack[1].state == State.IRS_BREAKIN):
                self._break_in(due)
            elif (self.state == State.ISS_HANDBACK and accepted
                    and incoming.kind == Kind.BODY and incoming.frame.status.protocol == 2
                    and not source_payload(incoming.frame.status, incoming.frame.data)
                    and self.last is not None
                    and pending.get('responds_to') == self.last.stream_sample):
                self._next = Proposal(Kind.WORD, due, word='V27', state=State.IRS_ANSWER)
            elif (pending.get('data_reply') and incoming.kind == Kind.BODY
                    and incoming.frame.status.protocol == 2
                    and not source_payload(incoming.frame.status, incoming.frame.data)):
                self._body(due, 0x40, State.ISS_GRANT)
            elif (accepted and self.state == State.IRS_BREAKIN and invitation
                  and incoming.kind == Kind.BODY and self.last is not None
                  and pending.get('responds_to') == self.last.stream_sample
                  and not incoming.provisional and not incoming.weak):
                self._begin_data_turn(due, State.ISS_DATA_ACK)
            elif (identity is not None and self._last_ack is not None
                    and identity == self._last_ack[0]
                    and not (incoming.frame.status.protocol == 2
                             and not self._last_ack[1].payload and self.sends_data
                             and self.state in ISS_TURN and self._bytes_waiting())):
                response = self._last_ack[1]
                repeat = True
                if response.kind == Kind.WORD and pending.get('answer') is not None:
                    response = pending['answer']
                if not deferred:
                    self._log_answer('acked-repeat', sample, response, incoming.kind, repeat=True)
                if response.kind == Kind.WORD:
                    state = (State.IRS_ANSWER if self.state == State.IRS_BREAKIN
                             and incoming.kind == Kind.NORMAL
                             and incoming.frame.status.protocol == 0 else self.state)
                    self._next = replace(response, stream_sample=due, state=state)
                    selected_repeat_answer = accepted
                elif response.payload:
                    if (self._data_packet is not None and self.outbox.inflight == response.payload
                            and self._data_packet.payload == response.payload):
                        self._retry_data(due)
                    else:
                        self._last_ack = None
                        self._body(due, 0x40, State.ISS_GRANT)
                elif self.grants >= 4 and self.ladder:
                    self._requeue()
                    self._short(due, 0)
                else:
                    self._next = replace(response, stream_sample=due)
            elif (accepted and incoming.frame.status.protocol == 2
                    and (incoming.kind == Kind.NORMAL and self.state == State.IRS_BREAKIN
                    and not self._long_peer_open() and self.peer_watch is None or
                    incoming.kind == Kind.BODY and not source_payload(incoming.frame.status, incoming.frame.data) and (
                    self.state == State.IRS_BREAKIN and (not self._long_peer_open()
                    or self.last is not None and pending.get('responds_to') == self.last.stream_sample) or
                    self.state == State.ISS_GRANT and self.last is not None
                    and self.last.state == State.ISS_GRANT
                    and pending.get('responds_to') == self.last.stream_sample))):
                self._begin_data_turn(due, State.ISS_DATA_ACK)
            elif accepted and pending.get('answer') is not None:
                self._next = replace(pending['answer'], stream_sample=due)
            elif accepted and incoming.frame.status.protocol == 1:
                self._next = Proposal(Kind.WORD, due, word='V27', state=State.IRS_CLOSE,
                                      ack_identity=identity, current_slot=sample)
            elif bare or accepted and incoming.frame.status.protocol == 2:
                self._grant(due, incoming)
            answering_current = (self._next is not None
                                 and (self._next.accepted_target in (None, sample)
                                      or selected_repeat_answer))
            if (answering_current and accepted and identity is not None
                    and self._next.kind == Kind.WORD):
                self._next = replace(self._next, receive_epoch=pending.get('receive_epoch'),
                                     source_root=pending.get('source_root'))
            if (answering_current and accepted and self._next.kind == Kind.WORD
                    and (status.protocol in (0, 3) or status.protocol == 2
                         and (incoming.kind in (Kind.LONG, Kind.CHIRP)
                              or incoming.kind == Kind.BODY and source_payload(status, field)))):
                self._answer_kinds[self._next.stream_sample] = (
                    incoming.kind, identity, self._next.receive_epoch, self._peer_generation,
                    self._packet_form(incoming))
            if answering_current and identity is not None and self._next.kind != Kind.SHORT:
                self._next = replace(self._next, ack_identity=identity)
            if (answering_current and accepted and self._next.kind == Kind.WORD
                    and self._next.state == State.IRS_ANSWER):
                slot = pending.get('current_slot', due-round(self.t[incoming.kind]*self.fs))
                if (deferred and self.state == State.IRS_ANSWER
                        and not pending.get('current_slot_accepted')):
                    self._next = self._held_proposal(due, incoming, slot=slot, log=False)
                self._next = replace(self._next, accepted_target=sample, current_slot=slot,
                                     receive_epoch=pending.get('receive_epoch'),
                                     source_root=pending.get('source_root'),
                                     current_slot_accepted=pending.get('current_slot_accepted', not deferred))
                self._accepted_proposal = (self._next, sample, dict(pending),
                                           self._peer_generation)
            elif (self._next is not None and accepted and self._next.kind == Kind.BODY
                    and self._next.state == State.ISS_GRANT and self._next.status == 0x40
                    and not self._next.payload and self._next.stream_sample == due):
                self._accepted_proposal = (self._next, sample, dict(pending),
                                           self._peer_generation)
            if answering_current and identity is not None and self._next.kind != Kind.SHORT:
                self._log_answer('long-answer' if incoming.kind == Kind.LONG else 'normal-answer',
                                 sample, self._next, incoming.kind, repeat=repeat or deferred)
        if (self._closing_turn(now) and self.peer_phase is not None
                and (self._next is None or self._next.state == State.IRS_BREAKIN)):
            # This decision's slot, even before its long decodes, and not one a
            # cycle on that an earlier request was placed at.
            self._break_in(now + self.lead - round(.035 * self.fs))
        if self._train_expired(now):
            return self._next
        if self._opening_wait and not self.pending:
            return None
        handback_due = (self._handback_attempted and self._hold_until is not None
                        and self.next_raster(now) >= self._hold_until)
        if (self.state in (State.ISS_HOLD, State.ISS_GRANT, State.ISS_HANDBACK)
                and self._hold_until is not None and (now >= self._hold_until or handback_due)
                and (self._next is None or self._next.state in (
                    State.ISS_HOLD, State.ISS_GRANT, State.ISS_HANDBACK))):
            close_at = (max(now + self.lead,
                            min(self.next_raster(now), self._hold_until))
                        if handback_due else now + self.lead)
            if self.managed_application and not self.app_done:
                self._handback_exit(self.next_raster(now))
                self._hold_until = None
            else:
                self._handback_exit(close_at)
        if (self.managed_application and not self.app_done
                and self.state == State.ISS_HOLD and not self._bytes_waiting()
                and self.next_offer_at is not None and self.next_raster(now) >= self.next_offer_at
                and self._waiting_ack is None
                and (self._next is None or self._next.state == State.ISS_HOLD)):
            self._body(self.next_raster(now), 0x40, State.ISS_HANDBACK)
        if (self._next is not None or self._deferred is not None
                or any(s not in self._decided for s in self.pending)):
            return self._next
        if self.state == State.ISS_HOLD and self._waiting_ack is not None:
            return None
        if self.state == State.ISS_HOLD and self.sends_data and self._bytes_waiting():
            if self._handback_attempted:
                self._begin_data_turn(self.next_raster(now), word_yield=True)
            else:
                self._data(self.next_raster(now))
        elif self.state == State.ISS_HOLD and self.sends_data:
            self._idle(self.next_raster(now))
        return self._next

    def _ack_policy(self, kind):
        return (self.long_ack if kind == Kind.LONG else
                self.chirp_ack if kind == Kind.CHIRP else self.normal_ack)

    @staticmethod
    def _packet_form(packet):
        if packet.kind == Kind.LONG and (
                (getattr(packet, 'form', None) or '').startswith('normal-long')
                or getattr(packet, 'speedlevel', None) in (
                    'SL5-long', 'SL6-long', 'SL7-long', 'SL8-long', 'SL9-long', 'SL10-long')):
            return 'normal long'
        return 'robust' if packet.kind in (Kind.LONG, Kind.CHIRP) else 'normal short'

    def _chain_starts(self, form):
        return (self._ack_chain is None or self._ack_boundary
                or 'form' in self.chain_reset.split(',') and self._ack_form != form)

    def _ack_partner(self, kind, form):
        policy = self._ack_policy(kind)
        return ('T27' if kind == Kind.LONG and (
            policy == 'toggle-t27' or policy == 'toggle'
            and form == 'normal long') else 'V27')

    def _counted_ack(self, obs):
        status = obs.frame.status
        counter = status.counter
        policy = self._ack_policy(obs.kind)
        if policy == 'x27':
            return 'X27', 'x27 every new packet'
        if not policy.startswith('toggle'):
            if obs.kind == Kind.BODY:
                held = self._held_answer
                if (held is not None and held.kind == Kind.BODY
                        and held.answer.ack_identity == (status.byte, obs.frame.data)):
                    return held.answer.word, held.answer.ack_reason
                if status.protocol == 3:
                    return 'U27', None
                return ('X27', 'opening body') if self.first_body else ('V27', None)
            if obs.kind == Kind.LONG:
                t27 = counter % 2 == 0
                return ('T27' if t27 else 'X27'), f'{policy} counter {counter}'
            if obs.kind == Kind.CHIRP:
                return ('X27' if counter & 1 else 'V27'), None
            partner = 'V27'
            return (partner if counter & 1 else 'X27'), f'counter {counter}'
        if self._ack_boundary:
            return self.first_ack, 'chain start'
        if (self._ack_chain is not None and self._ack_epoch == self._receive_epoch
                and counter == self._ack_chain[0]):
            word = self._ack_chain[1]
            return word, f'repeat keeps {word}'
        if self._chain_starts(self._packet_form(obs)):
            return self.first_ack, 'chain start'
        previous_word = self._ack_chain[1]
        partner = self._ack_partner(obs.kind, self._packet_form(obs))
        word = partner if previous_word == 'X27' else 'X27'
        return word, f'new packet after {previous_word}'

    def _log_answer(self, event, at, answer, kind, *, repeat):
        identity = answer.ack_identity
        if answer.purpose == 'repeat-request':
            event = 'repeat-request'
        self.link_events.append(dict(event=event, at=at, kind=answer.kind,
            ack_policy=self._ack_policy(kind), word=answer.word,
            reason=answer.ack_reason, previous=self._ack_chain[1] if self._ack_chain else None,
            counter=identity[0] & 3 if identity else None,
            status=identity[0] if identity else None, repeat=bool(repeat),
            accepted_target=answer.accepted_target, current_slot=answer.current_slot,
            current_slot_accepted=answer.current_slot_accepted))

    def peer_cycle(self, silent, at):
        if self.state == State.IRS_YIELD and silent >= 2:
            self.state = State.ISS_GRANT
            self._repeat(at)

    def _grant(self, due, incoming):
        if self.close_context is not None and self.close_context.grants:
            return
        if self._long_peer_open():
            due = self.breakin_due(due-self.lead)
            if self._wants_turn(due):
                self._break_in(due)
            elif self.peer_watch is not None:
                self._recover_peer_turn(due)
            return
        # Once we hold the turn, a peer's empty protocol-2 body is its grant
        # whatever it followed; as IRS the same body can be its invitation.
        if self.breakin == 'y27' and self.state == State.IRS_ANSWER and self._wants_turn():
            self._break_in(due)
        elif (incoming is not None and incoming.kind == Kind.BODY and self.sends_data and self.state in ISS_TURN
                and not source_payload(incoming.frame.status, incoming.frame.data) and self._bytes_waiting()):
            self._begin_data_turn(due, State.ISS_DATA_ACK)
        elif self.grants >= 4 and self.ladder:
            self._requeue()
            self._short(due, 0)
        else:
            self._body(due, 0x40, State.ISS_GRANT)

    def sf8_source(self, proposal):
        # A held ACK repeated in a later slot is not bound by the SF8 copy's own deadline.
        return self._sf8_acks.get(proposal.ack_identity) if proposal.current_slot_accepted else None

    def can_key(self, proposal):
        if self._application_limits(proposal.stream_sample):
            return False
        if proposal is self._withdrawn:
            return False
        if self.managed_application and proposal is not self._next:
            return False
        if proposal.peer_watch_generation is not None:
            current = self._next
            if (current is None or replace(proposal, stream_sample=current.stream_sample) != current
                    or self.peer_watch is None
                    or proposal.peer_watch_generation != self.peer_watch.generation):
                return False
            if self.next_offer_at is not None and proposal.stream_sample < self.next_offer_at:
                return False
            if not self._wants_turn(proposal.stream_sample):
                self._recover_peer_turn(self.breakin_due(proposal.stream_sample-self.lead))
                return False
        obs = self.sf8_source(proposal)
        return obs is None or self.sf8_eligible is None or self.sf8_eligible(obs)

    def keyed(self, at=None):
        proposal = self._next
        if proposal is None:
            raise RuntimeError("no proposal to key")
        origin = getattr(self, '_grant_receive_origin', None)
        if not (origin is not None and self.last is origin.actual
                and proposal.repeat and proposal.kind == self.last.kind == Kind.BODY
                and proposal.state == self.last.state == State.ISS_GRANT
                and proposal.status == self.last.status == 0x40
                and not proposal.payload and not self.last.payload
                and proposal.purpose is None and self.close_context is None):
            self._grant_receive_origin = None
        if proposal.receive_epoch is not None and proposal.receive_epoch != self._receive_epoch:
            self._next = None
            raise RuntimeError('stale receive answer')
        # Bind admitted source metadata before the physical key replaces its slot.
        answer = self._answer_kinds.get(proposal.stream_sample)
        packet_kind = (answer[0] if answer is not None
                       and answer[1:4] == (proposal.ack_identity, proposal.receive_epoch,
                                          self._peer_generation) else None)
        self.keyed_ack = {}
        if proposal.kind == Kind.WORD:
            form = answer[4] if packet_kind is not None else self._ack_form
            starts = (proposal.ack_identity is not None and proposal.purpose != 'repeat-request'
                      and packet_kind is not None and self._chain_starts(form) and (
                proposal.ack_reason == 'chain start' or
                not self._ack_policy(packet_kind).startswith('toggle')))
            if starts:
                self._chain_epoch += 1
                if not self._ack_boundary and self._ack_chain is not None:
                    self._chain_reason = 'form'
            source_kind = (packet_kind or
                           (self._held_answer.kind if self._held_answer else None) or
                           (self.peer_phase.kind if self.peer_phase else Kind.BODY))
            self.keyed_ack = dict(
                ack_policy='peer' if proposal.peer_control is not None else self._ack_policy(source_kind),
                chain_epoch=self._chain_epoch, chain_reason=self._chain_reason,
                answers_counter=proposal.ack_identity[0] & 3 if proposal.ack_identity else None)
        proposal = replace(
            proposal, stream_sample=proposal.stream_sample if at is None else at
        )
        train_root = self.train_root_for(proposal)
        if train_root is not None:
            self._train_keyed[proposal.stream_sample] = train_root
        if train_root is not None and train_root == proposal.stream_sample:
            self._train_root = train_root
            self._train_active = False
            self._train_fallback = False
            self._train_fallback_keyed = False
            self._train_elapsed = 0
            self._train_heads.clear()
            self._train_wall_end = self._wall_clock() + 9*1.25
            if proposal.kind == Kind.SHORT and proposal.value == 38:
                self._opening_end = min(
                    train_root + round(10*1.25*self.fs),
                    self.session_end or train_root + round(10*1.25*self.fs))
                self._opening_wall_end = self._wall_clock() + 10*1.25
        elif (train_root is not None and self._train_active
              and proposal.stream_sample >= train_root + round(8*1.25*self.fs)):
            self._train_fallback_keyed = True
        self._next = None
        waiting_emission = (proposal.state == State.ISS_HOLD and not proposal.payload
                            and proposal.kind == Kind.NORMAL)
        unresolved = (self._long_peer_open() and not proposal.payload
                      and proposal.state in (State.ISS_GRANT, State.IRS_BREAKIN)
                      and self.peer_phase is not None
                      and abs(proposal.stream_sample-self.peer_phase.onset-round(self.t[self.peer_phase.kind]*self.fs)
                              - round((proposal.stream_sample-self.peer_phase.onset
                                       -round(self.t[self.peer_phase.kind]*self.fs))/self.peer_phase.cycle)
                              * self.peer_phase.cycle) <= round(.035*self.fs))
        if unresolved and self.peer_watch is not None:
            renewing = self.next_offer_at is not None and proposal.stream_sample >= self.next_offer_at
            self.peer_watch = replace(self.peer_watch, attempts=1 if renewing else self.peer_watch.attempts+1,
                                      last_attempt=proposal.stream_sample)
            if renewing:
                self.next_offer_at = None
        elif (not unresolved and (proposal.payload or proposal.state in ISS_TURN
              or proposal.state in (State.ISS_CLOSE, State.IRS_CLOSE))):
            self.revoke_peer_watch()
        acquisition = ((proposal.state in ISS_TURN and not waiting_emission)
                       or proposal.state == State.IRS_BREAKIN)
        if self.peer_phase is not None and proposal.kind in (Kind.WORD, Kind.BODY):
            if proposal.state in (State.IRS_ANSWER, State.IRS_BREAKIN, State.IRS_CLOSE):
                self._peer_consumed = proposal.stream_sample
        if (proposal.state in ISS_TURN and not waiting_emission and not unresolved
                or proposal.state in (State.ISS_CLOSE, State.IRS_CLOSE)):
            self.peer_phase = None
            self._peer_consumed = None
            self._peer_generation += 1
        if proposal.state in (State.ISS_CLOSE, State.IRS_CLOSE):
            self._close_receive_epoch('keyed close')
            if proposal.state == State.ISS_CLOSE and self.close_context is not None:
                self.close_context.emission = proposal.stream_sample
                self.close_context.generation = self._peer_generation
                self.close_context.first_request = None
                self.close_context.last_request = None
                self._close_candidates.clear()
        elif proposal.purpose == 'close-grant':
            self._open_receive_epoch(proposal.stream_sample)
        elif acquisition and not unresolved:
            self._close_receive_epoch('local acquisition')
            self._receive_boundary = proposal.stream_sample
            if (proposal.kind == Kind.BODY and proposal.state in
                    (State.ISS_GRANT, State.ISS_HANDBACK) and not proposal.payload):
                self._handoff_ticket = proposal.stream_sample
                self._granted = proposal.ack_identity
        if proposal.current_slot_accepted:
            self._cancel_deferred('our key')
        self._accepted_proposal = None
        prior = self.state
        if proposal.state == State.ISS_CLOSE and proposal.kind == Kind.BODY:
            # An odd-counter data ACK is V27 too; a refused word-yield
            # turn or its exhausted ladder establishes the held seed.
            self._close_held_v27 = (self._peer_word == 'V27' and self._word_yield
                                    and (bool(self._data_refusals) or prior == State.ISS_LADDER
                                         or self.close_context is not None
                                         and self.close_context.held_word_yield is not None))
        if proposal.state:
            self.state = proposal.state
        if proposal.purpose == 'close-grant' and self.close_context is not None:
            self.close_context.grants += 1
            self.close_context.grant_at = proposal.stream_sample
            self._close_refusals = 0
        if self.state == State.ISS_HANDBACK:
            self._handback_attempted = True
            self.first_body = True
            if prior != State.ISS_HANDBACK:
                if self._offer_at is None:
                    self.counter = 0
                self._first_data = True
                self._word_yield = True
                self._handback_refusals.clear()
                self.link_events.append(dict(event='handback-offer', at=proposal.stream_sample,
                                             reason='peer IRS seed', prior=prior))
            if not proposal.repeat or self._offer_at is None:
                self._offer_at = proposal.stream_sample
                self._hold_until = proposal.stream_sample + round(4*1.25*self.fs)
                self.next_offer_at = proposal.stream_sample + round(HANDBACK_REOFFER_SECONDS*self.fs)
        elif self.state not in (State.ISS_HOLD, State.ISS_GRANT) and self._turn_body is None:
            self._clear_offer()
        if proposal.payload or self.state == State.IRS_ANSWER:
            self._hold_until = None
        opening_probe = (proposal.kind == Kind.SHORT and self.state == State.ISS_LADDER
                         and self._opening_prefix is None)
        if (not opening_probe and proposal.purpose != 'repeat-request'
                and (self.state != State.IRS_ANSWER or proposal.payload)):
            self._clear_opening_credit()
        elif (self._opening_prefix is not None and proposal.kind == Kind.WORD
              and proposal.peer_control is not None):
            self._opening_negotiated = (proposal.peer_control in (49, 60, 61)
                                        and proposal.responds_to is not None)
        self.index = (
            proposal.index
            if proposal.kind == Kind.SHORT and proposal.value != 37
            else self.index
        )
        answered_short = (self.last is not None and self.last.kind == Kind.SHORT
                          and self.last.stream_sample == self._ladder_answered)
        self.repeats = self.repeats + 1 if proposal.repeat and not answered_short else 0
        self.last = proposal
        self._emissions[proposal.stream_sample] = proposal
        self._emission_epoch[proposal.stream_sample] = (
            self._receive_epoch + 1 if self._handoff_ticket == proposal.stream_sample
            else self._receive_epoch)
        if (proposal.kind == Kind.WORD and proposal.receive_epoch == self._receive_epoch
                and self._receipt is not None):
            if (self._negotiation == 0 and proposal.ack_identity == self._receipt.identity
                    and proposal.source_root == self._receipt.root
                    and proposal.word not in (None, 'Z27', 'Y27')):
                self._negotiation = 1
            elif (proposal.peer_control == 49 and self._negotiation == 1
                    and self._control_roots.get(proposal.responds_to) == proposal.source_root
                    and proposal.source_root is not None
                    and proposal.word not in (None, 'Z27', 'Y27')):
                self._negotiation = 2
                self._v49_root = proposal.source_root
            elif (proposal.peer_control == 53 and self._negotiation == 2
                    and self._control_roots.get(proposal.responds_to) == proposal.source_root
                    and proposal.source_root is not None
                    and proposal.source_root != self._v49_root
                    and proposal.word not in (None, 'Z27', 'Y27')):
                self.adapter.arm(self._receipt)
                self._negotiation = 3
        if proposal.kind == Kind.WORD:
            if proposal.ack_identity is not None and proposal.purpose != 'repeat-request':
                self._unread_data_due = None
                if self._read_unacknowledged():
                    self._held_answer = replace(self._held_answer, acknowledged=True)
            if proposal.peer_control is not None:
                if 'ladder' in self.chain_reset.split(','):
                    self._ack_boundary = True
                    self._chain_reason = 'ladder'
            elif (packet_kind is not None and proposal.ack_identity is not None
                  and proposal.receive_epoch == self._receive_epoch
                  and proposal.word in ('X27', 'V27', 'T27', 'U27')):
                self._ack_chain = proposal.ack_identity[0] & 3, proposal.word
                self._ack_form = answer[4]
                self._ack_epoch = proposal.receive_epoch
                self._ack_boundary = False
        if (proposal.ack_identity is not None
                and (self._last_ack is None or proposal.ack_identity != self._last_ack[0]
                     or proposal.current_slot_accepted)):
            self.acked_counter = proposal.ack_identity[0] & 3
            self._last_ack = proposal.ack_identity, proposal
            if proposal.kind == Kind.WORD:
                self._last_body_ack = proposal.ack_identity, proposal.word
                if source_payload(Status.from_byte(proposal.ack_identity[0]), proposal.ack_identity[1]):
                    self._handback_attempted = False
        self._blocked_redue = False
        self._rollback_until = None
        self._refused_due = None
        if proposal.kind in DATA_KINDS and proposal.payload:
            if (not proposal.repeat or self._data_episode is None
                    or not self._data_episode.matches(proposal)):
                self._retire_data_episode()
                if self._word_yield and proposal.kind != Kind.CHIRP:
                    self._data_episode_serial += 1
                    self._data_episode = DataEpisode(
                        self._data_episode_serial, proposal.stream_sample,
                        proposal.kind, proposal.status, bytes(proposal.payload))
            self.data_repeats = (
                self.data_repeats + 1 if self._data_packet is not None else 0
            )
            if not proposal.repeat or self._data_packet is None:
                self._data_packet = proposal
                self._data_refusals.clear()
            if proposal.kind == Kind.CHIRP:
                self._last_chirp = proposal.status, proposal.payload
            self.breakins = 0
        else:
            # A ladder/grant/receive key is not an identical data retry. Close
            # evidence was already transferred to its bounded CloseContext.
            self._retire_data_episode()
        if proposal.kind == Kind.SHORT and proposal.index == 0:
            self.grants = 0
        if self.state == State.ISS_GRANT:
            self.grants += 1
            self.unread = 0
        if self.state == State.IRS_BREAKIN:
            self.breakins += 1
        if proposal.kind == Kind.WORD and proposal.current_slot_accepted:
            if (self.close_context is not None and prior == State.ISS_CLOSE
                    and proposal.state == State.IRS_ANSWER):
                self.close_context.receive_at = proposal.stream_sample
            self.first_body = False
            if (self._handback_attempted and proposal.ack_identity is not None
                    and proposal.word == 'V27'
                    and proposal.ack_identity[0] & 0x60 == 0x40
                    and not source_payload(Status.from_byte(proposal.ack_identity[0]), proposal.ack_identity[1])):
                self.first_body = True
        return proposal

    def refused(self, reason, now=None):
        if self._next is None:
            return
        if (self._train_active and self._train_fallback
                and self._next.stream_sample == self._train_root + round(8*1.25*self.fs)):
            self._next = None
            return
        self.weak_ack = None
        context, self._accepted_proposal = self._accepted_proposal, None
        if (reason == 'deadline' and context is not None and context[0] is self._next
                and context[3] == self._peer_generation
                and context[2].get('receive_epoch') == self._receive_epoch):
            _, sample, pending, _ = context
            self._next = None
            if not pending.get('fallback'):
                self._defer(sample, pending, sample+round(self.t[pending['body'].kind]*self.fs))
            return
        if self._next.purpose == 'close-grant':
            self._next = None
            return
        if self._next.state == State.IRS_BREAKIN:
            self._blocked_redue = False
            refused_at = self._next.stream_sample
            self._refused_due = refused_at + 1
            self._next = replace(self._next, stream_sample=self.breakin_due(refused_at))
        elif self._next.kind == Kind.WORD:
            self._next = None
        else:
            cycle = (self.peer_watch.cycle if self.peer_watch is not None
                     else round(CYCLE[self._next.kind] * self.fs))
            # A refusal found late steps to the first slot still keyable.
            steps = 1 if now is None else max(1, -(-(now + self.lead - self._next.stream_sample) // cycle))
            at = self._next.stream_sample + steps * cycle
            self._next = replace(
                self._next,
                stream_sample=at,
                chirp_slot=self._chirp_slot(at) if self._next.kind == Kind.CHIRP else 0,
            )
            self._refused_due = self._next.stream_sample

    def _exhaust_data(self, at):
        self._train_initial_data = False
        queued, inflight = bytes(self.outbox.arq._outbuf), self.outbox.inflight
        first_close = self.close_context is None
        self.reason = "retry budget exhausted"
        self._clear_ack_wait()
        self._speculative.clear()
        self._cancel_deferred(self.reason)
        self.pending.clear()
        self._rollback_until = None
        self._blocked_redue = False
        self._requeue()
        self._data_packet = None
        self.data_repeats = 0
        self._body(at, 0x20, State.ISS_CLOSE)
        if first_close:
            self.close_context.queued = queued
            self.close_context.inflight = inflight

    def _retry_data(self, at):
        if self.data_repeats >= 7:
            if self.ladder and (self.data_kind != Kind.LONG or not self._ladder_returned):
                if self.data_kind == Kind.LONG:
                    self._ladder_returned = True
                    self._restart_packet = self._data_packet
                self._requeue()
                self._rollback_until = None
                self._data_packet = None
                self.data_repeats = 0
                self._short(at, 0)
            else:
                self._exhaust_data(at)
        else:
            self._next = replace(self._data_packet, stream_sample=at, repeat=True,
                                 chirp_slot=self._chirp_slot(at),
                                 retry_of=self.last.stream_sample if self.last else None)

    def retry_budget(self):
        """(repeats of the last keying, the budget they run against)."""
        budget = (
            4
            if self.state in (State.ISS_GRANT, State.IRS_BREAKIN, State.ISS_HANDBACK)
            or self.last.value == 49
            else 7
            if self.last.kind == Kind.CHIRP
            else 8
        )
        repeats = (
            self.data_repeats
            if self.state in (State.ISS_DATA, State.ISS_DATA_ACK)
            else self.repeats
        )
        return repeats, budget

    def _repeat(self, now=None):
        if self.peer_watch is not None and self.state in (State.ISS_GRANT, State.IRS_BREAKIN):
            at = self.next_raster(now if now is not None else self.last.stream_sample)
            if self.peer_watch.attempts >= 4 or not self._wants_turn(at):
                self._recover_peer_turn(at)
            else:
                self._break_in(at)
            return
        if not self.last or self.last.kind == Kind.WORD:
            return
        at = (self.last.stream_sample + round(CYCLE[self.last.kind] * self.fs)
              if now is None else self.next_raster(now))
        if (self.state == State.ISS_LADDER and not self._opening and self.data_kind == Kind.CHIRP
                and self.ladder == LADDER_CHIRP and self.index < len(self.ladder) - 1
                and self.last.header == 'B'):
            self._short(at, self.index + 1)
            return
        if (self.state in (State.ISS_DATA, State.ISS_DATA_ACK)
                and self._data_packet is not None and self._data_packet.kind == Kind.LONG):
            self._retry_data(at)
            return
        repeats, budget = self.retry_budget()
        if repeats >= budget - 1:
            if self.state == State.ISS_LADDER and self._turn_body is not None:
                self._handback_exit(at)
            elif (self.state == State.ISS_LADDER
                  and (self._word_yield or self.data_kind == Kind.LONG and not self._opening)):
                self._requeue()
                self._body(at, 0x20, State.ISS_CLOSE)
            elif self.state == State.ISS_HANDBACK:
                self._handback_exit(at)
            elif (self.state in (State.ISS_DATA_ACK, State.ISS_GRANT) and self.ladder
                  and (self.data_kind != Kind.LONG or not self._ladder_returned)):
                if self.data_kind == Kind.LONG:
                    self._ladder_returned = True
                    self._restart_packet = self._data_packet
                self._requeue()
                self._data_packet = None
                self.data_repeats = 0
                self._short(at, 0)
            elif self.state == State.ISS_GRANT and not self.ladder:
                origin = getattr(self, '_grant_receive_origin', None)
                if not (origin is not None and self.last is origin.actual
                        and origin.turn_generation == self._peer_generation
                        and origin.receive_epoch == self._receive_epoch
                        and self.close_context is None and not self.app_done
                        and not self._closing_failure):
                    self._handback_exit(at)
            elif (self.state in (State.ISS_DATA, State.ISS_DATA_ACK)
                  and self._data_packet is not None
                  and self._data_packet.kind in DATA_KINDS):
                self._exhaust_data(at)
            elif self.state != State.ISS_GRANT:
                self.link_down("retry budget exhausted")
            return
        self._next = replace(
            self.last,
            value=self.ladder[self.index] if self._opening else self.last.value,
            index=self.index if self._opening else self.last.index,
            stream_sample=at,
            header=(
                "B"
                if self.last.kind == Kind.SHORT and self.last.value != 37
                or self.last.kind == Kind.NORMAL
                else self.last.header
            ),
            repeat=True,
            chirp_slot=self._chirp_slot(at) if self.last.kind == Kind.CHIRP else 0,
        )

    def silence(self, at, *, answer_due=None):
        if self._application_limits(at):
            return
        if self._close_expired(at):
            return
        if self._train_expired(at) or self._opening_wait or self._train_active:
            return
        ctx = self.close_context
        if ctx is not None and self.state == State.ISS_CLOSE and ctx.first_request is not None:
            lease_end = ctx.emission + round((RESPONSE_S + 2*1.25 + .810)*self.fs)
            if at < min(ctx.end, lease_end):
                return
        if (ctx is not None and self.state == State.IRS_ANSWER and not ctx.resumed
                and (ctx.grant_at is not None or ctx.receive_at is not None)):
            start = ctx.progress_at or ctx.grant_at or ctx.receive_at
            if at < start + 5*self.fs:
                return
            self._resume_close(at)
            return
        if (self.weak_ack is not None and self.last is not None
                and self.last.stream_sample > self.weak_ack and self._waiting_ack is None):
            self.weak_ack = None
        expired = None
        if self.state in (State.IRS_CLOSE, State.ISS_CLOSE, State.DOWN):
            self._clear_ack_wait()
        elif self._waiting_ack is not None:
            if at < self._waiting_until:
                self._deferred_ack = self._waiting_ack
                if self._deferred_slot is None:
                    self._deferred_slot = self.next_raster(at)
                return
            expired = self._waiting_observation
            self.observe(RegisteredObservation(expired, Registered.EXPIRED, expired), now=at)
            self._expired_acks.add(expired.stream_sample)
        if self._blocked_redue:
            if (self.state != State.IRS_ANSWER or self.pending and
                    max(self.pending) + round(self.t[Kind.BODY]*self.fs)
                    >= at + self.lead - self.fs//10):
                return
            self._blocked_redue = False
        if self._deferred or self.pending and any(s not in self._decided for s in self.pending):
            return
        if self._next is not None:
            return
        if self.peer_watch is not None and self.state in (State.ISS_GRANT, State.IRS_BREAKIN):
            self._repeat(at)
            return
        if self.state == State.IRS_CLOSE and self._closing_failure:
            self._break_in(self.next_raster(at))
        elif self.state == State.IRS_CLOSE:
            self.link_down("peer closed")
        elif self.state == State.ISS_CLOSE and self.last:
            if self.last.kind == Kind.BODY:
                self._next = Proposal(
                    Kind.SHORT,
                    self.last.stream_sample + round(1.25 * self.fs),
                    value=37,
                    state=State.ISS_CLOSE,
                )
            elif self._close_refusals:
                if self.repeats >= 7:
                    self.link_down("close refused")
                else:
                    self._repeat(at)
            elif self.repeats < 2:
                self._repeat(at)
            else:
                self.link_down("close silence")
        elif self._breaking_in():
            self._break_in(self.next_raster(at))
        elif self.state == State.IRS_ANSWER or self._unread_opening_breakin():
            due = answer_due if answer_due is not None else self.next_raster(at)
            if self._long_owns_answer(due):
                return
            if (self.peer_watch is not None and self.next_offer_at is not None
                    and due >= self.next_offer_at and self.managed_application
                    and not self.app_done and self.sends_data and self._bytes_waiting()):
                self._break_in(due)
            else:
                self._next = self._held_proposal(due)
                if self._unread_opening_breakin():
                    self._next = replace(self._next, state=None)
        elif self.state not in (State.ISS_HOLD, State.IRS_YIELD, State.DOWN):
            self._repeat(at)
        if expired is not None and self._next is not None:
            while self._next.stream_sample < at + self.lead:
                due = self._next.stream_sample + round(CYCLE[self._next.kind] * self.fs)
                self._next = replace(
                    self._next,
                    stream_sample=due,
                    chirp_slot=self._chirp_slot(due) if self._next.kind == Kind.CHIRP else 0,
                )

    def link_down(self, reason):
        arq = getattr(self.outbox, 'arq', None)
        if arq is not None:
            contact_ended(arq, reason)
        self._grant_receive_origin = None
        self.revoke_peer_watch()
        self._clear_offer()
        self._clear_held_word_yield()
        self._retire_data_episode()
        self._train_active = False
        self._train_initial_data = False
        self._train_heads.clear()
        try:
            self._close_receive_epoch(reason, terminal=True)
        except P4LateFrame:
            # Keep held source and the completed-callback journal for diagnosis;
            # shutdown is not permission to publish bytes after admission.
            self.adapter.failure = self.adapter.failure or "P4 source delivery incomplete at link end"
            self.link_events.append(dict(event="source-release-deferred", reason=reason))
        self._cancel_deferred('link down')
        self._clear_opening_credit()
        self.weak_ack = None
        self._clear_ack_wait()
        self._requeue()
        self.reason = reason
        self.state = State.DOWN
        self.peer_phase = None
        self._peer_consumed = None
        self._peer_generation += 1
        self._next = None
