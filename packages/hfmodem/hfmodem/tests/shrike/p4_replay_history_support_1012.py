# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math

from hfmodem.shrike import p4frame, p4turn, p4robusttx
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation


class HistoryConflict(AssertionError):
    """Preserve the one candidate and return a precise admission conflict."""


def primitive(value):
    """Detached strict JSON values; object relationships use traversal IDs."""
    objects = {}

    def freeze(v):
        if isinstance(v, Enum):
            return freeze(v.value)
        if v is None or type(v) in (str, bool, int):
            return v
        if type(v) is float and math.isfinite(v):
            return v
        if isinstance(v, bytes):
            return v.hex()
        if is_dataclass(v) and not isinstance(v, type):
            if id(v) in objects:
                return {'audit_ref': objects[id(v)]}
            label = f'object-{len(objects)+1}'
            objects[id(v)] = label
            return dict(audit_id=label, **{f.name: freeze(getattr(v, f.name)) for f in fields(v)})
        if type(v) in (list, tuple):
            return [freeze(x) for x in v]
        if type(v) in (set, frozenset):
            return sorted((freeze(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
        if type(v) is dict:
            if all(type(k) is str for k in v):
                return {k: freeze(x) for k, x in v.items()}
            return [dict(key=freeze(k), value=freeze(x)) for k, x in sorted(
                v.items(), key=lambda item: json.dumps(primitive(item[0]), sort_keys=True))]
        raise TypeError(f'nonprimitive audit value: {type(v).__name__}')

    return freeze(value)


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def completed_history(heard, start_sample, fs):
    """Archived completed CRC state; never a fresh receive/registration claim."""
    physical, excluded = {}, []
    for index, h in enumerate(heard):
        reasons = []
        if h.get('kind') != 'chirp':
            reasons.append('not-chirp')
        if h.get('status') is None or h.get('data') is None:
            reasons.append('no-completed-status-data')
        for flag in ('provisional', 'weak', 'unexpected'):
            if h.get(flag):
                reasons.append(flag)
        if h.get('outcome') not in ('new', 'confirmed', 'enriched'):
            reasons.append('registration-not-accepted')
        at, receipt, support = (h.get(k) for k in
                               ('onset_sample', 'received_at_sample', 'capture_end_sample'))
        if any(type(v) is not int for v in (at, receipt, support)):
            reasons.append('missing-complete-coordinates')
        else:
            wave_end = at+round(3.3075*fs)
            if receipt < support or support < wave_end:
                reasons.append('receipt-support-does-not-prove-completion')
            if receipt >= start_sample:
                reasons.append('receipt-not-before-start')
            if support >= start_sample:
                reasons.append('capture-end-not-before-start')
            if wave_end >= start_sample:
                reasons.append('wave-end-not-before-start')
            if at+round(3.4*fs) >= start_sample:
                reasons.append('onset-plus-3.4-not-before-start')
        if reasons:
            excluded.append(dict(row_index=index, reasons=reasons, row=h))
            continue
        root = ('chirp', at)
        explicit = h.get('source_root')
        replaced = h.get('replaced')
        if explicit is not None:
            if (type(explicit) not in (list, tuple) or len(explicit) != 2
                    or explicit[0] != 'chirp' or type(explicit[1]) is not int):
                raise HistoryConflict(f'row {index}: unresolved explicit physical root')
            root = tuple(explicit)
        if replaced is not None:
            if (type(replaced) is not dict or replaced.get('kind') != 'chirp'
                    or type(replaced.get('onset_sample')) is not int):
                raise HistoryConflict(f'row {index}: unresolved replacement identity')
            prior = ('chirp', replaced['onset_sample'])
            if explicit is not None and root != prior:
                raise HistoryConflict(f'row {index}: conflicting replacement root')
            root = prior
        identity = h['status'], h['data']
        old = physical.get(root)
        if old is not None:
            if identity != (old['row']['status'], old['row']['data']):
                raise HistoryConflict(f'rows {old["row_index"]}/{index}: conflicting identities for root {root}')
            # No onset/CFO transplantation into the preserved observation path.
            if (at, h['cfo_hz']) != (old['row']['onset_sample'], old['row']['cfo_hz']):
                raise HistoryConflict(f'row {index}: unresolved onset/CFO refinement for root {root}')
            excluded.append(dict(row_index=index, reasons=['same-physical-root-refinement'], row=h))
            continue
        if root != ('chirp', at):
            raise HistoryConflict(f'row {index}: replacement root cannot be retained by original lifecycle')
        physical[root] = dict(row_index=index, root=root, row=h)
    return sorted(physical.values(), key=lambda x: x['root'][1]), excluded


TURN_FIELDS = ('state', '_receive_epoch', '_receive_active', '_receive_boundary',
    '_root_epoch', '_accepted_roots', '_last_source_onset', '_emission_epoch',
    '_receipt', '_handoff_ticket', 'counter', 'acked_counter', '_last_ack',
    '_last_body_ack', '_held_answer', '_accepted_proposal', 'peer_phase',
    'peer_watch', '_peer_generation', '_peer_consumed', 'pending', '_deferred',
    '_next', 'last', '_emissions', '_decided', 'unread', 'app_done', 'reason')
HOST_FIELDS = ('last', 'result', 'last_credit', 'failure', '_receipts', '_active',
    '_changeover_receipt', '_delivery_journal', '_transaction', '_effect_cursor',
    '_source_kind', '_source_counter', '_source_width', '_covered', '_resegmented')
RX_FIELDS = ('clock', 'peer_watch', 'chirp', 'expected', 'cfo_hz', 'admitted',
    'observations', 'registrations', 'emissions', 'last_emission', 'consumed',
    'pending_body', '_answered', '_responds_to', '_response_evidence',
    '_answered_generations', '_watch_read_generation', '_peer_watch_evidence',
    'live_onsets', 'peer_onset', 'live_events', 'silent_cycles', '_counted_cycle',
    '_counted_live_onset', '_missed_due', '_body_heads', '_body_attempted',
    '_chirp_headers', '_chirp_prior_cycles', 'turnarounds', 'capture_gaps',
    'acked_counter', 'close_lease', 'train_lease', 'grant_emission_context',
    '_train_generation', '_close_grant_word', '_body_carrier_known', '_word_cfo_hz',
    '_scan_generation_number', 'normal_waiting', 'peer_normal_pending', 'peer_normal_grant')


def actual_turn(machine):
    return dict(machine={k: getattr(machine, k) for k in TURN_FIELDS},
        observation_roots=list(machine._observation_roots.values()),
        adapter={k: getattr(machine.adapter, k) for k in HOST_FIELDS},
        outbox=dict(queued=machine.outbox.queued, inflight=machine.outbox.inflight,
                    queued_bytes=bytes(machine.outbox.arq._outbuf)))


def seed_history(machine, heard, start_sample, fs, audit):
    # Preserve the existing IRS fixture and synthetic onset+3.31/+3.4-lead convention.
    machine.state = p4turn.State.IRS_ANSWER
    selected, excluded = completed_history(heard, start_sample, fs)
    audit.update(selected=primitive(selected), excluded=primitive(excluded), actions=[])
    for selected_row in selected:
        h = selected_row['row']
        field = p4frame.decode_field(p4frame.encode_field(
            bytes.fromhex(h['data']), p4frame.Status.from_byte(h['status'])), data_bytes=22)
        at = h['onset_sample']
        obs = Observation(Kind.CHIRP, fs, at, 0, at, 3.3075, h['cfo_hz'], 0., frame=field)
        before_epoch, previous_identity = machine._receive_epoch, machine.adapter.last
        machine.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(3.31*fs))
        identity, root = (field.status.byte, field.data), (Kind.CHIRP, at)
        action = primitive(dict(row_index=selected_row['row_index'], root=root,
            recorded_receipt=h['received_at_sample'], recorded_capture_end=h['capture_end_sample'],
            waveform_end=at+round(3.3075*fs), synthetic_observation=at+round(3.31*fs),
            synthetic_decision=at+round(3.4*fs)-machine.lead,
            archived_outcome=h['outcome'], fixture_registration='new',
            legacy_response_association=None, after_observe=actual_turn(machine)))
        audit['actions'].append(action)
        if (root not in machine._accepted_roots or machine._root_epoch.get(root) != before_epoch
                or machine._receive_epoch != before_epoch or machine.state != p4turn.State.IRS_ANSWER
                or machine.adapter.last != identity or machine.adapter.failure is not None
                or machine.adapter.result.duplicate != (previous_identity == identity)
                or machine.peer_phase is None or machine.peer_phase.identity != (Kind.CHIRP, at, *identity)
                or machine.peer_watch is None or machine.peer_watch.onset != at):
            raise HistoryConflict(f'row {selected_row["row_index"]}: actual source/phase admission conflict')
        proposal = machine.decide(at+round(3.4*fs)-machine.lead)
        action['proposal'] = primitive(proposal)
        if (proposal is None or proposal.kind != Kind.WORD
                or proposal.receive_epoch != machine._receive_epoch or proposal.source_root != root
                or proposal.ack_identity != identity or proposal.accepted_target != at
                or proposal.current_slot != at or not proposal.current_slot_accepted
                or machine._held_answer is None or machine._held_answer.onset != at):
            raise HistoryConflict(f'row {selected_row["row_index"]}: actual held/proposal association conflict')
        word_samples = len(p4robusttx.robust_word(proposal.word, fs))
        action.update(synthetic_key=proposal.stream_sample, word_samples=word_samples,
                      synthetic_word_end=proposal.stream_sample+word_samples)
        if proposal.stream_sample+word_samples >= start_sample:
            raise HistoryConflict(f'row {selected_row["row_index"]}: full proposed answer does not finish before start')
        sent = machine.keyed()
        action['after_keyed'] = primitive(actual_turn(machine))
        if (sent != proposal or machine._last_ack != (identity, sent)
                or machine.acked_counter != field.status.counter
                or machine._held_answer.answer.source_root != root):
            raise HistoryConflict(f'row {selected_row["row_index"]}: actual consumed ACK/held conflict')
    machine.link_events.clear()


def boundary_snapshot(machine, rx, delivered, history, previous, head_index, start_sample, inputs):
    record = primitive(dict(stage='after-seed-and-existing-clock-before-read-and-delivery-clear',
        start_sample=start_sample, pinned_inputs=inputs,
        assumed_existing_receiver_start=dict(label='existing assumed heads cadence; no acquired-marker claim',
            heads_row_index=head_index, heads_row=previous,
            formula='advance onset by 3.75 until >= start; Clock(PACKET, at, .330, 3.75, CFO)',
            actual={k: getattr(rx, k) for k in RX_FIELDS}),
        derived_completed_turn_history=dict(
            convention='archived completed CRC state; synthetic onset+3.31 and onset+3.4-lead; not RF feasibility',
            selection=history, delivery_journal=list(delivered), actual=actual_turn(machine))))
    return record, serialized(record)
