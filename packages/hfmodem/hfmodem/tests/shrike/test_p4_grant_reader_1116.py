# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Grant-side reader admission, ownership, and authenticated 1116 driver pins."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p4receiver import Emission, Expect, Kind, P4Receiver, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.p4_grant_1116_support import CLEAR, FILL_HASH, GRANT, replay
from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
from hfmodem.tests.shrike.test_p4turn import FS, machine


def armed():
    m, a = machine(role='irs', data='body', ladder='none', breakin='body')
    m._body(3 * FS, 0x40, State.ISS_GRANT)
    m.keyed()
    r = P4Receiver(FS, admitted=True)
    r.enable_hold()
    r.peer_onset = 2 * FS
    r.note_emission(Emission(Kind.BODY, Role.ISS, 3 * FS, .798, packet_reply=True))
    onair._p4_peer_normal_reader(m, r)
    return m, a, r



@pytest.mark.parametrize('state', [State.ISS_GRANT, State.IRS_ANSWER])
def test_grant_timing_stays_with_actual_key_across_receive_state_change(state):
    m, _, r = armed()
    m.state = state
    assert onair._p4_grant_read_owned(m, r)


@pytest.mark.parametrize('bad', ['unkeyed', 'breakin-key', 'wrong-kind', 'old-emission',
                                'stale-generation', 'no-raster', 'no-grant', 'wrong-role'])
def test_unrelated_grant_history_cannot_change_read_schedule(bad):
    m, _, r = armed()
    if bad == 'unkeyed':
        m.last = None
    elif bad == 'breakin-key':
        m.last = replace(m.last, state=State.IRS_BREAKIN)
    elif bad == 'wrong-kind':
        m.last = replace(m.last, kind=Kind.WORD)
    elif bad == 'old-emission':
        r.last_emission = replace(r.last_emission, stream_sample=FS)
    elif bad == 'stale-generation':
        r._train_generation += 1
    elif bad == 'no-raster':
        r.peer_onset = None
    elif bad == 'no-grant':
        r.peer_normal_grant = None
    else:
        r.last_emission = replace(r.last_emission, role=Role.IRS)
    assert not onair._p4_grant_read_owned(m, r)


@pytest.mark.parametrize('queued,allowed', [(False, False), (True, False), (True, True)])
def test_side_reader_is_independent_of_host_queue_and_breakin(queued, allowed):
    m, a, r = armed()
    if queued:
        a.on_host_data(b'queued proposal')
    m.allow_breakin = allowed
    onair._p4_peer_normal_reader(m, r)
    assert r.peer_normal_pending
    assert r.peer_normal_grant == (m.last.stream_sample, r._train_generation)


@pytest.mark.parametrize('bad', ['unkeyed', 'missing-raster', 'old-key', 'wrong-state', 'non-grant'])
def test_side_reader_requires_current_actually_keyed_grant(bad):
    m, _, r = armed()
    if bad == 'unkeyed':
        m.last = None
    elif bad == 'missing-raster':
        r.peer_onset = None
    elif bad == 'old-key':
        r.last_emission = replace(r.last_emission, stream_sample=FS)
    elif bad == 'wrong-state':
        m.state = State.ISS_DATA
    else:
        r.last_emission = replace(r.last_emission, packet_reply=False)
    onair._p4_peer_normal_reader(m, r)
    assert not r.peer_normal_pending
    assert r.peer_normal_grant is None


@pytest.mark.parametrize('case', ['current', 'stale-before', 'stale-during', 'gap', 'tx-overlap', 'unread', 'deadline'])
def test_side_reader_generation_gaps_and_word_alternative(monkeypatch, case):
    m, _, r = armed()
    at = r.peer_onset + round(2 * 1.25 * FS)
    origin, end = at - FS // 20, at + round(.810 * FS)
    calls = []
    obs = normal(at, 0, b'\x1e' * 40)

    def read(*args, **kwargs):
        calls.append(kwargs['predicted'])
        if case == 'stale-during':
            r._train_generation += 1
        return replace(obs, frame=None) if case == 'unread' else obs

    monkeypatch.setattr(r, '_normal', read)
    if case == 'stale-before':
        r._train_generation += 1
    elif case == 'gap':
        r.note_gap(at, at + 1)
    elif case == 'tx-overlap':
        r.emissions.append(Emission(Kind.BODY, Role.ISS, at - 1, .1))
    elif case == 'deadline':
        r.normal_deadline = lambda at: 0.
    clock = replace(r.clock)
    result = []
    r._read_peer_normal(np.zeros(end - origin), origin, result)
    assert len(result) == (1 if case == 'current' else 0)
    assert bool(calls) == (case in ('current', 'stale-during', 'unread'))
    assert r.clock == clock and r.clock.expects == Expect.WORD
    assert not m.outbox.inflight


@pytest.mark.parametrize('offset', [0, .300])
def test_off_raster_tone_cannot_become_checked_packet(offset):
    _, _, r = armed()
    at = r.peer_onset + round(2 * 1.25 * FS)
    origin, end = at - FS // 20, at + round(.810 * FS)
    audio = np.zeros(end - origin)
    start = round((.05 + offset) * FS)
    audio[start:] = .5 * np.cos(2 * np.pi * 1500 * np.arange(len(audio) - start) / FS)
    result = []
    r._read_peer_normal(audio, origin, result)
    assert not result
    assert r.clock.expects == Expect.WORD


@pytest.fixture(scope='module')
def empty():
    return replay(clock="declared")


@pytest.fixture(scope='module')
def queued():
    return replay(queued=True, clock="declared")


def tails(result):
    return [h for h in result['heard'] if h['kind'] == Kind.NORMAL
            and h.get('status') == 0 and h['onset_sample'] >= round(67.2 * FS)]


def legal(result):
    assert result['keys'] and all(k['margin'] >= 0 for k in result['keys'])
    assert not any(e.get('payload') for e in result['emissions'])
    assert not any(abs(h['onset_sample'] - round(67.260333 * FS)) < 50
                   for h in tails(result))
    for h in tails(result):
        assert (h['field_bytes'], h['field_sha256']) == (40, FILL_HASH)
        assert all(not (left <= h['onset_sample'] < right) for left, right in result['gaps'])
        assert any(silent == 0 and h['onset_sample'] in live
                   for _, silent, live in result['cycles'])


def first_answer(result):
    at = CLEAR[0] + round(.920 * FS)
    assert at == 3332660
    assert any(e['audio_start'] == at for e in result['emissions'])
    assert any(r['wall_end'] > r['wall_start'] and r['capture_end'] >= CLEAR[0] + round(.810 * FS)
               and r['wall_end'] <= at - result['lead'] for r in result['reads'])


def test_1116_nine_clear_tails_register_once_and_first_answer_is_on_time(empty):
    legal(empty)
    first_answer(empty)
    onsets = [h['onset_sample'] for h in tails(empty)]
    assert all(onsets.count(at) == 1 for at in CLEAR)
    assert empty['grants'] == 4 and empty['breakins'] == 0
    assert empty['buffered'] == 0
    # The accepted NORMAL changes current state before its bounded pending read;
    # the actual grant key continues to own that reader schedule.
    assert any(c['last_state'] == State.ISS_GRANT
               and c['last_key'] == c['emission'] == GRANT + 3 * round(1.25 * FS)
               and c['grant'] == (c['last_key'], c['generation'])
               and c['requested'] <= CLEAR[0] + round(.810 * FS) + 5
               for c in empty['captures'])


def test_1116_actual_clock_first_answer_is_on_time():
    result = replay()
    assert result['clock'] == 'actual'
    legal(result)
    first_answer(result)  # Strict actual positive; refusal cannot satisfy it.
    assert all(sum(h['onset_sample'] == at for h in tails(result)) == 1 for at in CLEAR)


def test_1116_queued_proposal_reads_then_requests_without_payload_permission(queued):
    legal(queued)
    first_answer(queued)
    assert [h['onset_sample'] for h in tails(queued)] == list(CLEAR[:2])
    requests = [e for e in queued['events'] if e['event'] == 'Keyed'
                and e.get('state') == State.IRS_BREAKIN]
    assert len(requests) == queued['breakins'] == 4
    assert requests[0]['at'] == CLEAR[1] + round(.920 * FS)
    assert queued['buffered'] == 35 and queued['queued'] and not queued['inflight']
    assert queued['reason'] == 'break-in unanswered'
    # Later modeled requests cover historical clear heads. They remain unavailable.
    assert any(e['audio_start'] <= CLEAR[2] < e['audio_end'] for e in queued['emissions'])


def test_1116_disabled_breakin_keeps_queue_and_receives_clear_tails():
    result = replay(queued=True, breakin=False, clock="declared")
    legal(result)
    first_answer(result)
    assert all(sum(h['onset_sample'] == at for h in tails(result)) == 1 for at in CLEAR)
    assert result['breakins'] == 0 and result['buffered'] == 35 and not result['inflight']


def test_1116_irs_only_reader_mutation_loses_clear_tails():
    result = replay(legacy=True)
    assert not tails(result)
    assert result['grants'] == 4 and result['reason'] == 'peer silent'
    assert not any(e['audio_start'] == CLEAR[0] + round(.920 * FS) for e in result['emissions'])


def test_1116_no_readable_return_retains_four_grant_bound():
    result = replay(silent=True)
    assert not tails(result)
    assert result['grants'] == 4 and result['breakins'] == 0
    assert result['reason'] == 'peer silent'


def test_1116_work_overrun_cannot_key_first_answer_late():
    result = replay(stall_ms=100)
    assert any(h['onset_sample'] == CLEAR[0] for h in tails(result))
    assert not any(e['audio_start'] == CLEAR[0] + round(.920 * FS) for e in result['emissions'])
    assert all(k['margin'] >= 0 for k in result['keys'])
    first = CLEAR[0] + round(.920 * FS)
    assert any(r['wall_end'] > first - result['lead'] and
               any(o['onset'] == CLEAR[0] and o['crc'] for o in r['registered'])
               for r in result['reads'])
    assert any(c['last_state'] == State.ISS_GRANT for c in result['captures'])


def test_grant_revoked_after_partial_cannot_lend_its_pending_read_time():
    result = replay(clock='declared', revoke_after_partial=True)
    assert result['revoked']
    assert any(h['onset_sample'] == CLEAR[0] for h in tails(result))
    assert any(c['last_state'] == State.ISS_GRANT and c['grant'] is not None
               and c['grant'][1] != c['generation'] and
               c['requested'] > CLEAR[0] + round(.920 * FS)
               for c in result['captures'])
    assert not any(e['audio_start'] == CLEAR[0] + round(.920 * FS)
                   for e in result['emissions'])
    assert all(k['margin'] >= 0 for k in result['keys'])
    assert result['buffered'] == 0 and result['breakins'] == 0
