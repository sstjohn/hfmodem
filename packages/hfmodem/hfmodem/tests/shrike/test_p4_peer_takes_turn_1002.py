# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Peer-raster turn after an empty grant."""

import pytest
import numpy as np

from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Emission, P4Receiver, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.evidence import CAPTURES
from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
from hfmodem.tests.shrike.test_p4_data_turn_0930 import CYCLE, T, receive
from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
from hfmodem.tests.shrike.test_p4turn import FS, machine, word


def test_1018_late_burst_does_not_place_peer_raster_packet():
    # The recorded burst rises about 40 ms after our carrier ends.
    path = CAPTURES/'onair-1002-1018'/'stream.flac'
    if not path.is_file():
        pytest.skip('local N9VC recording absent')
    audio = onair.session.load_wav(str(path), FS)
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576
    for n in range(4):
        grant = 1952736+n*CYCLE
        r.note_emission(Emission(Kind.BODY, Role.ISS, grant, .798, packet_reply=True))
        at = 1968576+n*CYCLE
        origin = grant-round(.1*FS)
        early = grant+round((.935+.035+.180)*FS)
        first = r.read_partial(audio[origin:early], origin, deadline=early)
        assert not any(x.observation.kind == Kind.WORD for x in first)
        end = at+round(.835*FS)
        records = r.read_partial(audio[origin:end], origin, deadline=end)
        assert not any(x.observation.kind == Kind.WORD for x in records)
        assert not any(x.observation.peer_raster_tail for x in records)
        assert r.clock.expects.value == 'word'


def test_1018_late_burst_does_not_enrich_prior_unread_normal():
    # A late burst cannot turn an unread packet into a peer-raster tail.
    path = CAPTURES/'onair-1002-1018'/'stream.flac'
    if not path.is_file():
        pytest.skip('local N9VC recording absent')
    audio = onair.session.load_wav(str(path), FS)
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576
    grant = 1952736
    at = 1968576
    r.note_emission(Emission(Kind.BODY, Role.ISS, grant, .798, packet_reply=True))
    unread = Observation(Kind.NORMAL, FS, 0, at, at, .805, 0., .61, placed=True)
    r.register(unread)
    origin = grant-round(.1*FS)
    end = at+round(.835*FS)
    records = r.read_partial(audio[origin:end], origin, deadline=end)
    assert not any(x.outcome == Registered.ENRICHED and x.observation.peer_raster_tail
                   for x in records)


@pytest.mark.parametrize('late_ms,expected', [(0, True), (40, False)])
def test_peer_raster_tail_must_be_present_at_carrier_end(late_ms, expected):
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576
    grant = Emission(Kind.BODY, Role.ISS, 1952736, .798, packet_reply=True)
    r.note_emission(grant)
    carrier_end = grant.stream_sample + round(grant.duration_s*FS)
    end = r.peer_onset + CYCLE + round(.805*FS)
    start = round(late_ms*FS/1000)
    audio = np.zeros(end+round(.030*FS)-carrier_end)
    t = np.arange(end-carrier_end-start)
    audio[start:start+len(t)] = .5*np.cos(2*np.pi*1500*t/FS)
    assert r._word(audio, carrier_end, provisional=True) is None
    result = []
    r._grant_take(audio, carrier_end, result)
    assert any(x.observation.peer_raster_tail for x in result) is expected


@pytest.mark.parametrize('end_offset_ms,observed', [(0, True), (20, True),
                                                     (25, True), (29, False)])
def test_continuous_tail_within_peer_raster_end_tolerance(end_offset_ms, observed):
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576
    grant = Emission(Kind.BODY, Role.ISS, 1952736, .798, packet_reply=True)
    r.note_emission(grant)
    carrier_end = grant.stream_sample + round(grant.duration_s*FS)
    end = r.peer_onset + CYCLE + round(.805*FS)
    audio = np.zeros(end+round(.030*FS)-carrier_end)
    n = end+round(end_offset_ms*FS/1000)-carrier_end
    audio[:n] = .5*np.cos(2*np.pi*1500*np.arange(n)/FS)
    assert r._word(audio, carrier_end, provisional=True) is None
    early = []
    r._grant_take(audio[:-1], carrier_end, early)
    assert not early
    result = []
    r._grant_take(audio, carrier_end, result)
    # At +29 ms, the +30 ms capture contains no complete quiet frame.
    assert any(x.observation.peer_raster_tail for x in result) is observed


@pytest.mark.parametrize('end_offset_ms', [26, 27, 28, 29, 30, 31, 40])
def test_live_cutoff_requires_observed_burst_end(end_offset_ms):
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576+144
    grant = Emission(Kind.BODY, Role.ISS, 1952736, .798, packet_reply=True)
    r.note_emission(grant)
    carrier_end = grant.stream_sample+round(grant.duration_s*FS)
    at = r.peer_onset+CYCLE
    end = at+round(.805*FS)
    deadline = at+round(.840*FS)
    t = np.arange(deadline-carrier_end)
    audio = .01*np.cos(2*np.pi*700*t/FS)
    n = min(len(audio), end+round(end_offset_ms*FS/1000)-carrier_end)
    audio[:n] += .5*np.cos(2*np.pi*1500*np.arange(n)/FS)
    records = r.read_partial(audio, carrier_end, deadline=deadline)
    observed = end_offset_ms <= 30
    assert any(x.observation.peer_raster_tail for x in records) is observed
    assert (r.clock.expects.value == 'packet') is observed


def test_quiet_continuous_tail_uses_measured_floor():
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576
    grant = Emission(Kind.BODY, Role.ISS, 1952736, .798, packet_reply=True)
    r.note_emission(grant)
    carrier_end = grant.stream_sample+round(grant.duration_s*FS)
    end = r.peer_onset+CYCLE+round(.805*FS)
    t = np.arange(end+round(.100*FS)-carrier_end)
    audio = .0002*np.cos(2*np.pi*700*t/FS)
    n = end-carrier_end
    audio[:n] += .05*np.cos(2*np.pi*1500*np.arange(n)/FS)
    assert r._word(audio, carrier_end, provisional=True) is None
    result = []
    r._grant_take(audio, carrier_end, result)
    assert any(x.observation.peer_raster_tail for x in result)


@pytest.mark.parametrize('end_offset_ms,expected', [(29, True), (30, True), (31, False)])
@pytest.mark.parametrize('jitter,quiet_amplitude', [(0, 0.), (48, .02), (144, .01)])
def test_peer_raster_tail_latest_allowed_end(end_offset_ms, expected, jitter, quiet_amplitude):
    r = P4Receiver(FS, admitted=True, breakin=True)
    r.enable_hold()
    r.peer_onset = 1908576+jitter
    grant = Emission(Kind.BODY, Role.ISS, 1952736, .798, packet_reply=True)
    r.note_emission(grant)
    carrier_end = grant.stream_sample + round(grant.duration_s*FS)
    end = r.peer_onset + CYCLE + round(.805*FS)
    t = np.arange(end+round(.040*FS)-carrier_end)
    audio = quiet_amplitude*np.cos(2*np.pi*700*t/FS)
    n = end+round(end_offset_ms*FS/1000)-carrier_end
    audio[:n] += .5*np.cos(2*np.pi*1500*np.arange(n)/FS)
    result = []
    r._grant_take(audio, carrier_end, result)
    assert any(x.observation.peer_raster_tail for x in result) is expected


@pytest.mark.parametrize('tail_ms', [0, 26, 30])
def test_synthetic_continuous_tail_yields_before_driver_capture_ends(monkeypatch, capsys, tail_ms):
    load = onair.session.load_wav

    def synthetic(*args, **kwargs):
        audio = load(*args, **kwargs).copy()
        carrier = 1952736+CYCLE+round(.798*FS)
        end = 1908576+2*CYCLE+round((.805+tail_ms/1000)*FS)
        audio[carrier:] = 0
        audio[carrier:end] = .5*np.cos(2*np.pi*1500*np.arange(end-carrier)/FS)
        return audio

    monkeypatch.setattr(onair.session, 'load_wav', synthetic)
    calls = []
    original = p4turn.P4Turn.peer_cycle

    def cycle(self, silent, at):
        before, grants = self.state, self.grants
        original(self, silent, at)
        if before == State.IRS_YIELD:
            calls.append((silent, before, self.state, grants, self.grants))

    monkeypatch.setattr(p4turn.P4Turn, 'peer_cycle', cycle)
    result = replay('onair-1002-1018', stop=49*FS)
    assert any(h.get('placed') for h in result['heard'] if h.get('kind') == 'normal')
    fallback = [c for c in calls if c[2] == State.ISS_GRANT]
    assert fallback and fallback[0][0] == 2 and fallback[0][3] == fallback[0][4] == 1
    grants = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('state') == State.ISS_GRANT]
    assert len(grants) > 1
    assert not any(e['event'] == 'Refused' and e.get('reason') == 'deadline'
                   for e in result['events'])
    assert all(e['now_sample'] <= e['decision_at'] and e['missed_slots'] == 0
               for e in result['events'] if e['event'] == 'Decision' and e['proposed_at'] is not None)
    assert 'state=ISS_GRANT expect=word' in capsys.readouterr().out


def test_1018_replay_keeps_grant_retry_budget_without_peer_tail():
    result = replay('onair-1002-1018', stop=47*FS)
    grants = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('kind') == Kind.BODY and e.get('status') == 0x40
              and not e.get('payload') and e.get('state') == State.ISS_GRANT]
    assert len(grants) == 4 and abs(grants[0]['at']-(1952736+CYCLE)) <= 48
    heard = [h for h in result['heard'] if h.get('kind') == Kind.NORMAL
             and h.get('placed') and h.get('onset_sample', 0) >= 1968576]
    assert not heard
    assert not any(e['event'] == 'Decision' and e.get('state') == State.IRS_YIELD
                   for e in result['events'])


def grant_machine(normal_ack='parity'):
    m, a = machine(data='chirp', ladder='chirp', normal_ack=normal_ack, breakin='body')
    short = round(38.803*FS)
    m._short(short, 0)
    m.keyed()
    request = round(39.762*FS)
    grant = receive(m, request, responds_to=short)
    assert (grant.kind, grant.status, grant.payload, grant.state) == (
        Kind.BODY, 0x40, b'', State.ISS_GRANT)
    assert abs(grant.stream_sample-round(40.682*FS)) <= 48
    m.keyed()
    assert m.grants == 1
    return m, a, grant


def take(m):
    at = 1968576
    shared = (m.ack_word, m._last_ack, m.acked_counter,
              m.first_body, m._peer_generation)
    obs = Observation(Kind.NORMAL, FS, 0, at, at, .805, 0., 0., placed=True,
                      peer_raster_tail=True)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.81*FS))
    m.silence(round(41.932*FS)-m.lead)
    assert m.decide(round(41.932*FS)-m.lead) is None
    assert m.state == State.IRS_YIELD and m.grants == 1
    assert (m.ack_word, m._last_ack, m.acked_counter,
            m.first_body, m._peer_generation) == shared


def test_peer_raster_take_yields_then_acks_first_normal():
    m, _, _ = grant_machine()
    take(m)
    at = round(42.273*FS)
    masked = Observation(Kind.NORMAL, FS, 0, at, at, .805, 0., 0., placed=True)
    m.observe(RegisteredObservation(masked, Registered.NEW), now=at+round(.81*FS))
    assert m.state == State.IRS_YIELD and m.decide(at+T-m.lead) is None
    obs = normal(at, 0, b'greeting')
    m.observe(RegisteredObservation(obs, Registered.ENRICHED, masked),
              now=at+round(.81*FS))
    p = m.decide(at+T-m.lead)
    assert (p.kind, p.word, p.state, p.stream_sample) == (
        Kind.WORD, 'X27', State.IRS_ANSWER, at+T)
    assert m.grants == 1
    assert m.peer_phase is not None and m.peer_phase.onset == at


def test_peer_raster_take_falls_back_after_two_empty_cycles():
    m, _, grant = grant_machine()
    take(m)
    m.peer_cycle(1, 1968576+CYCLE)
    assert m.state == State.IRS_YIELD and m.decide(grant.stream_sample+CYCLE) is None
    m.peer_cycle(2, 1968576+2*CYCLE)
    p = m.decide(grant.stream_sample+2*CYCLE)
    assert m.state == State.ISS_GRANT
    assert (p.kind, p.status, p.state, m.grants) == (Kind.BODY, 0x40, State.ISS_GRANT, 1)


@pytest.mark.parametrize('peer_slot,coherence', [
    (False, 0.), (False, .61), (True, .61),
])
def test_placed_unread_normal_without_tail_signature_does_not_yield(peer_slot, coherence):
    m, _, grant = grant_machine()
    at = 1968576 if peer_slot else grant.stream_sample + round(.935*FS)
    obs = Observation(Kind.NORMAL, FS, 0, at, at, .805, 0., coherence, placed=True)
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.81*FS))
    assert m.state == State.ISS_GRANT


def test_1018_muted_take_restores_grant_clock(monkeypatch):
    load = onair.session.load_wav

    def muted(*args, **kwargs):
        audio = load(*args, **kwargs).copy()
        audio[round(41.9*FS):] = 0
        return audio

    monkeypatch.setattr(onair.session, 'load_wav', muted)
    result = replay('onair-1002-1018', stop=48*FS)
    grants = [e for e in result['events'] if e['event'] == 'Keyed'
              and e.get('state') == State.ISS_GRANT]
    assert len(grants) >= 2
    assert not [e for e in result['events'] if e['event'] == 'Refused'
                and e.get('reason') == 'deadline'
                and e['at'] > grants[0]['at']]


def test_0058_slot_reply_keeps_existing_grant_path():
    m, _ = machine(data='hold', ladder='long', normal_ack='v27')
    at = round(67.516*FS)
    m._body(at, 0x40, State.ISS_GRANT)
    m.keyed()
    m.first_body = False
    p = receive(m, round(68.446*FS), status=0, responds_to=at)
    assert (p.kind, p.word, p.state) == (Kind.WORD, 'V27', State.IRS_ANSWER)
    assert m.grants == 1


def test_0355_two_empty_grants_still_reach_third_slot_reply():
    m, _ = machine(data='hold', ladder='long')
    grant_at = round(34.412*FS)
    m._body(grant_at, 0x40, State.ISS_GRANT)
    m.keyed()
    for count, seconds in ((2, 35.662), (3, 36.912)):
        due = round(seconds*FS)
        m.silence(due-m.lead)
        grant = m.decide(due-m.lead)
        assert (grant.kind, grant.status, grant.state, grant.stream_sample) == (
            Kind.BODY, 0x40, State.ISS_GRANT, due)
        m.keyed()
        assert m.grants == count and m.state == State.ISS_GRANT
    at = round(37.849*FS)
    m.observe(word('V27', at, extent=.245, responds_to=grant.stream_sample),
              now=at+round(.28*FS))
    p = m.decide(at+round(.30*FS))
    assert p is not None and p.state == State.ISS_LADDER
    assert any(e['event'] == 'turn' and e['responds_to'] == grant.stream_sample
               for e in m.link_events)


def test_mail_breakin_after_peer_raster_take():
    m, a, _ = grant_machine()
    take(m)
    at = round(42.273*FS)
    m.observe(RegisteredObservation(normal(at, 0, b'greeting'), Registered.NEW))
    m.decide(at+T-m.lead)
    m.keyed()
    a.on_host_data(b'mail bytes')
    m.allow_breakin = True
    p = m.decide(at+CYCLE)
    if p is not None:
        m.keyed()
    idle = normal(at+CYCLE, 1, b'')
    m.observe(RegisteredObservation(idle, Registered.NEW))
    p = m.decide(idle.stream_sample+T-m.lead)
    assert (p.kind, p.status, p.state) == (Kind.BODY, 0x40, State.IRS_BREAKIN)
