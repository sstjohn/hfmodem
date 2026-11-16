# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import FrozenInstanceError

import pytest

from hfmodem.kestrel.vara.ledger import PayloadLedger
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.tests.kestrel.test_compression import IO


def test_original_offsets_survive_text_rebinding_and_files_reframing():
    ledger = PayloadLedger()
    ledger.queue(b'abcdefgh', b'abcdefgh')
    before = ledger.evidence()
    ledger.reframe()
    ledger.queue(b'zip', b'abcdefgh', atomic=True, accept=False)
    assert ledger.evidence() == before
    ledger.settle(2)
    assert ledger.evidence().acknowledged == ()
    ledger.settle(1)
    assert ledger.evidence().acknowledged == ((0, 8),)
    ledger.queue(b'ijklmnop', b'ijklmnop')
    ledger.bind_prefix(b'encoded', b'ijkl')
    ledger.settle(6)
    assert ledger.evidence().acknowledged == ((0, 8),)
    ledger.settle(1)
    ledger.clear()
    assert ledger.evidence().acknowledged == ((0, 12),)
    assert ledger.evidence().discarded == ((12, 16),)
    assert ledger.evidence().sequence > before.sequence
    with pytest.raises(FrozenInstanceError):
        before.accepted = 90


def test_modem_epochs_are_explicit_and_retirement_retains_range_evidence():
    modem = VaraHostModem(IO())
    original = modem.station.payload_ledger
    original.queue(b'abcdef', b'abcdef')
    original.settle(2)
    modem._replace_station(reason='ABORT')
    old, current = modem.payload_evidence()
    assert old.epoch != current.epoch
    assert old.retired and old.acknowledged == ((0, 2),) and old.discarded == ((2, 6),)
    assert current.accepted == 0 and current.epoch != str(modem.generation)
    modem.station.payload_ledger.queue(b'new', b'new')
    modem._replace_station(preserve_data=True)
    assert modem.payload_evidence()[-1].epoch == current.epoch


def test_bounded_fragmented_history_exposes_loss_instead_of_inventing_ranges():
    ledger = PayloadLedger()
    for _ in range(4100):
        ledger.queue(b'ab', b'ab')
        ledger.settle(1)
        ledger.clear()
    evidence = ledger.evidence()
    assert not evidence.complete
    assert len(evidence.acknowledged) == len(evidence.discarded) == 4096
    assert ledger.confirmed == ledger.discarded == 4100


def test_reframing_after_clear_keeps_absolute_original_offsets():
    ledger = PayloadLedger()
    ledger.queue(b'old', b'old')
    ledger.clear()
    ledger.queue(b'fresh', b'fresh')
    ledger.reframe()
    ledger.queue(b'zip', b'fresh', atomic=True, accept=False)
    ledger.settle(3)
    assert ledger.evidence().discarded == ((0, 3),)
    assert ledger.evidence().acknowledged == ((3, 8),)


def test_runtime_builds_snapshots_only_for_subscribers_and_changed_sequences(monkeypatch):
    from hfmodem.tests.kestrel.test_idle_tick_reply_window import owner
    runtime, station, _ = owner()
    ledger = PayloadLedger()
    station.payload_ledger = ledger
    builds = []
    def evidence():
        builds.append(station.payload_ledger.sequence)
        return (station.payload_ledger.evidence(),)
    modem_type = type(runtime.modem)
    monkeypatch.setattr(modem_type, 'payload_evidence_sequence',
                        property(lambda self: (self.station.payload_ledger.epoch,
                                               self.station.payload_ledger.sequence)), raising=False)
    runtime.modem.payload_evidence = evidence
    runtime.step()
    ledger.queue(b'abc', b'abc')
    runtime.step()
    assert builds == [] and runtime.payload_evidence() == ()
    with runtime.subscribe_payload_evidence() as read:
        runtime.step()
        first = read()
        assert len(builds) == 1 and first[0].accepted == 3
        runtime.step()
        assert len(builds) == 1 and read() is first
        ledger.settle(1)
        runtime.step()
        assert len(builds) == 2 and read()[0].acknowledged == ((0, 1),)
        with runtime.subscribe_payload_evidence() as second:
            runtime.step()
            assert len(builds) == 2 and second() is read()
        ledger.clear()
        runtime.step()
        assert len(builds) == 3 and read()[0].discarded == ((1, 3),)
    ledger.queue(b'd', b'd')
    runtime.step()
    assert len(builds) == 3 and runtime.payload_evidence() == ()
    with runtime.subscribe_payload_evidence() as read:
        runtime.step()
        assert len(builds) == 4 and read()[0].accepted == 4
        station.payload_ledger = PayloadLedger()
        runtime.step()
        assert len(builds) == 5 and read()[0].epoch == station.payload_ledger.epoch


def test_modem_evidence_sequence_tracks_retirement_and_preserved_ledger():
    modem = VaraHostModem(IO())
    initial = modem.payload_evidence_sequence
    modem.station.payload_ledger.queue(b'abc', b'abc')
    accepted = modem.payload_evidence_sequence
    assert accepted != initial and accepted[0] == initial[0]
    modem._replace_station(preserve_data=True)
    assert modem.payload_evidence_sequence == accepted
    modem._replace_station()
    assert modem.payload_evidence_sequence[0] != initial[0]
