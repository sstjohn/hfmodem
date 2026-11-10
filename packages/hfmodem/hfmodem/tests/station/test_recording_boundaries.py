# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Keys are (run, protocol, endpoint, local, peer_slot, generation)."""
import pytest

from hfmodem.tests.station.recording_support import flush, index, module


def key(generation=1, *, run='run', protocol='ale', endpoint='ep', local='K9ZZT', slot='W0QXR'):
    return run, protocol, endpoint, local, slot, generation


def boundaries(sessions, **kwargs):
    rec, root = sessions('contacts', **kwargs)
    rec.stream('rx', rate=8, clock='card')
    return module().ContactBoundaries(rec, run='run'), rec, root


def test_duplicate_establishment_and_peer_enrichment_keep_one_id(sessions):
    edges, rec, root = boundaries(sessions)
    positions = {'card': 100}
    edges.established(key(), '', positions)
    positions['card'] = 999
    flush(rec)
    [contact_id] = index(root)['contacts']
    edges.established(key(), 'W0QXR', {'card': 120})
    edges.established(key(), 'W0QXR', {'card': 130})
    flush(rec)
    assert list(index(root)['contacts']) == [contact_id]
    contact = index(root)['contacts'][contact_id]
    assert contact['peer'] == 'W0QXR'
    assert contact['positions'] == {'card': 100}


def test_replacement_retires_old_generation_and_ignores_stale_edges(sessions):
    edges, rec, root = boundaries(sessions)
    edges.established(key(1), 'W0QXR', {'card': 100})
    flush(rec)
    [old] = index(root)['contacts']
    edges.established(key(2), 'W0QXR', {'card': 150})
    edges.ended(key(1), {'card': 999}, 'stale')
    edges.established(key(1), 'W0QXR', {'card': 999})
    flush(rec)
    rows = index(root)['contacts']
    assert len(rows) == 2
    assert rows[old]['state'] == 'partial'
    assert rows[old]['end_positions'] == {'card': 150}
    new = next(k for k in rows if k != old)
    assert rows[new].get('end_positions') is None
    edges.ended(key(2), {'card': 200}, 'disconnect')
    edges.ended(key(2), {'card': 900}, 'duplicate')
    edges.established(key(2), 'W0QXR', {'card': 901})
    flush(rec)
    assert len(index(root)['contacts']) == 2
    assert index(root)['contacts'][new]['end_positions'] == {'card': 200}


@pytest.mark.parametrize('change', [
    {'protocol': 's5066'}, {'endpoint': 'ep2'}, {'local': 'K9ZZT-1'}, {'slot': 'N0CALL'},
])
def test_independent_keys_overlap_without_reference_counting(sessions, change):
    edges, rec, root = boundaries(sessions)
    edges.established(key(), 'W0QXR', {'card': 10})
    edges.established(key(**change), 'N0CALL', {'card': 20})
    edges.ended(key(), {'card': 30}, 'disconnect')
    flush(rec)
    rows = list(index(root)['contacts'].values())
    assert len(rows) == 2
    assert sum(row.get('end_positions') is None for row in rows) == 1
    edges.close_owner({'card': 40}, 'owner teardown')
    edges.close_owner({'card': 50}, 'duplicate')
    flush(rec)
    rows = list(index(root)['contacts'].values())
    [partial] = [row for row in rows if row['state'] == 'partial']
    assert partial['end_positions'] == {'card': 40}


def test_foreign_run_and_unknown_end_are_ignored(sessions):
    edges, rec, root = boundaries(sessions)
    edges.established(key(run='other'), 'W0QXR', {'card': 10})
    edges.ended(key(), {'card': 20}, 'unknown')
    flush(rec)
    assert index(root)['contacts'] == {}


def test_recording_exceptions_do_not_escape_edge_callbacks():
    recording = module()
    reports = []
    class FailedSession:
        report = staticmethod(reports.append)
        def begin(self, *args, **kwargs):
            raise OSError('synthetic recording failure')
        def end(self, *args, **kwargs):
            raise OSError('synthetic recording failure')
    edges = recording.ContactBoundaries(FailedSession(), run='run')
    edges.established(key(), 'W0QXR', {'card': 1})
    edges.ended(key(), {'card': 2}, 'disconnect')
    edges.close_owner({'card': 3}, 'teardown')
    assert reports and all('synthetic recording failure' in str(r) for r in reports)
