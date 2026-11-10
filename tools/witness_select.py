#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The contact spans a contact-only witness can be aligned against, one per line."""
import json
from pathlib import Path

from hfmodem.station import sessionlog
from hfhost.recordings import available


def selected(root, witness):
    """(local path, witness path, selection) for every overlap; whatever was found before a fault."""
    root, witness, found = Path(root), Path(witness), []
    try:
        index = json.loads((root / 'recording.json').read_text())
        side = json.loads(witness.with_suffix('.json').read_text())
        if side.get('recording_index'):
            child_path = (witness.parent / side['recording_index']).resolve()
            if not child_path.is_relative_to(witness.parent.resolve()):
                raise ValueError('Witness index outside run')
            child = json.loads(child_path.read_text())
            for cid, rec in index.get('contacts', {}).items():
                remote = child.get('contacts', {}).get(cid, {})
                for name, ref in remote.get('lanes', {}).items():
                    anchor = remote.get('anchors', {}).get(name, {})
                    witness_path = child_path.parent / ref['source']
                    if not anchor or not available(witness_path):
                        continue
                    for local_ref in rec.get('lanes', {}).values():
                        if local_ref['clock'] != anchor.get('parent_clock'):
                            continue
                        path = Path(rec.get('run_directory', sessionlog.source_root(rec, root / local_ref['source']))) / local_ref['source']
                        chosen = sessionlog.select(rec, path)
                        if not chosen.complete or not available(path):
                            continue
                        wa, wb = ref.get('selection', ref['actual'])
                        spans = [(wa, wb)]
                        for g0, g1 in ref.get('gaps', []):
                            spans = [(c, d) for x, y in spans for c, d in ((x, min(y, g0)), (max(x, g1), y)) if c < d]
                        def local(at):
                            return anchor['sample'] + (at - anchor['position']) * anchor['parent_rate'] / anchor['rate']
                        def remote_at(at):
                            return round(anchor['position'] + (at - anchor['sample']) * anchor['rate'] / anchor['parent_rate'])
                        for a, b in spans:
                            for span in chosen.spans:
                                lo, hi = max(local(a), span.bounds[0]), min(local(b), span.bounds[1])
                                if lo < hi:
                                    found.append((path, witness_path, dict(recording=rec,
                                        local=[round(lo), round(hi)], witness=[remote_at(lo), remote_at(hi)])))
            return found
        if not available(witness):
            raise ValueError('witness unavailable')
        for rec in index.get('contacts', {}).values():
            bound = side.get('recording', {})
            if bound.get('id') != rec.get('id') or bound.get('context', {}).get('id') != rec.get('context', {}).get('id'):
                continue
            for ref in rec.get('lanes', {}).values():
                path = Path(rec.get('run_directory', sessionlog.source_root(rec, root / ref['source']))) / ref['source']
                chosen = sessionlog.select(rec, path)
                if not chosen.complete or not available(path):
                    continue
                anchor = side.get('source_anchor', {})
                if anchor.get('clock') != ref['clock']:
                    continue
                rate = side.get('rate', side.get('samplerate'))
                wa, wb = side['actual']
                def local(at):
                    return anchor['sample'] + (at - anchor['position']) * ref['rate'] / rate
                def remote(at):
                    return round(anchor['position'] + (at - anchor['sample']) * rate / ref['rate'])
                lo, hi = max(chosen.bounds[0], local(wa)), min(chosen.bounds[1], local(wb))
                if lo < hi:
                    # Gaps are separate alignments, never concatenated audio.
                    for span in chosen.spans:
                        a, b = max(lo, span.bounds[0]), min(hi, span.bounds[1])
                        if a < b:
                            found.append((path, witness, dict(recording=rec,
                                local=[round(a), round(b)], witness=[remote(a), remote(b)])))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return found


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='directory holding the modem recording.json')
    parser.add_argument('witness', help='the witness recording')
    args = parser.parse_args()
    for path, witness_path, selection in selected(args.root, args.witness):
        print(f'{path}|{witness_path}|{json.dumps(selection)}')
