# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Amended sender pairing rule, per delivery and level.

The count restarts at a level change, at an over the sender re-keys with BUFFER
unchanged, and at the end of a delivery: the over after which BUFFER reads 0, unless
the sender keys again within 5 s (a close the receiver did not take, re-keyed).
Polls and slow answers do not restart it. At the previous over's level, with r bytes
left before the over:
  - six delivered singles there: pair when r >= one frame's payload;
  - previous over a delivered pair: pair when r >= one frame's payload, and also when
    r is short enough for the record below (its last full frame announced the close one
    record down, 0x81) and the receiver held the level anyway: the sender keys the
    remainder with field 0x80 and an empty close behind it.
Keyings of 4.1-4.25 s are the end-of-session bursts both stations key at once, not DATA."""
import collections
import json
import sys

P = {4: 43, 3: 34, 2: 22, 1: 9}
BELOW = {4: 34, 3: 22, 2: 9, 1: 0}
rows = [json.loads(line) for line in open(sys.argv[1])]
seqs = collections.defaultdict(list)
for r in rows:
    if not r['setup'] and r['level'] in P and not 4.1 < r['dur'] < 4.25:
        seqs[(r['file'], r['who'])].append(r)
conf = collections.Counter()
miss = []
for (f, who), rs in seqs.items():
    run, paired, prev = 0, False, None
    for i, r in enumerate(rs):
        lv, b = r['level'], r['buffer']
        nxt = rs[i + 1]['buffer'] if i + 1 < len(rs) else None
        if prev != lv:
            run, paired = 0, False
        state = run >= 6 or paired
        act = r['frames'] == 2
        if state and b is not None:
            pred = b >= P[lv] or (paired and b < BELOW[lv])
            conf[(lv, 'paired' if paired else 'run6', 'pair' if pred else 'single', 'pair' if act else 'single')] += 1
        else:
            pred = False
            conf[(lv, 'nostate', 'pair' if act else 'single')] += 1
        if pred != act:
            miss.append((f[-50:], who, r['t0'], lv, r['frames'], b, run, paired))
        delivered = not (b is not None and nxt == b and b > 0)
        rekey = i + 1 < len(rs) and rs[i + 1]['t0'] - (r['t0'] + r['dur']) < 5.0
        closes = nxt == 0 and not rekey
        if closes or not delivered:
            run, paired = 0, False
        elif act:
            run, paired = 0, True
        else:
            run, paired = run + 1, False
        prev = lv
for k in sorted(conf, key=str):
    print(k, conf[k])
print('mismatches', len(miss))
for m in miss:
    print(m)
