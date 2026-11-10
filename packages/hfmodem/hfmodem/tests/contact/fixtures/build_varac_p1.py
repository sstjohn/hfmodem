# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Build sanitized application fixtures from P1 transcripts/messages.json."""
import argparse
import json
from pathlib import Path
import re

INDICES = (0, 2, 3, 6, 10, 74, 75, 76, 79, 94, 120, 123, 126, 127,
           131, 133, 135, 136, 137, 138, 139, 142, 164)


def synthetic(value, seed):
    if seed == b'a@example.test':
        return b'x' * (len(value) - len(b'@example.test')) + b'@example.test'
    return (seed + b'x' * len(value))[:len(value)]


def sanitize(content):
    for tag, seed in ((b'NAME', b'Alex'), (b'QTH', b'Test Town'),
                      (b'LOC', b'AA00AA'), (b'EM', b'a@example.test')):
        content = re.sub(rb'<' + tag + rb':([^>]+)>', lambda m:
                         b'<' + tag + b':' + synthetic(m[1], seed) + b'>', content)
    for pattern in (rb"73's ([A-Za-z]+)!", rb'Good evening ([A-Za-z]+)', rb'Hello ([A-Za-z]+),'):
        content = re.sub(pattern, lambda m: m[0].replace(m[1], synthetic(m[1], b'Alex')), content)
    for label in (b'Age', b'Profession', b'Hobbies'):
        content = re.sub(label + rb': ([^\r\n]+)', lambda m:
                         label + b': ' + synthetic(m[1], b'00' if label == b'Age' else b'Test'), content)
    return content


def build(source):
    rows = json.loads(source.read_text())
    result = []
    for index in INDICES:
        row = rows[index]
        original = bytes.fromhex(row['text_hex'])
        content = sanitize(original)
        assert len(content) == len(original) == row['declared_length']
        wire = str(row['declared_length']).encode() + b' ' + content
        decoded = b''.join(bytes.fromhex(s['decoded_fragment_hex']) for s in row['sources'])
        assert decoded.startswith(str(len(original)).encode() + b' ' + original)
        fragments, offset = [], 0
        for source_frame in row['sources']:
            size = len(bytes.fromhex(source_frame['decoded_fragment_hex']))
            fragments.append(wire[offset:offset + size].hex())
            offset += size
        result.append(dict(message_index=index, declared_length=row['declared_length'],
                           content=content.decode(), fragments_hex=fragments,
                           sources=[{key: s[key] for key in ('session', 'file', 'slot', 'start_s', 'stop_s')}
                                    for s in row['sources']]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    Path(__file__).with_name('varac-p1.json').write_text(json.dumps(build(args.source), indent=2) + '\n')
