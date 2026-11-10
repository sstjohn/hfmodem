# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""`nc` to an FTDX10 behind rigctld, answering in Hamlib's own extended-response
framing (tests/rigctl_parse.c, the Hamlib 5.0 tree the station builds):

- each `+\\cmd args` is echoed as a `cmd: args` header and ended by `RPRT 0`;
- `send_cmd` prints `Reply: ` and then, for a query, the rig's answer and a NUL (the
  command holds a `;`); a setter gets no answer, so its read times out and the status
  follows the `Reply: ` on the same line;
- `q` answers `RPRT 0` too.

The rig keeps its state in $FIXTURE/rig.json across connections, answers RF0; with
the roofing filter's read code, not the code that set it, and leaves unchanged what a
setter named in $RIG_IGNORES asks for. Like the FTDX10's band stack, a dial on another
band brings back the mode and filters that band was last left at: USB, WIDTH 3000 and
R. FIL 3 kHz on a band not yet visited."""
import json
import os
import sys
from pathlib import Path

RIG = {'FA;': 'FA007105000;', 'MD0;': 'MD02;', 'SH0;': 'SH0018;', 'RF0;': 'RF07;',
       'IS0;': 'IS00+0000;', 'AC;': 'AC001;', 'AGC': '6', 'PREAMP': '10', 'VFO': 'VFOA'}
ROOF_READ = {'1': '6', '2': '7', '4': '9', '5': 'A'}
STACKED = ('MD0;', 'SH0;', 'RF0;', 'IS0;')
FRESH = {'MD0;': 'MD02;', 'SH0;': 'SH0018;', 'RF0;': 'RF07;', 'IS0;': 'IS00+0000;'}


def band(fa):
    return int(fa[2:11]) // 1_000_000


def answer(requests, rig, ignores=(), log=lambda request: None):
    """rigctld's reply to one connection's requests, `rig` changed as the rig would be."""
    out = []
    for request in requests:
        log(request)
        if request == 'q':
            out.append('RPRT 0\n')
            break
        name, _, args = request.removeprefix('+\\').partition(' ')
        out.append(f'{name}: {args}\n' if args else f'{name}:\n')
        if name == 'send_cmd':
            out.append('Reply: ')
            if args in rig:
                out.append(rig[args] + '\0')
            elif not args.startswith(tuple(ignores)):
                query = next(key for key in rig if key.endswith(';') and args.startswith(key[:-1]))
                code = args[len(query) - 1:-1]
                if query == 'FA;' and band(args) != band(rig['FA;']):
                    stack = rig.setdefault('stack', {})
                    stack[str(band(rig['FA;']))] = {key: rig[key] for key in STACKED}
                    rig.update(stack.get(str(band(args)), FRESH))
                rig[query] = (f'RF0{ROOF_READ[code]};' if query == 'RF0;' else args)
                if query == 'RF0;':
                    rig['SH0;'] = 'SH0018;'  # the FTDX10 puts the width back to 3000 on a roofing change
        elif name == 'set_level':
            level, value = args.split()
            rig[level] = value
        elif name == 'get_level':
            out.append(rig[args] + '\n')
        elif name == 'get_vfo':
            out.append(f'VFO: {rig["VFO"]}\n')
        out.append('RPRT 0\n')
    return ''.join(out)


if __name__ == '__main__':
    if '-z' in sys.argv:
        sys.exit(0)
    fixture = Path(os.environ['FIXTURE'])
    store = fixture / 'rig.json'
    rig = json.loads(store.read_text()) if store.exists() else dict(RIG)

    def log(request):
        with open(fixture / 'nc.requests', 'a') as requests:
            requests.write(request + '\n')
    sys.stdout.write(answer(sys.stdin.read().splitlines(), rig,
                            os.environ.get('RIG_IGNORES', '').split(), log))
    store.write_text(json.dumps(rig))
