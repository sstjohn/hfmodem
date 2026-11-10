# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Seeded sample-level bootstrap timing, including fast probe and floor recovery."""
from pathlib import Path
import numpy as np
from hfmodem.sabir.arq import ArqConfig, LinkModem, SessionState, wire
from hfmodem.sabir.arq.profiles import EXTENDED
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.m2 import add_noise_snr3k


def handshake(extensions=0, snr_db=20., seed=0, fast_first=True, drop_first_ack=False):
    rng = np.random.default_rng(seed)
    identifier = int.from_bytes(np.random.default_rng([seed, 1]).bytes(8), 'big') or 1
    state = {'time': 0.}
    clock = lambda: state['time']
    a = LinkModem(ArqConfig(callsign='ALICE', advertise_gears=EXTENDED, fast_ctrl=fast_first),
                  clock=clock, session_id_factory=lambda: identifier)
    b = LinkModem(ArqConfig(callsign='BOB', advertise_gears=EXTENDED), clock=clock)
    for modem in (a, b):
        if extensions:
            modem.fsm._my_caps = lambda m=modem: [wire.Caps.build(m.fsm.session,
                wire.pack_tlvs([(wire.IMPL, b'sabir measurement')]), extensions, i)
                for i in range(extensions)]
    b.fsm.on_host_listen(True)
    a.fsm.on_host_connect('BOB')
    air = idle = turns = 0.
    count = [0, 0]
    for _ in range(30):
        moved = False
        for direction, (src, dst) in enumerate(((a, b), (b, a))):
            while src.outbox:
                wav = src.outbox.pop(0)
                duration = wav.size / FS
                air += duration
                state['time'] += duration
                count[direction] += 1
                snr = snr_db[direction] if isinstance(snr_db, tuple) else snr_db
                if not (drop_first_ack and direction == 1 and count[1] == 1):
                    dst.on_air(add_noise_snr3k(wav, snr, rng))
                state['time'] += .25
                turns += .25
                moved = True
        if a.fsm.state == b.fsm.state == SessionState.CONNECTED:
            return {'ok': True, 'air': air, 'idle': idle, 'turnaround': turns, 'elapsed': state['time']}
        if not moved:
            deadline = a.fsm.next_deadline()
            if deadline is None:
                break
            target = max(state['time'], deadline) + .001
            idle += target - state['time']
            state['time'] = target
            a.fsm.on_timer()
            b.fsm.on_timer()
    return {'ok': False, 'air': air, 'idle': idle, 'turnaround': turns, 'elapsed': state['time']}


def main():
    lines = ['Sabir bootstrap: exact current-profile bitmap, optional extensions',
             'Actual 44-byte controls; airtime includes guards, synchronization, FEC and CRC.',
             'AWGN in 3 kHz referenced to transmitted active power; 250 ms per turnaround.',
             '10 trials per row, session/noise seeds 0..9; retry and timer time included.',
             '', 'mode extensions/side SNR-forward/reverse successes mean-air(s) mean-idle(s) mean-elapsed(s)']
    cases = [(fast, 0, snr, False) for fast in (True, False) for snr in (20., -10., (20., -10.))]
    cases += [(True, 0, 20., True)]
    cases += [(True, count, 20., False) for count in (1, 2, 3)]
    for fast, extensions, snr, drop in cases:
        results = [handshake(extensions, snr, seed, fast, drop) for seed in range(10)]
        label = 'fast-dropACK' if drop else 'fast-first' if fast else 'floor-only'
        mean = lambda key: sum(r[key] for r in results) / len(results)
        lines.append(f'{label:12s} {extensions:15d} {str(snr):>19s} {sum(r["ok"] for r in results):3d}/10'
                     f' {mean("air"):11.3f} {mean("idle"):12.3f} {mean("elapsed"):15.3f}')
    text = '\n'.join(lines) + '\n'
    print(text)
    root = next(p for p in Path(__file__).resolve().parents if (p / 'docs/protocols/sabir').is_dir())
    (root / 'docs/protocols/sabir/negotiation_campaign_results.txt').write_text(text)


if __name__ == '__main__':
    main()
