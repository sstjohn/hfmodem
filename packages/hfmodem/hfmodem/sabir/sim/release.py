# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Deterministic full-session measurements, without radio or sockets.

python -m hfmodem.sabir.sim.release --bytes 250000 --snr 40
python -m hfmodem.sabir.sim.release --channel good --snr 30 --bytes 30000

Goodput counts uncompressed ARQ stream bytes. Time includes emitted headers,
FEC, pilots, preambles, guards, ACKs, setup, disconnect, retries, timer waits
and modeled 250 ms turnarounds. Host record envelopes, CPU time and additional
radio delays are excluded. Fading traces are independent per direction and
continuous across gaps. SNR is referenced to transmitted power before fading.
"""
import argparse
import json
import time

import numpy as np

from hfmodem.sabir.arq import ArqConfig, LinkModem
from hfmodem.sabir.arq.profiles import PROFILES, EXTENDED
from hfmodem.sabir.phy.rate import FS
from .m3 import run_pair
from .watterson import ContinuousWatterson, PROFILES as CHANNELS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bytes', type=int, default=250000)
    ap.add_argument('--reverse-bytes', type=int, default=0)
    ap.add_argument('--snr', type=float, default=40)
    ap.add_argument('--reverse-snr', type=float)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--channel', choices=tuple(CHANNELS))
    ap.add_argument('--reverse-channel', choices=('awgn', *CHANNELS))
    ap.add_argument('--profile', choices=tuple(PROFILES), default='wide256')
    ap.add_argument('--drop-burst', action='append', default=[], metavar='a:1',
                    help='erase this one-based transmitted burst in direction a or b')
    args = ap.parse_args()
    if args.bytes < 0 or args.reverse_bytes < 0:
        ap.error('payload byte counts must be nonnegative')
    drops = set()
    for item in args.drop_burst:
        try:
            direction, number = item.split(':')
            number = int(number)
            if direction not in ('a', 'b') or number < 1:
                raise ValueError()
        except ValueError:
            ap.error('--drop-burst requires a:1 or b:1 with a positive index')
        drops.add((direction, number))
    rng = np.random.default_rng(args.seed)
    payload, reverse = rng.bytes(args.bytes), rng.bytes(args.reverse_bytes)
    reverse_channel = args.reverse_channel or args.channel
    channels = tuple(ContinuousWatterson(name, FS, seed=args.seed + i)
                     if name and name != 'awgn' else lambda x, t: x
                     for i, name in enumerate((args.channel, reverse_channel)))
    start = time.perf_counter()
    result = run_pair(payload, None, (args.snr, args.reverse_snr if args.reverse_snr is not None else args.snr),
                      seed=args.seed, payload_b=reverse,
                      cfg_kw=dict(advertise_gears=EXTENDED, data_profile=args.profile, loading=False),
                      channel_transform=channels, drop_bursts=frozenset(drops), max_exchanges=3000)
    modem = LinkModem(ArqConfig())
    profile = PROFILES[args.profile]
    codec = modem.fsm.profile_codec(profile.name)
    coded = np.stack([codec.encode_cw(bytes(codec.data_bytes)) for _ in range(profile.frame_cws)])
    body, _ = modem.encode_body(profile, coded)
    control_s = modem.fast.n_samples(1) / FS
    cycle = dict(data_body_s=len(body) / FS, header_s=control_s,
                 ack_s=control_s, turnaround_s=.5)
    cycle['total_s'] = sum(cycle.values())
    cycle['payload_bytes'] = profile.frame_cws * codec.data_bytes
    cycle['payload_bps'] = cycle['payload_bytes'] * 8 / cycle['total_s']
    output = dict(parameters=vars(args), complete_session=result.ok,
                  byte_exact=result.byte_exact, terminal_states=result.terminal_states,
                  delivered_bytes=len(result.delivered), reverse_delivered_bytes=len(result.delivered_a),
                  virtual_seconds=result.wall_s, airtime_seconds=result.airtime,
                  timing=result.timing, forward_goodput_bps=result.throughput_bps,
                  total_goodput_bps=8 * (len(result.delivered) + len(result.delivered_a)) / result.wall_s,
                  rate_scope='uncompressed ARQ stream, full session; excludes host records and CPU time',
                  error_free_fast_control_cycle=cycle,
                  cpu_wall_seconds=time.perf_counter() - start,
                  gears=sorted(set(result.gears)), rebuilds=result.stats_a['rebuilds'],
                  ack_timeouts=result.stats_a['timeouts'])
    print(json.dumps(output, indent=2))
    return 0 if result.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
