# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p4short
from hfmodem.shrike.p4receiver import P4Receiver, Emission, Role, Kind, Registered

FS = 48000
ROOT = Path(__file__).resolve().parents[5]
SEED = 92804


def inputs():
    result = []
    for arm, phase, cut in [('0927-1544', 35.9772, None), ('0927-1547', 12.2445, None),
                             ('0927-1549', 25.9974, None), ('0927-1553', 15.999, None),
                             ('0928-0951', None, 28.5)]:
        path = ROOT/'captures'/f'onair-{arm}'/'stream.flac'
        if not path.exists():
            pytest.skip(f'local corpus absent: {path}')
        audio = onair.session.load_wav(str(path), FS)
        if cut:
            audio = audio[:round(cut*FS)]
        result.append((path, audio, phase))
    return result


def receiver(at, cfo, wide):
    r = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    r.enable_hold()
    r.acked_counter = 1
    r.note_emission(Emission(Kind.WORD, Role.IRS, at-round(.330*FS), .244))
    if not wide:
        r.peer_onset = at-60000
        r._peer_heads = [at-120000, at-60000]
    return r


def read(r, audio, at):
    origin = at-round(.05*FS)
    regs = r.read_partial(audio[origin:at+round(.81*FS)], origin, deadline=at+round(.81*FS))
    regs += r.read_window(audio[origin:at+round(.846*FS)], origin, whole=True)
    regs += r.refine()
    normal = [x.observation for x in regs if x.observation.kind == Kind.NORMAL]
    assert not any(o.frame for o in normal), [(o.stream_sample, o.frame) for o in normal]
    r.finish_cycle(at)
    r.new_cycle()
    return any(x.observation.placed and not x.observation.provisional
               and x.outcome not in (Registered.EXPIRED, Registered.DUPLICATE)
               for x in regs if x.observation.kind == Kind.NORMAL)


def run(epochs=300):
    assert p4short.NORMAL_COHERENCE_KNEE == .30
    assert p4short.NORMAL_ACQUIRE_KNEE == .35
    source = inputs()
    rng = np.random.default_rng(SEED)
    tally = Counter()
    for epoch in range(epochs):
        if epoch < 200:
            _, audio, phase = source[epoch % 5]
            span = len(audio)/FS-11*1.25-1.1
            while True:
                start = .05*int(rng.integers(2, int(span/.05)))
                if phase is None or abs(((start-phase)+.625) % 1.25-.625) >= .100:
                    break
            cfo = float(rng.uniform(-8, 8))
            copies = [(audio, round((start+k*1.25)*FS)) for k in range(12)]
        else:
            copies = [(rng.normal(0, .1, round(1.2*FS)), round(.1*FS)) for _ in range(12)]
            cfo = 0.
        for wide in (False, True):
            name = 'wide' if wide else 'narrow'
            r = receiver(copies[0][1], cfo, wide)
            for k, (audio, at) in enumerate(copies):
                if epoch >= 200:
                    at = round((.1+1.25*k)*FS)
                    audio = np.r_[np.zeros(k*60000), audio]
                live = read(r, audio, at)
                tally[name+' corpus copies'] += 1
                tally[name+' corpus normal'] += live
            assert not tally[name+' corpus normal'], dict(tally)
        if epoch % 50 == 0:
            print('normal corpus', epoch, dict(tally), flush=True)
    rng = np.random.default_rng(SEED)
    for path, audio, phase in source:
        for offset in np.arange(.05, 1.25, .05):
            if phase is not None and abs(((offset-phase)+.625) % 1.25-.625) < .100:
                continue
            cfo = float(rng.uniform(-8, 8))
            for wide in (False, True):
                name = 'wide' if wide else 'narrow'
                r = receiver(round(offset*FS), cfo, wide)
                for k in range(int((len(audio)/FS-offset-1.1)/1.25)):
                    at = round((offset+k*1.25)*FS)
                    live = read(r, audio, at)
                    tally[name+' liveness cycles'] += 1
                    tally[name+' false normal'] += live
                assert not tally[name+' false normal'], dict(tally)
        print('normal liveness', path.parent.name, dict(tally), flush=True)
    audio = source[-1][1]
    audio = onair.session.load_wav(str(source[-1][0]), FS)
    for displacement in (-.2312, -.1156, .1156, .2312, .3, -.4, .625):
        for wide in (False, True):
            at = 1667751+round(displacement*FS)
            r = receiver(at, -.7, wide)
            for k in range(15):
                read(r, audio, at+k*60000)
                tally['displaced copies'] += 1
    return dict(seed=SEED, policy=dict(narrow_knee=.30, acquisition_knee=.35,
                                     short_polarity=1, acked_counter=1,
                                     precedence=True, complete_bracket=True),
                longest_false_live_run=0,
                inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p, _, _ in source}, counts=dict(tally))
