# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR corpus detector. Runs in its own process; never opens radio/audio IO."""
import json
from functools import lru_cache
from pathlib import Path
import signal
import sys

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from creance.monitor.runners.shrike_runner import fsk_offset

FS = 48000
BLOCK = 4 * FS
LOOKBACK = 4 * FS  # a complete 3.3075 s entry chirp can straddle a block boundary
#: A header this strong earns a CRC attempt, and only the CRC names a chirp.
#: N9VC's eleven good 08-02 chirps, 400 Hz up, peak 0.36-0.81 on this two-lane
#: header, and eight non-P4 recordings reach 0.44 at worst, median 0.25: the
#: header cannot separate them, so it only chooses where the CRC looks.
CHIRP_HEADER_KNEE = .33


@lru_cache(maxsize=4)
def chirp_templates(marker, rotation=0.):
    from hfmodem.shrike import p4chirp, pactor2
    templates = []
    for steps, delay in zip(pactor2.marker_steps(marker), (0, 4)):
        impulses = np.zeros(104, complex)
        impulses[delay:delay + 65:8] = np.exp(1j*np.r_[0., np.cumsum(steps + rotation)])
        shaped = np.convolve(impulses, p4chirp.QUASI_RRC)
        templates.append(resample_poly(shaped, 9, 4)[:180])  # 150 ms at 1200 Hz
    return templates


def chirp_events(audio, max_offset):
    """Blind dual-lane header acquisition followed by the existing CRC reader.

    Two independent complex phases are allowed, one per physical carrier;
    timing and CFO must agree. A header alone earns only a bounded candidate
    hold. The known full-field decoder decides whether a whole chirp copied.
    """
    from hfmodem.shrike import p4chirp, p4chirprx
    t = np.arange(len(audio)) / FS
    lanes = [resample_poly(audio*np.exp(-2j*np.pi*tone*t), 1, 40)
             for tone in (1530., 550.)]
    if len(lanes[0]) < 180:
        return []
    n = len(lanes[0])
    tt = np.arange(n)/1200
    energy = [np.convolve(abs(lane)**2, np.ones(180), mode='valid') for lane in lanes]
    matches = []
    # The recorded Dragon headers have an additional pi/4 per symbol relative
    # to our renderer. Differential header correlation tolerates that constant
    # rotation; coherent acquisition must search both explicit constructions.
    for marker, rotation in ((8, 0.), (9, 0.), (8, np.pi/4), (9, np.pi/4)):
        templates = chirp_templates(marker, rotation)
        for cfo in np.arange(-max_offset, max_offset + .1, 4.):
            scores = []
            for lane, template, power in zip(lanes, templates, energy):
                corr = abs(fftconvolve(lane*np.exp(-2j*np.pi*cfo*tt), template[::-1].conj(), mode='valid'))
                denom = np.sqrt(power * np.sum(abs(template)**2))
                scores.append(np.divide(corr, denom, out=np.zeros_like(corr),
                                        where=denom > max(float(denom.max())*1e-5, 1e-15)))
            score = np.minimum(*scores)
            # Keep separated maxima, so a strong repeated caller cannot hide
            # a different chirp in the same block. At most two CRC attempts.
            for _ in range(2):
                at = int(np.argmax(score))
                if score[at] < CHIRP_HEADER_KNEE:
                    break
                matches.append((float(score[at]), at, float(cfo), marker))
                score[max(0, at-1200):at+1200] = 0
    found = []
    attempted = []
    for score, at, cfo, marker in sorted(matches, reverse=True):
        start = round(at * FS / 1200)
        if any(abs(start - previous) < FS for previous in attempted):
            continue
        attempted.append(start)
        decoded = None
        if start + round(p4chirp.PACKET_S * FS) <= len(audio):
            for shift in (0, -40, 40, -80, 80, -120, 120, -160, 160):
                if start + shift < 0:
                    continue
                decoded = p4chirprx.decode_packet(audio, start_sample=start+shift,
                                                 frequency_offset=cfo)
                if decoded is not None:
                    break
        if decoded is None and score < .87:
            if len(attempted) >= 2:
                break
            continue
        found.append(dict(t=start/FS, protocol='PACTOR-4', offset_hz=cfo,
            kind='P4-CHIRP-CRC' if decoded is not None else 'P4-CHIRP-HEADER',
            grade='confirmed' if decoded is not None else 'tentative',
            detail=f'entry chirp marker={marker} header={score:.3f} cfo={cfo:+.1f} Hz; '
            + (f'CRC valid status=0x{decoded.frame.status.byte:02x}' if decoded is not None
               else 'no complete CRC-valid field')))
        if len(found) == 2:
            break
    return found


def analyze(audio, max_offset=60):
    from hfmodem.shrike import p4sig, rxfront
    events = chirp_events(audio, max_offset)
    for ev in rxfront.decode_events(audio, p3_envelope=False):
        proto = ev.protocol or ('PACTOR-1' if ev.kind in ('connect', 'callb', 'fsk', 'p1reply') else 'UNKNOWN')
        kinds = {'connect': 'P1-CONNECT', 'callb': 'P1-CALLB',
                 'packet': 'P1-PACKET' if proto == 'PACTOR-1' else 'P3-HEADER',
                 'cs': 'P1-CS' if proto == 'PACTOR-1' else 'P3-CS'}
        kind = kinds.get(ev.kind, 'SHAPE')
        detail = ev.text
        if proto == 'PACTOR-1':
            at = max(0, int(ev.t * FS))
            span = int(.18 * FS) if ev.kind in ('cs', 'unassigned') else FS
            offset = fsk_offset(audio[at:at + span])
            detail += f'; FSK pair offset={offset} Hz'
            if offset is None:
                kind = 'UNRESOLVED-CENTER'
            elif abs(offset) > max_offset:
                kind = 'OFFCENTER'
        events.append(dict(t=ev.t, protocol=proto, kind=kind, detail=detail,
                           grade='confirmed' if kind.startswith(('P1-', 'P3-')) else 'tentative'))
    # The spread statistic aliases every 112.5 Hz and cannot establish center.
    # Only a complete Chu header matched in its narrow CFO search is eligible.
    candidates = []
    for start in range(0, max(0, len(audio) - FS // 2 + 1), FS // 4):
        seg = audio[start:start + FS // 2]
        score = p4sig.spread_score(seg)
        if score >= p4sig.SPREAD_KNEE:
            candidates.append((score, start))
    # One full header search per block bounds CPU cost. Distinct blocks, not
    # overlapping windows over one burst, must establish repeated candidates.
    if candidates:
        spread, start = max(candidates)
        score, root, shift, cfo = p4sig.header_score(audio[start:start + FS // 2])
        if score >= .35 and abs(cfo) <= max_offset:
            events.append(dict(t=start / FS, protocol='PACTOR-4', kind='P4-HEADER',
                grade='tentative', detail=f'Chu candidate score={score:.3f} spread={spread:.3f} '
                f'root={root} shift={shift} cfo={cfo:+.1f} Hz; no CRC/payload claim'))
    return sorted(events, key=lambda ev: ev['t'])


def main():
    # The parent handles terminal Ctrl-C, closes audio, then drains this reader
    # through EOF. A simultaneous child KeyboardInterrupt would lose the tail.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    max_offset = float(sys.argv[1]) if len(sys.argv) > 1 else 60.
    buf = np.zeros(0)
    base = 0
    seen = {}
    while True:
        raw = sys.stdin.buffer.read(BLOCK * 2)
        if not raw:
            break
        buf = np.concatenate((buf, np.frombuffer(raw, '<i2').astype(float) / 32768))
        for event in analyze(buf, max_offset):
            event['t'] += base / FS
            key = (event['protocol'], event['kind'])
            if event['t'] <= seen.get(key, -1e9) + .75:
                continue
            seen[key] = event['t']
            event['modem'] = 'pactor-corpus'
            print(json.dumps(event), flush=True)
        print(json.dumps(dict(t=(base + len(buf))/FS, modem='pactor-corpus',
                              protocol='UNKNOWN', kind='ANALYZED', grade='tentative')), flush=True)
        cut = max(0, len(buf) - LOOKBACK)
        buf = buf[cut:]
        base += cut


if __name__ == '__main__':
    # Allows a direct runner invocation through creance's process supervisor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
