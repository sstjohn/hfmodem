"""Measured packet views, including a separately clocked local TX monitor tap."""
import json

import numpy as np

FS = 48000


def signal_view(x, start, end, out, name, source):
    """Bounded, calibrated STFT plus envelope; no sliced/guessed symbols."""
    lo, hi = max(0, round(start*FS)), min(len(x), round(end*FS))
    if hi <= lo:
        return None
    a = x[lo:hi]
    nfft = 4096
    count = min(180, max(2, int(len(a)/480)))
    centers = np.linspace(0, len(a)-1, count).astype(int)
    padded = np.pad(a, (nfft//2, nfft//2))
    blocks = padded[centers[:, None]+np.arange(nfft)]
    win = np.hanning(nfft)
    power = (np.abs(np.fft.rfft(blocks*win, axis=1))[:, :257]*2/win.sum())**2
    db = 10*np.log10(np.maximum(power, 1e-12))
    spectrum = 10*np.log10(np.maximum(power.mean(axis=0), 1e-12))
    envelope = 20*np.log10(np.maximum(np.sqrt(np.mean(blocks**2, axis=1)), 1e-6))
    peak = float(db.max())
    payload = {'source': source, 'duration_s': len(a)/FS, 'hz_per_bin': FS/nfft,
               'rows': count, 'bins': 257, 'db_min': -100, 'db_max': 0,
               'peak_dbfs': round(peak, 1), 'rms_dbfs': round(float(20*np.log10(max(np.sqrt(np.mean(a*a)), 1e-6))), 1),
               'map': np.clip((db+100)*2.55, 0, 255).astype(np.uint8).ravel().tolist(),
               'spectrum': spectrum.round(1).tolist(), 'envelope': envelope.round(1).tolist(),
               'note': 'Measured audio energy; tone tracks are not decoded symbol decisions.'}
    (out/name).write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    return name


def align_audio(reference, other, events):
    """Fit a constant offset using independently matching RX envelopes.

    At least three distinct windows must agree. Repetitive/silent windows and
    ambiguous clocks are rejected by agreement across windows. This is audio alignment, not wall-clock time.
    """
    from scipy.signal import correlate
    def envelope(x):
        n = len(x)//480
        return np.log10(np.maximum(np.mean(x[:n*480].reshape(n,480)**2, axis=1), 1e-12))
    a, b = envelope(reference), envelope(other)
    votes = []
    windows = [e for e in events if e.get('direction')=='rx' and e['t1']-e['t0']>=2]
    # Prefer decoded frames but also permit independently classified bursts.
    windows.sort(key=lambda e: e['status']!='CRC pass')
    used = []
    for e in windows:
        lo, hi = max(0,int(e['t0']*100)), min(len(a),int(e['t1']*100))
        if any(abs(lo-v)<100 for v in used):
            continue
        q = a[lo:hi]; q = q-q.mean(); n = len(q)
        if n<200 or len(b)<n or np.std(q)<.05:
            continue
        dot = correlate(b,q,mode='valid',method='fft')
        sums = np.convolve(b,np.ones(n),'valid')
        squares = np.convolve(b*b,np.ones(n),'valid')
        scores = dot/np.sqrt(np.maximum(squares-sums*sums/n,1e-10)*np.sum(q*q))
        k = int(np.argmax(scores))
        if scores[k]<.65:
            continue
        votes.append(((k-lo)/100, float(scores[k]))); used.append(lo)
        if len(votes)>=12:
            break
    if len(votes)<3:
        return None, f'Unaligned: only {len(votes)} distinctive audio windows matched; inspect on its own clock.'
    shift = float(np.median([v[0] for v in votes]))
    agreeing = [v for v in votes if abs(v[0]-shift)<=.08]
    if len(agreeing)<3 or len(agreeing)<.8*len(votes):
        return None, 'Unaligned: audio matches disagree; inspect on its own clock.'
    return shift, (f'Audio-aligned by {len(agreeing)} receive windows; offset {shift:+.2f}s; '
                   f'10 ms envelope grid, maximum residual {max(abs(v[0]-shift) for v in agreeing):.2f}s. '
                   'A local TX monitor shows our audio, not reception at the peer.')


def align_packets(reference, other):
    """CRC payload identity aligns independent decodes even with different audio."""
    def unique(events):
        groups = {}
        for e in events:
            key = e.get('packet_sha256')
            if key and e.get('status')=='CRC pass' and not e.get('timing_precision'):
                groups.setdefault(key, []).append(e['t0'])
        return {key: times[0] for key,times in groups.items() if len(times)==1}
    a, b = unique(reference), unique(other)
    offsets = [b[key]-a[key] for key in a.keys() & b.keys()]
    if len(offsets)<2:
        return None, 'Fewer than two unique CRC-valid packet anchors.'
    shift = float(np.median(offsets))
    residual = max(abs(v-shift) for v in offsets)
    if residual>.08:
        return None, 'Decoded packet offsets disagree; clock drift or ambiguous frame timing.'
    return shift, f'Aligned by {len(offsets)} unique CRC-valid packet payloads; offset {shift:+.3f}s, maximum timing residual {residual:.3f}s.'


def align_handshakes(reference, other):
    """Match named scanner locks; repeated calls must beat their timing aliases."""
    def locks(events):
        return [(e['t0'],e['handshake_key']) for e in events if e.get('handshake_key')]
    a,b = locks(reference),locks(other)
    votes = sorted((tb-ta,label,ta,tb) for ta,label in a for tb,other_label in b if label==other_label)
    def cluster(items):
        groups=[]
        for first in items:
            group=[v for v in items if first[0]<=v[0]<=first[0]+.08]
            # One physical lock gets at most one vote in each recording.
            if len({v[2] for v in group})==len(group)==len({v[3] for v in group}):
                groups.append(group)
        return max(groups,key=lambda g:(len(g),-(g[-1][0]-g[0][0])),default=[])
    best=cluster(votes)
    rival=cluster([v for v in votes if v not in best])
    if len(best)<3 or len(best)<=len(rival):
        return None, f'Handshake alignment inconclusive: {len(best)} anchors, {len(rival)} at a competing offset.'
    shift=float(np.median([v[0] for v in best]))
    return shift, (f'Aligned by {len(best)} matching named handshake locks; offset {shift:+.3f}s; '
                   f'anchor spread {(best[-1][0]-best[0][0])*1000:.0f} ms, competing offset has {len(rival)} anchors.')
