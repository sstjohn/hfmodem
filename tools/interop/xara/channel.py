# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json

import numpy as np
from scipy import fft, signal
from hfmodem.sabir.sim.watterson import ContinuousWatterson, PROFILES

RATE = 48000
PART = 480
REFERENCE = 2500
NOISE_RMS = .05


def subseed(seed, bw, caller, condition, replicate, direction, component):
    fields = ['xara-channel-v1', int(seed), int(bw), caller, condition,
              int(replicate), direction, component]
    return int.from_bytes(hashlib.sha256(json.dumps(fields, separators=(',', ':')).encode()).digest()[:16], 'big')


@dataclass(frozen=True)
class Condition:
    snr: str = 'clean'
    fading: str = 'none'
    offset: float = 0
    drift: float = 0
    clock_ppm: float = 0
    qrn: str = 'off'

    def __post_init__(self):
        if self.snr not in ('clean', 'inf') and not np.isfinite(float(self.snr)):
            raise ValueError('--snr must be clean, inf, or finite dB')
        if self.snr not in ('clean', 'inf'):
            object.__setattr__(self, 'snr', format(float(self.snr), 'g'))
            try:
                rms = self.rms
            except OverflowError as error:
                raise ValueError('SNR gain is outside floating-point range') from error
            if not np.isfinite(rms) or rms == 0:
                raise ValueError('SNR gain is outside floating-point range')
        if self.fading not in ('none', 'good', 'moderate', 'poor') or self.qrn not in ('off', 'standard'):
            raise ValueError('unsupported fading or QRN')
        if not all(np.isfinite(v) for v in (self.offset, self.drift, self.clock_ppm)) or abs(self.clock_ppm) > 50:
            raise ValueError('finite offsets and |clock-ppm| <= 50 required')
        if self.snr == 'clean' and (self.fading != 'none' or self.offset or self.drift or self.clock_ppm or self.qrn != 'off'):
            raise ValueError('clean rejects impairments; select --snr inf or finite dB')

    @property
    def rms(self):
        return .12 if self.snr in ('clean', 'inf') else NOISE_RMS * 10**(float(self.snr)/20)

    @property
    def identity(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))


def keyed(samples):
    x = np.asarray(samples, dtype=float)
    nz = np.flatnonzero(x)
    if not len(nz) or not np.all(np.isfinite(x)):
        raise ValueError('calibration needs finite nonzero keyed audio')
    lo, hi = int(nz[0]), int(nz[-1])+1
    return lo, hi, float(np.sqrt(np.mean(x[lo:hi]**2)))


def calibrate(samples, rms):
    lo, hi, native = keyed(samples)
    gain = rms/native
    return np.asarray(samples, float)*gain, dict(native_rms=native, calibrated_rms=rms,
        gain=gain, samples=hi-lo, first_nonzero=lo, last_nonzero=hi-1)


@lru_cache(maxsize=1)
def filters():
    image = signal.remez(1055, [0, 1450, 1575, RATE/2], [1, 0],
                         weight=[1, 5.76], fs=RATE, grid_density=32)
    order, critical = signal.ellipord([60, 3150], [20, 3500], .099, 60.01, fs=RATE)
    receive = signal.ellip(order, .099, 60.01, critical, btype='bandpass', fs=RATE, output='sos')
    def response(h, passed, stopped, sos=False):
        f, z = (signal.sosfreqz if sos else signal.freqz)(h, worN=262144, fs=RATE)
        p = np.abs(z)**2
        db = 10*np.log10(np.maximum(p, 1e-30))
        if sos:
            w = 2*np.pi*1500/RATE + np.array([-1e-6, 1e-6])
            delay = -np.diff(np.unwrap(np.angle(signal.sosfreqz(h, worN=w)[1])))[0]/2e-6
        else:
            delay = (len(h)-1)/2
        return dict(taps=int(h.size), sha256=hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
            kind='elliptic-sos' if sos else 'equiripple-fir',
            group_delay_samples=round(delay), group_delay_at_1500_samples=float(delay),
            ripple_db=float(np.ptp(db[passed(f)])), rejection_db=float(-np.max(db[stopped(f)])),
            enbw_hz=float(np.mean(p)*RATE/2),
            reference_gain=float(np.mean(p[(f >= 250) & (f <= 2750)])))
    meta = dict(image=response(image, lambda f: f <= 1450, lambda f: f >= 1575),
                receive=response(receive, lambda f: (f >= 60) & (f <= 3150),
                                 lambda f: (f <= 20) | (f >= 3500), True))
    meta['receive'].update(order=2*order, sections=len(receive))
    if any(m['ripple_db'] > .1 or m['rejection_db'] < 60 for m in meta.values()):
        raise ValueError('channel filter specification exceeded')
    return image, receive, meta


class FIR:
    def __init__(self, coefficients):
        self.h = np.asarray(coefficients)
        self.size = fft.next_fast_len(len(self.h)-1+PART)
        self.response = fft.fft(self.h, self.size)
        self.history = np.zeros(len(self.h)-1, complex)

    def process(self, samples):
        assert len(samples) == PART
        x = np.concatenate((self.history, samples))
        y = fft.ifft(fft.fft(x, self.size)*self.response)[len(self.h)-1:len(self.h)-1+PART]
        self.history = x[-len(self.history):].copy()
        return y


class IIR:
    def __init__(self, sos):
        self.sos = sos
        self.state = np.zeros((len(sos), 2))

    def process(self, samples):
        out, self.state = signal.sosfilt(self.sos, samples, zi=self.state)
        return out


class Fading:
    def __init__(self, profile, seed):
        self.delay = round(max(PROFILES[profile][0])*RATE/1000)
        self.history = np.zeros(self.delay, complex)
        self.model = ContinuousWatterson(profile, RATE, duration_s=3602, seed=seed)

    def process(self, samples, at):
        x = np.concatenate((self.history, samples))
        y = self.model(x, 1+(at-self.delay)/RATE)[self.delay:]
        self.history = x[-self.delay:].copy() if self.delay else x[:0]
        return y


class Clock:
    def __init__(self, ppm, horizon):
        if not 0 < horizon <= 3600 or abs(ppm) > 50:
            raise ValueError('clock supports |ppm| <= 50 and 0 < horizon <= 3600 s')
        self.ppm, self.horizon = ppm, horizon
        self.headroom = int(np.ceil(abs(ppm)*1e-6*horizon*RATE))+PART if ppm else 0
        self.delay = self.headroom+16
        self.capacity = 2*self.delay+4*PART+64
        self.ring = np.zeros(self.capacity)
        self.offsets = np.arange(-15, 17)
        self.window = np.kaiser(32, 8.6)

    def process(self, samples, at):
        n = at+np.arange(len(samples))
        self.ring[n % self.capacity] = samples
        u = n-self.delay+self.ppm*1e-6*n
        idx = np.floor(u).astype(np.int64)[:, None]+self.offsets
        if np.max(idx) > n[-1] or np.min(idx) < n[-1]-self.capacity+1:
            raise ValueError('clock history/causality exhausted')
        w = np.sinc(u[:, None]-idx)*self.window
        w /= w.sum(axis=1)[:, None]
        return np.sum(np.where(idx >= 0, self.ring[idx % self.capacity], 0)*w, axis=1)


class QRN:
    def __init__(self, seeds, power):
        self.arrival = np.random.default_rng(seeds['qrn-arrival'])
        self.duration = np.random.default_rng(seeds['qrn-duration'])
        self.amplitude = np.random.default_rng(seeds['qrn-amplitude'])
        self.next = self.arrival.exponential(RATE/20)
        self.tail = np.zeros(32)
        self.power, self.events, self.energy, self.raw_energy = power, 0, 0., 0.

    def process(self, at):
        out = np.zeros(PART+32)
        out[:32] += self.tail
        while self.next < at+PART:
            start = int(self.next)-at
            duration = max(1, int(np.exp(self.duration.uniform(np.log(.00005), np.log(.0003)))*RATE))
            slab = self.amplitude.standard_normal(duration)*np.sqrt(self.power)
            out[start:start+duration] += slab
            self.raw_energy += float(np.sum(slab*slab))
            self.events += 1
            self.next += self.arrival.exponential(RATE/20)
        self.tail = out[PART:].copy()
        return out[:PART]


class Channel:
    def __init__(self, condition=Condition(), *, seed=20261025, bw=2300, caller='kestrel',
                 replicate=0, direction='A->B', horizon=3600):
        self.condition, self.horizon = condition, horizon
        self.direction = direction
        if not 0 < horizon <= 3600:
            raise ValueError('channel horizon must be in (0, 3600] seconds')
        self.seeds = {name: subseed(seed, bw, caller, condition.identity, replicate, direction, name)
                      for name in ('fade', 'awgn', 'qrn-arrival', 'qrn-duration', 'qrn-amplitude')}
        self.cursor = self.at = 0
        self.pending = np.empty(0)
        self.output = np.zeros(PART if condition.snr != 'clean' else 0)
        self.padding = np.zeros(2400)
        self.noise_energy = self.signal_energy = self.qrn_energy = 0.
        self.noise_pending = np.empty(0)
        self.reference_energy = self.reference_samples = 0
        self.bursts = []
        self.samples = 0
        if condition.snr == 'clean':
            self.delay = 2400
            return
        image, receive, self.filter_info = filters()
        self.image, self.receive = FIR(image), IIR(receive)
        self.noise_filter, self.qrn_filter = IIR(receive), IIR(receive)
        sign = 1 if direction == 'A->B' else -1
        self.offset, self.drift = sign*condition.offset, sign*condition.drift
        self.clock = Clock(sign*condition.clock_ppm, horizon)
        self.fade = Fading(condition.fading, self.seeds['fade']) if condition.fading != 'none' else None
        gain = self.filter_info['receive']['reference_gain']
        self.variance = 0 if condition.snr == 'inf' else NOISE_RMS**2*RATE/(2*REFERENCE*gain)
        self.noise = np.random.default_rng(self.seeds['awgn'])
        self.qrn = QRN(self.seeds, condition.rms**2*10**2.5*RATE/(2*REFERENCE*gain)) if condition.qrn == 'standard' else None
        processing = PART+self.window(0, 0)[0]
        self.propagation = max(0, 2400-processing)
        self.padding = np.zeros(self.propagation)
        self.delay = self.propagation+processing

    def window(self, start, count, *, output=False):
        if self.condition.snr == 'clean':
            return start+2400, start+count+2400
        image = self.filter_info['image']['group_delay_samples']
        receive = self.filter_info['receive']['group_delay_samples']
        shift = receive+(self.propagation+PART if output else 0)
        scale = 1+self.clock.ppm*1e-6
        return tuple(int(np.ceil((n+image+self.clock.delay)/scale+shift))
                     for n in (start, start+count))

    def register(self, start, count, label):
        burst = dict(start=start, count=count, label=label, energy=0., samples=0)
        if count is None:
            burst['pending'] = []
        self.bursts.append(burst)
        return burst

    def finish(self, burst, count):
        burst['count'] = count
        for at, samples in burst.pop('pending'):
            self._measure(burst, samples, at)

    def _measure(self, burst, samples, at):
        if burst['count'] is None:
            burst['pending'].append((at, samples.copy()))
            return
        start, end = self.window(burst['start'], burst['count'])
        lo, hi = max(0, start-at), min(len(samples), end-at)
        if hi > lo:
            burst['energy'] += float(samples[lo:hi]@samples[lo:hi])
            burst['samples'] += hi-lo

    def _block(self, x):
        n = self.at+np.arange(PART)
        t = n/RATE
        base = self.image.process(2*x*np.exp(-2j*np.pi*1500*t))
        if self.fade:
            base = self.fade.process(base, self.at)
        phase = 2*np.pi*((1500+self.offset)*t+.5*self.drift*t*t)
        warped = self.clock.process((base*np.exp(1j*phase)).real, self.at)
        s = self.receive.process(warped).real
        noise = self.noise_filter.process(self.noise.standard_normal(PART)*np.sqrt(self.variance)).real
        qrn = self.qrn_filter.process(self.qrn.process(self.at) if self.qrn else np.zeros(PART)).real
        self.signal_energy += float(s@s)
        self.noise_energy += float(noise@noise)
        self.qrn_energy += float(qrn@qrn)
        self.noise_pending = np.concatenate((self.noise_pending, noise))
        if len(self.noise_pending) >= RATE:
            block = self.noise_pending[:RATE]
            spectrum = np.fft.rfft(block)
            self.reference_energy += float(2*np.sum(np.abs(spectrum[250:2750])**2)/RATE)
            self.reference_samples += RATE
            self.noise_pending = self.noise_pending[RATE:]
        for burst in self.bursts:
            self._measure(burst, s, self.at)
        self.samples += PART
        self.at += PART
        out = np.concatenate((self.padding, s+noise+qrn))
        self.padding = out[PART:]
        return out[:PART]

    def process(self, samples, absolute_sample):
        x = np.asarray(samples, dtype=float)
        if absolute_sample != self.cursor or x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError('channel requires contiguous finite mono samples')
        if self.cursor+len(x) > self.horizon*RATE+PART:
            raise ValueError('channel horizon exhausted')
        self.cursor += len(x)
        if self.condition.snr == 'clean':
            out = np.concatenate((self.padding, x))
            self.padding = out[len(x):]
            return out[:len(x)]
        self.pending = np.concatenate((self.pending, x))
        blocks = [self.output]
        used = len(self.pending)//PART*PART
        for start in range(0, used, PART):
            blocks.append(self._block(self.pending[start:start+PART]))
        self.pending = self.pending[used:].copy()
        ready = np.concatenate(blocks)
        out, self.output = ready[:len(x)], ready[len(x):]
        return out

    def next_block(self):
        if self.cursor % PART or len(self.pending):
            raise ValueError('scheduler preview requires partition alignment')
        # Both causal paths retain at least one partition ahead of their input.
        return (self.padding if self.condition.snr == 'clean' else self.output)[:PART].copy()

    def evidence(self):
        result = dict(condition=asdict(self.condition), seeds=self.seeds, rate=RATE,
            direction=self.direction,
            partition_samples=PART, delay_samples=self.delay, horizon_seconds=self.horizon,
            signal_rms=self.condition.rms, reference_hz=REFERENCE, native_xara_drive_db=-5)
        if self.condition.snr != 'clean':
            result.update(filters={name: {k: m[k] for k in ('taps', 'group_delay_samples', 'enbw_hz', 'sha256')}
                                   for name, m in self.filter_info.items()},
                clock_headroom_samples=self.clock.headroom,
                propagation_samples=self.propagation,
                interpolation_taps=32, interpolation_margin_samples=16, interpolation_group_delay_samples=0,
                effective_offset_hz=self.offset, effective_drift_hz_per_second=self.drift,
                effective_clock_ppm=self.clock.ppm, clock_audio_shift_hz_per_hz=self.clock.ppm*1e-6,
                raw_noise_variance=self.variance, noise_density=self.variance*2/RATE,
                reference_noise_power=self.variance*2/RATE*REFERENCE*self.filter_info['receive']['reference_gain'],
                measured_reference_noise_power=self.reference_energy/max(1, self.reference_samples),
                measured_total_noise_power=self.noise_energy/max(1, self.samples),
                realized_signal_energy=self.signal_energy, realized_qrn_energy=self.qrn_energy,
                qrn_events=self.qrn.events if self.qrn else 0,
                qrn_slab_reference_power=self.qrn.power*2/RATE*REFERENCE*self.filter_info['receive']['reference_gain'] if self.qrn else 0)
            result['bursts'] = [dict({k: v for k, v in b.items() if k != 'pending'}, realized_keyed_power=b['energy']/max(1, b['samples']),
                realized_snr2500_db=10*np.log10(max(1e-300, b['energy']/max(1, b['samples']))/(NOISE_RMS**2))
                if self.variance else None) for b in self.bursts]
        return result
