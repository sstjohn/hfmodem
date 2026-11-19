// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// ABI v2: all buffers are caller-owned, aligned, contiguous and disjoint.
// No borrowed pointer survives the call.
use std::{
    mem::{align_of, size_of},
    panic::{AssertUnwindSafe, catch_unwind},
    slice,
    time::Instant,
};

// Shared with p4longdecode.TIE.
const TIE: f64 = 1e-9;
const WIDTH: usize = 20700;
const LEVELS: usize = 9;
const TIERS: usize = 10;
const FIELDS: usize = 3;
const STATE_WORDS: usize = 6;
const TIER: usize = 0;
const DONE: usize = 1;
const COMPONENT_CALLS: usize = 2;
const TRELLIS_POSITIONS: usize = 3;
const MEMBER_INDEX: usize = 4;
const SUPPORTED_HIT: usize = 5;
const TIMINGS_PER_TIER: usize = 2;
const WHITENING_BYTES: usize = 2587;
const MAX_MEMBERS: usize = 4096;
const LENGTHS: [usize; LEVELS] = [348, 580, 1160, 1656, 4140, 8280, 12420, 16560, 20700];
const BYTES: [usize; LEVELS] = [40, 69, 141, 203, 514, 1031, 1549, 2066, 2584];

fn geometry(g: u32) -> (usize, usize) {
    match g {
        2 => (2, 1),
        5 => (1, 0),
        _ => (10, 4),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffer<T> {
    data: *mut T,
    len: usize,
}
#[repr(C)]
pub struct Request {
    members: usize,
    llr_width: usize,
    state_width: usize,
    last_tier: u64,
    live: u64,
    budget: u64,
    deadline_ns: u64,
    llrs: Buffer<f64>,
    geometries: Buffer<u32>,
    orders: Buffer<u32>,
    whitening: Buffer<u8>,
    priors: Buffer<f64>,
    first: Buffer<f64>,
    state: Buffer<u64>,
    bits: Buffer<u8>,
    flags: Buffer<u8>,
    timings: Buffer<f64>,
}

#[inline(always)]
fn next(s: usize, x: usize) -> usize {
    ((x ^ ((s >> 1) & 1) ^ (s & 1)) << 2) | (s >> 1)
}
#[inline(always)]
fn parity(s: usize, x: usize) -> usize {
    x ^ ((s >> 1) & 1) ^ ((s >> 2) & 1)
}
fn probabilities(llr: f64) -> [f64; 2] {
    // Punctured parity supplies exact zero at most positions of SL6--10.
    // This is exactly exp(0)/(1+exp(0)), without a transcendental call.
    if llr == 0. {
        return [0.5; 2];
    }
    let tail = (-llr.abs()).exp();
    let hi = 1. / (1. + tail);
    let lo = tail / (1. + tail);
    if llr >= 0. { [hi, lo] } else { [lo, hi] }
}

struct Workspace {
    gamma: Vec<[[f64; 2]; 8]>,
    pp: Vec<[f64; 2]>,
    a: Vec<[f64; 8]>,
    b: Vec<[f64; 8]>,
    local: Vec<f64>,
    par: Vec<f64>,
    ext: Vec<f64>,
    hard: Vec<u8>,
}
impl Workspace {
    fn new(n: usize) -> Self {
        Self {
            gamma: vec![[[0.; 2]; 8]; n],
            pp: vec![[0.; 2]; n],
            a: vec![[0.; 8]; n + 1],
            b: vec![[0.; 8]; n + 1],
            local: vec![0.; n],
            par: vec![0.; n],
            ext: vec![0.; n],
            hard: vec![0; n],
        }
    }
    fn component(&mut self, n: usize, terminated: bool) {
        let gamma = &mut self.gamma[..n];
        let pp = &mut self.pp[..n];
        let a = &mut self.a[..n + 1];
        let b = &mut self.b[..n + 1];
        let local_llrs = &mut self.local[..n];
        let par = &mut self.par[..n];
        let ext = &mut self.ext[..n];
        let hard = &mut self.hard[..n];
        for t in 0..n {
            local_llrs[t] = local_llrs[t].clamp(-40., 40.);
            let ip = probabilities(local_llrs[t]);
            let probabilities = probabilities(par[t].clamp(-40., 40.));
            pp[t] = probabilities;
            let even = [ip[0] * probabilities[0], ip[1] * probabilities[1]];
            let odd = [ip[0] * probabilities[1], ip[1] * probabilities[0]];
            gamma[t] = [even, even, odd, odd, odd, odd, even, even];
        }
        a[0] = [0.; 8];
        a[0][0] = 1.;
        b[n] = if terminated {
            [1., 0., 0., 0., 0., 0., 0., 0.]
        } else {
            [0.125; 8]
        };
        for t in 0..n {
            let r = n - 1 - t;
            let previous = a[t];
            let following = b[r + 1];
            let forward = gamma[t];
            let backward = gamma[r];
            let mut anext = [0.; 8];
            let mut bnext = [0.; 8];
            for s in 0..8 {
                let p0 = (s & 3) << 1;
                let p1 = p0 | 1;
                let x0 = ((s >> 2) & 1) ^ (s & 1);
                anext[s] = previous[p0] * forward[p0][x0] + previous[p1] * forward[p1][x0 ^ 1];
                bnext[s] =
                    following[next(s, 0)] * backward[s][0] + following[next(s, 1)] * backward[s][1];
            }
            let an: f64 = anext.iter().sum();
            let bn: f64 = bnext.iter().sum();
            for value in &mut anext {
                *value /= an;
            }
            for value in &mut bnext {
                *value /= bn;
            }
            a[t + 1] = anext;
            b[r] = bnext;
        }
        for t in 0..n {
            let alpha = a[t];
            let beta = b[t + 1];
            let probabilities = pp[t];
            let mut marginal = [0.; 2];
            for s in 0..8 {
                for (x, value) in marginal.iter_mut().enumerate() {
                    *value += alpha[s] * probabilities[parity(s, x)] * beta[next(s, x)];
                }
            }
            ext[t] = marginal[0].ln() - marginal[1].ln();
            hard[t] = (ext[t] + local_llrs[t] < -TIE) as u8;
        }
    }
}

#[cfg(unix)]
fn monotonic_ns() -> u64 {
    use std::os::raw::{c_int, c_long};
    #[repr(C)]
    struct Timespec {
        sec: c_long,
        nsec: c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: c_int, ts: *mut Timespec) -> c_int;
    }
    #[cfg(target_os = "macos")]
    const CLOCK: c_int = 8; // CLOCK_UPTIME_RAW: mach_absolute_time, the clock time.monotonic reads
    #[cfg(not(target_os = "macos"))]
    const CLOCK: c_int = 1;
    let mut ts = Timespec { sec: 0, nsec: 0 };
    if unsafe { clock_gettime(CLOCK, &mut ts) } != 0 {
        return u64::MAX;
    }
    (ts.sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.nsec as u64)
}
#[cfg(windows)]
fn monotonic_ns() -> u64 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn QueryPerformanceCounter(value: *mut i64) -> i32;
        fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    }
    let (mut ticks, mut frequency) = (0, 0);
    unsafe {
        if QueryPerformanceCounter(&mut ticks) == 0
            || QueryPerformanceFrequency(&mut frequency) == 0
        {
            return u64::MAX;
        }
    }
    ((ticks as u128 * 1_000_000_000) / frequency as u128) as u64
}

fn region(
    buffer: usize,
    count: usize,
    item: usize,
    alignment: usize,
) -> Result<(usize, usize), i32> {
    let bytes = count.checked_mul(item).ok_or(-2)?;
    if buffer == 0 || buffer % alignment != 0 || bytes > isize::MAX as usize {
        return Err(-2);
    }
    Ok((buffer, buffer.checked_add(bytes).ok_or(-2)?))
}
fn buffer<T>(b: Buffer<T>, expected: usize, regions: &mut Vec<(usize, usize)>) -> Result<(), i32> {
    if b.len != expected {
        return Err(-2);
    }
    regions.push(region(
        b.data as usize,
        b.len,
        size_of::<T>(),
        align_of::<T>(),
    )?);
    Ok(())
}
unsafe fn read<'a, T>(b: Buffer<T>) -> &'a [T] {
    unsafe { slice::from_raw_parts(b.data, b.len) }
}
unsafe fn write<'a, T>(b: Buffer<T>) -> &'a mut [T] {
    unsafe { slice::from_raw_parts_mut(b.data, b.len) }
}

fn crc_valid(bits: &[u8], bytes: usize, whitening: &[u8]) -> bool {
    let mut crc = 0xffffu16;
    for j in 0..bytes {
        let mut value = whitening[j];
        for k in 0..8 {
            value ^= bits[8 * j + k] << k;
        }
        crc ^= value as u16;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x8408 } else { 0 };
        }
    }
    crc == 0xf0b8
}

// The stop prefilter mirrors positive re-encoding support. Python independently
// checks each CRC hit and alone owns acceptance and identity collapse.
fn supported(bits: &[u8], fields: &[f64], order: &[u32], period: usize, phase: usize) -> bool {
    let n = bits.len();
    let p = n / period;
    let mut scores = [0.; 3];
    let mut totals = [0.; 3];
    let mut scales = [0f64; 3];
    for (r, values) in [&fields[..n], &fields[n..n + p], &fields[n + p..]]
        .iter()
        .enumerate()
    {
        for v in *values {
            scales[r] = scales[r].max(v.abs());
        }
        if scales[r] == 0. {
            return false;
        }
        for v in *values {
            totals[r] += (v / scales[r]).abs();
        }
    }
    let (mut s1, mut s2) = (0, 0);
    for t in 0..n {
        let x = bits[t] as usize;
        scores[0] += fields[t] / scales[0] * (1. - 2. * x as f64);
        if t % period == 0 {
            scores[1] += fields[n + t / period] / scales[1] * (1. - 2. * parity(s1, x) as f64);
        }
        s1 = next(s1, x);
        let y = bits[order[t] as usize] as usize;
        if t % period == phase {
            scores[2] += fields[n + p + t / period] / scales[2] * (1. - 2. * parity(s2, y) as f64);
        }
        s2 = next(s2, y);
    }
    scores
        .iter()
        .zip(totals)
        .all(|(score, total)| score / total > 0.)
}

unsafe fn advance(r: &Request, address: usize) -> Result<(), i32> {
    let (m, lw, w) = (r.members, r.llr_width, r.state_width);
    if m == 0
        || m > MAX_MEMBERS
        || w == 0
        || w > WIDTH
        || lw == 0
        || lw > WIDTH * FIELDS
        || r.last_tier >= TIERS as u64
        || r.live > 1
    {
        return Err(-2);
    }
    let mut regions = vec![region(
        address,
        1,
        size_of::<Request>(),
        align_of::<Request>(),
    )?];
    buffer(r.llrs, m * lw, &mut regions)?;
    buffer(r.geometries, m, &mut regions)?;
    buffer(r.orders, LEVELS * WIDTH, &mut regions)?;
    buffer(r.whitening, WHITENING_BYTES, &mut regions)?;
    buffer(r.priors, m * w, &mut regions)?;
    buffer(r.first, m * w, &mut regions)?;
    buffer(r.state, STATE_WORDS, &mut regions)?;
    buffer(r.bits, TIERS * m * w, &mut regions)?;
    buffer(r.flags, TIERS * m * FIELDS, &mut regions)?;
    buffer(r.timings, TIERS * TIMINGS_PER_TIER, &mut regions)?;
    regions.sort_unstable();
    if regions.windows(2).any(|a| a[0].1 > a[1].0) {
        return Err(-2);
    }
    // No array is read until every buffer's extent/alignment/aliasing is checked.
    let gs = unsafe { read(r.geometries) };
    let orders = unsafe { read(r.orders) };
    let mut validated = [false; LEVELS];
    for &g in gs {
        if !(2..=10).contains(&g) {
            return Err(-2);
        }
        let k = g as usize - 2;
        let n = LENGTHS[k];
        let (period, _) = geometry(g);
        if n > w || n + 2 * (n / period) > lw {
            return Err(-2);
        }
        if validated[k] {
            continue;
        }
        let mut seen = vec![false; n];
        for &j in &orders[k * WIDTH..k * WIDTH + n] {
            if j as usize >= n || seen[j as usize] {
                return Err(-2);
            }
            seen[j as usize] = true;
        }
        validated[k] = true;
    }
    let fields = unsafe { read(r.llrs) };
    if fields.iter().any(|x| !x.is_finite())
        || unsafe { read(r.priors) }
            .iter()
            .chain(unsafe { read(r.first) })
            .any(|x| !x.is_finite())
    {
        return Err(-2);
    }
    let input_state = unsafe { read(r.state) };
    if input_state[TIER] > TIERS as u64
        || input_state[DONE] > 1
        || input_state[COMPONENT_CALLS] > 16 * m as u64
        || input_state[TRELLIS_POSITIONS] > 16 * (m * w) as u64
        || input_state[MEMBER_INDEX] >= m as u64
        || input_state[SUPPORTED_HIT] > 1
    {
        return Err(-2);
    }
    let whitening = unsafe { read(r.whitening) };
    let priors = unsafe { write(r.priors) };
    let first = unsafe { write(r.first) };
    let state = unsafe { write(r.state) };
    let bits = unsafe { write(r.bits) };
    let flags = unsafe { write(r.flags) };
    let timings = unsafe { write(r.timings) };
    let mut work = Workspace::new(w);
    while state[DONE] == 0 && state[TIER] <= r.last_tier {
        let tier = state[TIER] as usize;
        let calls = if tier == 0 {
            0
        } else if tier <= 2 {
            1
        } else {
            2
        };
        while (state[MEMBER_INDEX] as usize) < m {
            if (r.deadline_ns != 0 && monotonic_ns() >= r.deadline_ns)
                || state[COMPONENT_CALLS] + calls as u64 > r.budget
            {
                return Ok(());
            }
            let started = Instant::now();
            let i = state[MEMBER_INDEX] as usize;
            let g = gs[i];
            let k = g as usize - 2;
            let n = LENGTHS[k];
            let (period, phase) = geometry(g);
            let p = n / period;
            let f = &fields[i * lw..i * lw + n + 2 * p];
            let order = &orders[k * WIDTH..k * WIDTH + n];
            let prior = &mut priors[i * w..i * w + n];
            let ex = &mut first[i * w..i * w + n];
            let hard = &mut bits[(tier * m + i) * w..(tier * m + i) * w + n];
            if tier == 0 {
                for t in 0..n {
                    hard[t] = (f[t] < -TIE) as u8;
                }
            }
            if tier != 0 && tier != 2 {
                work.par[..n].fill(0.);
                for (t, value) in f[n..n + p].iter().enumerate() {
                    work.par[t * period] = *value;
                }
                for ((value, field), message) in
                    work.local[..n].iter_mut().zip(&f[..n]).zip(prior.iter())
                {
                    *value = field + message;
                }
                work.component(n, true);
                ex.copy_from_slice(&work.ext[..n]);
                state[COMPONENT_CALLS] += 1;
                state[TRELLIS_POSITIONS] += n as u64;
                if tier == 1 {
                    hard.copy_from_slice(&work.hard[..n]);
                }
            }
            if tier >= 2 {
                work.par[..n].fill(0.);
                for (t, value) in f[n + p..n + 2 * p].iter().enumerate() {
                    work.par[phase + t * period] = *value;
                }
                for (value, &j) in work.local[..n].iter_mut().zip(order) {
                    let j = j as usize;
                    // The shared permutation was validated against n above.
                    *value = unsafe { *f.get_unchecked(j) + *ex.get_unchecked(j) };
                }
                work.component(n, false);
                for ((&j, &message), &bit) in order.iter().zip(&work.ext[..n]).zip(&work.hard[..n])
                {
                    let j = j as usize;
                    // Unique in-range indices also keep these writes disjoint.
                    unsafe {
                        *prior.get_unchecked_mut(j) = message;
                        *hard.get_unchecked_mut(j) = bit;
                    }
                }
                state[COMPONENT_CALLS] += 1;
                state[TRELLIS_POSITIONS] += n as u64;
            }
            timings[tier * TIMINGS_PER_TIER] += started.elapsed().as_secs_f64();
            let checked = Instant::now();
            let crc = crc_valid(hard, BYTES[k] + 3, whitening);
            let pad = hard[8 * (BYTES[k] + 3)..n - 3].iter().all(|b| *b == 0);
            let flush = crc && hard.iter().fold(0, |s, x| next(s, *x as usize)) == 0;
            flags[(tier * m + i) * FIELDS..(tier * m + i) * FIELDS + FIELDS].copy_from_slice(&[
                crc as u8,
                pad as u8,
                flush as u8,
            ]);
            if crc && pad && flush {
                state[SUPPORTED_HIT] |= supported(hard, f, order, period, phase) as u64;
            }
            timings[tier * TIMINGS_PER_TIER + 1] += checked.elapsed().as_secs_f64();
            state[MEMBER_INDEX] += 1;
        }
        state[MEMBER_INDEX] = 0;
        state[TIER] += 1;
        state[DONE] =
            ((r.live != 0 && state[SUPPORTED_HIT] != 0) || state[TIER] > r.last_tier) as u64;
        state[SUPPORTED_HIT] = 0;
    }
    Ok(())
}

/// Query with (0, null, 0) returns ABI 2. Otherwise returns 0 on success,
/// -1 for ABI mismatch, -2 for invalid buffers/state, -3 for a caught panic.
///
/// # Safety
/// Except for the ABI query, `request` must point to an initialized `Request`.
/// Its buffers must be valid allocations matching their declared extents and
/// remain accessible for the call, with writable buffers exclusively borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn p4long_advance_v2(abi: u32, request: *const Request, bytes: usize) -> i32 {
    if abi == 0 && request.is_null() && bytes == 0 {
        return 2;
    }
    if abi != 2 || bytes != size_of::<Request>() {
        return -1;
    }
    if request.is_null() || (request as usize) % align_of::<Request>() != 0 {
        return -2;
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        advance(&*request, request as usize)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(code)) => code,
        Err(_) => -3,
    }
}
