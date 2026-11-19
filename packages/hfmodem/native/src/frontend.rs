// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::Complex;

fn buffer<T: Element>(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<PyBuffer<T>> {
    let buf = PyBuffer::get(obj).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let prefix = buf.format().to_bytes().first().copied();
    let foreign_endian = match prefix {
        Some(b'>') | Some(b'!') => cfg!(target_endian = "little"),
        Some(b'<') => cfg!(target_endian = "big"),
        _ => false,
    };
    if foreign_endian || buf.dimensions() != ndim || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "expected a C-contiguous buffer of the required rank",
        ));
    }
    Ok(buf)
}

fn overlap<A: Element, B: Element>(a: &PyBuffer<A>, b: &PyBuffer<B>) -> bool {
    let ap = a.buf_ptr() as usize;
    let bp = b.buf_ptr() as usize;
    a.len_bytes() != 0
        && b.len_bytes() != 0
        && ap < bp.saturating_add(b.len_bytes())
        && bp < ap.saturating_add(a.len_bytes())
}

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

// SAFETY: callers validate format, alignment, rank and contiguity. Buffer
// exports live across detach; Python callers must not mutate inputs or access
// outputs during a call, as with NumPy's own GIL-free numerical loops.
unsafe fn input<T: Element>(buf: &PyBuffer<T>) -> &[T] {
    unsafe { buf.as_slice_ptr().unwrap().as_ref() }
}

unsafe fn output<T: Element>(buf: &PyBuffer<T>) -> &mut [T] {
    unsafe { buf.as_slice_ptr().unwrap().as_mut() }
}

#[derive(Clone, Copy)]
enum Magnitude {
    Hypot,
    Fused,
    Plain,
}

impl Magnitude {
    fn parse(mode: &str) -> PyResult<Self> {
        match mode {
            "hypot" => Ok(Self::Hypot),
            "fused" => Ok(Self::Fused),
            "plain" => Ok(Self::Plain),
            _ => Err(invalid("unknown complex magnitude mode")),
        }
    }

    fn apply(self, re: f64, im: f64) -> f64 {
        match self {
            Self::Hypot => re.hypot(im),
            Self::Fused | Self::Plain => {
                let large = re.abs().max(im.abs());
                let small = re.abs().min(im.abs());
                let r = if large == 0.0 { 0.0 } else { small / large };
                let square = match self {
                    Self::Fused => r.mul_add(r, 1.0),
                    _ => r * r + 1.0,
                };
                large * square.sqrt()
            }
        }
    }
}

fn sliding(
    audio: &[f64],
    n: usize,
    oscillator: &[Complex],
    out: &mut [f64],
    mode: Magnitude,
) -> bool {
    let mut prefix = vec![[0.0, 0.0]; n];
    let mut acc = [0.0, 0.0];
    let mut phase = 0;
    let mut slot = 0;
    for (i, &x) in audio.iter().enumerate() {
        let [c, d] = oscillator[phase].0;
        phase += 1;
        if phase == oscillator.len() {
            phase = 0;
        }
        let re = x * c - 0.0 * d;
        let im = x * d + 0.0 * c;
        if i == 0 {
            acc = [re, im];
        } else {
            acc[0] += re;
            acc[1] += im;
        }
        if !re.is_finite() || !im.is_finite() || !acc[0].is_finite() || !acc[1].is_finite() {
            return false;
        }
        slot += 1;
        if slot == n {
            slot = 0;
        }
        if i + 1 >= n {
            let re = acc[0] - prefix[slot][0];
            let im = acc[1] - prefix[slot][1];
            let magnitude = mode.apply(re, im);
            if !re.is_finite() || !im.is_finite() || !magnitude.is_finite() {
                return false;
            }
            out[i + 1 - n] = magnitude;
        }
        prefix[slot] = acc;
    }
    true
}

#[pyfunction]
pub fn sliding_magnitudes(
    py: Python<'_>,
    audio: &Bound<'_, PyAny>,
    n: usize,
    oscillators: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<bool> {
    let audio = buffer::<f64>(audio, 1)?;
    let out = buffer::<f64>(out, 2)?;
    let oscillators = oscillators
        .try_iter()?
        .map(|x| buffer::<Complex>(&x?, 1))
        .collect::<PyResult<Vec<_>>>()?;
    let mode = Magnitude::parse(mode)?;
    if n == 0
        || n > audio.item_count()
        || out.readonly()
        || out.shape() != [oscillators.len(), audio.item_count() - n + 1]
        || oscillators.iter().any(|x| x.item_count() == 0)
    {
        return Err(invalid("inconsistent window, oscillator or output shape"));
    }
    if overlap(&audio, &out) || oscillators.iter().any(|x| overlap(x, &out)) {
        return Err(invalid("out must not overlap any input buffer"));
    }
    Ok(py.detach(|| {
        let audio = unsafe { input(&audio) };
        if !audio.iter().all(|x| x.is_finite()) {
            return false;
        }
        let width = audio.len() - n + 1;
        let out = unsafe { output(&out) };
        for (oscillator, row) in oscillators.iter().zip(out.chunks_mut(width)) {
            let oscillator = unsafe { input(oscillator) };
            if !oscillator
                .iter()
                .all(|z| z.0[0].is_finite() && z.0[1].is_finite())
                || !sliding(audio, n, oscillator, row, mode)
            {
                return false;
            }
        }
        true
    }))
}

fn check_starts(starts: &[i64], len: usize, sps: usize, slots: usize) -> PyResult<()> {
    if sps == 0 || slots == 0 {
        return Err(invalid("stride and slot count must be positive"));
    }
    let span = (slots - 1)
        .checked_mul(sps)
        .ok_or_else(|| invalid("index overflow"))?;
    for &s in starts {
        if s < 0
            || usize::try_from(s)
                .ok()
                .and_then(|s| s.checked_add(span))
                .is_none_or(|end| end >= len)
        {
            return Err(invalid("candidate indices out of range"));
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gather_frames(
    py: Python<'_>,
    mark: &Bound<'_, PyAny>,
    space: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    sps: usize,
    nbytes: usize,
    skips: Vec<usize>,
    frames: &Bound<'_, PyAny>,
    ok: &Bound<'_, PyAny>,
    polynomial: u16,
    init: u16,
    xorout: u16,
    reflected: bool,
) -> PyResult<bool> {
    let mark = buffer::<f64>(mark, 1)?;
    let space = buffer::<f64>(space, 1)?;
    let starts = buffer::<i64>(starts, 1)?;
    let frames = buffer::<u8>(frames, 2)?;
    let ok = buffer::<u8>(ok, 3)?;
    let k = starts.item_count();
    if mark.shape() != space.shape()
        || nbytes < 2
        || skips.iter().any(|&s| s > nbytes - 2)
        || frames.shape() != [k, nbytes]
        || ok.shape() != [skips.len(), 2, k]
        || frames.readonly()
        || ok.readonly()
    {
        return Err(invalid("inconsistent frame, tone or CRC shapes"));
    }
    if overlap(&frames, &ok)
        || overlap(&frames, &mark)
        || overlap(&frames, &space)
        || overlap(&frames, &starts)
        || overlap(&ok, &mark)
        || overlap(&ok, &space)
        || overlap(&ok, &starts)
    {
        return Err(invalid("outputs must not overlap inputs or each other"));
    }
    check_starts(
        unsafe { input(&starts) },
        mark.item_count(),
        sps,
        nbytes
            .checked_mul(8)
            .ok_or_else(|| invalid("frame width overflow"))?,
    )?;
    Ok(py.detach(|| {
        let mark = unsafe { input(&mark) };
        let space = unsafe { input(&space) };
        if !mark.iter().chain(space).all(|x| x.is_finite()) {
            return false;
        }
        let starts = unsafe { input(&starts) };
        let frames = unsafe { output(&frames) };
        let ok = unsafe { output(&ok) };
        let mut table = [0u16; 256];
        for (byte, slot) in table.iter_mut().enumerate() {
            let mut reg = if reflected {
                byte as u16
            } else {
                (byte as u16) << 8
            };
            for _ in 0..8 {
                reg = if reflected {
                    (reg >> 1) ^ if reg & 1 != 0 { polynomial } else { 0 }
                } else {
                    (reg << 1) ^ if reg & 0x8000 != 0 { polynomial } else { 0 }
                };
            }
            *slot = reg;
        }
        for (i, (&start, row)) in starts.iter().zip(frames.chunks_mut(nbytes)).enumerate() {
            for (b, byte) in row.iter_mut().enumerate() {
                *byte = 0;
                for bit in 0..8 {
                    let j = start as usize + (b * 8 + bit) * sps;
                    *byte |= u8::from(mark[j] > space[j]) << bit;
                }
            }
            for (s, &skip) in skips.iter().enumerate() {
                for polarity in 0..2 {
                    let mask = if polarity == 0 { 0 } else { 0xff };
                    let mut reg = if reflected { init.reverse_bits() } else { init };
                    for &byte in &row[skip..nbytes - 2] {
                        let byte = byte ^ mask;
                        reg = if reflected {
                            (reg >> 8) ^ table[((reg ^ byte as u16) & 0xff) as usize]
                        } else {
                            (reg << 8) ^ table[((reg >> 8) ^ byte as u16) as usize]
                        };
                    }
                    reg ^= xorout;
                    let trailer =
                        u16::from_le_bytes([row[nbytes - 2] ^ mask, row[nbytes - 1] ^ mask]);
                    ok[(s * 2 + polarity) * k + i] = u8::from(reg == trailer);
                }
            }
        }
        true
    }))
}

fn pairwise(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut result = -0.0;
        for &v in values {
            result += v;
        }
        result
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&values[..8]);
        let end = n - n % 8;
        for block in values[8..end].chunks_exact(8) {
            for j in 0..8 {
                r[j] += block[j];
            }
        }
        let mut result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[end..] {
            result += v;
        }
        result
    } else {
        let half = (n / 2) & !7;
        pairwise(&values[..half]) + pairwise(&values[half..])
    }
}

fn position(
    mark: &[f64],
    space: &[f64],
    env: &[f64],
    starts: &[i64],
    sps: usize,
    nbits: usize,
    tail: usize,
) -> (bool, usize) {
    let mut eye = vec![0.0; nbits];
    let mut tails = vec![0.0; tail];
    let mut best = 0;
    let mut maximum = f64::NEG_INFINITY;
    for (i, &start) in starts.iter().enumerate() {
        for (j, value) in eye.iter_mut().enumerate() {
            let p = start as usize + j * sps;
            *value = (mark[p] - space[p]).abs();
        }
        for (j, value) in tails.iter_mut().enumerate() {
            *value = env[start as usize + (nbits + j) * sps];
        }
        let score = (0.0 + pairwise(&eye)) - (0.0 + pairwise(&tails));
        if !score.is_finite() {
            return (false, 0);
        }
        if score > maximum {
            maximum = score;
            best = i;
        }
    }
    (true, best)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn eye_argmax(
    py: Python<'_>,
    mark: &Bound<'_, PyAny>,
    space: &Bound<'_, PyAny>,
    env: &Bound<'_, PyAny>,
    starts: &Bound<'_, PyAny>,
    sps: usize,
    nbits: usize,
    tail: usize,
) -> PyResult<(bool, usize)> {
    let mark = buffer::<f64>(mark, 1)?;
    let space = buffer::<f64>(space, 1)?;
    let env = buffer::<f64>(env, 1)?;
    let starts = buffer::<i64>(starts, 1)?;
    if mark.shape() != space.shape() || mark.shape() != env.shape() || starts.item_count() == 0 {
        return Err(invalid("inconsistent tone shapes or empty candidates"));
    }
    check_starts(
        unsafe { input(&starts) },
        mark.item_count(),
        sps,
        nbits
            .checked_add(tail)
            .ok_or_else(|| invalid("slot count overflow"))?,
    )?;
    Ok(py.detach(|| {
        let mark = unsafe { input(&mark) };
        let space = unsafe { input(&space) };
        let env = unsafe { input(&env) };
        if !mark.iter().chain(space).chain(env).all(|x| x.is_finite()) {
            return (false, 0);
        }
        position(
            mark,
            space,
            env,
            unsafe { input(&starts) },
            sps,
            nbits,
            tail,
        )
    }))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn slice_eyes(
    py: Python<'_>,
    audio: &Bound<'_, PyAny>,
    bounds: &Bound<'_, PyAny>,
    n: usize,
    oscillators: &Bound<'_, PyAny>,
    nbits: usize,
    tail: usize,
    bits: &Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<bool> {
    let audio = buffer::<f64>(audio, 1)?;
    let bounds = buffer::<i64>(bounds, 2)?;
    let bits = buffer::<u8>(bits, 2)?;
    let oscillators = oscillators
        .try_iter()?
        .map(|x| buffer::<Complex>(&x?, 1))
        .collect::<PyResult<Vec<_>>>()?;
    let mode = Magnitude::parse(mode)?;
    if n == 0
        || nbits == 0
        || bounds.shape()[1] != 2
        || oscillators.len() != 2
        || oscillators.iter().any(|x| x.item_count() == 0)
        || bits.shape() != [bounds.shape()[0], nbits]
        || bits.readonly()
    {
        return Err(invalid("inconsistent slice, tone or bit shapes"));
    }
    if overlap(&bits, &audio)
        || overlap(&bits, &bounds)
        || oscillators.iter().any(|x| overlap(&bits, x))
    {
        return Err(invalid("bits must not overlap any input buffer"));
    }
    let pad = tail
        .checked_mul(n)
        .ok_or_else(|| invalid("padding overflow"))?;
    let minimum = nbits
        .checked_mul(n)
        .ok_or_else(|| invalid("bit span overflow"))?;
    for pair in unsafe { input(&bounds) }.chunks_exact(2) {
        if pair[0] < 0
            || pair[1] < pair[0]
            || pair[1] as u64 > audio.item_count() as u64
            || ((pair[1] - pair[0]) as u64) < minimum as u64
        {
            return Err(invalid("slice bounds out of range or too short"));
        }
        ((pair[1] - pair[0]) as usize)
            .checked_add(pad)
            .ok_or_else(|| invalid("padded slice overflow"))?;
    }
    Ok(py.detach(|| {
        let audio = unsafe { input(&audio) };
        let oscillators: Vec<_> = oscillators.iter().map(|x| unsafe { input(x) }).collect();
        if !audio.iter().all(|x| x.is_finite())
            || !oscillators
                .iter()
                .all(|x| x.iter().all(|z| z.0[0].is_finite() && z.0[1].is_finite()))
        {
            return false;
        }
        let bits = unsafe { output(&bits) };
        for (pair, row) in unsafe { input(&bounds) }
            .chunks_exact(2)
            .zip(bits.chunks_mut(nbits))
        {
            let slice = &audio[pair[0] as usize..pair[1] as usize];
            let mut padded = Vec::with_capacity(slice.len() + pad);
            padded.extend_from_slice(slice);
            padded.resize(slice.len() + pad, 0.0);
            let len = padded.len() - n + 1;
            let mut mark = vec![0.0; len];
            let mut space = vec![0.0; len];
            if !sliding(&padded, n, oscillators[0], &mut mark, mode)
                || !sliding(&padded, n, oscillators[1], &mut space, mode)
            {
                return false;
            }
            let env: Vec<_> = mark.iter().zip(&space).map(|(m, s)| m + s).collect();
            if !env.iter().all(|x| x.is_finite()) {
                return false;
            }
            let starts: Vec<_> = (0..slice.len() - minimum + 1).map(|x| x as i64).collect();
            let (finite, best) = position(&mark, &space, &env, &starts, n, nbits, tail);
            if !finite {
                return false;
            }
            for (j, bit) in row.iter_mut().enumerate() {
                *bit = u8::from(mark[best + j * n] > space[best + j * n]);
            }
        }
        true
    }))
}
