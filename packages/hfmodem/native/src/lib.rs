// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use std::ffi::CStr;

mod frontend;

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct Complex([f64; 2]);

// SAFETY: complex128 is two native doubles, with every bit pattern valid.
unsafe impl Element for Complex {
    fn is_compatible_format(format: &CStr) -> bool {
        matches!(format.to_bytes(), b"Zd" | b"@Zd" | b"=Zd")
            || (cfg!(target_endian = "little") && format.to_bytes() == b"<Zd")
    }
}

fn buffer(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<PyBuffer<Complex>> {
    let buf = PyBuffer::get(obj)?;
    if buf.dimensions() != ndim || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "expected a C-contiguous buffer of the required rank",
        ));
    }
    Ok(buf)
}

fn indices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    obj.try_iter()?.map(|item| item?.extract()).collect()
}

fn residues(obj: &Bound<'_, PyAny>, nfft: usize) -> PyResult<Vec<usize>> {
    let n = i64::try_from(nfft)?;
    obj.try_iter()?
        .map(|item| Ok(item?.extract::<i64>()?.rem_euclid(n) as usize))
        .collect()
}

fn overlaps(left: &PyBuffer<Complex>, right: &PyBuffer<Complex>) -> bool {
    let a = left.buf_ptr() as usize;
    let b = right.buf_ptr() as usize;
    left.len_bytes() != 0
        && right.len_bytes() != 0
        && a < b.saturating_add(right.len_bytes())
        && b < a.saturating_add(left.len_bytes())
}

fn compute<const FUSED: bool>(
    spectra: &[PyBuffer<Complex>],
    picks: &[usize],
    shifts: &[usize],
    tones: &[usize],
    weight: &PyBuffer<Complex>,
    out: &PyBuffer<Complex>,
    width: usize,
) -> bool {
    // SAFETY: validation proved format, alignment, shape, contiguity and no
    // output/input overlap. Exports keep storage alive throughout detach.
    // As with NumPy's own GIL-free loops, callers must not mutate inputs or
    // touch out from another thread during the call; the only caller,
    // grid_fold, passes buffers private to one `_grid`.
    let weight = unsafe { weight.as_slice_ptr().unwrap().as_ref() };
    let out = unsafe { out.as_slice_ptr().unwrap().as_mut() };
    let nfft = weight.len();
    let mut finite = true;
    for (c, &tone) in tones.iter().enumerate() {
        for (i, (&pick, &shift)) in picks.iter().zip(shifts).enumerate() {
            let spectrum = unsafe { spectra[pick].as_slice_ptr().unwrap().as_ref() };
            let row = &mut out[(c * picks.len() + i) * width..][..width];
            row.fill(Complex([0.0, 0.0]));
            let mut a = (shift + tone) % nfft;
            let b = (tone + nfft - shift) % nfft;
            let mut y = if b == 0 { 0 } else { nfft - b };
            for weights in weight.chunks_exact(width) {
                let mut j = 0;
                while j < width {
                    // Split at either wrap, so both input streams are contiguous
                    // and the product loop has no modulo or wrap branches.
                    let len = (width - j).min(nfft - a).min(y + 1);
                    let xs = &spectrum[a..a + len];
                    let ys = &spectrum[y + 1 - len..y + 1];
                    for (((acc, w), x), y) in row[j..j + len]
                        .iter_mut()
                        .zip(&weights[j..j + len])
                        .zip(xs)
                        .zip(ys.iter().rev())
                    {
                        let zr = x.0[0] + y.0[0];
                        let zi = x.0[1] + (-y.0[1]);
                        let [wr, wi] = w.0;
                        let (pr, pi) = if FUSED {
                            (zr.mul_add(wr, -(zi * wi)), zr.mul_add(wi, zi * wr))
                        } else {
                            (zr * wr - zi * wi, zr * wi + zi * wr)
                        };
                        acc.0[0] += pr;
                        acc.0[1] += pi;
                    }
                    j += len;
                    a += len;
                    if a == nfft {
                        a = 0;
                    }
                    y = if len == y + 1 { nfft - 1 } else { y - len };
                }
            }
            // A finite row had only finite intermediates, so its bits are
            // NumPy's; NaN signs and overflow paths need not be.
            finite &= row.iter().all(|z| z.0[0].is_finite() && z.0[1].is_finite());
        }
    }
    finite
}

#[allow(clippy::too_many_arguments)]
fn validated_fold(
    py: Python<'_>,
    spectra: &Bound<'_, PyAny>,
    picks: &Bound<'_, PyAny>,
    shifts: &Bound<'_, PyAny>,
    tones: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    fused: bool,
) -> PyResult<bool> {
    let weight = buffer(weight, 2)?;
    let out = buffer(out, 3)?;
    let width = weight.shape()[1];
    let nfft = width
        .checked_mul(10)
        .ok_or_else(|| PyValueError::new_err("width overflow"))?;
    if weight.shape()[0] != 10 || width == 0 || out.readonly() {
        return Err(PyValueError::new_err(
            "weight must have shape (10, width > 0); out must be writable",
        ));
    }
    let spectra = spectra
        .try_iter()?
        .map(|s| buffer(&s?, 1))
        .collect::<PyResult<Vec<_>>>()?;
    let picks = indices(picks)?;
    let shifts = residues(shifts, nfft)?;
    let tones = residues(tones, nfft)?;
    if picks.len() != shifts.len()
        || out.shape() != [tones.len(), picks.len(), width]
        || picks.iter().any(|&p| p >= spectra.len())
        || spectra.iter().any(|s| s.shape() != [nfft])
    {
        return Err(PyValueError::new_err(
            "inconsistent shapes, lengths or spectrum picks",
        ));
    }
    if overlaps(&out, &weight) || spectra.iter().any(|s| overlaps(&out, s)) {
        return Err(PyValueError::new_err(
            "out must not overlap any input buffer",
        ));
    }
    Ok(py.detach(|| {
        if fused {
            compute::<true>(&spectra, &picks, &shifts, &tones, &weight, &out, width)
        } else {
            compute::<false>(&spectra, &picks, &shifts, &tones, &weight, &out, width)
        }
    }))
}

/// Fold a complete grid into out; true when every result is finite.
/// Do not mutate inputs or access out until return.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fold_grid(
    py: Python<'_>,
    spectra: &Bound<'_, PyAny>,
    picks: &Bound<'_, PyAny>,
    shifts: &Bound<'_, PyAny>,
    tones: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    fused: bool,
) -> PyResult<bool> {
    validated_fold(py, spectra, picks, shifts, tones, weight, out, fused).map_err(|err| {
        if err.is_instance_of::<PyBufferError>(py) || err.is_instance_of::<PyValueError>(py) {
            err
        } else {
            PyValueError::new_err(err.to_string())
        }
    })
}

#[pymodule]
fn _dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fold_grid, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::sliding_magnitudes, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::gather_frames, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::eye_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(frontend::slice_eyes, m)?)
}
