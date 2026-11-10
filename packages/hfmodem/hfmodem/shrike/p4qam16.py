# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline normal-short 16-QAM decoder; no live receive integration."""
from __future__ import annotations

import numpy as np

from . import p4c2, p4coding, p4decode, p4frame, p4rx, p4short, tablegen

# STANAG 4539 Table C-V, held here because a protocol never imports another.
QAM16 = np.array([
    0.866025 + 0.500000j, 0.500000 + 0.866025j, 1.000000 + 0.000000j, 0.258819 + 0.258819j,
    -0.500000 + 0.866025j, 0.000000 + 1.000000j, -0.866025 + 0.500000j, -0.258819 + 0.258819j,
    0.500000 - 0.866025j, 0.000000 - 1.000000j, 0.866025 - 0.500000j, 0.258819 - 0.258819j,
    -0.866025 - 0.500000j, -0.500000 - 0.866025j, -1.000000 + 0.000000j, -0.258819 - 0.258819j])

_SYMBOL_ORDER = tablegen.INTERLEAVERS['0xbaf20_1056']()
_LABELS = (np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1
_LOWER_POINTS = tuple(np.exp(2j*np.pi*np.arange(1 << width)/(1 << width))
                      for width in (1, 2, 3))
_MAX_REENCODED_DISTANCE = .15
_LOWER_ORDER_MARGIN = .01
_COMPONENT_ORDER = p4c2.permutation(1408)


def component_permutation() -> np.ndarray:
    return _COMPONENT_ORDER.copy()


def demap(symbols, noise: float, *, exact=False, prior=None) -> np.ndarray:
    points = np.asarray(symbols, dtype=complex)
    variance = np.asarray(noise)
    if (points.shape != (1056,) or not np.all(np.isfinite(points))
            or np.iscomplexobj(variance) or variance.shape not in ((), (1056,))
            or not np.all(np.isfinite(variance)) or np.any(variance <= 0)):
        raise ValueError('1056 finite symbols and positive finite noise required')
    if prior is not None:
        if not exact:
            raise ValueError('prior scores require exact demapping')
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (4224,) or not np.all(np.isfinite(prior)):
            raise ValueError('4224 finite prior scores required')
        prior = prior.reshape(1056, 4)[np.argsort(_SYMBOL_ORDER)]
    distance = abs(points[:, None] - QAM16)**2
    variance = np.maximum(variance, 1e-4)
    if exact:
        likelihood = -distance / np.broadcast_to(variance, (1056,))[:, None]
        if prior is not None:
            likelihood += ((1 - 2 * _LABELS)[None] * prior[:, None, :] / 2).sum(axis=2)
        scores = np.stack([
            np.logaddexp.reduce(likelihood[:, _LABELS[:, bit] == 0], axis=1)
            - np.logaddexp.reduce(likelihood[:, _LABELS[:, bit] == 1], axis=1)
            for bit in range(4)
        ], axis=1)
        if prior is not None:
            scores -= prior
    else:
        scores = np.stack([
            distance[:, _LABELS[:, bit] == 1].min(axis=1)
            - distance[:, _LABELS[:, bit] == 0].min(axis=1)
            for bit in range(4)
        ], axis=1) / np.broadcast_to(variance, (1056,))[:, None]
    return scores[_SYMBOL_ORDER].ravel()


def _zero_state_component(bits):
    source = np.asarray(bits, dtype=np.uint8)
    lane = np.pad(source, (0, -len(source) % 7))
    # Over GF(2), (1+D²+D³)(1+D²+D³+D⁴) = 1+D⁷.
    for delay in (2, 3, 4):
        lane[delay:len(source)] ^= source[:-delay]
    feedback = np.bitwise_xor.accumulate(lane.reshape(-1, 7), axis=0).ravel()[:len(source)]
    parity = feedback.copy()
    parity[1:] ^= feedback[:-1]
    parity[3:] ^= feedback[:-3]
    state = tuple(int(feedback[-i]) if i <= len(feedback) else 0 for i in (1, 2, 3))
    return parity, state


def codeword(frame: p4frame.Frame) -> np.ndarray:
    wire = p4frame.encode_field(frame.data, frame.status)
    if len(wire) != 175:
        raise ValueError('normal-short QAM requires a 175-byte net field')
    systematic = np.unpackbits(np.frombuffer(wire, dtype=np.uint8),
        bitorder='little').tolist() + [0]*5
    prefix, state = _zero_state_component(systematic)
    tail = p4coding.rsc_terminate(state)
    systematic += list(tail.systematic)
    c2, _ = _zero_state_component(np.asarray(systematic)[_COMPONENT_ORDER])
    return np.r_[systematic, prefix, tail.parity, c2]


def decode_fields(fields, *, iterations=8, status_ok=None, collect=None):
    scores = np.asarray(fields, dtype=float)
    if scores.shape != (4224,) or not np.all(np.isfinite(scores)):
        raise ValueError('4224 finite scores required')
    systematic, c1, c2 = np.split(scores, 3)
    c1, c2 = c1.copy(), c2.copy()
    c1[-8:] = c2[-8:] = 0
    accepted = {}
    references = {}
    for result in p4decode.turbo_iterations(systematic, c1, c2, _COMPONENT_ORDER,
                                            iterations=iterations):
        wire = np.packbits(result.bits[:1400], bitorder='little').tobytes()
        frame = p4frame.decode_field(wire, data_bytes=172)
        if frame is None or status_ok is not None and not status_ok(frame):
            continue
        key = frame.status.byte, frame.data, frame.crc
        if key not in references:
            references[key] = codeword(frame)
        reference = references[key]
        weight = np.sum(abs(scores))
        mismatch = np.sum(abs(scores)[(scores < 0) != reference])
        if weight and mismatch <= _MAX_REENCODED_DISTANCE * weight:
            accepted[key] = (
                frame, result.diagnostics[-1].iteration, float(mismatch / weight))
    if collect is not None:
        for result in accepted.values():
            collect(result)
    return next(iter(accepted.values())) if len(accepted) == 1 else None


def decode_equalized(symbols, noise: float, *, iterations=8, status_ok=None,
                     exact=False, collect=None):
    return decode_fields(demap(symbols, noise, exact=exact), iterations=iterations,
                         status_ok=status_ok, collect=collect)


def _lower_order_body(symbols):
    points = np.asarray(symbols)
    qam = np.mean(np.min(abs(points[:, None]-QAM16)**2, axis=1))
    psk = min(np.mean(np.min(abs(points[:, None]-alphabet)**2, axis=1))
              for alphabet in _LOWER_POINTS)
    return psk + _LOWER_ORDER_MARGIN < qam


def _soft_symbols(scores):
    ordered = np.asarray(scores).reshape(1056, 4)[np.argsort(_SYMBOL_ORDER)]
    probability = 1 / (1 + np.exp(-np.clip(ordered, -40, 40)))
    weights = np.prod(np.where(_LABELS[None] == 0, probability[:, None],
                               1 - probability[:, None]), axis=2)
    means = weights @ QAM16
    return means, weights.max(axis=1), weights @ abs(QAM16)**2 - abs(means)**2


def _decoder_extrinsic(fields, iterations):
    systematic, c1, c2 = np.split(fields, 3)
    c1, c2 = c1.copy(), c2.copy()
    c1[-8:] = c2[-8:] = 0
    prior = np.zeros_like(systematic)
    for _ in range(iterations):
        first = p4decode.component_decode(systematic, c1, prior)
        second = p4decode.component_decode(systematic[_COMPONENT_ORDER], c2,
                                            first.extrinsic[_COMPONENT_ORDER])
        prior[_COMPONENT_ORDER] = second.extrinsic
    first = p4decode.component_decode(systematic, c1, prior, parity_output=True)
    second = p4decode.component_decode(systematic[_COMPONENT_ORDER], c2,
                                        first.extrinsic[_COMPONENT_ORDER],
                                        parity_output=True)
    posterior = np.empty_like(systematic)
    posterior[_COMPONENT_ORDER] = second.posterior
    extrinsic = np.r_[posterior - systematic,
                      first.parity_extrinsic, second.parity_extrinsic]
    extrinsic[2808:2816] = extrinsic[4216:] = 0
    return np.clip(extrinsic, -16, 16)


def _block_equalized(baseband, start, cfo, offset, soft=None):
    z = np.asarray(baseband, dtype=complex)
    if not p4short.complete(z, start + 2) or not np.any(z):
        return None
    z = z / np.max(abs(z)) * np.exp(-2j * np.pi * cfo * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)

    def sample(coordinates):
        indices = np.asarray(coordinates, dtype=int)
        return matched[indices] * (1 - offset) + matched[indices + 1] * offset

    taps = np.arange(-2, 3)
    targets = np.arange(10, 22)
    body = 208 * np.arange(6)[:, None] + 32 + np.arange(176)
    observations = sample(start + 128 + 16 * (208 * np.arange(7)[:, None]
                                               + targets))
    design = p4short.TRAIN[:, targets[:, None] + taps]
    pilot_matrix = design.reshape(-1, 5)
    global_gram = pilot_matrix.conj().T @ pilot_matrix
    global_channel = np.linalg.solve(global_gram + 1e-5*np.trace(global_gram).real/5*np.eye(5),
                                     pilot_matrix.conj().T @ observations.ravel())
    channels = []
    for block in range(6):
        matrix = design[block:block+2].reshape(-1, 5)
        received = observations[block:block+2].ravel()
        weights = np.ones(24)
        uncertainty_penalty = np.zeros(5)
        if soft is not None:
            means, confidence, uncertainty = soft
            chosen = np.arange(2, 174)
            local = means[176*block:176*(block+1)]
            reliable = confidence[176*block:176*(block+1)]
            quality = np.mean(reliable[chosen[:, None] + taps], axis=1)
            row_weights = .35*np.clip((quality-.25)/.75, 0, 1)
            selected = row_weights > 0
            chosen = chosen[selected]
            if len(chosen):
                rows = local[chosen[:, None] + taps]
                values = sample(start + 128 + 16 * body[block, chosen])
                weight = row_weights[selected]
                matrix = np.concatenate((matrix, rows))
                received = np.r_[received, values]
                weights = np.r_[weights, weight]
                uncertainty_penalty = (weight[:, None]
                                       * uncertainty[176*block + chosen[:, None] + taps]).sum(axis=0)
        weighted = matrix * np.sqrt(weights[:, None])
        response = received * np.sqrt(weights)
        gram = weighted.conj().T @ weighted + np.diag(uncertainty_penalty)
        ridge = 12.
        channel = np.linalg.solve(gram + ridge*np.eye(5),
                                  weighted.conj().T @ response + ridge*global_channel)
        variance = max(float(np.mean(abs(observations[block:block+2].ravel()
                                          - design[block:block+2].reshape(-1, 5) @ channel)**2)),
                       1e-8)
        channels.append((channel, variance))
    windows = start + 128 + 16*(body[:, :, None] + np.arange(-5, 6))
    symbols = np.empty((6, 176), dtype=complex)
    noise = np.zeros((6, 176))
    target = np.zeros(15)
    target[7] = 1
    for block, (channel, variance) in enumerate(channels):
        convolution = np.column_stack([np.convolve(channel, row) for row in np.eye(11)])
        inverse = np.linalg.solve(convolution.conj().T @ convolution
                                  + variance*np.eye(11), convolution.conj().T @ target)
        residual = np.convolve(channel, inverse)
        gain = residual[7]
        residual[7] = 0
        if abs(gain) < 1e-8:
            return None
        symbols[block] = (sample(windows[block]) @ inverse) / gain
        block_noise = (variance*np.vdot(inverse, inverse).real
                       + np.vdot(residual, residual).real) / abs(gain)**2
        if soft is not None:
            means, _, uncertainty = soft
            local = means[176*block:176*(block+1)]
            local_uncertainty = uncertainty[176*block:176*(block+1)]
            for tap in range(15):
                shift = tap - 7
                if shift == 0:
                    continue
                destination = np.arange(max(0, -shift), min(176, 176-shift))
                coefficient = residual[tap] / gain
                symbols[block, destination] -= coefficient * local[destination+shift]
                correction = abs(coefficient)**2 * (local_uncertainty[destination+shift]-1)
                noise[block, destination] += correction
        noise[block] = np.maximum(block_noise + noise[block], 1e-4)
    return symbols.ravel(), noise.ravel()


def _wide_equalized(baseband, start, cfo, tap_count=9, ridge=12.,
                    equalizer=15, soft=None):
    z = np.asarray(baseband, dtype=complex)
    if not p4short.complete(z, start) or not np.any(z):
        return None
    z = z / np.max(abs(z)) * np.exp(-2j * np.pi * cfo * np.arange(len(z)) / 28800)
    matched = p4rx.matched_filter(z)
    observed = matched[start + 128 + 16 * np.arange(1280)]
    taps = np.arange(-(tap_count//2), tap_count//2+1)
    targets = np.arange(8, 24)
    training = 208 * np.arange(7)[:, None] + targets
    design = p4short.TRAIN[:, targets[:, None] + taps]
    matrix = design.reshape(-1, tap_count)
    values = observed[training].ravel()
    global_channel = np.linalg.solve(matrix.conj().T @ matrix + 1e-5*np.eye(tap_count),
                                     matrix.conj().T @ values)
    symbols = np.empty((6, 176), complex)
    noise = np.empty((6, 176))
    for block in range(6):
        local = design[block:block+2].reshape(-1, tap_count)
        received = observed[training[block:block+2]].ravel()
        matrix = local
        values = received
        weights = np.ones(len(received))
        penalty = np.zeros(tap_count)
        if soft is not None:
            means, confidence, uncertainty = soft
            select = np.arange(tap_count, 176-tap_count)
            quality = np.mean(confidence[176*block+select[:, None]+taps], axis=1)
            select = select[quality > .3]
            if len(select):
                weight = .5*((quality[quality > .3]-.3)/.7)**2
                matrix = np.concatenate((matrix,
                                         means[176*block+select[:, None]+taps]))
                values = np.r_[values, observed[208*block+32+select]]
                weights = np.r_[weights, weight]
                penalty = (weight[:, None]
                           * uncertainty[176*block+select[:, None]+taps]).sum(axis=0)
        weighted = matrix*np.sqrt(weights[:, None])
        channel = np.linalg.solve(weighted.conj().T @ weighted
                                  + np.diag(penalty) + ridge*np.eye(tap_count),
                                  weighted.conj().T @ (values*np.sqrt(weights))
                                  + ridge*global_channel)
        variance = max(float(np.mean(abs(received - local @ channel)**2)), 1e-8)
        convolution = np.column_stack([np.convolve(channel, row)
                                       for row in np.eye(equalizer)])
        target = np.zeros(tap_count+equalizer-1)
        center = (tap_count+equalizer-2)//2
        target[center] = 1
        inverse = np.linalg.solve(convolution.conj().T @ convolution
                                  + variance*np.eye(equalizer),
                                  convolution.conj().T @ target)
        residual = np.convolve(channel, inverse)
        gain = residual[center]
        residual[center] = 0
        if abs(gain) < 1e-8:
            return None
        body = 208*block + 32 + np.arange(176)
        windows = body[:, None] + np.arange(-(equalizer//2), equalizer//2+1)
        symbols[block] = (observed[windows] @ inverse) / gain
        noise[block] = max(float((variance*np.vdot(inverse, inverse).real
                                  + np.vdot(residual, residual).real) / abs(gain)**2), 1e-4)
        if soft is not None:
            means, _, uncertainty = soft
            local_means = means[176*block:176*(block+1)]
            local_uncertainty = uncertainty[176*block:176*(block+1)]
            for tap in range(len(residual)):
                shift = tap-center
                if shift == 0:
                    continue
                destination = np.arange(max(0, -shift), min(176, 176-shift))
                coefficient = residual[tap]/gain
                symbols[block, destination] -= coefficient*local_means[destination+shift]
                noise[block, destination] += (abs(coefficient)**2
                                               * (local_uncertainty[destination+shift]-1))
            noise[block] = np.maximum(noise[block], 1e-4)
    return symbols.ravel(), noise.ravel()


def _wide_iterative_candidate(baseband, start, cfo, iterations, status_ok,
                              collect=None):
    prior = np.zeros(4224)
    soft = None
    for _ in range(5):
        equalized = _wide_equalized(baseband, start, cfo, 13, 30., 31, soft)
        if equalized is None:
            return None
        fields = demap(*equalized, exact=True, prior=prior)
        result = decode_fields(fields, iterations=iterations, status_ok=status_ok,
                               collect=collect)
        if result is not None:
            if collect is None:
                return result
            collect(result)
        feedback = _decoder_extrinsic(fields, min(iterations, 16))
        soft = _soft_symbols(fields+feedback)
        prior = .7*feedback
    return None


def _body_em_candidate(baseband, start, cfo, iterations, status_ok, collect=None):
    z = np.asarray(baseband, dtype=complex)
    if not p4short.complete(z, start) or not np.any(z):
        return None
    z = z / np.max(abs(z)) * np.exp(-2j*np.pi*cfo*np.arange(len(z))/28800)
    matched = p4rx.matched_filter(z)
    observed = matched[start+128+16*np.arange(1280)]
    taps = np.arange(-7, 8)
    targets = np.arange(8, 24)
    training = 208*np.arange(7)[:, None]+targets
    design = p4short.TRAIN[:, targets[:, None]+taps]
    matrix = design.reshape(-1, 15)
    values = observed[training].ravel()
    global_channel = np.linalg.solve(matrix.conj().T@matrix+1e-5*np.eye(15),
                                     matrix.conj().T@values)
    channels = []
    variances = []
    equalized = np.empty((6, 176), complex)
    noise = np.empty((6, 176))
    source = np.zeros(1280, complex)
    for block in range(7):
        source[208*block:208*block+32] = p4short.TRAIN[block]
    for block in range(6):
        local = design[block:block+2].reshape(-1, 15)
        received = observed[training[block:block+2]].ravel()
        channel = np.linalg.solve(local.conj().T@local+20*np.eye(15),
                                  local.conj().T@received+20*global_channel)
        variance = max(float(np.mean(abs(received-local@channel)**2)), 1e-8)
        channel[-1] = 0
        channels.append(channel)
        variances.append(variance)
        convolution = np.column_stack([np.convolve(channel, row)
                                       for row in np.eye(31)])
        target = np.zeros(45)
        target[22] = 1
        inverse = np.linalg.solve(convolution.conj().T@convolution
                                  + variance*np.eye(31), convolution.conj().T@target)
        residual = np.convolve(channel, inverse)
        gain = residual[22]
        residual[22] = 0
        if abs(gain) < 1e-8:
            return None
        body = 208*block+32+np.arange(176)
        equalized[block] = (observed[body[:, None]+np.arange(-15, 16)]@inverse)/gain
        noise[block] = max(float((variance*np.vdot(inverse, inverse).real
                                  + np.vdot(residual, residual).real)/abs(gain)**2), 1e-4)
    scores = demap(equalized.ravel(), 2*noise.ravel(), exact=True)
    means, _, uncertainty = _soft_symbols(scores)
    bodies = 208*np.arange(6)[:, None]+32+np.arange(176)
    source[bodies] = means.reshape(6, 176)
    source_uncertainty = np.zeros(1280)
    source_uncertainty[bodies] = uncertainty.reshape(6, 176)
    relative = np.arange(176)[:, None]-taps
    valid = (relative >= 0) & (relative < 176)
    adjacent = np.clip(relative, 0, 175)
    signs = 1-2*_LABELS
    prior = np.zeros((1056, 4))
    for round_index in range(8):
        padded = np.pad(source, 7)
        rows = padded[np.arange(1280)[:, None]+7+taps]
        scores = np.empty((1056, 4))
        next_means = np.empty(1056, complex)
        next_uncertainty = np.empty(1056)
        width = max((2.5 if round_index < 4 else 2.0)*np.mean(variances),
                    .06 if round_index < 4 else .05)
        for block in range(6):
            body = bodies[block]
            channel = channels[block]
            residual = observed[body] - rows[body]@channel
            neighboring = residual[adjacent]
            differences = neighboring[:, :, None] + channel[None, :, None]*(
                source[body, None, None]-QAM16[None, None, :])
            cost = np.sum(np.where(valid[:, :, None], abs(differences)**2, 0), axis=1)/width
            part = slice(176*block, 176*(block+1))
            likelihood = -cost
            if round_index >= 3:
                likelihood += (signs[None]*prior[part, None, :]/2).sum(axis=2)
            likelihood -= np.logaddexp.reduce(likelihood, axis=1)[:, None]
            weights = np.exp(likelihood)
            next_means[part] = weights@QAM16
            next_uncertainty[part] = weights@abs(QAM16)**2-abs(next_means[part])**2
            scores[part] = np.stack([
                np.logaddexp.reduce(likelihood[:, _LABELS[:, bit] == 0], axis=1)
                - np.logaddexp.reduce(likelihood[:, _LABELS[:, bit] == 1], axis=1)
                - prior[part, bit]
                for bit in range(4)
            ], axis=1)
            source[body] = .5*source[body]+.5*next_means[part]
            source_uncertainty[body] = next_uncertainty[part]
            if round_index >= 2:
                selected = body[15:-15]
                soft_rows = np.pad(source, 7)[selected[:, None]+7+taps]
                penalty = np.pad(source_uncertainty, 7)[selected[:, None]+7+taps].sum(axis=0)
                updated = np.linalg.solve(soft_rows.conj().T@soft_rows
                                          + np.diag(penalty)+40*np.eye(15),
                                          soft_rows.conj().T@observed[selected]+40*channel)
                channels[block] = .95*channel+.05*updated
                channels[block][-1] = 0
        fields = scores[_SYMBOL_ORDER].ravel()
        if round_index >= 2:
            feedback = _decoder_extrinsic(fields, min(iterations, 8))
            prior = .5*np.clip(feedback, -8, 8).reshape(1056, 4)[np.argsort(_SYMBOL_ORDER)]
        result = decode_fields(fields, iterations=iterations, status_ok=status_ok,
                               collect=collect)
        if result is not None:
            if collect is None:
                return result
            collect(result)
    return None


def _reference_equalized(baseband, start, cfo, symbols):
    z = np.asarray(baseband, dtype=complex)
    if not p4short.complete(z, start) or not np.any(z):
        return None
    z = z / np.max(abs(z)) * np.exp(-2j*np.pi*cfo*np.arange(len(z))/28800)
    matched = p4rx.matched_filter(z)
    observed = matched[start+128+16*np.arange(1280)]
    source = np.zeros(1280, complex)
    for block in range(7):
        source[208*block:208*block+32] = p4short.TRAIN[block]
    for block in range(6):
        source[208*block+32:208*block+208] = symbols[176*block:176*(block+1)]
    design = np.pad(source, 15)[np.arange(1280)[:, None]+15+np.arange(-7, 8)]
    equalized = np.empty(1056, complex)
    variance = np.empty(1056)
    for block in range(6):
        fit = np.arange(208*block+15, 208*block+193)
        channel = np.linalg.lstsq(design[fit], observed[fit], rcond=None)[0]
        noise = max(float(np.mean(abs(observed[fit]-design[fit]@channel)**2)), 1e-8)
        convolution = np.column_stack([np.convolve(channel, row)
                                       for row in np.eye(31)])
        target = np.zeros(45)
        target[22] = 1
        inverse = np.linalg.solve(convolution.conj().T@convolution
                                  + noise*np.eye(31), convolution.conj().T@target)
        residual = np.convolve(channel, inverse)
        gain = residual[22]
        residual[22] = 0
        if abs(gain) < 1e-8:
            return None
        body = 208*block+32+np.arange(176)
        windows = body[:, None]+np.arange(-15, 16)
        equalized[176*block:176*(block+1)] = (observed[windows]@inverse)/gain
        variance[176*block:176*(block+1)] = max(float(
            (noise*np.vdot(inverse, inverse).real
             + np.vdot(residual, residual).real)/abs(gain)**2), 1e-4)
    return equalized, variance


def decode_repeated_body(baseband, *, first_training_start, repeats,
                         frequency_offset=0., iterations=32, status_ok=None):
    """Decode a short using only analog observations of the same repeated field."""
    estimates = []
    for other, start, cfo in repeats:
        equalized = _wide_equalized(other, start, cfo, 9, 12.)
        if equalized is not None:
            estimates.append(equalized[0])
    if len(estimates) < 2:
        return None
    mean = np.mean(estimates, axis=0)
    symbols = QAM16[abs(mean[:, None]-QAM16).argmin(axis=1)]
    accepted = {}

    def add(result):
        frame = result[0]
        key = frame.status.byte, frame.data, frame.crc
        if key not in accepted or result[2] < accepted[key][2]:
            accepted[key] = result

    for shift in (0, 2, 4, 6, 8):
        equalized = _reference_equalized(baseband, first_training_start+shift,
                                         frequency_offset, symbols)
        if equalized is not None:
            decode_equalized(*equalized, iterations=iterations, status_ok=status_ok,
                             exact=True, collect=add)
    return next(iter(accepted.values())) if len(accepted) == 1 else None


def _iterative_candidate(baseband, start, cfo, offset, iterations, status_ok,
                         collect=None):
    prior = np.zeros(4224)
    soft = None
    for round_index in range(3):
        if round_index == 0 and offset == 0:
            output = []
            p4short.short_scores(baseband, first_training_start=start,
                                 frequency_offset=cfo, refine=0,
                                 on_equalized=lambda s, n: output.extend((s, n)))
            equalized = tuple(output) if output else None
        else:
            equalized = _block_equalized(baseband, start, cfo, offset, soft)
        if equalized is None:
            return None
        fields = demap(*equalized, exact=True, prior=prior)
        result = decode_fields(fields, iterations=iterations, status_ok=status_ok,
                               collect=collect)
        if result is not None:
            if collect is None:
                return result
            collect(result)
        feedback = _decoder_extrinsic(fields, min(iterations, 12))
        posterior = fields + feedback
        soft = _soft_symbols(posterior)
        prior = .7 * feedback
    return None


def decode_body(baseband, *, first_training_start: int, frequency_offset=0.,
                iterations=32, status_ok=None):
    accepted = {}

    def add(result):
        if result is None:
            return
        frame = result[0]
        key = frame.status.byte, frame.data, frame.crc
        if key not in accepted or result[2] < accepted[key][2]:
            accepted[key] = result

    for shift, exact in ((0, False), (0, True), (2, True), (4, True), (6, True)):
        equalized = []
        observation = p4short.short_scores(
            baseband, first_training_start=first_training_start + shift,
            frequency_offset=frequency_offset, refine=0,
            on_equalized=lambda symbols, noise: equalized.extend((symbols, noise)))
        if observation is None:
            continue
        if shift == 0 and not exact and equalized and _lower_order_body(equalized[0]):
            return None
        result = decode_equalized(*equalized, iterations=iterations,
                                  status_ok=status_ok, exact=exact, collect=add)
        add(result)
    for tap_count, ridge in ((9, 12.), (13, 30.)):
        for shift in (0, 2, 4, 6):
            equalized = _wide_equalized(baseband, first_training_start + shift,
                                        frequency_offset, tap_count, ridge)
            if equalized is not None:
                result = decode_equalized(*equalized, iterations=iterations,
                                          status_ok=status_ok, exact=True,
                                          collect=add)
                add(result)
    for shift in (6, 4, 8):
        result = _body_em_candidate(baseband, first_training_start+shift,
                                    frequency_offset, iterations, status_ok,
                                    collect=add)
        add(result)
    for shift in (2, 4, 6):
        result = _wide_iterative_candidate(baseband,
                                           first_training_start + shift,
                                           frequency_offset, iterations, status_ok,
                                           collect=add)
        add(result)
    for shift in (0, 2, 4, 6):
        for offset in (0., .5):
            result = _iterative_candidate(baseband, first_training_start + shift,
                                          frequency_offset, offset, iterations,
                                          status_ok, collect=add)
            add(result)
    return next(iter(accepted.values())) if len(accepted) == 1 else None
