# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the PACTOR-2 transmitter and the shared coding chain it rides.

What this proves on its own, with no external decoder (self-consistent and
deterministic):
  * the two-carrier DPSK modulator round-trips bit-exact through a clean demod,
    in both DBPSK and DQPSK, on both tones;
  * each of the four speed levels, on both cycle lengths, through the coded-bit
    count, helical stride and puncture `pactor2.PATHS` records for it, round-trips
    field -> code -> channel order -> depuncture -> Viterbi -> CRC back to the
    same info bytes;
  * the K=9 taps and the two puncturing vectors read back as the binary strings
    Annex I prints, so a transcription slip in either fails here;
  * every one of the eight paths, short and long, has a field length that divides
    exactly and lands on the published payload table -- which is a check against
    an outside source rather than against ourselves;
  * the emitted waveform lands on the measured P2 geometry (1400/1600 Hz, 100 Bd).

And one thing it proves with an outside referee: `test_accepted_vector` rebuilds
the exact SL1 frame an independent decoder took through its own trellis, byte for
byte, and passed its own CRC on -- so the code, the packing, the flush, the
interleaver and its depth, the geometry and the absence of a whitener are all
checked against something that is not this package.

Acquisition is checked separately, in test_p2rx.py: `pactor2.frame_marker` emits
the nine-pulse header, a conforming receiver forms a frame descriptor off it with
the codeword index we sent, and `p2rx.find_markers` reads the three real markers
on the published FEC recording at the same times and indices that receiver
reports. pactor2.md §5.4.1.
"""
import numpy as np


from hfmodem.shrike import coding, p2rx, pactor2
from hfmodem.tests.kestrel import corpora

FS = 48000
ORDER = {1: "DBPSK", 2: "DQPSK", 3: "8-DPSK", 4: "16-DPSK"}
PASS = 0
FAILURES: list[str] = []


def check(name, ok, detail=""):
    global PASS
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def test_modulator_roundtrip():
    rng = np.random.default_rng(7)
    for bpc in (1, 2, 3, 4):
        lo = rng.integers(0, 2, 240 * bpc)
        hi = rng.integers(0, 2, 240 * bpc)
        x = pactor2.modulate(lo, hi, bpc)
        r_lo, r_hi = pactor2.demod_clean(x, bpc)
        n = min(len(lo), len(r_lo))
        check(f"{ORDER[bpc]} tone-lo round-trips bit-exact",
              np.array_equal(lo[:n], r_lo[:n]),
              f"{(lo[:n] != r_lo[:n]).sum()} bit errors")
        m = min(len(hi), len(r_hi))
        check(f"{ORDER[bpc]} tone-hi round-trips bit-exact",
              np.array_equal(hi[:m], r_hi[:m]),
              f"{(hi[:m] != r_hi[:m]).sum()} bit errors")


def test_low_level_alphabets_are_on_the_diagonals():
    """The two low levels' alphabets are antipodal, Gray-ordered, and diagonal.

    The DIAGONALS are published -- SCS, *The PACTOR-4 Protocol* s11.7, of the
    level s11 states is based on PACTOR-2 speed level 1: "The utilized
    differential phases are PI/4 when a 0 is being transferred, and 5*PI/4 when a
    1 is being transferred - just like P2." Which diagonal carries the zero is
    that document's own phase reference and not a reference decoder's; both
    origins are measured through one, 90 and 270 degrees from the pair as
    printed (`_DBPSK_STEP`, `_DQPSK_STEP`). What survives from the page is the
    geometry, and that is what this holds: odd multiples of 45 degrees, one Gray
    step per sector, antipodes where the level puts them."""
    lo, hi = (np.degrees(pactor2._DBPSK_STEP[(b,)]) % 360 for b in (0, 1))
    check("DBPSK is an antipodal pair on the diagonals",
          abs(lo % 90 - 45) < 1e-9 and abs((hi - lo) % 360 - 180) < 1e-9,
          f"{lo:.1f} and {hi:.1f} deg")
    ladder = sorted((np.degrees(s) % 360, d)
                    for d, s in pactor2._DQPSK_STEP.items())
    check("DQPSK sits on the four diagonals",
          all(abs(deg % 90 - 45) < 1e-9 for deg, _ in ladder),
          ", ".join(f"{deg:.1f}" for deg, _ in ladder))
    check("...one Gray step per 90 degrees round the circle",
          all(sum(a != b for a, b in zip(ladder[i][1], ladder[(i + 1) % 4][1])) == 1
              for i in range(4)),
          " ".join(f"{d}@{deg:.0f}" for deg, d in ladder))


def test_soft_demapper_inverts_the_modulator():
    """One convention has to serve the whole modem: the demapper reads the
    constellation in the CONJUGATE sense of what the modulator emits, and
    `soft_bits(conj(d), bpc)` must then return the bits, at no rotation, for both
    orders.

    The conjugation is not a free choice -- it is the difference between the
    transmit alphabet as SCS prints it and the demapper as measured off a
    receiver's soft-decision stage, and no rotation composes one into the other.
    Against the multiples of 90 degrees this file used to emit, the demapper
    recovered 0.545 of the bits at the best of all eight half-step rotations: one
    bit of each pair right and the other inverted. Nothing caught it because the
    round trip above goes through `demod_clean`, which inverts the modulator by
    construction, while every decode path on real audio goes through
    `soft_bits`."""
    rng = np.random.default_rng(19)
    for bpc in (1, 2, 3, 4):
        lo = rng.integers(0, 2, 200 * bpc).astype(np.uint8)
        hi = rng.integers(0, 2, 200 * bpc).astype(np.uint8)
        x = pactor2.modulate(lo, hi, bpc)
        L = FS // int(p2rx.SYMBOL_RATE)
        pulse = pactor2.tx_pulse(FS)
        n = np.arange(x.size)
        for f0, want, tag in ((1400.0, lo, "lo"), (1600.0, hi, "hi")):
            y = np.convolve(x * np.exp(-2j * np.pi * f0 * n / FS), pulse)
            sym = y[pulse.size - 1::L]
            d = np.conj(sym[1:] * np.conj(sym[:-1]))
            got = (pactor2.soft_bits(d, bpc) < 0).astype(np.uint8).ravel()
            k = min(got.size, want.size)
            check(f"{ORDER[bpc]} tone-{tag}: soft_bits inverts modulate",
                  np.array_equal(got[:k], want[:k]),
                  f"{(got[:k] != want[:k]).mean():.3f} of bits differ")


def test_published_interleaver_listing():
    """`interleave_pointer` must be Annex II's routine, run as printed."""
    for n_buf, depth in ((144, 16), (288, 8), (432, 4), (576, 2)):
        p, s, want = 0, 1, []
        for _ in range(n_buf):
            want.append(p)
            p += depth
            if p >= n_buf:
                p, s = s, s + 1
        got = pactor2.interleave_pointer(n_buf, depth)
        check(f"Annex II pointer walk, {n_buf} bits depth {depth}",
              got.tolist() == want, "differs from the printed loop")
        check(f"Annex II pointer walk, {n_buf} bits depth {depth}, is a bijection",
              sorted(got.tolist()) == list(range(n_buf)), "not a permutation")


def test_symbol_period_is_measured_not_assumed():
    """A recording carries its own symbol clock, and the receiver has to find it.

    The published FEC sample runs 0.77 % fast, which walks a 72-symbol frame
    nearly half a symbol end to end. Planted here at the same error."""
    rng = np.random.default_rng(23)
    bits = rng.integers(0, 2, 1200).astype(np.uint8)
    x = pactor2.modulate(bits, bits[::-1], 2)
    # resample to 100.77 Bd on the same grid, the FEC sample's measured error
    k = 480 / 476.35
    idx = np.arange(0, x.size / k) * k
    y = np.interp(idx, np.arange(x.size), x)
    period, _ = p2rx.measure_symbol_period(y, FS, 1400.0 * k)
    check("off-nominal symbol period is recovered within 0.1 %",
          abs(period - 476.35) / 476.35 < 1e-3, f"got {period:.2f}, want 476.35")


def test_symbol_period_at_every_dpsk_order():
    """The front end has to lock all four levels, not just the two sparse ones.

    An M-ary differential concentrates under the Mth power and under no lower
    one, so a statistic that tries only the 2nd and 4th cannot see 8-DPSK or
    16-DPSK -- and `measure_symbol_period` then fits a period to nothing. That
    is not hypothetical: a planted 8-DPSK frame was not recovered until this was
    widened."""
    rng = np.random.default_rng(29)
    L = FS / 100.0
    for order in p2rx.DPSK_ORDERS:
        steps = rng.choice(np.arange(order) * 2 * np.pi / order, 900)
        ph = np.concatenate([[0.0], np.cumsum(steps)]) % (2 * np.pi)
        n = np.arange(int(len(ph) * L))
        x = np.cos(2 * np.pi * 1400.0 * n / FS + ph[np.minimum((n / L).astype(int),
                                                               len(ph) - 1)])
        sym = p2rx.symbol_stream(x, FS, 1400.0, L, 0)
        check(f"{order}-DPSK concentrates under the widened statistic",
              p2rx._concentration(sym) > 0.9, f"{p2rx._concentration(sym):.3f}")
        period, _ = p2rx.measure_symbol_period(x, FS, 1400.0)
        check(f"{order}-DPSK symbol period recovered", abs(period - L) < 0.5,
              f"got {period:.2f}, want {L:.2f}")


def test_waveform_geometry():
    rng = np.random.default_rng(8)
    x = pactor2.modulate(rng.integers(0, 2, 400), rng.integers(0, 2, 400), 2)
    lo, hi = p2rx.measure_carriers(x, FS)
    check("emitted tones ~1400/1600 Hz", abs(lo - 1400) < 20 and abs(hi - 1600) < 20,
          f"{lo:.1f}/{hi:.1f}")
    check("emitted spacing ~200 Hz", abs((hi - lo) - 200) < 25, f"{hi - lo:.1f}")


def test_emitted_bandwidth():
    """The emission has to fit a 500 Hz CW filter, which is what the shaping is for.

    [SCS] s1 -- "a two-tone DPSK system with raised cosine pulse shaping, which
    reduces the required bandwidth to less than 500 Hz" -- and s3 puts the figure
    at "around 450 Hz at minus 50 dB". This is the only check on what leaves the
    transmitter rather than on what comes back through our own receiver, and a
    round trip is blind to occupancy by construction: the rectangular symbols
    this file emitted until 2026-09-02 decoded byte-exact at every level and put
    5.0 per cent of their power outside 1250-1750 Hz, with 1899 Hz between the
    -26 dB edges and the whole audio band above -50.

    Measured on six SL2 bursts, the arrangement alternating, 48 kHz, a
    65536-point periodogram: 360 Hz at -26, 366 at -30, 401 at -40, 559 at -50,
    and 0.005 per cent of the power out of band. The bounds below sit clear of
    those and far under what the unshaped waveform could reach.
    """
    path = pactor2.PATHS[1]
    rng = np.random.default_rng(3)
    x = np.concatenate([
        pactor2.data_burst(pactor2.build_field(
            bytes(rng.integers(0, 256, path.crc_bytes - 2, dtype=np.uint8)), path),
            path, swapped=bool(i & 1), fs=FS) for i in range(6)])

    n = 65536
    seg = np.zeros(n)
    seg[:min(x.size, n)] = x[:n]
    psd = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    db = 10 * np.log10(psd / psd.max() + 1e-30)
    f = np.fft.rfftfreq(n, 1 / FS)
    for level, limit in ((-26, 400.0), (-40, 500.0)):
        edges = np.nonzero(db >= level)[0]
        bw = f[edges.max()] - f[edges.min()]
        check(f"emitted bandwidth at {level} dB is under {limit:.0f} Hz",
              bw <= limit, f"{bw:.0f} Hz")

    power = np.abs(np.fft.rfft(x)) ** 2
    fx = np.fft.rfftfreq(x.size, 1 / FS)
    out = power[(fx < 1250) | (fx > 1750)].sum() / power.sum()
    check("under 0.05 per cent of the power sits outside 1250-1750 Hz",
          out < 5e-4, f"{100 * out:.4f} per cent")


def test_carrier_swap_is_read_not_assumed():
    """The arrangement a marker is in comes back out of acquisition.

    The two virtual carriers exchange tones every ARQ cycle, taking their channel
    rank and their T/2 stagger with them, so `frame_marker` with its tone pair
    reversed IS a swapped-cycle marker: the codebook that leads moves to the upper
    tone. Half the real bursts of a link are in that arrangement, and a receiver
    that assumes the home one reads nothing from them (pactor2.md 7.11).

    The separation is the point, not just the detection: on real audio the wrong
    arrangement scores 0.52-0.66 against 0.97-1.00 for the right one, which is
    what lets `decode_bursts` take one arrangement per burst instead of offering
    the CRC both.
    """
    home = p2rx.lane_tones(12, False)
    check("lane_tones is an involution", p2rx.lane_tones(12, True) == home[::-1],
          f"{home} vs {p2rx.lane_tones(12, True)}")
    check("rank 0 is the upper tone at home", home[0] == p2rx.carrier_bins(12)[1],
          f"{home[0]} vs {p2rx.carrier_bins(12)}")

    for k in (0, 6, 15):
        for swapped in (False, True):
            tones = pactor2.NOMINAL_TONES_HZ
            sig = pactor2.frame_marker(k, tones[::-1] if swapped else tones)
            audio = np.concatenate([np.zeros(FS // 10), sig, np.zeros(FS // 10)])
            hits = p2rx.find_markers(audio, FS, threshold=0.95)
            arm = "swapped" if swapped else "home"
            check(f"{arm} marker k={k} read back, arrangement and all",
                  len(hits) == 1 and hits[0][2] == k and hits[0][5] == swapped,
                  str([(round(h[0], 3), h[2], round(h[4], 3), h[5]) for h in hits]))
            other = [h for h in p2rx.find_markers(audio, FS, threshold=0.80)
                     if h[5] != swapped]
            check(f"{arm} marker does not also arm the other arrangement",
                  not other, str([(round(h[0], 3), round(h[4], 3)) for h in other]))


PUBLISHED_PAYLOAD = {"short": (5, 14, 32, 59), "long": (36, 76, 156, 276)}
"""Data bytes per packet, SL1..SL4: SCS, *The PACTOR-2 Protocol -- A Technical
Description* (1996), Table 1 p. 5, "standard" and "data" cycles. `crc_bytes` less
the status byte and the two CRC bytes has to reproduce it."""


def test_published_generators():
    """The taps must be Annex I's two polynomials, read newest-bit-first.

    Compared against the binary strings the document prints rather than against a
    second octal constant, so the reversal itself is what is under test."""
    for octal, printed in zip(pactor2.CODE.generators, ("111101011", "101110001")):
        got = f"{octal:0{pactor2.CODE.constraint_length}b}"[::-1]
        check(f"generator {octal:#o} reverses to G={printed}", got == printed, got)


def _row_major(punc):
    """The mask read back as Annex I prints a vector: G0's row, then G1's."""
    return "".join(str(b) for row in punc.mask for b in row)


def _interleaved(punc):
    """The mask read back the other way: even positions G0, odd positions G1."""
    return "".join(str(b) for pair in zip(*punc.mask) for b in pair)


def test_published_puncture_vectors():
    """Both punctured levels must read back as Annex I's printed vectors.

    Annex I prints one serial vector per level and does not say how to fold it
    into per-generator rows. Under the ROW-MAJOR reading -- first half G0, second
    half G1 -- both shipped masks reproduce the printed vector EXACTLY, with no
    rotation, which is the whole reason that reading is the one shipped: SL3's
    mask is measured off the air (twelve real frames byte-exact, no other
    rotation returning any of them) and it is `1011` split down the middle.

    The other reading is carried and checked here too, so that neither candidate
    can drift: `PUNCTURE_7_8_INTERLEAVED` serialises even-odd to the printed
    vector rotated one place, which is the rotation that reading needs to explain
    SL3. Free distance separates none of them -- 3 for every rate-7/8 candidate
    over this K=9 code -- so only level-4 material can, and there is none."""
    for vector, punc in (("1011", coding.PUNCTURE_2_3),
                         ("10100110101011", coding.PUNCTURE_7_8)):
        check(f"{vector} read row-major is the shipped mask, unrotated",
              _row_major(punc) == vector, _row_major(punc))

    vector = "10100110101011"
    rotated = vector[-1] + vector[:-1]
    got = _interleaved(coding.PUNCTURE_7_8_INTERLEAVED)
    check("the alternative reading is that vector rotated one place, interleaved",
          got == rotated, f"{got} against {rotated}")


def test_published_geometry():
    """The field length every path derives must be whole, and must be the one
    published for that level -- the only check here with an outside referee."""
    for tag, paths in (("short", pactor2.PATHS), ("long", pactor2.PATHS_LONG)):
        for path, data_bytes in zip(paths, PUBLISHED_PAYLOAD[tag]):
            info = path.n_pairs - pactor2.FLUSH_BITS
            check(f"{path.name}: {path.n_buf} coded bits -> {info} info bits, whole bytes",
                  info % 8 == 0, f"{info / 8} bytes")
            check(f"{path.name}: payload is the published {data_bytes} bytes",
                  path.crc_bytes - 3 == data_bytes, f"got {path.crc_bytes - 3}")
            check(f"{path.name}: {path.n_pairs} steps is whole puncture periods",
                  path.n_pairs % path.puncture.period == 0,
                  f"period {path.puncture.period}")


def test_frame_chain():
    """Every speed level, both cycle lengths, back to the same bytes."""
    for path in pactor2.PATHS + pactor2.PATHS_LONG:
        info = bytes((i + 1) & 0xFF for i in range(path.crc_bytes - 2))
        coded = pactor2.encode_frame(pactor2.build_field(info, path), path)
        check(f"{path.name}: encoder fills {path.n_buf} coded bits exactly",
              coded.size == path.n_buf, f"got {coded.size}")
        src = pactor2.channel_of_code(path)
        check(f"{path.name}: helical stride {path.stride} is a permutation",
              sorted(src.tolist()) == list(range(path.n_buf)), "not a bijection")
        channel = np.zeros(path.n_buf)
        channel[src] = 1.0 - 2.0 * coded.astype(float)      # 0->+1, 1->-1
        field = pactor2.decode_frame(channel[src], path)
        body, crc = field[:-2], int.from_bytes(field[-2:], "little")
        check(f"{path.name}: {path.n_buf} coded bits -> CRC-16/X25 verifies",
              coding.crc16(body) == crc, f"crc {crc:#06x} vs {coding.crc16(body):#06x}")
        check(f"{path.name}: {len(info)} info bytes recovered", body == info,
              f"{body.hex()} != {info.hex()}")


ACCEPTED_FIELD = bytes.fromhex("534852494b45fee1")
ACCEPTED_CHANNEL_BITS = bytes.fromhex("a8d658b05d31530c117c8811668ef58dbbe9")
"""The SL1 short frame an independent decoder accepted, and the 144 channel-order
bits it accepted it as.

The field and channel bits pin the K=9 taps, LSB-first packing, the 8-bit
flush, Annex II's walk at depth 16, and the absence of a whitener."""


def test_accepted_vector():
    """Rebuild the exact frame an independent decoder read, from the top."""
    path = pactor2.PATHS[0]
    field = pactor2.build_field(b"SHRIKE", path)
    check("SL1 field an independent decoder accepted", field == ACCEPTED_FIELD, field.hex())
    coded = pactor2.encode_frame(field, path)
    air = coded[pactor2.interleave_pointer(path.n_buf, path.stride)]
    got = np.packbits(air).tobytes()
    check("SL1 channel bits an independent decoder accepted", got == ACCEPTED_CHANNEL_BITS, got.hex())
    check("the chain does not whiten",
          pactor2.decode_frame(1.0 - 2.0 * coded.astype(float), path) == field,
          "trellis output is not the field")


def test_data_burst_stagger_and_swap():
    """The transmitted data field carries the T/2 stagger, and the carrier swap
    carries the stagger with it.

    Graded by the shipped receiver, whose model of both is validated off real
    air (`INTEROP_OPEN`): `decode_expected_burst` places the leading lane half a
    symbol ahead of the other and reads the arrangement off the marker, so a
    transmitter that staggers wrongly, or swaps the tones without the stagger,
    hands it a field half a symbol from where its own marker says it is. The
    negative arm proves that sensitivity: the old construction -- the marker
    concatenated with an unstaggered `modulate` -- must not decode, or the
    positive arms are round-tripping something this test is not about.

    SIXTEEN FIELDS PER ARM, not one, because one is a draw rather than a rate.
    Acquisition scores a neighbouring 25 Hz bin pair within a few per cent of the
    right one on a single burst -- adjacent-bin leakage through a 32-tap window
    is large -- and when it takes the neighbour the field behind it does not come
    back. That costs about one field in sixteen on noiseless audio and it is a
    receiver's problem, not a transmitter's; grading a transmit convention on one
    draw of it turns a 6 per cent receive miss into a coin flip over the whole
    check. The floor sits below the measured rate and far above the negative arm,
    which returns nothing at all.
    """
    path = pactor2.PATHS[2]
    pad = np.zeros(FS // 2)
    fields = [pactor2.build_field(
        bytes(np.random.default_rng(s).integers(
            0, 256, path.crc_bytes - 2, dtype=np.uint8)), path)
        for s in range(16)]

    for swapped in (False, True):
        got = [p2rx.decode_expected_burst(np.concatenate(
            [pad, pactor2.data_burst(f, path, swapped=swapped, fs=FS), pad]), FS)
            for f in fields]
        exact = sum(g is not None and g[2] == f for g, f in zip(got, fields))
        arm = "swapped" if swapped else "home"
        check(f"SL3 data fields decode byte-exact, {arm} arrangement",
              exact >= 14, f"{exact} of {len(fields)}")

    flat = 0
    for field in fields:
        channel = np.zeros(path.n_buf, np.uint8)
        channel[pactor2.channel_of_code(path)] = pactor2.encode_frame(field, path)
        lanes = [channel.reshape(path.n_symbols, 2, path.bits_per_cell)[:, r, ::-1]
                 .ravel() for r in range(2)]
        got = p2rx.decode_expected_burst(np.concatenate([
            pad, pactor2.frame_marker(pactor2.marker_index(path.level), fs=FS),
            pactor2.modulate(lanes[1], lanes[0], path.bits_per_cell, fs=FS),
            pad]), FS)
        flat += got is not None and got[2] == field
    check("the unstaggered construction does not decode", flat == 0,
          f"{flat} of {len(fields)} decoded, so the positive arms prove nothing")


CAPTURE = corpora.RF_CORPUS / "p2hunt" / "hb9ak_055246_c1500.wav"


def test_dense_alphabet_origin_is_measured_off_air():
    """`cell_steps`' 8-DPSK origin must be the one a real transmitter emits.

    Every other convention in the chain is graded by a decode, and an origin is
    the one that cannot be: the receive path enumerates every half-sector
    rotation, so it absorbs any constant turn of the alphabet and a round trip
    stays green while the waveform is wrong. The measurement has to come from
    outside, and the frame marker supplies it -- eight known chips on the same
    carrier a few symbols before the field, which fix the rotation a carrier
    offset puts on every one-symbol differential.

    So: decode the recording, which gives the cell value behind every data
    symbol; fit each lane's marker to read off its rotation; de-rotate the field
    by it; and bin the measured differentials by the value that produced them.
    Each of the eight lands within a couple of degrees of `cell_steps`, at a
    concentration that leaves no room for a second answer. Against the origin
    this file emitted until 2026-08-03 every one of them is 45 degrees out -- one
    whole sector -- and a reference decoder printed no payload for any of it.
    """
    if not CAPTURE.exists():
        print(f"  skip alphabet origin ({CAPTURE} absent)")
        return
    audio, fs = p2rx._read_wav_mono(str(CAPTURE))
    grid, bin_pair, arms = p2rx.burst_grid(audio, fs)
    fields = dict(p2rx.decode_bursts(audio, fs))
    marks = {t: k for t, _b, k, _s, sc, _sw in p2rx.find_markers(audio, fs, 0.90)
             if sc >= p2rx.MARKER_ARM}
    path = pactor2.PATHS[2]
    m = 1 << path.bits_per_cell
    step = p2rx.FRAMES_PER_SYMBOL

    z = p2rx.bin_phasors(audio, fs)
    diff = np.zeros_like(z)
    diff[step:] = z[step:] * np.conj(z[:-step])
    diff /= np.abs(diff) + 1e-12

    seen = {v: [] for v in range(m)}
    for t, swapped in zip(grid, arms):
        if t not in fields or t not in marks:
            continue
        channel = np.zeros(path.n_buf, np.uint8)
        channel[pactor2.channel_of_code(path)] = pactor2.encode_frame(
            fields[t], path)
        cells = [pactor2._cell_values(
            channel.reshape(path.n_symbols, 2, path.bits_per_cell)[:, r, ::-1]
            .ravel(), path.bits_per_cell) for r in range(2)]
        chips = pactor2.marker_steps(marks[t])
        f0 = int(round(t * p2rx.ACQ_RATE))
        for rank, b in enumerate(p2rx.lane_tones(bin_pair, swapped)):
            lead = 0 if rank == 0 else p2rx.LANE_LEAD_FRAMES
            want = np.exp(1j * chips[1 - rank])
            fit = max(
                ((abs(np.mean(diff[idx, b] * np.conj(want))),
                  np.mean(diff[idx, b] * np.conj(want)), s)
                 for s in range(-2 * step, 2 * step + 1)
                 for idx in [f0 - lead + s - step * (7 - np.arange(8))]
                 if idx.min() >= 0 and idx.max() < len(diff)),
                default=(0.0, 0j, 0))
            if fit[0] < 0.85:
                continue
            at = f0 - lead + fit[2] + step * (np.arange(path.n_symbols) + 1)
            if at.max() >= len(diff):
                continue
            obs = diff[at, b] * np.conj(fit[1] / abs(fit[1]))
            for v in range(m):
                on = cells[rank] == v
                if on.any():
                    seen[v].append(obs[on])

    emitted = pactor2.cell_steps(path.bits_per_cell)
    checked = 0
    for v in range(m):
        if not seen[v]:
            continue
        pooled = np.concatenate(seen[v])
        off = np.degrees(np.angle(pooled.mean() * np.exp(-1j * emitted[v])))
        checked += 1
        check(f"real SL3 cell {v} sits where cell_steps puts it",
              abs(off) < 5.0 and abs(pooled.mean()) > 0.9,
              f"{off:+.1f} deg off over {pooled.size} symbols, "
              f"|r|={abs(pooled.mean()):.3f}")
    check("...and all eight cell values were measured", checked == m,
          f"{checked} of {m}")


INTEROP_OPEN = """\
[PACTOR-2 RECEIVE DECODES REAL OFF-AIR AUDIO -- speed level 3, 2026-08-02]
  Thirty-one frames of `hb9ak_055246_c1500.wav` decode to fields that are
  BYTE-EXACT against the fields an independent decoder's CRC accepted for the
  same bursts, unioned over three of its runs: Hamming distance 0 of 280 graded
  bits, thirty-one times over. Reproduce it with

      working/pactor/analysis/p2/verify.py

  which prints, per burst, the anchor, the arrangement, the field this package
  decodes, that decoder's own payload for it, and the distance between them. It
  goes through `p2rx.decode_bursts`, so what it prints comes out of the shipped
  receiver.

  THE NUMBER IS 31 OF 31, AND THE TRUTH FILE IS A UNION. The reference decoder's
  yield under emulation is nondeterministic -- three runs on identical audio
  accepted different subsets of the bursts -- so the grading set is the union of
  its CRC-ACCEPTED fields across runs, all 35 bytes of each.
  The run offers 8,192 alignments -- the same number it offered when it returned
  14, because the arrangement is READ and not searched -- so it buys 0.125
  expected false CRC accepts. A CRC accept is worth sixteen bits; 280 bits of
  agreement are not something chance does, which is why the graded count is the
  result and the accept count is not.

  THE OTHER HALF OF THE LINK WAS THE CARRIER SWAP. The two virtual carriers
  exchange tones every ARQ cycle, taking their channel rank and their T/2 stagger
  with them -- PACTOR-3's `spec.CARRIER_SWAP` mechanism, on two carriers instead
  of eighteen. It is one fact that shows up as two changes, which is why neither
  alone reads anything: on those 32 bursts, home arrangement 14 fields, stagger
  reversed by itself 0, lanes swapped by themselves 1, both together the other 15.
  The frame marker rides the same two carriers, so acquisition reads which
  arrangement a burst is in rather than trying both: the arrangement a burst is
  actually in scores 0.966-0.999 against 0.52-0.66 for the other, at all 30 of
  the 32 anchors that read above the arming threshold either way, and it names the
  arrangement that decodes at all 31 anchors that decode at all. `find_markers`
  was deaf on every other cycle before that, which is why these recordings' marker
  spacing measured 2.5 s where the link runs a 1.25 s cycle.

  THE OTHER TWO CAPTURES OF THE SAME QSO NOW DECODE TOO, and their counts are what
  an independent decoder reports for them: 26 of 27 bursts on `hb9ak_055208`,
  where that decoder reads 26 frames, and 9 of 9 on `hb9ak_055156`, where it reads
  9. All nine are byte-exact against the union of its CRC-accepted fields.
  055208 has no byte-level reference, so its 26 are a count agreement and not a graded
  one.

  TWO FAULTS HELD THIS OPEN, and neither is visible to a round trip.
    * THE DATA FIELD WAS DEMODULATED WITH A BOXCAR. The transmitted pulse is
      shaped across some three symbols, and section 5.1 had already measured a
      symbol-length boxcar at the carrier -- the textbook DPSK detector --
      recovering 0.36 of the frame-marker correlator's output where the shipped
      window gets 1.00. Nobody had applied that to the DATA. Boxcar: 0.33-0.55
      concentration under the 8th power, on a clock and grid phase fitted per
      window. `p2rx.bin_phasors`: 0.90-0.94, which is what a plant at 10 dB reads.
      A 3 dB frame and a 10 dB one, from the same audio.
    * THE SPEED-LEVEL-3 PUNCTURE WAS AT THE WRONG PHASE. Annex I prints `1011`;
      starting that at trellis step 0 gives ((1,1),(0,1)), and the truth is
      ((1,0),(1,1)) -- the same four-position pattern rotated one place right. A
      printed vector fixes the pattern, not where it starts. No other rotation
      returns any frame. `test_published_puncture_vectors` now asserts the
      rotation, not just the pattern.
  And one convention, measured the same way: `channel_buffer` takes a cell's three
  bits LEAST SIGNIFICANT FIRST, the order the field's bytes are packed in.

  WHAT THE SIGNAL ITSELF SETTLES, from the same material (pactor2.md 7.10):
    * SPEED LEVEL 3 IS 8-DPSK. Over the 72 symbols after a marker the differential
      concentrates at C8 = 0.90-0.94 with C1, C2 and C4 all at 0.01-0.16. An
      eighth-power-only concentration is what an eight-phase alphabet gives and
      what nothing else does.
    * THE CLOCK IS NOMINAL. The strong markers sit on an exact 2.5 s grid across
      40 s, so the period is 480 samples and the marker's frame grid is the
      data's; one residue then fixes every burst, including those whose own marker
      scores too low to anchor. `measure_symbol_period` fitted 481.92 across the
      whole file, which is a fit to the gaps as much as to the bursts.
    * THE FRAMES REPEAT AND THE SYMBOLS ARE FAITHFUL. Correlating the 76 symbols
      after each marker between every pair of bursts -- a statistic a constant
      rotation leaves alone, so there is nothing to fit -- gives 0.94-0.975 for
      many pairs against 0.35 for chance, and both carriers group the bursts
      identically. Nothing that mangled the symbols would do that 40 s apart.
    * AND THE REPEATS MEASURE THE GEOMETRY, INDEPENDENTLY OF THE DECODE. Two
      families differing only in a short run put every differing cell in four
      blocks at symbols 16-17, 34-35, 52-53 and 70-71: 18 symbols, so 108 channel
      positions, so six positions per symbol in symbol-major order and a depth-4
      interleaver whose columns are contiguous 108-position blocks. The run is 48
      positions, at the END of each column, which at rate 2/3 is 32 trellis steps,
      which at the end of a terminated K=9 trellis is the last 24 information bits
      and the 8 flush bits -- the status byte, the CRC and the flush, exactly what
      two copies of one idle packet differ by. THIS PACKAGE'S OWN ENCODER
      REPRODUCES THAT FINGERPRINT EXACTLY, for a status-only change and for a
      payload-two-bytes-earlier change, which is six chain parameters corroborated
      from the signal rather than from a document.

  WHAT IS STILL OPEN, stated as such.
    * ONE BURST OF THE 32 RETURNS NOTHING, at 38.150 s, in either arrangement: a
      selective fade sweeps the pair mid-field and takes each carrier in turn,
      and the reference decoder fails it too (pactor2.md 7.13). The 4.400
      and 33.150 bursts, unread when this was first written, decode since the
      magnitude-weighted softs and the anchored comb phase of 7.13.
    * SPEED LEVEL 4 IS UNTESTED. No level-4 material exists here, so its puncture
      vector keeps the phase it is printed in, unchecked, and its alphabet's
      origin rides SL3's rule with no measurement of its own. Levels 1, 2 and 3
      are read from outside -- see below.

[PACTOR-2 TRANSMIT IS READ FROM OUTSIDE -- speed levels 1, 2 and 3, 2026-08-03]
  An independent monitor prints the payload of what `pactor2.data_burst` renders.
  The 32 bursts of `hb9ak_055246_c1500.wav` re-rendered field for field -- same
  anchors, same arrangement, none of the recording's audio in them -- come back as
  that link's own payload stream, in order, from the same monitor.

  WHAT STOOD IN THE WAY WAS THE CONSTELLATION'S ORIGIN, at every level, and
  nothing in this file could have found it. `p2rx.decode_burst` enumerates every
  half-sector rotation of each carrier and `p2rx.burst_window` measures the data
  grid from the marker time our own transmitter put there, so transmit and receive
  cancelled a constant turn of the alphabet and the round trip stayed green while
  the waveform was wrong -- the same shape of fault as the interleaver depth and
  the code family, and the reason a clean round trip is necessary and never
  sufficient. `test_dense_alphabet_origin_is_measured_off_air` closes SL3 off the
  air with no outside decoder in the loop; `test_p2_oracle.py` closes all three
  against the monitor, and reads NO LEVEL AT ALL against the alphabets it
  carried until 2026-08-03.

  IT IS A REGRESSION FIXTURE NOW. `rf-corpus/regress` carries both recordings that
  have byte-level reference fields, as `pos_p2_sl3_hb9ak` and `pos_p2_sl3_hb9ak_055156`, with the
  known fields beside each, and the `p2fields` decoder grades decodes against
  those bytes rather than against a verdict -- the only fixtures in that suite
  that a plausible-looking wrong answer fails. Restoring the old puncture phase
  takes the first from 31 lines and 31 graded to 0 and 0; pinning the receiver
  back to the home arrangement takes it to 14 lines, and the short fixture to
  5, both below their gates.

  WHAT FOLLOWS IS THE EARLIER MATERIAL, on the two published recordings, which
  still do not decode. It is kept because the negatives are real and the
  calibration behind them is what made this recognisable when it arrived.

  FOUR THINGS THE SPECIFICATIONS SETTLE, that this package had been inferring.
    * The carriers are STAGGERED BY T/2 -- "Symbols on the carrier with the lower
      frequency always appear delayed by T / 2", of a two-carrier level the same
      document says is based on P2 speed level 1 and whose phase alphabet it
      calls "just like P2". Measured here at 0.315 and 0.554 of a symbol by two
      estimators before the sentence was found. The SIGN does not transfer:
      PACTOR-3 runs the same mechanism with the low carriers LEADING.
    * The DBPSK phases are PI/4 and 5*PI/4 -- diagonals, not 0 and PI.
    * The CRC is CCITT-CRC16, register preset 0xFFFF, complemented at the end,
      low byte transmitted first. That is what `coding.crc16` computes and
      `build_field` appends; it had been carried from PACTOR-3 on faith.
    * NO PACTOR DOCUMENT DESCRIBES A WHITENER, and PACTOR-2 has none. Settled
      since, against a decoder rather than by argument -- see below.

  THREE FAULTS ARE FIXED, each measured and each now gated above.
    * The symbol clock. A recording carries its own, and `fs / 100` is only the
      nominal. Both tones of the FEC sample independently measure 476.3 samples
      per symbol -- 100.77 Bd on that file's 48 kHz grid -- which walks a
      72-symbol frame 0.43 of a symbol end to end. Every earlier scan of these
      clips integrated on the nominal grid.
    * The constellation. `soft_bits` labels its two-bit case in the opposite
      rotational sense from the steps `modulate` emitted, and no rotation
      composes one into the other: the demapper recovered 0.545 of the bits at
      the best of eight half-step rotations. The alphabet sits on ODD MULTIPLES
      OF 45 deg, which is independently what pactor2.md §5.5 measures for
      the marker codeword targets.
    * The interleaver. Annex II prints its permutation routine outright and it is
      a FORWARD pointer walk; this package carried PACTOR-3's backward helical
      walk, a different permutation at all four depths.

  THE SCANNER IS CHECKED BY A PLANT, not trusted. A frame built with the Annex II
  permutation, sent at 100.77 Bd with a 7.3 Hz carrier offset and noise, comes
  back byte-exact at its planted start -- but only once the SENSE of the
  differential was added as a search axis. It came back at neither M nor 2M
  rotations without it, and that is what put the demapper fault in view.

  AND THE SCAN IS STILL AT CHANCE. Both whitening readings, both senses of the
  differential, both cell layouts, both carrier pairings, and the carriers on the
  published T/2 grid -- 4.4M trials, every accept a singleton with no ASCII
  structure, none surviving a one-symbol shift of the frame start. Counts and
  expected-false per arm are in pactor2.md §7.6.

  IT NO LONGER NEEDS TO BE BLIND, AND ANCHORED IT IS STILL AT CHANCE. Acquisition
  hands over three real frame starts on the FEC recording, so the search is a few
  dozen labellings at a known position rather than a million alignments. Enumerated
  there -- differential sense, per-lane constellation rotation, carrier pairing,
  cell layout, lane lag, both cycle lengths -- 63,360 trials for 0.97 expected
  false accepts, and ONE accept: a non-ASCII singleton whose neighbours at plus and
  minus one symbol reject under the same enumeration, and which recurs at neither
  other anchor. Exactly where chance puts it. Every arm is plant-validated first:
  SL1 and SL2, short and long, transmitted at the clip's own 100.77 Bd with a
  carrier offset and noise, found through the marker correlator rather than at a
  handed-over start, and recovered byte-exact. Levels 3 and 4 are not scanned on
  THAT clip -- its audio measures M = 4 in 34 of 34 packet-carrier pairs, which
  excludes them there. The level-3 material is separate, and is above.

  AND THE INSTRUMENT IS NOW CALIBRATED, which is what turns that from an absence
  into a result. Re-encoding the trellis output and correlating it back against its
  own softs tests every coded bit instead of the CRC's sixteen, and carries no
  false-accept budget, so it can be offered 40 symbols either side of each anchor:
  over 81 starts per anchor it returns medians that meet a permuted-symbol null's
  to the third decimal. A planted frame demodulated NO BETTER THAN THIS RECORDING
  -- concentration 0.807/0.814 against the clip's 0.802/0.842 -- scores 1.000 on
  that metric and returns its payload byte-exact through the CRC. So a conforming
  frame at this clip's own demodulation quality would have been found, at the start
  acquisition names, with the clock untouched. What is left is a two-way split:
  either the material carries no frame this chain can read, which is what the same
  decoder finding no valid field on the same clip says from the other side, or
  there is an axis the search does not span. It is no longer SNR, timing, the front
  end, the carrier assignment, the interleaver, the whitener or the geometry.

  AND THERE IS NOTHING TO COMBINE. The 362-symbol period is corroborated exactly by
  the marker spacing -- both gaps are 361.998 symbols on the measured clock -- but
  the three anchored data windows do not agree with each other: all twelve
  carrier-by-pair correlations sit below their own rolled null. What recurs at that
  period is the cycle structure, not the packet.

  ONLY THE FEC RECORDING COUNTS. The ARQ clip does not demodulate: at packet
  granularity with the period and phase fitted freely inside each packet, and a
  null built by phase-randomising the same audio, the FEC clip clears the null's
  maximum in 17 of 17 packets on both tones and the ARQ clip in 6 of 23 and 4 of
  23. Not a wrong rate -- a 60 to 160 Bd sweep peaks at 0.22, at rates the two
  tones disagree on. Not a wrong band, and not turnaround gaps. A clip that
  cannot be demodulated cannot support a negative, so its rows are withdrawn,
  and so is everything else ever taken through it.

  NO RECORDING CAN ADJUDICATE THIS. A conforming receiver builds its packet field
  before the +-5 Hz frequency gate, and on these two recordings that field is not
  CRC-valid for any frame, spliced into a live session or not. So an independent
  decoder is not reading them either, and the frame descriptor it forms is a
  header-codeword decision rather than corroborated ground truth: measured on the
  audio both tones carry DQPSK throughout, where that descriptor says DBPSK.

  The coding chain is checked by `test_accepted_vector`, which fixes the field
  and channel bits. Independent known-payload transmitters remain useful for
  receive tests that do not share our encoder's assumptions.

  ACQUISITION IS CLOSED TOO, from 2026-08-01. The marker is eight chips of a
  complex 45 deg codeword on each carrier's differential phase -- the chip table
  read as 32 COMPLEX codewords rather than 64 real ones -- nine pulses, the lower
  carrier leading by T/2. Sixteen markers played into a conforming receiver
  formed thirteen frame descriptors, every one on the right bin pair and carrying
  back the codeword index sent; and our own correlator reads the three real
  markers on the published FEC recording at the same times and indices that
  receiver reports. What was read as the packet-confirm state above the frame was
  never a gate: a plain stream of bursts prints from its FIRST frame, with no
  connect in front of it and no session to continue, once the constellation
  origin is right. Every CRC was failing, so nothing above the CRC was reached.
See docs/protocols/pactor/pactor2.md §1.3 for the symbol clock, §3 for the
interleaver, §5.10 for the coding chain and its accepted test vector, §7.6 for the blind scan and its trial counts, and §7.9 for the
anchored scan, its calibration and the arbiter question, and §7.12 for the three
measured alphabet origins and the monitor that reads what they emit.
working/pactor/analysis/p2/anchored.py runs the anchored arms."""


def test_verdict() -> None:
    """Every check above only accumulates, so the verdict is drawn here, last."""
    print(f"\n{PASS} passed, {len(FAILURES)} failed")
    print()
    print(INTEROP_OPEN)
    assert not FAILURES, (f"{len(FAILURES)} of {PASS + len(FAILURES)} FAILED: "
                          + "; ".join(FAILURES))
