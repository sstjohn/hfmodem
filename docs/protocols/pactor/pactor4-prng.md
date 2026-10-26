<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-4 C2 generator and PRNG

The recovered Coded Modulation Library (CML) recipes construct the second turbo component's input
permutation. They reproduce twelve of the thirteen runtime sizes. The
1160-entry permutation is available exactly, but its original seed and call
history remain unresolved.

This PRNG belongs to offline table construction. Normal modem operation uses
fixed permutations; it does not negotiate a random seed for each packet.
The packet whitener is a separate Galois register, described in
[packet framing](pactor4-framing.md).

## ran2 state and recurrence

The constructor uses Numerical Recipes ran2 as implemented by CML. Its state
is two integer recurrence states, a 32-entry shuffle array and the previous
combined output. All indices below are zero-based.

| Parameter | First recurrence | Second recurrence |
|---|---:|---:|
| Modulus | 2147483563 | 2147483399 |
| Multiplier | 40014 | 40692 |
| Quotient divisor | 53668 | 52774 |
| Remainder coefficient | 12211 | 3791 |

One draw advances both recurrences. The quotient form avoids multiplication
overflow in the original implementation:

```text
k = x // 53668
x = 40014 * (x - k * 53668) - k * 12211
if x < 0: x += 2147483563

k = y // 52774
y = 40692 * (y - k * 52774) - k * 3791
if y < 0: y += 2147483399

j = previous // 67108862
previous = shuffle[j] - y
shuffle[j] = x
if previous < 1: previous += 2147483562

u = min((1.0 / 2147483563) * previous, 1.0 - 1.2e-7)
```

The multiplication and clamp use double precision. A proposed position is
`floor(N * u)`. Replacing this with integer modulo, a different floating-point
rounding rule or another generator changes the proposal stream. The bundled
constructor must not be compiled with `-ffast-math`.

## Initialization and call history

For an explicit nonnegative seed, initialize `x = y = max(seed, 1)`.
Advance the first recurrence 40 times, with loop index 39 down to 0; store
its last 32 outputs at `shuffle[31]` through `shuffle[0]`. Set
`previous = shuffle[0]`, then execute and discard one complete ran2 draw.
The recovered seeds are listed in the [recipe table](pactor4-c2-derivation.md#recovered-recipes).

CML's time seed is the sum of zero-based month, day of month, hour, minute
and second. A recovered sum does not identify a unique capture date or time.
The offline tool supplies that sum directly.

Initialize once per session. Each constructor call resets its permutation
array, but retains both recurrence states, the shuffle array and `previous`.
An earlier discarded table therefore changes every later table in the session.

## Permutation construction

Start with a column walk. For `w = floor(sqrt(N))`, concatenate
`range(r, N, w)` for `r = 0 … w−1`.

For each proposal:

1. Draw `i = floor(N * ran2())`, then `j = floor(N * ran2())`.
2. Reject equal positions. Both draws have already been consumed.
3. Test the old value at j against i's neighborhood, and the old value at i
   against j's neighborhood. Both tests read the unchanged array.
4. Exchange the values only if both tests pass. Count one accepted swap.

At destination p, the neighborhood is the half-open interval
`[max(0, p−S), min(N, p+S))`. An incoming value v fails if any old entry a
in that interval has `abs(v−a) < S`. The old entries at the proposed positions
are included. A symmetric endpoint convention or testing an already-swapped
array produces a different constructor.

Rejected proposals still consume two draws. The recovered recipes stop after
20N **accepted swaps**, while upstream CML defaults to 10N. A proposal-count
limit is an execution limit, not an alternative recipe.

## Sessions that must stay intact

| Kept output | Seed | Calls in order: (N, S, accepted swaps) |
|---|---:|---|
| 352 | 81 | (352, 17, 7040) |
| 1656 | 99 | (2484, 38, 49680) → (2484, 40, 49680) → (1656, 32, 33120) |
| 20700 | 51 | (8280, 55, 165600) → (12420, 65, 248400) → (20700, 85, 414000) |
| 1160 | Unresolved | No original-generator recipe claimed |

The first seed-99 output is discarded, but the call consumes random state. The second yields
a research-only 2484-entry table; 2484 is not a runtime selector. Seed 51 is
shared by the 8280, 12420 and 20700 outputs, taken at successive call boundaries.
See the [complete recipes and hashes](pactor4-c2-derivation.md).

## Runtime tables and evidence

The runtime starts from the column walk and applies stored position swaps.
Those swaps encode the final permutation; they are not the sequence of accepted
CML proposals. All thirteen runtime tables have canonical hashes, including
1160. Exact runtime reconstruction does not resolve 1160's original history.

The convention is direct gather:
`component2_input[j] = systematic[permutation[j]]`.
Canonical bytes are unsigned 16-bit little-endian entries in that order.
Using the inverse permutation changes the turbo code.

[Reproduction commands](pactor4-c2-derivation.md#reproduce-a-proof) distinguish
complete entry-by-entry comparison, unresolved recipes and interrupted runs.
The small self-test checks algorithm agreement; it does not rerun all twelve
recovered recipes or demonstrate radio interoperability.

## Small reproducibility control

Seed 1 with one call `(N=9, S=2, accepted swaps=4)` completes after 127
proposals and 254 draws, excluding the discarded initialization draw.
Its canonical u16-LE SHA-256 is
`bbc4cf10386548b3c02ff9f38423310212ab41310d82a1b94bc514b887ed4db3`.
This is an algorithm control, not a PACTOR-4 packet size. The bundled
`python -m hfmodem.shrike.c2proof self-test` compares this and other small
cases across the upstream, scalar, cached and audit implementations.

## Source definitions

- [Standalone constructor](../../../packages/hfmodem/hfmodem/shrike/c2proof/cml/cml_generate.c)
- [CML source](../../../packages/hfmodem/hfmodem/shrike/c2proof/cml/upstream/srandom.h)
- [Recovered recipes and verification](../../../packages/hfmodem/hfmodem/shrike/c2proof/__main__.py)
- [Runtime reconstruction](../../../packages/hfmodem/hfmodem/shrike/p4c2.py)
- [CML attribution and license](../../../packages/hfmodem/hfmodem/shrike/c2proof/NOTICE.md)
