# PACTOR-4 C2 permutations: runtime representation and offline derivation

The runtime needs **13** distinct C2 permutation sizes:
280, 348, 352, 704, 1056, 1160, 1408, 1656, 4140, 8280, 12420, 16560, 20700.
Twelve have recovered deterministic CML recipes. **1160 does not yet have a
recovered CML seed/history.** The 2484 table has a recovered recipe useful for
understanding the 1656 session, but is research-only and is not a runtime table.

The permutation is a **direct gather order**:
`component2_input[j] = systematic[permutation[j]]`. Inverting the permutation
changes the coding operation. Canonical hashes encode every entry as unsigned
16-bit little-endian, in that direct order, with no header.

## Fast runtime representation

`hfmodem.shrike.p4c2.permutation(size)` builds a deterministic column walk and
applies an ordered list of position swaps. All thirteen lists are explicit,
packed, compressed and hash-pinned resources. Some large permutations need
nearly N swaps, so “compact” refers to the representation, not a claim that
every permutation differs at only a few positions. For `w = isqrt(N)`, the walk
is the concatenation of `range(r,N,w)` for `r=0,…,w−1`. Thus each value 0…N−1
appears exactly once. A correction swap exchanges two positions, preserving
bijection. The complete canonical digest is checked against `TABLE_SHA256`.

A permutation relative to the walk can be decomposed into cycles; each cycle of
length k needs k−1 transpositions. These compact correction swaps describe the
final permutation. **They are not the accepted stochastic swaps in the original
CML run.** In particular, representing 1160 this way does not solve its original
seed, spread or earlier session calls. No random generation or C compiler is
needed during normal modem use.

## Public runtime inputs

[The runtime constructor](../../../packages/hfmodem/hfmodem/shrike/p4c2.py)
contains both the compressed-resource hashes and canonical permutation hashes.
Download the corresponding resource below to reproduce a runtime table in
another implementation. Inflate with zlib; read each 4-byte record as two
unsigned little-endian 16-bit **position indices**; apply records in order to
the base walk. Check indices, record count, bijection and canonical digest.
The input bits are then gathered directly, never through the inverse table.

| N | Correction swaps | Resource | Canonical u16-LE SHA-256 |
|---|---|---|---|
| 280 | 6 | [c2-280.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-280.swaps.z) | `8b24b531066d9e020b423c3a05b22304bce162470d40dd4b5600480232a53e3a` |
| 348 | 12 | [c2-348.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-348.swaps.z) | `c5e38726a9247ae94b619c0c64762d706e62314f8ff9582aa9440e8a915c8b0c` |
| 352 | 4 | [c2-352.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-352.swaps.z) | `a96675e46a0e83610c759bf188b1a1829f5ad1bea64c58b79eb7e599624ed0cb` |
| 704 | 14 | [c2-704.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-704.swaps.z) | `70aa4076ce903714266abf094d987226d5ffe8496eb22d467e04c567081cf323` |
| 1056 | 86 | [c2-1056.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-1056.swaps.z) | `17c2726165e352e9e7e9547d99ce85a05520dd9e8805b61482883c3857eba1b3` |
| 1160 | 22 | [c2-1160.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-1160.swaps.z) | `ab46bf3a863a8831fa5ad56ab8d57ea3b415af38139e8b3dccd1cb2355a8e621` |
| 1408 | 62 | [c2-1408.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-1408.swaps.z) | `ae62d0c442d7a252418c4a27f1d2c72c4db5901f80e03859155c76b25c078481` |
| 1656 | 158 | [c2-1656.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-1656.swaps.z) | `8b6afde2f0b64cef83d314b44a927f4b7a1592237d25c3fb96af6c038585f132` |
| 4140 | 4130 | [c2-4140.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-4140.swaps.z) | `c07e40058cc80445f12749c119e2e09fc6a69099a0ed9cda0e4f9d61ee85d504` |
| 8280 | 8270 | [c2-8280.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-8280.swaps.z) | `5ce426b7fcb363ee9fda0e560a7c4b3c8ae958d79374222e7d52d603a7007c91` |
| 12420 | 12412 | [c2-12420.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-12420.swaps.z) | `4c4b450028c6ac66b87647702dfeeafa04f0621583f60fc3161a1ecc31036a17` |
| 16560 | 16548 | [c2-16560.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-16560.swaps.z) | `c2fadc1e3764cc08f66e0286751eb201012277aed8424dedd3a77e620a55cfff` |
| 20700 | 20688 | [c2-20700.swaps.z](../../../packages/hfmodem/hfmodem/shrike/assets/c2-20700.swaps.z) | `2a3bd42d9d423fc7ad84df654c4e80f9e801dc842d2c57364cf4965b35c4565b` |

## Original CML constructor

The offline package is
[`hfmodem.shrike.c2proof`](../../../packages/hfmodem/hfmodem/shrike/c2proof/__main__.py).
It includes the original public CML `srandom.h`, its attribution/license, and a
standalone C adaptation. See its `NOTICE.md` and `COPYING.LESSER`.

The CML implementation by Matthew C. Valenti and Yufei Wu (2005–2006) creates the
same column walk, then proposes two positions using Numerical Recipes ran2. It
accepts their exchange only when both incoming values pass the **old** array's
neighborhood tests. At position p the scanned window is
`[max(0,p−S), min(N,p+S))`; a value v is rejected if any window entry a has
`abs(v−a) < S`. The upper endpoint is excluded and the lower included. Tests
include the old entries at the proposed positions; neither is removed before
checking. Replacing this with a symmetric window or a general “S-random” test
can change the table.

CML initializes ran2 once per session, discards that initialization call's
returned value, then consumes two further values for every proposal, including
rejections and equal indices. Its original time seed sums zero-based month,
day, hour, minute and second. The offline tool supplies the recovered sum
explicitly. It preserves `floor(N * min((1.0/2147483563)*iy, 1−1.2e−7))` in
double precision; do not compile with `-ffast-math`. Each subsequent constructor
call resets the walk, **not the RNG**.

Upstream's default budget is 10N accepted swaps. The recovered recipes below use
20N. This is an explicit constructor parameter, not a claim that upstream's
unmodified default produces these permutations.

## Recovered recipes

Each entry `(N,S,B)` is one complete call with B accepted swaps. Calls in a row
share a seed and continuous RNG state.

| Runtime size | Seed | Ordered calls through the kept output |
|---:|---:|---|
|280|5|(280,15,5600)|
|348|68|(348,16,6960)|
|352|81|(352,17,7040)|
|704|30|(704,24,14080)|
|1056|82|(1056,28,21120)|
|1160|—|**Unresolved; no recipe claimed**|
|1408|79|(1408,33,28160)|
|1656|99|(2484,38,49680) → (2484,40,49680) → (1656,32,33120)|
|4140|88|(4140,40,82800)|
|8280|51|(8280,55,165600)|
|12420|51|(8280,55,165600) → (12420,65,248400)|
|16560|91|(16560,70,331200)|
|20700|51|(8280,55,165600) → (12420,65,248400) → (20700,85,414000)|

280 also has matching seed 17 and 60 alternatives; the tool selects 5. The first
seed 99 / 2484 / S38 trial is unkept but consumes RNG and cannot be omitted. The
research-only 2484 output is the second call of that session. The normal runtime
has no 2484 selector.

## Reproduce a proof

With hfmodem and a C compiler installed:

```sh
python -m hfmodem.shrike.c2proof list
python -m hfmodem.shrike.c2proof self-test
python -m hfmodem.shrike.c2proof verify 280
python -m hfmodem.shrike.c2proof verify 348 --method scalar
```

`list` does not compile or regenerate anything. `verify` compiles the bundled C
source in a temporary directory, executes the complete selected recipe, checks
length/range/bijection, compares **every entry and every canonical byte** with
`p4c2.permutation(size)`, and compares generated/reference/runtime-declared
SHA-256 values. Its JSON record includes all stage proposal/accepted counts,
source and actual binary hashes, the method and full recipe. A stored expected
digest alone never sets `cml_proof: true`.

`verify 1160` reports `unresolved_recipe` with `cml_proof: false` and exits 3.
Runtime representation consistency is distinct from original-generator proof.
`verify --all-solved` explicitly requests all twelve solved runtime sizes; it
may take hours or longer. Separate selectors with shared session prefixes
currently recompute those prefixes. Do not confuse the tiny self-test with
having rerun every recovered recipe.

An optional per-stage proposal cap is available:

```sh
python -m hfmodem.shrike.c2proof verify 280 --max-proposals 100000000
```

Default 0 means no cap. A cap exits 3, emits no final table, and reports
`incomplete_cap`, never a mismatch or negative proof. Arbitrary histories may
be unable to reach their requested accepted-swap budget, so callers should
choose a cap for exploratory recipes. An operating-system interruption likewise
cannot produce a successful proof record.

The standalone generator also provides a slow original-algorithm path for any
explicit seed/history, independently of recovered selectors:

```sh
python -m hfmodem.shrike.c2proof generate --seed 99 \
  --call 2484,38,49680 --call 2484,40,49680 --call 1656,32,33120 \
  --method scalar --output generated.u16le
```

Output must be a new file. `generate` reports `generated_not_compared`; producing
a permutation is not proof that it matches a runtime table. Research-only2484
can be compared against an explicitly supplied canonical reference with
`verify 2484 --reference reference.u16le`; no research table is embedded here.

## Exact acceleration and small controls

`scalar` uses the literal CML guard for every unequal proposal. Default `cached`
maintains a necessary-condition flag for each position: whether **any** value
0…N−1 could pass its window. Window entry a forbids the integer interval
`[a−S+1,a+S−1]`, clipped to 0…N−1. A coverage count of zero identifies a value
outside their union. Since every actual incoming value is in that range, no
legal proposal can touch a position with a false flag. Candidates still undergo
both literal guards; a flag is never sufficient for acceptance.

Changing array position i only changes windows centered in
`[i−S+1,i+S]`, clipped to valid positions. Recomputing those flags after each
accepted exchange restores the invariant. Cache work consumes no RNG. `audit`
also computes the literal scalar decision on **every** proposal and fails if
it differs, including a skipped proposal.

`self-test` compares scalar/cached/audit final bytes against the bundled upstream
implementation with only its seed and budget made explicit. It covers a small
lower-boundary case, ordinary accepted/rejected proposals, a continuous two-call
session, zero-budget initialization, and incomplete-cap behavior. All three
adapted modes must also agree on every stage's proposal/draw counts. These are
small deterministic equivalence controls; they do not replace full selected
runtime comparisons or prove an unrecovered 1160 recipe.
