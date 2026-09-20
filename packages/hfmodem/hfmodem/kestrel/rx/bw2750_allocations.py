# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Measured BW2750 record0/1/2 allocation bins, stock VARA HF 4.9.0.

Records1/2 have228 columns over bins12..51; reference positions and stride2 are shared with
BW2300, but these allocations are not. Record1 is the unique table common to
all256 trailer-field trials over two independent22-byte payloads. Record2 was
inverted from a known37-byte closing frame then validated on distinct47-byte
frames and an empty close. No closed-form allocation law is claimed.

Native evidence and sample/hash provenance:
    tests/kestrel/fixtures/bw2750-low-levels/{README.md,provenance.json}
Record0 has124 columns over bins24..103; its independent two-payload
provenance is in fixtures/bw2750-floor/. These tables support native receive
and waveform generation; selectable TX interoperability needs separate ARQ
qualification.
"""

# Native host L1: unique intersection of all 256 field-byte hypotheses over
# two independently known nine-byte payloads. Both recordings decode CRC-clean
# with 24/24 references. See fixtures/bw2750-floor/{provenance,allocation}.json.
ALLOC_REC0 = (
    65, 60, 43, 44, 65, 52, 25, 78, 55, 98, 79, 76, 73, 98, 63, 98,
    91, 44, 61, 74, 93, 30, 77, 46, 93, 48, 59, 88, 25, 40, 83, 52,
    35, 56, 97, 56, 91, 40, 39, 24, 75, 54, 63, 88, 61, 68, 77, 96,
    87, 100, 35, 56, 29, 90, 47, 26, 63, 40, 53, 40, 43, 34, 57, 76,
    45, 40, 37, 38, 61, 46, 99, 34, 87, 40, 83, 24, 89, 86, 71, 24,
    61, 80, 83, 84, 35, 42, 45, 48, 97, 28, 73, 24, 99, 24, 101, 78,
    67, 32, 43, 46, 63, 72, 51, 34, 65, 82, 43, 66, 73, 94, 65, 70,
    57, 80, 67, 40, 33, 46, 47, 80, 71, 48, 31, 60,
)

ALLOC_REC1 = (
    15, 12, 19, 24, 17, 50, 21, 42, 47, 14, 43, 40, 45, 28, 41, 44,
    35, 18, 41, 22, 43, 44, 27, 18, 45, 32, 15, 12, 51, 44, 23, 20,
    13, 34, 37, 38, 33, 46, 43, 28, 17, 40, 47, 32, 41, 28, 17, 46,
    21, 38, 17, 46, 43, 32, 19, 26, 33, 48, 27, 24, 13, 22, 37, 12,
    21, 36, 37, 32, 17, 26, 39, 12, 13, 36, 37, 36, 21, 38, 31, 18,
    29, 40, 49, 32, 47, 30, 33, 36, 21, 32, 29, 46, 43, 22, 49, 20,
    21, 16, 47, 12, 25, 44, 19, 30, 39, 36, 43, 48, 25, 32, 33, 20,
    47, 18, 41, 24, 39, 50, 31, 30, 27, 24, 43, 38, 51, 44, 23, 32,
    27, 42, 51, 26, 49, 28, 29, 26, 29, 30, 49, 28, 17, 20, 21, 36,
    13, 46, 21, 22, 29, 20, 25, 14, 25, 28, 15, 40, 37, 28, 47, 32,
    15, 48, 35, 48, 43, 30, 35, 16, 41, 20, 25, 22, 45, 14, 43, 44,
    25, 30, 15, 26, 47, 36, 13, 14, 33, 22, 13, 14, 37, 24, 45, 22,
    43, 28, 27, 24, 43, 24, 41, 12, 51, 46, 43, 28, 13, 26, 19, 36,
    43, 46, 13, 40, 49, 26, 49, 50, 35, 30, 25, 44, 41, 44, 25, 46,
    45, 28, 13, 32,
)

ALLOC_REC2 = (
    33, 36, 41, 38, 35, 30, 21, 50, 41, 50, 43, 46, 39, 44, 13, 12,
    49, 34, 35, 22, 17, 40, 39, 14, 35, 50, 41, 16, 21, 12, 31, 22,
    47, 24, 35, 14, 15, 12, 47, 40, 39, 12, 21, 12, 33, 16, 25, 22,
    45, 26, 39, 20, 49, 30, 19, 46, 29, 20, 27, 38, 23, 12, 51, 16,
    37, 26, 13, 14, 37, 34, 35, 20, 43, 38, 13, 48, 29, 40, 29, 16,
    25, 46, 15, 24, 13, 48, 13, 42, 49, 46, 21, 12, 23, 14, 33, 44,
    21, 20, 37, 42, 17, 28, 39, 40, 21, 30, 25, 34, 21, 44, 39, 12,
    33, 28, 33, 34, 51, 22, 17, 28, 29, 24, 49, 20, 17, 32, 29, 44,
    15, 12, 13, 42, 47, 46, 19, 40, 29, 26, 21, 12, 33, 38, 43, 30,
    51, 44, 39, 12, 13, 46, 31, 40, 29, 22, 13, 22, 43, 32, 45, 46,
    39, 12, 13, 16, 43, 18, 27, 40, 45, 40, 13, 26, 21, 36, 51, 14,
    47, 46, 41, 38, 39, 12, 21, 16, 21, 38, 23, 16, 25, 40, 45, 48,
    23, 12, 19, 50, 39, 38, 25, 38, 41, 24, 19, 34, 21, 48, 23, 46,
    13, 24, 37, 34, 45, 48, 33, 18, 49, 34, 25, 48, 51, 28, 35, 12,
    27, 22, 19, 32,
)
