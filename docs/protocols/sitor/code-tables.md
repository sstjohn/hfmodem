<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# HF telegraphy reference tables: ITA2, CCIR 476 / SITOR, NAVTEX

Sources (primary, fetched and extracted with `pdftotext -layout` for exact
transcription, not OCR/vision-only reads):

- **[ITU-T Rec. S.1 (03/93)](https://www.itu.int/rec/T-REC-S.1-199303-I/en)**, "International Telegraph Alphabet No. 2" —
  `T-REC-S.1-199303-I!!PDF-E.pdf`
- **[ITU-R Rec. M.476-5 (10/1995)](https://www.itu.int/rec/R-REC-M.476-5-199510-I/en)**, "Direct-printing telegraph equipment in
  the maritime mobile service" — `R-REC-M.476-5-199510-I!!PDF-E.pdf`
  (legacy equipment conventions; listed by ITU as in force)
- **[ITU-R Rec. M.625-4 (03/2012)](https://www.itu.int/rec/R-REC-M.625-4-201203-I/en)**, "Direct-printing telegraph equipment
  employing automatic identification in the maritime mobile service" —
  `R-REC-M.625-4-201203-I!!PDF-E.pdf` (automatic-identification SITOR/AMTOR specification;
  listed by ITU as in force)
- **ITU-R Rec. M.540-2 (06/1990)**, "...automated direct-printing telegraph
  system for promulgation of navigational and meteorological warnings..." —
  `R-REC-M.540-2-199006-I!!PDF-E.pdf`
- **IMO MSC.1/Circ.1403/Rev.2 (28 Nov 2022), NAVTEX Manual, 2023 edition**
  (current operative document for B1/B2/B3B4 semantics; M.540 itself
  delegates these assignments to IMO and does not enumerate them)

Cross-checks against independent, non-ITU sources:

- **Wikipedia, "CCIR 476"** (fetched via WebFetch) — gives the CS1–CS5,
  idle-α/β and RQ patterns as plain binary strings.
- **`dl-fldigi`** (`ukhas/dl-fldigi`, a public fork of the fldigi amateur
  digital-modes program), `src/cw_rtty/rtty.cxx` — real open-source Baudot
  letters/figures tables, used to confirm the US-TTY figure-case variant.

No unresolved disagreements were found between ITU-R M.476 and M.625 on the
7-unit code tables (they are byte-for-byte identical where both define a
signal). One **version difference**, not a disagreement, is flagged in §3.

---

## 1. ITA2 (ITU-T S.1)

### 1.1 Coding rules

- 32 combinations, 5 units each. **Condition A** = start polarity = no tape
  perforation = binary `0`; **condition Z** = stop polarity = perforation =
  binary `1`. ITU-T S.1 §3.2, note 2: "start"/"stop" and "space"/"mark" are
  synonyms for A and Z respectively.
- **Bit order:** "In serial transmission, code element 1 is transmitted
  first" (Table 1/S.1 note). So on the wire: start bit (space) → unit 1 →
  unit 2 → unit 3 → unit 4 → unit 5 → stop (mark).
- S.1 does not itself specify baud rate or stop-bit length (that's a
  transmission-standard matter, e.g. ITU-T R-series / national practice).
  **Amateur 45.45 Bd RTTY convention** (not from S.1; corroborated by
  `w1hkj.org/modes/rtty.htm`, `kloth.net/radio/rtty.php`, sigidwiki): 1 start
  bit + 5 data bits + **1.5 stop bits**, 170 Hz shift.
- Combinations 6, 7, 8 (F, G, H) are **unassigned in figure case
  internationally** (S.1 §4.2) — reception must not trigger a repeat
  request. National/US practice fills them (see 1.3).

### 1.2 Full table (`combo_no, letters, figure_intl, bits12345`)

```python
ITA2_TABLE = [
    # (combination_no, letter_case, figure_case_ITA2, bits "12345" A/Z)
    (1,  "A", "-",              "ZZAAA"),
    (2,  "B", "?",              "ZAAZZ"),
    (3,  "C", ":",              "AZZZA"),
    (4,  "D", "WRU",            "ZAAZA"),   # who-are-you / answerback, printed as an enquiry glyph
    (5,  "E", "3",              "ZAAAA"),
    (6,  "F", None,             "ZAZZA"),   # unassigned internationally
    (7,  "G", None,             "AZAZZ"),   # unassigned internationally
    (8,  "H", None,             "AAZAZ"),   # unassigned internationally
    (9,  "I", "8",              "AZZAA"),
    (10, "J", "BEL",            "ZZAZA"),   # audible signal
    (11, "K", "(",              "ZZZZA"),
    (12, "L", ")",              "AZAAZ"),
    (13, "M", ".",              "AAZZZ"),
    (14, "N", ",",              "AAZZA"),
    (15, "O", "9",              "AAAZZ"),
    (16, "P", "0",              "AZZAZ"),
    (17, "Q", "1",              "ZZZAZ"),
    (18, "R", "4",              "AZAZA"),
    (19, "S", "'",              "ZAZAA"),
    (20, "T", "5",              "AAAAZ"),
    (21, "U", "7",              "ZZZAA"),
    (22, "V", "=",              "AZZZZ"),
    (23, "W", "2",              "ZZAAZ"),
    (24, "X", "/",              "ZAZZZ"),
    (25, "Y", "6",              "ZAZAZ"),
    (26, "Z", "+",              "ZAAAZ"),
    (27, "CR", None,            "AAAZA"),   # carriage return, either case
    (28, "LF", None,            "AZAAA"),   # line feed, either case
    (29, "LTRS", None,          "ZZZZZ"),   # letter shift
    (30, "FIGS", None,          "ZZAZZ"),   # figure shift
    (31, "SPACE", None,         "AAZAA"),
    (32, "NUL", None,           "AAAAA"),   # all-space / unperforated tape / blank
]
```

### 1.3 US-TTY figure-case variant

Confirmed against `dl-fldigi/src/cw_rtty/rtty.cxx` (`letters[32]` /
`figures[32]` arrays — the figures table is explicitly the "U.S. version").
Exactly **8 combinations** differ from the ITA2 figure case above:

```python
USTTY_FIGURE_OVERRIDES = {
    # combo_no: (ITA2_intl_figure, USTTY_figure)
    4:  ("WRU", "$"),
    6:  (None,  "!"),
    7:  (None,  "&"),
    8:  (None,  "#"),
    10: ("BEL", "'"),   # apostrophe moves here...
    19: ("'",   "BEL"), # ...and BEL moves here
    22: ("=",   ";"),
    26: ("+",   '"'),
}
```

All other letters/figures (including A–C, E, I, K–P, R, T, U, W, X, Y and
all control characters) are identical between ITA2 and US-TTY.

---

## 2. CCIR 476 / ITU-R M.476 & M.625 seven-unit code

### 2.1 Coding rules

- **Constant-ratio code**: every valid signal has exactly 4 marks (`B`) and
  3 spaces (`Y`) among its 7 units — C(7,3) = 35 possible codewords total.
  Guarantees detection of any single-bit error.
- Table footnotes (identical wording in M.476 and M.625):
  1. "A represents start polarity, Z represents stop polarity" — same A/Z
     convention as ITA2, applied to the *ITA2 code* column only.
  2. **"B represents the higher emitted frequency and Y the lower."**
  3. **"The bit in bit position 1 is transmitted first; B = 0, Y = 1."**
     (M.476 only states rule 2 explicitly in its own footnotes; M.625 adds
     rule 3 (`B=0,Y=1`, bit-1-first) explicitly. Not a disagreement — M.625
     is simply more explicit and is the automatic-identification recommendation.)
- FSK: 100 Bd, 170 Hz shift, class F1B/J2B, audio centre 1700 Hz (M.476
  notes some legacy equipment instead centred on 1500 Hz).
- 32 of the 35 possible codewords are used for **traffic** (Table 1); the
  remaining 3 plus 5 of the traffic codewords are reused for **service
  signals** (Table 2) — see the dual-meaning table in §2.4.

### 2.2 Table 1 — the 32 traffic combinations

Verified via `pdftotext -layout` text-layer extraction of both M.476-5 and
M.625-4 — **byte-for-byte identical** between the two Recommendations for
every row.

```python
# (combo_no, letter_case, figure_case, ITA2_bits12345, seven_unit_BY, seven_unit_binary)
# seven_unit_binary uses B=0, Y=1, bit position 1 transmitted first (leftmost char).
CCIR476_TABLE = [
    (1,  "A",    "-",       "ZZAAA", "BBBYYYB", "0001110"),
    (2,  "B",    "?",       "ZAAZZ", "YBYYBBB", "1011000"),
    (3,  "C",    ":",       "AZZZA", "BYBBBYY", "0100011"),
    (4,  "D",    "WRU",     "ZAAZA", "BBYYBYB", "0011010"),
    (5,  "E",    "3",       "ZAAAA", "YBBYBYB", "1001010"),
    (6,  "F",    None,      "ZAZZA", "BBYBBYY", "0010011"),
    (7,  "G",    None,      "AZAZZ", "BYBYBBY", "0101001"),
    (8,  "H",    None,      "AAZAZ", "BYYBYBB", "0110100"),
    (9,  "I",    "8",       "AZZAA", "BYBBYYB", "0100110"),
    (10, "J",    "BEL",     "ZZAZA", "BBBYBYY", "0001011"),
    (11, "K",    "(",       "ZZZZA", "YBBBBYY", "1000011"),
    (12, "L",    ")",       "AZAAZ", "BYBYYBB", "0101100"),
    (13, "M",    ".",       "AAZZZ", "BYYBBBY", "0110001"),
    (14, "N",    ",",       "AAZZA", "BYYBBYB", "0110010"),
    (15, "O",    "9",       "AAAZZ", "BYYYBBB", "0111000"),
    (16, "P",    "0",       "AZZAZ", "BYBBYBY", "0100101"),
    (17, "Q",    "1",       "ZZZAZ", "YBBBYBY", "1000101"),
    (18, "R",    "4",       "AZAZA", "BYBYBYB", "0101010"),
    (19, "S",    "'",       "ZAZAA", "BBYBYYB", "0010110"),
    (20, "T",    "5",       "AAAAZ", "YYBYBBB", "1101000"),
    (21, "U",    "7",       "ZZZAA", "YBBBYYB", "1000110"),
    (22, "V",    "=",       "AZZZZ", "YYBBBBY", "1100001"),
    (23, "W",    "2",       "ZZAAZ", "BBBYYBY", "0001101"),
    (24, "X",    "/",       "ZAZZZ", "YBYBBBY", "1010001"),
    (25, "Y",    "6",       "ZAZAZ", "BBYBYBY", "0010101"),
    (26, "Z",    "+",       "ZAAAZ", "BBYYYBB", "0011100"),
    (27, "CR",   None,      "AAAZA", "YYYBBBB", "1110000"),
    (28, "LF",   None,      "AZAAA", "YYBBYBB", "1100100"),
    (29, "LTRS", None,      "ZZZZZ", "YBYBBYB", "1010010"),
    (30, "FIGS", None,      "ZZAZZ", "YBBYBBY", "1001001"),
    (31, "SPACE",None,      "AAZAA", "YYBBBYB", "1100010"),
    (32, "NUL",  None,      "AAAAA", "YBYBYBB", "1010100"),
]
```

All 32 rows independently verified to have exactly 4×B/3×Y; all 32
seven-unit codewords are distinct.

### 2.3 Table 2 — service information signals

```python
# Mode A (ARQ) control/idle/repeat signals, and Mode B (FEC) phasing/idle signals.
# Same B=0/Y=1, bit-1-first convention as Table 1.
SERVICE_SIGNALS = {
    "CS1":              "BYBYYBB",  # mode A only
    "CS2":              "YBYBYBB",  # mode A only
    "CS3":              "BYYBBYB",  # mode A only
    "CS4":              "BYBYBBY",  # mode A only — M.625 addition, NOT in M.476
    "CS5":              "BYYBYBB",  # mode A only — M.625 addition, NOT in M.476
    "idle_beta":        "BBYYBBY",  # both modes A and B, same role in both
    "idle_alpha":       "BBBBYYY",  # mode A idle-alpha == mode B phasing signal 1 / idle alpha
    "signal_repetition":"YBBYYBB",  # mode A RQ == mode B phasing signal 2
}
```

CS4 and CS5 exist **only in M.625** (added for the automatic
MMSI-identification handshake — "call signal correctly received during
phasing" / "...during rephasing"). M.476 (the legacy standard) defines only
CS1–CS3 plus idle-α, idle-β and RQ.

### 2.4 Dual-meaning combinations (flagged per request)

Checking Table 2's 8 patterns against Table 1's 32 shows **5 of the 8
service signals reuse a traffic codeword**, disambiguated only by protocol
state (control-signal slot vs. information-block slot in the synchronous
ARQ cycle), and **3 are spare codewords** (of the 35 total, 32−35=3 left
over) not used for traffic at all:

```python
DUAL_MEANING = {
    "CS1": {"also_means": "combination 12 (L / ')')"},
    "CS2": {"also_means": "combination 32 (NUL / blank)"},
    "CS3": {"also_means": "combination 14 (N / ',')"},
    "CS4": {"also_means": "combination 7  (G, unassigned figure)"},
    "CS5": {"also_means": "combination 8  (H, unassigned figure)"},
    "idle_alpha":         {"also_means": "one of the 3 spare (non-traffic) codewords; "
                                          "in mode B this SAME pattern is also Phasing signal 1"},
    "signal_repetition":  {"also_means": "spare codeword; in mode B this SAME pattern is Phasing signal 2"},
    "idle_beta":          {"also_means": "spare codeword; same role (filler) in both mode A and mode B"},
}
```

**RQ's pattern is phasing signal 2**, and it is **idle-α's pattern** that doubles
as **phasing signal 1** (M.625 Table 2 literally labels that Mode B cell
"Phasing signal 1, Idle signal α").

### 2.5 Cross-check against Wikipedia

Wikipedia's "CCIR 476" article gives the same 8 service-signal patterns as
plain binary, e.g. CS1 = `1100101`. These are **not** written in ITU's
bit-position-1-first convention — they are the *reverse* of the ITU string
with B=1/Y=0 instead of B=0/Y=1. Reversing the ITU string
`BYBYYBB`→(B=1,Y=0)→`1010011`→reverse→`1100101` reproduces Wikipedia's
value exactly, and this holds for all 8 signals (CS1–CS5, idle-α, idle-β,
RQ). **This is a documentation-convention difference, not a technical
disagreement** — flagged per the instructions, but resolved: both sources
describe the identical on-air signal.

---

## 3. SITOR mode B (FEC)

Per M.625-4 §4 (current) and M.476-5 §3.2 (legacy, consistent except where
noted):

- **Time-diversity interleave:** every character is sent twice. The first
  transmission (**DX**) is followed by four other characters' DX slots,
  then the retransmission (**RX**) of the first character follows — i.e. a
  5-character-deep interleave, with the DX/RX pair of the same character
  separated by **280 ms** (4 × 70 ms at 100 Bd, 7 units/char = 70 ms/char).
- **Phasing preamble:** the sending station transmits "phasing signal 1"
  and "phasing signal 2" alternately — **phasing signal 2 in the DX
  position, phasing signal 1 in the RX position**. **M.476 requires at
  least 4 signal pairs; M.625 raises this to at least 16 pairs** — a real
  version difference, not a measurement disagreement (M.625 is current).
  NAVTEX practice (IMO NAVTEX Manual, built on M.625) specifies phasing
  ≥10 s for a first message / ≥5 s for a repeat message in the same
  transmission, far exceeding either minimum (≈70/35 pairs at 140 ms per
  pair).
- **Receiver behaviour:** the receiving station "checks both signals (DX
  and RX), and uses the unmutilated one. When both signals appear as
  unmutilated but different, then both signals should be considered as
  mutilated" (M.625 §4.3, verbatim) — printed as `Δ` (combination 31,
  space) or a user-defined error glyph.
- **Selective B-mode (SBSS/SBRS):** the call sign (4- or 7-identification-
  signal call identity per Rec. M.491) is sent 6 times, each occurrence
  being (identification signals + idle-β), using the same time-diversity
  transmission. **All signals following the phasing preamble — call sign
  and traffic — are sent inverted, in a 3B/4Y ratio** (i.e. Table 1's B/Y
  values flipped) to distinguish selective from collective traffic; a
  receiver that locks onto its own (correctly inverted) call sign re-
  inverts subsequent traffic back to normal 4B/3Y before printing, but the
  bits remain inverted on the air for every other listening station.
- **End of emission:** at least 3 consecutive "idle signal α" characters
  in the DX position immediately after the last traffic signal; the
  receiver reverts to standby "not less than 210 ms after receipt of at
  least two consecutive idle signals α in the DX position" (M.625 §4.6.7).
  NAVTEX's own end-of-transmission spec (idle-α for ≥2 s) is consistent
  with this — it is simply a longer run of the same 3-signal minimum.

---

## 4. SITOR mode A (ARQ) timing

A synchronous system (M.625-4 §3, identical timing to M.476-5 §3.1): the
information sending station (ISS) transmits a **block of 3 characters
(3 × 7 units = 210 ms)**, then pauses 240 ms while listening; the
information receiving station (IRS) replies with **one 70 ms control
signal**, then pauses 380 ms — both sides thus share a **450 ms basic
timing cycle**, master-clocked by whichever station initiated the call
(the "master" station; the callee is "slave" for the life of the circuit
regardless of which side later becomes ISS/IRS). CS1 and CS2 alternately
acknowledge odd/even-numbered blocks; CS3 requests a change of traffic
direction; a mutilated block draws a "signal repetition" (RQ) triplet; a
constant equipment delay `t_E` (kept as short as practicable, occasionally
padded 20–40 ms to let a nearby receiver recover from transmitter
desensitisation) separates the end of a received signal from the start of
the reply at the phase-locked slave station.

---

## 5. NAVTEX (ITU-R M.540-2 + IMO NAVTEX Manual 2023)

### 5.1 Carrier frequencies

- **518 kHz** — International NAVTEX service, English only (worldwide).
- **490 kHz** and **4209.5 kHz** — National NAVTEX services, national
  language/content permitted.
- ITU delegated coordination authority for all three frequencies to IMO
  at WRC-95 (Resolution 339, reaffirmed WRC-97). M.540-2 §Annex II adds a
  ±10 Hz mark/space frequency tolerance requirement.
- Uses the collective B-mode (SITOR-B/FEC) of M.476/M.625: 100 Bd, 170 Hz
  shift.

### 5.2 Message format

```
ZCZC B1 B2 B3 B4 <CR><LF> <message text> NNNN <CR><LF><LF>
```

- **Phasing signals** precede `ZCZC`: ≥10 s for the first message in a
  transmission, ≥5 s for subsequent messages in the same transmission.
- **`ZCZC`** marks the end of the phasing period / start of the message
  identity group (need not be printed).
- **B1** — single letter (A–X only in practice; see §5.3), identifies the
  *transmitter*. Used by the receiver to accept/reject a whole station.
- **B2** — single letter, identifies the *subject* of the message (see
  §5.4). B2 = A, B, D, L cannot be deselected/rejected by the receiver
  (mandatory categories); B2 = D also triggers an audible alarm.
- **B3B4** — two-digit sequential serial number, 01–99 per (B1,B2)
  combination, used by the receiver to suppress duplicate printing of an
  already-received message for 72 hours. **B3B4 = 00 is reserved**: it
  cannot be rejected/deselected and overrides any B1/B2 filtering — used
  only for messages that must always reach every receiver.
- **`NNNN`** marks end of message text.
- Extra (redundant) letter/figure shifts are recommended within the
  message body to reduce the effect of garbles (M.540-2 Annex II §7).

### 5.3 B1 — transmitter identification character

From the current NAVTEX Manual (Table 2): B1 is allocated by the IMO
NAVTEX Coordinating Panel, one letter per transmitter, **A through X**
(24 letters; Y and Z are not allocated as B1 values in the current
scheme). Each letter has a fixed 10-minute slot every 4 hours, staggered
by 10 minutes per letter:

```python
# B1 letter -> six UTC transmission start times (HHMM), repeating every 4 h
NAVTEX_B1_START_TIMES = {
    "A": ["0000","0400","0800","1200","1600","2000"],
    "B": ["0010","0410","0810","1210","1610","2010"],
    "C": ["0020","0420","0820","1220","1620","2020"],
    "D": ["0030","0430","0830","1230","1630","2030"],
    "E": ["0040","0440","0840","1240","1640","2040"],
    "F": ["0050","0450","0850","1250","1650","2050"],
    "G": ["0100","0500","0900","1300","1700","2100"],
    "H": ["0110","0510","0910","1310","1710","2110"],
    "I": ["0120","0520","0920","1320","1720","2120"],
    "J": ["0130","0530","0930","1330","1730","2130"],
    "K": ["0140","0540","0940","1340","1740","2140"],
    "L": ["0150","0550","0950","1350","1750","2150"],
    "M": ["0200","0600","1000","1400","1800","2200"],
    "N": ["0210","0610","1010","1410","1810","2210"],
    "O": ["0220","0620","1020","1420","1820","2220"],
    "P": ["0230","0630","1030","1430","1830","2230"],
    "Q": ["0240","0640","1040","1440","1840","2240"],
    "R": ["0250","0650","1050","1450","1850","2250"],
    "S": ["0300","0700","1100","1500","1900","2300"],
    "T": ["0310","0710","1110","1510","1910","2310"],
    "U": ["0320","0720","1120","1520","1920","2320"],
    "V": ["0330","0730","1130","1530","1930","2330"],
    "W": ["0340","0740","1140","1540","1940","2340"],
    "X": ["0350","0750","1150","1550","1950","2350"],
}
```

Note: M.540-2 itself only says "B1 is a letter (A–Z) identifying the
transmitter coverage area" and delegates the actual assignment scheme to
IMO/the NAVTEX Manual — this timing table is from the 2023 NAVTEX Manual,
not from the ITU-R Recommendation.

### 5.4 B2 — subject indicator character

From the current NAVTEX Manual (Table 1):

```python
NAVTEX_B2_SUBJECTS = {
    "A": "Navigational warnings",                                    # mandatory, cannot be rejected
    "B": "Meteorological warnings",                                  # mandatory, cannot be rejected
    "C": "Ice reports",
    "D": "Search and rescue info, piracy, tsunami, other phenomena",  # mandatory, cannot be rejected; triggers alarm
    "E": "Meteorological forecasts",
    "F": "Pilot and VTS service messages",
    "G": "AIS service messages (non-navigational aid)",
    "H": "LORAN messages",
    "I": "Currently not used",
    "J": "GNSS messages regarding PRN status",
    "K": "Other electronic navigational aid system messages",
    "L": "Other navigational warnings — additional to B2 = A",        # mandatory, cannot be rejected
    "M": "Currently not used", "N": "Currently not used",
    "O": "Currently not used", "P": "Currently not used",
    "Q": "Currently not used", "R": "Currently not used",
    "S": "Currently not used", "T": "Currently not used",
    "U": "Currently not used",
    "V": "Special services allocation by IMO NAVTEX Coordinating Panel",
    "W": "Special services allocation by IMO NAVTEX Coordinating Panel",
    "X": "Special services allocation by IMO NAVTEX Coordinating Panel",
    "Y": "Special services allocation by IMO NAVTEX Coordinating Panel",
    "Z": "No message on hand",
}
```

M.540-2 itself does not enumerate this table (Annex II §2.1: "B2 characters
...are defined by IMO... combination numbers 1-26" of Table 1 of Rec. 476
and 625, i.e. just the ITA2 letters A–Z used as tags) — the semantic
mapping above is IMO's, from the current NAVTEX Manual, and has evolved
since 1990 (e.g. G = AIS, J = GNSS/PRN status did not exist when M.540-2
was last revised). If implementing against an older document, be aware
this table is the operative one now, not something derivable from M.540-2
alone.

---
