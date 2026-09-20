# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The register of what the published spec does NOT tell us.

This module exists to keep an *assumption* from ever being mistaken for a *fact*.
Anything in `spec.py` is published. Anything here is a hole, together with the
candidate set we intend to search and the evidence that would close it.

Each Unknown carries:
    status   : OPEN | ASSUMED | RESOLVED
    candidates: the search space (small, on purpose)
    evidence : what would settle it

Rule: a value here may be USED (we have to guess something to make a signal at
all), but every use is tagged, and `validate.py` prints the register so we can
never quietly ship a guess believing it was in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPEN = "OPEN"
ASSUMED = "ASSUMED"        # a working default, not yet confirmed by anything
RESOLVED = "RESOLVED"      # confirmed by an independent decoder or by capture


@dataclass
class Unknown:
    key: str
    what: str
    status: str
    value: Any = None
    candidates: list[Any] = field(default_factory=list)
    evidence: str = ""
    note: str = ""


REGISTER: dict[str, Unknown] = {}


def _reg(u: Unknown) -> Unknown:
    REGISTER[u.key] = u
    return u


# --- U1: convolutional generator polynomials ------------------------------
# The spec says only "an optimum rate 1/2 convolutional code with a constraint
# length of 7 or 9". "Optimum" is a term of art: for rate 1/2 these are the
# maximum-free-distance codes tabulated by Odenwalder, and they are what every
# other system uses. Very likely correct, but NOT stated -- so it lives here.
U1_GENERATORS = _reg(Unknown(
    key="U1_GENERATORS",
    what="Rate-1/2 convolutional generator polynomials for K=7 and K=9",
    status=RESOLVED,
    value={7: (0o133, 0o171), 9: (0o753, 0o561)},
    evidence="An independent decoder builds its trellis from exactly this pair at "
             "each constraint length -- K=9 = 0o753/0o561, K=7 = 0o133/0o171, held "
             "bit-reversed against a popcount-parity table for the branch outputs. "
             "They are also the textbook optimum codes, so the value is checkable "
             "from Odenwalder's table alone.",
    note="Build order is 0o753-first / 0o133-first. The transmit G0/G1 ORDER is "
         "fixed downstream at `placement.CONV_GENERATORS` and is no longer a coin "
         "flip: six candidate pairs against both bit orders were built into a "
         "case-1 header frame, and exactly one -- (0o171, 0o133) packed LSB-first "
         "-- passes an independent decoder's CRC and reads the field back "
         "byte-exact, against thirteen matched negatives.",
))

# --- U2: puncture patterns ------------------------------------------------
# Rate 3/4 (SL5) and 8/9 (SL6) are punctured from the rate-1/2 K=7 code. The
# spec says the pattern "is chosen carefully" but does not give it.
U2_PUNCTURE = _reg(Unknown(
    key="U2_PUNCTURE",
    what="Puncture patterns for rate 3/4 (SL5) and rate 8/9 (SL6)",
    status=RESOLVED,
    value={"3/4": ((1, 1, 0), (0, 1, 1)),
           "8/9": ((0, 1, 0, 1, 1, 0, 1, 0), (1, 0, 1, 0, 1, 0, 1, 1))},
    evidence="An independent decoder holds these as a (period, keep-mask) table: "
             "mask 0x5 for rate 3/4, 0xB366 for rate 8/9. Expanded into the tuples "
             "above as coding.PUNCTURE_3_4 / PUNCTURE_8_9.",
))

# --- U3 / U8: interleaver --------------------------------------------------
U3_INTERLEAVER = _reg(Unknown(
    key="U3_INTERLEAVER",
    what="'Full-frame bit-interleaving' geometry, and how the interleaved frame "
         "maps onto (tone x symbol)",
    status=OPEN,
    candidates=["rectangular block, rows x cols over the whole frame",
                "bit-reversal",
                "tone-major vs symbol-major fill order"],
    evidence="An independent decoder reading our payload back bit-exactly. Jointly "
             "searched with U1/U2 -- an interleaver error and a generator error "
             "are indistinguishable from outside the decoder, both showing up as a "
             "failed CRC, so vary one at a time.",
))

# --- U4: control signal codewords ------------------------------------------
U4_CS_CODEWORDS = _reg(Unknown(
    key="U4_CS_CODEWORDS",
    what="The six 20-bit control-signal codewords (CS1..CS6)",
    status=RESOLVED,
    value=(0x52D56, 0xAABA2, 0xAD45A, 0x95AC5, 0x74339, 0x4B4AD),
    evidence="All 15 "
             "pairwise Hamming distances are exactly 12/20 (=24/40), which meets "
             "the Plotkin bound with equality. Six 20-bit words cannot be spaced "
             "that evenly by accident, so the set checks itself. See "
             "spec.CONTROL_SIGNALS and docs/protocols/pactor/pactor3.md.",
    note="Sent DBPSK on BOTH tone 5 and tone 12 (same 20 bits each). The distance "
         "property above is what makes the set checkable without a live link.",
))

# --- U5: tone maps ---------------------------------------------------------
U5_TONE_MAPS = _reg(Unknown(
    key="U5_TONE_MAPS",
    what="Which channels each speed level uses",
    status=RESOLVED,
    value="see spec.SPEED_LEVELS",
    evidence="Recovered by column-position extraction from the source PDF figure, "
             "then triple-checked: tone counts match the published physical data "
             "rates; every map is symmetric about 1500 Hz; every map contains the "
             "header channels 5 and 12.",
))

# --- U6: header symbol mapping ---------------------------------------------
U6_HEADER_MAPPING = _reg(Unknown(
    key="U6_HEADER_MAPPING",
    what="How header bits map to symbols",
    status=OPEN,
    candidates=["VH = 32 bits split 16/16 across channels 5 and 12, 8 symbols "
                "each at 2 bits/symbol (DQPSK-like), CH = 16 bits in 8 symbols",
                "headers always DBPSK => 16 symbols, contradicting '8 symbols'"],
    evidence="Off-air capture; and an independent decoder reporting the correct "
             "speed level for our synthetic signal (M3 exit).",
    note="The source is genuinely ambiguous: it says headers are '8 symbols each' "
         "yet gives 32-bit VH and 16-bit CH codes, which only reconciles at "
         "2 bits/symbol. But it also says the CS are 'always sent in DBPSK in "
         "order to obtain maximum robustness', implying headers are NOT. "
         "The cycle graphic separately says '8 bytes / 4 bytes', which matches "
         "neither. Do not guess -- measure.\n"
         "\n"
         "ATTEMPTED off-air 2026-07-13, in the working record -- NOT solved. "
         "The model 'header = per-tone DQPSK, symbols 2..9 after the detected "
         "onset' does NOT fit real signal: a CONSTANT header (fixed on every "
         "packet) shows only 0.37 cross-packet agreement (chance 0.25). Three "
         "concrete leads for the next attempt, in priority order:\n"
         " 1. MATCHED FILTER, not boxcar. PACTOR-3 tones are 120 Hz apart at "
         "    100 Bd with pulse shaping and NO OFDM orthogonality, so a plain "
         "    downconvert+boxcar has heavy adjacent-tone bleed (ICI) that corrupts "
         "    per-tone symbols. Demodulate each tone through the raised-cosine "
         "    matched filter (modem.raised_cosine) to reject neighbours. This is "
         "    the most likely cause of the 0.37.\n"
         " 2. PACKET TIMING. Our onset detector fires ~0.85x/s (fading edges), far "
         "    more than the true 1.25/3.75 s packet grid -- so symbols are read "
         "    from the wrong place. Detect packet starts from the ENERGY-envelope "
         "    rise out of the inter-packet gap, and/or lock to the cycle grid a "
         "    monitor reports (CYC bits), then coherently AVERAGE the same header "
         "    across many aligned packets so fading cancels.\n"
         " 3. 'ALTERNATELY on tones 5 and 12' may mean the header symbols "
         "    INTERLEAVE between the two tones (sym0->t5, sym1->t12, ...), not the "
         "    32-bit code split 16/16 per tone (which is what shrike/frame.py "
         "    currently assumes). Untested.",
))

# --- U7: phase reference pulse ---------------------------------------------
U7_PHASE_REF = _reg(Unknown(
    key="U7_PHASE_REF",
    what="Definition of the 10 ms phase-reference pulse (per-tone phase, amplitude)",
    status=OPEN,
    candidates=["all tones at phase 0", "per-tone phase schedule chosen to "
                "minimise crest factor (the spec makes a point of the low CFR)"],
    evidence="Off-air capture: measure the first symbol of a real packet.",
    note="A differential receiver needs *a* reference; ours must match closely "
         "enough for a real receiver to lock.\n"
         "\n"
         "MEASURED 2026-07-13 (tests/shrike/test_modem.py): our naive schedule (every "
         "tone starting at phase 0) yields a crest factor 6.3 dB ABOVE the published "
         "table -- 12.0 dB vs 5.7 dB at SL6. The spec says PACTOR-III is "
         "'designed to ... minimiz[e] the CF', so a per-tone phase schedule is "
         "certainly missing. The published CFR column is therefore an OBJECTIVE "
         "FUNCTION: candidate schedules can be scored offline, needing neither a "
         "decoder nor captures.\n"
         "\n"
         "UNRESOLVED PUZZLE, worth understanding before optimising blindly: SL1 "
         "uses just TWO tones, and the published CFR is 1.9 dB. Two equal-amplitude "
         "carriers have an envelope |1 + exp(j.dw.t)| that sweeps 0..2, giving a "
         "~6 dB peak-to-RMS FLOOR. 1.9 dB is below what two independent tones can "
         "physically achieve. So either (a) SCS measures crest factor differently "
         "from peak-to-RMS (e.g. against peak envelope power, or a different "
         "reference), or (b) the two SL1 tones are not independent -- e.g. they "
         "carry related data, or unequal amplitudes. Do NOT tune a phase schedule "
         "against this column until the definition is pinned down, or we will be "
         "optimising toward a number that means something else.",
))

# --- CRC variant -----------------------------------------------------------
CRC_VARIANT = _reg(Unknown(
    key="CRC_VARIANT",
    what="Exact CCITT-CRC16 variant: init, reflection, xorout, and the byte span covered",
    status=RESOLVED,
    value="CRC-16/X-25 (poly 0x1021, init 0xFFFF, refin/refout true, xorout 0xFFFF, residue 0xF0B8)",
    evidence="coding.reflected_crc_ccitt_table generates the reflected table "
             "from polynomial 0x8408; the good-frame residue is 0xF0B8. "
             "shrike.coding.crc16 defaults to 'x25'.",
    note="Appended as 2 bytes little-endian.",
))

# --- Data whitening / scrambler --------------------------------------------
SCRAMBLER = _reg(Unknown(
    key="SCRAMBLER",
    what="Data whitening/scrambler -- table or LFSR, and where in the chain it applies",
    status=RESOLVED,
    value="word[i] ^= reflected_CRC_CCITT_table[i], on info bytes pre-encode, reset per packet",
    evidence="The 16-bit whitening words are the reflected CRC-16/CCITT table "
             "generated from its polynomial. See coding.whiten and "
             "coding.reflected_crc_ccitt_table.",
    note="Its own inverse. Applies to PACTOR-3, not PACTOR-2.",
))

# --- The published duplicate ------------------------------------------------
CH_DUPLICATE = _reg(Unknown(
    key="CH_DUPLICATE",
    what="CH7 and CH11 are both 0x5a3c in the primary source",
    status=RESOLVED,
    value="genuine duplicate -- 0x5a3c occupies two entries of an independent "
          "decoder's CH table",
    evidence="The two entries are distinct positions holding the same codeword, so "
             "the published pair is a duplicate rather than a typo. That table "
             "carries 18 entries; the published 16 are those minus 0xc328 and "
             "0xaa33.",
))


# --- Speed level 2, the one level a decoder will not read -------------------
SL2_HOME_ARRANGEMENT = _reg(Unknown(
    key="SL2_HOME_ARRANGEMENT",
    what="Why speed level 2 packets were read in the swapped carrier arrangement "
         "and never in the home one",
    status=RESOLVED,
    value="the two halves of the level 2 comb are not transmitted together -- "
          "carriers 3, 5 and 7 lead 10, 12 and 14 by half a symbol, and the lead "
          "belongs to the virtual carrier, so the swap moves it (spec.SUBBAND_LEAD)",
    evidence="Each lit channel's "
             "header block carries a known eight-dibit word, so the instant a "
             "channel best matches its own word is that carrier's symbol clock, and "
             "two real level 2 packets measure the low half 0.46 and 0.52 symbol "
             "early where five real level 3 and two real level 6 packets measure "
             "0.02 to 0.09. Staggered, the transmit path scores 8 of 8 accepted "
             "frames where it scored 0, and every level now passes every arm of "
             "tests/shrike/test_p3_oracle.py with no level exempt.",
    note="THE SIGN IS OPPOSITE to the only published statement of this mechanism. "
         "PACTOR-4 section 11.7 says of its own two-carrier speed level 1 that "
         "symbols on the lower-frequency carrier are DELAYED by half a symbol; "
         "here the lower carriers LEAD, and late was rendered, offered and scores "
         "zero. Do not reconcile the two. What propped the wrong reading up for so "
         "long: both a staggered and an unstaggered comb decode through our own "
         "receiver on clean audio -- the stagger costs the misread half its margin, "
         "not its decode -- so only an external decoder could see it.",
))
# === HOST INTERFACE ========================================================
# shrike.ptc emulates an SCS PTC over WA8DED/CRC hostmode. The framing is fully
# documented (PTC-IIIusb manual v4.1 ch. 10) and was driven end to end by a real
# client -- Pat 1.0.0 via harenber/ptc-go over a pty: terminal init, JHOST4,
# connect, a Winlink B2F handshake in both directions, disconnect. What is left
# here is the part no document pinned down and no client we could run exercised.

HOST_IDENT = _reg(Unknown(
    key="HOST_IDENT",
    what="Which identification/firmware string a Winlink client requires of a PACTOR-3 PTC",
    status=ASSUMED,
    value="PTC-IIIusb, firmware 4.1, BIOS 2.90 (%V -> '4.1 2.90')",
    candidates=["PTC-IIIusb 4.1", "PTC-IIusb 4.0", "PTC-IIpro 4.0", "DR-7800 2.x"],
    evidence="A capture of Winlink Express negotiating with a real PTC-IIIusb, or "
             "the RMS Express PTC driver, showing whether it gates on the banner "
             "or on %V at all.",
    note="Taken verbatim from the manual's own hostmode start banner (Display "
         "10.1.1), so it is a real shipped combination. Pat never reads it.",
))

HOST_LINK_MESSAGES = _reg(Unknown(
    key="HOST_LINK_MESSAGES",
    what="Exact wording of the code-3 link status strings an SCS PTC emits for PACTOR",
    status=ASSUMED,
    value="'(ch) CONNECTED to CALL' / '(ch) DISCONNECTED fm CALL'",
    candidates=["WA8DED AX.25 wording (what we use)",
                "PACTOR-specific wording, e.g. the terminal mode's CONNECTED banner"],
    evidence="A hostmode transcript of a real PTC taking a PACTOR connect. The "
             "PTC-IIIusb manual documents the channel and the code, never the text.",
    note="From the WA8DED host mode user's guide, which gives the full list -- "
         "BUSY fm, CONNECTED to, LINK RESET, DISCONNECTED fm, LINK FAILURE with, "
         "CONNECT REQUEST fm, FRAME REJECT -- all AX.25-flavoured. Of those, "
         "CONNECTED to, DISCONNECTED fm and LINK FAILURE with (which 10.4.10 also "
         "names for an exhausted repeat count) are emitted here. Pat reads link "
         "state from the L command instead and ignores these entirely, so a client "
         "that parses them is the risk.",
))

HOST_L_CODE = _reg(Unknown(
    key="HOST_L_CODE",
    what="Response code the PTC uses for the L (link status) and @B replies",
    status=ASSUMED,
    value=1,
    candidates=[1, 3],
    evidence="Any hostmode capture: the byte after the channel number in an L reply.",
    note="The six-field body and its NUL termination are confirmed (ptc-go parses "
         "them positionally); only the code byte is a guess, and every client we "
         "have seen skips it.",
))

HOST_WINLINK_COMMANDS = _reg(Unknown(
    key="HOST_WINLINK_COMMANDS",
    what="The command subset Winlink Express and Airmail actually exercise",
    status=OPEN,
    candidates=["ptc-go's set (verified)", "plus scanner sync %W", "plus %O/%I/%Q",
                "plus terminal-mode commands we accept but do not model"],
    evidence="A serial capture of Winlink Express or Airmail driving a real PTC. "
             "Both are Windows-only, so neither could be run here.",
    note="Verified against Pat/ptc-go only. shrike.ptc accepts and stores any "
         "unrecognised terminal command rather than rejecting it, so an unknown "
         "setting cannot abort a session -- but an unknown *hostmode* command "
         "answers OK with no effect, which could mislead a client.",
))

HOST_JHOST5 = _reg(Unknown(
    key="HOST_JHOST5",
    what="What JHOST5 (extended CRC hostmode) owes a client that asks for it",
    status=ASSUMED,
    value="served as JHOST4",
    candidates=["identical to JHOST4 for a modem with no FAX channel",
                "refuse it so the client falls back to JHOST4"],
    evidence="A capture of a client that sends JHOST5 and then uses the longer "
             "frames it buys.",
    note="10.9.1 says JHOST5 buys one thing: data packets up to 1024 bytes on the "
         "FAX channel 252, PTC to PC, with the two extra length bits in bits 4 and "
         "5 of the code byte. There is no channel 252 here, so every frame we emit "
         "is a legal JHOST4 frame that a JHOST5 master also parses. Neither ptc-go "
         "nor BPQ32's SCS driver sends JHOST5; both send JHOST4.",
))

HOST_UNDOCUMENTED_COMMANDS = _reg(Unknown(
    key="HOST_UNDOCUMENTED_COMMANDS",
    what="Hostmode verbs real clients send that manual chapter 10 does not list",
    status=ASSUMED,
    value="DD breaks the link immediately; a command starting '#' is run as a "
          "terminal-mode command",
    evidence="SCS's own documentation of either, or a capture of a real PTC "
             "answering them.",
    note="Chapter 10 gives neither. DD: harenber/ptc-go's forceDisconnect sends "
         "exactly `DD` on the PACTOR channel, and 10.4.2 describes D-twice as "
         "'corresponding to DD in PACTOR', so the verb exists. The '#' prefix: "
         "BPQ32's SCSPactor.c calls it a \"hidden feature where you can send any "
         "normal mode command in host mode by preceeding with a #\" and uses it "
         "for `#DD` and `#MYL <n>`. Both are served here because a client sends "
         "them; neither is in the manual.",
))

HOST_REFUSAL_TEXTS = _reg(Unknown(
    key="HOST_REFUSAL_TEXTS",
    what="Which code-2 failure texts an SCS PTC emits, and when",
    status=ASSUMED,
    value="the WA8DED guide's list: INVALID COMMAND, TNC BUSY - LINE IGNORED, "
          "CHANNEL ALREADY CONNECTED, STATION ALREADY CONNECTED",
    candidates=["the WA8DED four", "an SCS-specific set", "PACTOR-specific wording"],
    evidence="A hostmode capture of a real PTC refusing a command.",
    note="The PTC-IIIusb manual names code 2 (10.4.32) and never a text. The "
         "WA8DED host mode user's guide gives exactly the four above and no "
         "other. 'CHANNEL NOT CONNECTED' appears in neither, so a D or a data "
         "write on an unconnected channel is answered code 0 here rather than "
         "refused with a string nobody published.",
))

HOST_TRX_CHANNEL = _reg(Unknown(
    key="HOST_TRX_CHANNEL",
    what="What a PTC with no transceiver on its TRX port answers on channel 253",
    status=ASSUMED,
    value="code 2, empty text",
    candidates=["code 2 (what we do)", "code 0 and discard", "a documented text"],
    evidence="A capture of a real PTC with nothing wired to its TRX port taking "
             "data on channel 253.",
    note="10.7 makes 253 a transparent channel to the transceiver port and bounds "
         "it at 1000 buffered bytes in the TRX-to-PC direction; it does not say "
         "what a PTC with no TRX connection does. Nothing joins this emulation's "
         "hostmode to a rig -- shrike drives its own CAT -- so a silent OK would "
         "leave a host that keys through the modem believing its CAT bytes "
         "arrived. A failure is the recoverable answer; its wording is the guess. "
         "The NMEA channel 249 (10.8) is the same shape and is answered as an "
         "empty channel: no GPS is attached either.",
))

HOST_CHANGEOVER = _reg(Unknown(
    key="HOST_CHANGEOVER",
    what="What the hostmode changeover commands %O, %I and %Q mean to a client",
    status=RESOLVED,
    value="%O = changeover either way (break in when IRS, hand over when ISS once "
          "the transmit buffer is sent and confirmed); %I = break in, IRS only; "
          "%Q = hand over only, never breaks in",
    evidence="The manual chapter itself, SCS PTC-IIIusb manual 4.1 §10.4.30 "
             "(%I, 'works only in the receiving condition (IRS) (SEND bit in the "
             "status byte = 0)'), 10.4.33 (%O, changeover either way, executed "
             "when the transmit buffer is sent and confirmed) and 10.4.34 (%Q, "
             "'in contrast to %O, does not cause a breakin'). The same text is in "
             "the PTC-II 4.0 edition at lines 8649-8675.",
    note="Read from search-engine excerpts when this entry was opened; the printed "
         "chapter now confirms all three verbatim, including the SEND-bit gate on "
         "%I, which is the direction bit ptc._status_bytes reports. Wired up in "
         "shrike.ptc; the turnaround itself is ARQ_CHANGEOVER.",
))

ARQ_CHANGEOVER = _reg(Unknown(
    key="ARQ_CHANGEOVER",
    what="How an ISS<->IRS turnaround is completed, beyond the two published "
         "mechanisms (status-byte bit 6 changeover request, CS3 break-in)",
    status=RESOLVED,
    value="The requesting ISS keeps sending until the IRS transmits: every "
          "turnaround on the wire is the IRS's CS3-headed packet, and the ACK "
          "of a bit-6 packet acknowledges the packet only. Counters restart at "
          "0 with that packet; a break-in requeues the ISS's unacknowledged "
          "packet at the same sequence number; an ISS that exhausts its "
          "retries yields the link once and listens, so a turnaround lost on "
          "the air cannot strand both ends as ISS",
    candidates=["ACK completes the handover (refuted on air)",
                "a dedicated CS answers a changeover request",
                "the requesting ISS keeps sending until the IRS transmits "
                "(measured)",
                "counters restart at 0 on each turnaround"],
    evidence="Six live WS8EOC sessions, 2026-08-03, captures/p1-ws8eoc-0803-*: "
             "in every session that reached an acknowledged bit-6 packet the "
             "gateway acknowledged it (CS1->CS2 alternation) and remained the "
             "IRS -- 120-135 ms control signals every cycle afterwards (CS2 "
             "repeated, then CS4), zero 960 ms bursts and zero CRC-valid "
             "frames across every listening window. arq.py yielded on the ACK, "
             "so both ends sat receiving until the retry budget ended the "
             "link: one acknowledgement, then death, six times in six.",
    note="M.1798 §4 states only that bit 6 'indicates a changeover request', "
         "that bit 7 'initiates the QRT protocol' and that CS3 'forces a "
         "break-in'. The ACK reading was derived from 'no codeword is left to "
         "grant a changeover' -- the flaw was reading a REQUEST as needing a "
         "grant at all. Still assumed rather than measured: speed-level "
         "continuity across a turnaround, and the exact cycle in which a "
         "PACTOR-3 new ISS starts.",
))


P3_CHANGEOVER_PACKET = _reg(Unknown(
    key="P3_CHANGEOVER_PACKET",
    what="How CS3 joins the packet behind it in a PACTOR-3 changeover packet",
    status=OPEN,
    candidates=["the 20-bit codeword on tones 5 and 12 in place of the phase "
                "reference and header block, then the data rows",
                "the codeword ahead of an otherwise whole packet, which makes the "
                "packet longer than its slot",
                "the codeword carried inside the header block's own symbols"],
    evidence="One off-air PACTOR-3 turnaround in which the IRS seizes the channel. "
             "The recordings we hold carry a bit-6 handover and no break-in, so "
             "nothing in the corpus shows one.",
    note="PT-III §4 says CS3 'forces a break-in' and PACTOR-1's description says a "
         "changeover packet gives its head to the codeword, which is where the "
         "840 ms grid rotation comes from. Neither says how the two are joined at "
         "PACTOR-3's frame geometry, and every candidate above is renderable, so a "
         "plausible one would be transmitted and would look right from here. "
         "shrike therefore does not build one: `ptc.PtcHost.breakin_now` drops the "
         "link to PACTOR-1 to seize the channel and it climbs back on the next "
         "acknowledged packet, which costs one cycle of speed and uses only "
         "waveforms an independent decoder has read.",
))

P3_IDLE_PADDING = _reg(Unknown(
    key="P3_IDLE_PADDING",
    what="What fills a PACTOR-3 data field a station has not filled",
    status=ASSUMED,
    value="the walking template (spec.TEMPLATE) for a field with no data at all, "
          "measured; 0x1E (IDLE) behind a part-filled one, assumed",
    candidates=["spec.TEMPLATE from byte 0", "0x1E IDLE", "0x00",
                "the template resumed at some phase we cannot derive"],
    evidence="Fourteen fields of rf-corpus/PIII_Complete_1 carry the walking "
             "template and nothing else -- speed levels 1, 3 and 6, always from "
             "byte 0 of the pattern -- and the entry packet of pactor3.md §17.1 "
             "is one of them. That settles the EMPTY field. Two fields in the "
             "same session are part-filled and resume the template at byte 4 and "
             "byte 14 of it, which no offset rule reproduces, so what follows "
             "real data is still unknown and keeps 0x1E.",
    note="It is one session, which is all there is: PIII_Complete_1 is the only "
         "genuine PACTOR-III reference we hold. What makes the empty case worth "
         "acting on anyway is that it is the entry packet's case -- shrike keyed "
         "five bytes of user text at data type 0 where the one entry packet ever "
         "recorded being read carries the template at type 6, and four gateways "
         "across three bands answered every one of ours by asking again. An "
         "independent monitor reports a template-filled field as LEN: 0. "
         "Stripping cannot be undone: a field whose last byte is genuinely 0x1E "
         "loses it, which is the protocol's ambiguity and is measurable on real "
         "traffic. PACTOR-1 has made the same trade since it was written.",
))

GRANT_LEVEL = _reg(Unknown(
    key="GRANT_LEVEL",
    what="Which protocol level a 0x59A grant commands the caller up to",
    status=ASSUMED,
    value="PACTOR-3 (arq.GRANT_ENTRY_SL stages a P3 speed-level-1 entry)",
    candidates=["PACTOR-3, fixed", "the answering station's own maximum",
                "the highest level the announcement declared"],
    evidence="Every recorded completion of the handshake is against a granting "
             "station whose maximum was PACTOR-3: PIII_Complete_1's answerer "
             "greets as a PTC-II ('C-II DSP/QUICC System'), and "
             "pos_pactor1_local completes in PACTOR-3 with the gateway's "
             "hardware unrecorded. For those peers the three candidates "
             "coincide, so the tapes cannot separate them. The two stations "
             "that grant this station most are listed 'Pactor 3,4' and "
             "'Pactor 2,3,4', and both, on arms where they "
             "never granted, left PACTOR-1 unilaterally into PACTOR-4 "
             "robust-mode signaling (`p4sig` scores it) -- never into "
             "PACTOR-3. What would settle it: one "
             "recording of a P4-capable station granting a third party and "
             "what that party keyed next, or one arm keying a PACTOR-4 "
             "speed-level-1 chirp entry ([SCS-P4] §11) on a grant.",
    note="§17.1's own sentence is 'the answering station commands the change', "
         "and this register entry is the level of that command. WHAT THIS NOTE "
         "SAID FROM 2026-08-29 TO 2026-09-17, and no longer: that whatever "
         "receiver the granting P4dragons open, it does not detect a PACTOR-3 "
         "speed-level-1 entry. It was true of the record it was written against "
         "and the record overtook it. On the detector this reads -- a PACTOR-3 "
         "frame returned after the entry, off the session's own `control "
         "signals decoded N (M PACTOR-1 at zero errors, K PACTOR-3)` -- 29 of "
         "181 live granted sessions read it, 0.16, and every one of the 29 is "
         "dated 2026-09-09 or later. KB5LZK, the 'Pactor 2,3,4' station this "
         "sentence was aimed at, reads it in 8 of 44 and carries 7 of the 8 "
         "past a single cycle; the `the peer answered the entry packet` "
         "milestone fires in all 8. On the hardware reading of the Dragon "
         "label the Dragons run 10 of 51 against 19 of 130, AHEAD of the "
         "stations that are not. AND THE DENOMINATOR WAS WRONG AS WELL: the "
         "old figure counted entry PACKETS, of which those 181 sessions carry "
         "thousands, where the readout is bimodal per SESSION -- a gateway "
         "either follows the entry or it does not, and the ones that follow "
         "return control signals by the dozen. A per-packet rate counts our "
         "own repeats.",
))


CYCLE_ACROSS_CHANGEOVER = _reg(Unknown(
    key="CYCLE_ACROSS_CHANGEOVER",
    what="Whether the long cycle survives a changeover or resets to short",
    status=ASSUMED,
    value="kept (arq.PactorArq.cycle_long is untouched by a changeover)",
    candidates=["kept, as link state", "reset to short with the counter"],
    evidence="PIII_Complete_1's changeovers all happen on the short cycle, so "
             "the tape cannot separate the candidates. Keeping it is "
             "self-correcting either way: every packet's variable header "
             "declares its own geometry, and a disagreeing end renegotiates "
             "through status bit 5 and CS6 within a cycle. A recording of a "
             "changeover during a long-cycle train would settle it.",
))

CHANGEOVER_ANSWER_CS5 = _reg(Unknown(
    key="CHANGEOVER_ANSWER_CS5",
    what="What the CS5 answering every changeover packet means",
    status=OPEN,
    candidates=["an acknowledgement variant peculiar to the slot after a "
                "changeover", "a genuine NAK of the changeover frame",
                "a speed vote for the new direction's first packet"],
    evidence="All three changeover packets in PIII_Complete_1 are answered CS5 "
             "in the following answer slot (6.52, 65.87, 70.52 s) at zero bit "
             "errors, yet nothing is repeated and the new ISS transmits its "
             "first packet on schedule -- at SL5 after one of them, against "
             "CS5's published 'repeat and reduce'. shrike answers a changeover "
             "on the traffic behind it and does not key this; a second "
             "session's changeovers would say whether the pattern is the "
             "protocol's or this pair's.",
))


# === PACTOR-4: specification audit, 2026-09-19 =============================
# Source-page references, dimension checks and measurement priorities follow.
# These entries do not change runtime parameters. P2/P3 resolutions above do not automatically apply to P4.

P4_HEADER_VARIANTS = _reg(Unknown(
    key="P4_HEADER_VARIANTS",
    what="P4 Chu19 construction, header/reference boundary and variant meanings",
    status=OPEN,
    candidates=["19 Chu symbols plus reference account for the stated 20",
                "20-symbol header with a separate reference",
                "variant-dependent header/reference accounting"],
    evidence="Labelled clean robust and normal packets and connected retries: "
             "locate reference boundaries and map physical variants to known "
             "levels/roles across independent exchanges.",
    note="SCS-P4 pp.2,7,9: 20-symbol general header versus Chu19 and separate R. "
         "Referenced header description is absent. Root14/shift18 is a measured "
         "waveform candidate, not a decoded control meaning.",
))

P4_TURBO_INTERLEAVER = _reg(Unknown(
    key="P4_TURBO_INTERLEAVER",
    what="P4 permutation feeding component encoder C2, including tail domain",
    status=OPEN,
    candidates=["length-dependent permutation over information bits",
                "permutation over a padded information block"],
    evidence="Controlled distinct inputs with known systematic and parity "
             "outputs at one length; constrain permutations using the published "
             "component trellis, then validate on held-out inputs and lengths.",
    note="SCS-P4 Fig.7.1 p.11 draws the interleaver but gives no algorithm. "
         "The UMTS component-code comparison does not establish its interleaver. "
         "Candidates describe domains, not an exhaustive permutation search.",
))

P4_SYMBOL_INTERLEAVER = _reg(Unknown(
    key="P4_SYMBOL_INTERLEAVER",
    what="P4 post-mapper symbol interleaver, scope and permutation",
    status=OPEN,
    candidates=["packet-wide data-symbol permutation",
                "block-local data-symbol permutation",
                "variant-dependent permutation or bypass"],
    evidence="Controlled distinct codewords at a fixed variant, with labelled "
             "mapper output and recovered data-symbol positions; verify a "
             "candidate on new payloads without fitting it again.",
    note="SCS-P4 Fig.2.1 p.3 explicitly places this stage after mapping. It is "
         "separate from Fig.7.1's C2 input interleaver; neither is supplied by "
         "the chirp-only depth-16 rule of §11.6.",
))

P4_TAIL_PADDING = _reg(Unknown(
    key="P4_TAIL_PADDING",
    what="P4 component termination, extra systematic bits and puncture treatment",
    status=OPEN,
    candidates=["tail bits plus fixed padding inside V0/C1/C2 fields",
                "variant-specific termination/packing convention"],
    evidence="Known input-to-coded-bit captures resolving all extra bits, both "
             "component final states and puncturing phase; held-out payloads "
             "must predict parity as well as pass CRC.",
    note="SCS-P4 pp.3,11-13: Fig.7.2 publishes V0 then C1 then C2. V0 exceeds "
         "8*N by 4 or 8 bits on long packets and 8 on normal short packets. "
         "Three tail bits per component do not alone specify this packing. "
         "Keep published puncture patterns; missing tail treatment is separate.",
))

P4_SYMBOL_LABELS = _reg(Unknown(
    key="P4_SYMBOL_LABELS",
    what="P4 bit-to-constellation labels and robust differential phase mapping",
    status=OPEN,
    candidates=["Gray-labelled mapping with phase/conjugation variants",
                "natural-binary mapping with phase/conjugation variants",
                "other labelling consistent with published point geometry"],
    evidence="Known mapper-input bits and clean symbols, first at BPSK/DQPSK "
             "then each higher constellation; validate labels on new inputs.",
    note="SCS-P4 §9 pp.14-15 publishes bit grouping and UNLABELLED plots. "
         "16-QAM and 32-QAM plots are not ordinary square-grid constellations. "
         "Exact QAM coordinates/normalization also need verification. Geometry "
         "does not identify bit labels; candidate families are not exhaustive.",
))

P4_NORMAL_LAYOUT = _reg(Unknown(
    key="P4_NORMAL_LAYOUT",
    what="P4 training count/placement and conflicting normal-mode dimensions",
    status=ASSUMED,
    value="176 symbols per ordinary short data block; 32-symbol training from "
          "the printed C loops; packet training count remains open",
    candidates=["176 versus literal 175 short data symbols",
                "32 code-generated versus 33 prose-generated training symbols",
                "B+1 training blocks from figure versus B from numbering prose"],
    evidence="Locate training peaks and terminal training in clean short/long "
             "packets; compare symbol counts, alternating conjugation and full "
             "packet boundaries against both layout predictions.",
    note="SCS-P4 §6.4 pp.9-10: 6*175=1056 conflicts with 176 in table; C loops "
         "give C[9:16]+C[1:16]+C[1:8] (one-based inclusive). Fig.6.2 draws "
         "seven T blocks for six data blocks; prose numbers training 1..6. "
         "Working interpretations are not capture-confirmed or wired to runtime.",
))

P4_SPECIAL_PACKETS = _reg(Unknown(
    key="P4_SPECIAL_PACKETS",
    what="P4 robust short body and CS3/break-in packet coding/semantics",
    status=OPEN,
    candidates=["robust short body is fixed/control content without user bytes",
                "variant-dependent robust short content",
                "break-in uses separately sized net/parity fields"],
    evidence="Labelled robust-short and normal break-in captures with state "
             "context; recover their contents and reconcile dimensions with "
             "their distinct header/timing behavior.",
    note="SCS-P4 pp.2-4,7,9,12: robust short has 64 symbols but no user field; "
         "normal Short_BreaKin has 840 data symbols, unlike ordinary short's "
         "1056. Net/coding tables do not separately specify these cases.",
))

P4_EXTENDED_STATUS = _reg(Unknown(
    key="P4_EXTENDED_STATUS",
    what="P4 extended-status length, fields and payload boundary when E=1",
    status=OPEN,
    candidates=["fixed-size prefix", "variable or variant-dependent prefix"],
    evidence="CRC-valid E=1 packets with independently known payload and link "
             "state, compared with E=0 controls, to delimit and name fields.",
    note="SCS-P4 §5 pp.5-6 publishes the E bit and prefix location but its "
         "referenced layout is absent. Ordinary status is E/S1S0/M2M1M0/C1C0; "
         "do not copy P3 independent status-bit handling.",
))

P4_WHITENING = _reg(Unknown(
    key="P4_WHITENING",
    what="Whether P4 has whitening, and its layer, seed and reset scope",
    status=OPEN,
    candidates=["no separate whitening", "fixed per-packet mask",
                "state/address-dependent mask"],
    evidence="Compare controlled known fields against recovered systematic "
             "bits after resolving symbol labels/permutation; vary payload, "
             "packet counter and endpoint identities independently.",
    note="No whitening rule found in SCS-P4; Fig.2.1 has no named whitening "
         "stage. P3 SCRAMBLER above explicitly applies to P3 only. Repeated "
         "unknown plaintext does not identify a mask.",
))

P4_CHIRP_CRC = _reg(Unknown(
    key="P4_CHIRP_CRC",
    what="Scope and cause of P4 chirp CRC XOR 0x53E1",
    status=ASSUMED,
    value=0x53E1,
    candidates=["fixed variant-dependent CRC convention",
                "address/session-derived CRC initialization",
                "unresolved covered-field or transformation convention"],
    evidence="Independent chirp fields with varied source and destination, "
             "using a frozen CRC rule; distinguish address effects from "
             "variant and covered-span effects without per-packet fitting.",
    note="SCS-P4 §5 p.6 and §11.2 p.18 describe ordinary PACTOR CRC. "
         "p4chirp.CHIRP_CRC_XOR and test_p4chirp.REAL_FIELDS record four fields "
         "from one station. Their agreement does not establish other stations "
         "or SL2-10.",
))

P4_SOURCE_FRAMING = _reg(Unknown(
    key="P4_SOURCE_FRAMING",
    what="P4 source-stream fill, boundaries, escapes and reserved compression mode",
    status=OPEN,
    candidates=["P3 measured stream conventions apply unchanged",
                "same published tables with P4-specific framing conventions"],
    evidence="Independent P4 packets carrying known binary/text, partial fills "
             "and field-spanning runs/escapes; verify host bytes after packet "
             "CRC and extended-status removal. Preserve mode 3 as unknown.",
    note="SCS-P4 §§4,5,12 publishes P3-compatible source coding. Audit matches "
         "all 8382 HUFTAB and 128 REFTAB entries to compress.py. Its extra "
         "stream behaviors were measured on P2/P3; it is decode-side only for "
         "compressors, so it does not supply a known-plaintext encoder.",
))

P4_CONNECTED_ARQ = _reg(Unknown(
    key="P4_CONNECTED_ARQ",
    what="P4 connected RQ/MARQ, control roles, transitions and distance spreading",
    status=OPEN,
    candidates=["repeat variants need previous packet state",
                "some variants can be acquired independently",
                "state/variant-dependent combining or spreading"],
    evidence="Complete labelled connected links with retries, direction and "
             "speed changes; relate headers/counters to payload and independently "
             "observed peer decisions before applying any combining rule.",
    note="SCS-P4 §1 p.2 gives cycle times, not a full state machine. "
         "New_Commands_DR-7800_v1_10 PACTOR-Unproto section (text lines 475-508) "
         "explicitly contrasts Unproto normal-header repeats with connected "
         "P4-RQ/MARQ. Unproto behavior does not settle connected semantics.",
))


def summary() -> str:
    order = {OPEN: 0, ASSUMED: 1, RESOLVED: 2}
    lines = []
    for u in sorted(REGISTER.values(), key=lambda u: (order[u.status], u.key)):
        lines.append(f"  [{u.status:8}] {u.key:16} {u.what}")
    n_open = sum(1 for u in REGISTER.values() if u.status == OPEN)
    n_assumed = sum(1 for u in REGISTER.values() if u.status == ASSUMED)
    lines.append("")
    lines.append(f"  {n_open} open, {n_assumed} assumed-but-unconfirmed, "
                 f"{len(REGISTER) - n_open - n_assumed} resolved")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Unknown register\n")
    print(summary())
