# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-1 control signals, anchored to real off-air recordings.

shrike spent a whole on-air session unable to read a gateway's acknowledgement,
and then reported acknowledgements it had invented from noise. Both halves of that
are fixed, and this is what holds them fixed.

The anchor is ground truth, not our own encoder: two bursts 2.5 s apart in
`pos_pactor1_local.wav` carry the same 12-bit word, and it is one of the four in
`pactor1.CONTROL_SIGNALS`, at zero errors with the runner-up eight away.

Two properties are asserted, and the second matters more than the first:

  * a real control signal decodes, exactly;
  * a burst that is NOT one of the four is REFUSED. Several control-signal-length
    bursts in the corpus read with a wide-open eye and match nothing -- they are a
    different frame type. Accepting those is what produced a fictional link.

Run: python -m hfmodem.tests.shrike.test_p1cs
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, rxfront, spec
from hfmodem.tests import evidence

ARCHIVE = evidence.WORKING / "pactor"
CORPUS = evidence.CORPUS / "regress" / "fixtures"
SESSIONS = evidence.CAPTURES
"""Six sessions of 2026-07-30 in which a real gateway answered -- the positive
control any rule about connect evidence has to survive, and the only part of
`captures/` that is committed rather than kept in rf-corpus. They were reached
through a path that had never existed, so every positive control skipped and the
suite stayed green; the loop that reads them now fails when a checkout is missing
any of them, and skips by name only where all six are absent together -- the
distribution, which they do not cross."""
ok = True
CHECKS = 0


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok, CHECKS
    CHECKS += 1
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def main() -> int:
    # The four words are a perfect equidistant code. This is why a decode may
    # insist on zero errors, and why no fifth word can exist at that distance.
    d = [bin(a ^ b).count("1")
         for i, a in enumerate(pactor1.CONTROL_SIGNALS)
         for b in pactor1.CONTROL_SIGNALS[i + 1:]]
    check("the four control signals are mutually distance 8",
          set(d) == {8}, f"distances {sorted(set(d))}")

    # ...and they are closed under reversal, which is why the error count can never
    # choose the bit order and why `msb_first` has to be a switch rather than a
    # search. Both halves are pinned: the permutation, and that the alternative
    # reading reaches the decoder as a pure relabelling. If a future edit reverses
    # the 13-bit read before slicing twelve off the front, the second of these
    # fails -- that spelling is a one-bit origin slip wearing the reversal's name,
    # and the slip is measured below to cost a false accept.
    perm = [pactor1.CONTROL_SIGNALS.index(int(f"{w:012b}"[::-1], 2))
            for w in pactor1.CONTROL_SIGNALS]
    check("the four are closed under reversal, CS1<->CS2 and CS3<->CS4",
          perm == [1, 0, 3, 2], f"permutation {perm}")

    # Transmitter and receiver must agree on the NAME, not merely on the error
    # count. They are separate readings of the same convention and nothing made
    # them move together -- and because of the closure just asserted, a bit order
    # they BOTH get wrong reads back at zero errors under the wrong codeword. That
    # is not hypothetical: shrike sent every control signal of its on-air history
    # MSB-first, so each one reached the air as its reversal partner, and every
    # round trip in this suite stayed green throughout. This is the gate that was
    # missing.
    quiet = np.zeros(int(0.5 * rxfront.FS), np.float32)
    for i in range(len(pactor1.CS_WORDS)):
        for inv in (False, True):
            burst = np.concatenate(
                [quiet, pactor1.control_signal(i, invert=inv), quiet])
            r = p1rx.decode_control_signal(burst, 0.5, 0.120)
            # The SENSE is part of the round trip, not decoration on it. It is
            # what the transmitter's `invert` means read back off the air, and
            # the two have to be one convention or the peer's phase cannot be
            # compared with our own -- see p1rx.nearest_cs.
            check(f"{spec.P1_CS_NAMES[i]} round-trips through audio"
                  + (", shift inverted" if inv else ""),
                  r == (i, 0, int(inv)), f"got {r}")

    # -- the two words PACTOR-1 assigns no meaning ---------------------------
    #
    # They are matched so a receiver can SAY one arrived. Read against the four
    # alone, 0x59A is six errors from every one of them in every reading, so the
    # burst a station sends to grant PACTOR-3 reaches the log as a six-error CS1
    # and is thrown away as noise.
    for w in pactor1.UNASSIGNED_SIGNALS:
        d = sorted({bin(w ^ c).count("1") for c in pactor1.CONTROL_SIGNALS})
        check(f"0x{w:03X} is weight 6 and distance 6 from all four",
              bin(w).count("1") == 6 and d == [6], f"weight {bin(w).count('1')}, "
              f"distances {d}")
    check("...and from each other",
          bin(pactor1.UNASSIGNED_SIGNALS[0]
              ^ pactor1.UNASSIGNED_SIGNALS[1]).count("1") == 6)

    # THE THIRD WORD OF THE PUBLISHED TABLE IS NOT HERE, and this is why. 0xB2A is
    # the bit complement of CS1, every reader matches both shift senses, and by the
    # Shiftlage rule half of a peer's acknowledgements arrive complemented -- so a
    # table holding 0xB2A reports every second CS1 on the air as an unassigned
    # word. The exclusion is asserted rather than left to a comment.
    check("0xB2A is CS1 complemented, and is not in the table",
          0xB2A == pactor1.CONTROL_SIGNALS[pactor1.CS_ACK_A] ^ 0xFFF
          and 0xB2A not in pactor1.CS_WORDS)

    # THE WIDER TABLE COSTS THE FOUR NOTHING, over EVERY twelve-bit window there
    # is -- 4096 of them, so this is exhaustive rather than sampled. Whatever the
    # four-word matcher read at 2 errors or fewer, the six-word one reads
    # identically: same name, same count, same shift. Every caller in the receiver
    # insists on zero errors, so nothing the FSM has ever acted on moves.
    #
    # The margin the extras eat is real and is stated rather than hidden. Matched
    # in both shift senses the four words are only distance 4 apart -- a control
    # signal and ANOTHER one's complement -- so the four-word table never had the
    # radius its distance-8 spacing suggests, and adding two words at distance 6
    # does not take what was not there.
    def four_only(window):
        best, which, matched = 99, -1, 0
        for sense in (0, 1):
            read = [b ^ sense for b in window]
            for idx, w in enumerate(pactor1.CONTROL_SIGNALS):
                e = sum(x != ((w >> (pactor1.CS_BITS - 1 - j)) & 1)
                        for j, x in enumerate(read))
                if e < best:
                    best, which, matched = e, idx, sense
        return p1rx.CS(which, best, matched)

    moved, at = [], {0: 0, 1: 0, 2: 0}
    for v in range(1 << pactor1.CS_BITS):
        window = [(v >> (pactor1.CS_BITS - 1 - j)) & 1
                  for j in range(pactor1.CS_BITS)]
        old, new = four_only(window), p1rx.nearest_cs(window)
        if old.errors <= 2:
            at[old.errors] += 1
            if new != old:
                moved.append((v, old, new))
    check("every window the four-word table read within 2 bits reads the same",
          not moved, f"{len(moved)} moved, first {moved[:1]}")
    # The false-accept rate the four-word table actually had, counted rather than
    # inferred from its spacing: 8 windows of 4096 read a control signal at zero
    # errors, 96 at one error and 456 at two. The radius-1 and radius-2 balls
    # OVERLAP -- 8 x 13 is 104 and 8 x 67 is 536 -- which is the distance-4
    # crowding between a codeword and another one's complement, showing up as a
    # number.
    check("...and that is 8 windows at 0 errors, 96 at 1 and 456 at 2",
          (at[0], at[1], at[2]) == (8, 96, 456), f"{at}")

    # RECOGNISED, AND STILL NOT PASSED ON. The front end names the word and hands
    # the FSM no control signal for it, which is the whole of what the wider table
    # is allowed to do until something is known about what the word asks for.
    spare = np.concatenate(
        [quiet, pactor1.control_signal(pactor1.CS_59A), quiet])
    evs = [e for e in rxfront.decode_events(spare)
           if e.kind in ("cs", "unassigned")]
    check("a 0x59A burst is reported, and as no control signal",
          [(e.kind, e.cs) for e in evs] == [("unassigned", None)]
          and "0x59A" in evs[0].text,
          f"{[(e.kind, e.cs, e.text[:24]) for e in evs]}")
    ack = np.concatenate(
        [quiet, pactor1.control_signal(pactor1.CS_ACK_A), quiet])
    evs = [(e.kind, e.cs) for e in rxfront.decode_events(ack)
           if e.kind in ("cs", "unassigned")]
    check("...while a CS1 in the same place still is one",
          evs == [("cs", pactor1.CS_ACK_A)], f"{evs}")

    f = CORPUS / "pos_pactor1_local.wav"
    if not f.exists():
        print(f"  [SKIP] {f} absent -- set HFMODEM_CORPUS to the off-air "
              "recordings")
        return 2
    a = rxfront.load_wav(str(f))

    # The two anchor bursts, decoded through the shipped entry point.
    for t0, dur in ((8.69, 0.115), (11.19, 0.120)):
        r = p1rx.decode_control_signal(a, t0, dur)
        check(f"off-air burst at t={t0}s decodes at zero errors",
              r is not None and r[1] == 0,
              f"got {r}")
    r1 = p1rx.decode_control_signal(a, 8.69, 0.115)
    r2 = p1rx.decode_control_signal(a, 11.19, 0.120)
    check("the two bursts agree on the same codeword",
          r1 is not None and r2 is not None and r1[0] == r2[0],
          f"{r1} vs {r2}")
    # The SAME station, the SAME raster, one cycle on: 8.69 and 11.19 are CS4 two
    # cycles apart and 12.443 is 1.253 s after the second of them. It reads 0x59A,
    # and it is the one hit the corpus-wide sweep of pactor1-control-signals.md
    # sec 3 could only call "a search wide enough to manufacture exactly one" while
    # the word had no name. On the raster, behind two clean codewords from the
    # station that owns it, it is not a coincidence a sweep found.
    r3 = p1rx.decode_control_signal(a, 12.443, 0.120)
    check("the burst one cycle behind the second CS4 is 0x59A",
          r3 is not None and r3[:2] == (pactor1.CS_59A, 0), f"got {r3}")

    rev = [p1rx.decode_control_signal(a, t, d, msb_first=True)
           for t, d in ((8.69, 0.115), (11.19, 0.120))]
    check("the reversed order is a relabelling: same errors, permuted name",
          all(r is not None and s is not None
              and r[1] == s[1] and s[0] == perm[r[0]]
              for r, s in zip((r1, r2), rev)),
          f"{(r1, r2)} vs {tuple(rev)}")

    # -- ONE TONE SENSE, AND IT IS THE FRAME'S -------------------------------
    #
    # `CS.sense` is handed straight to the packet renderer by
    # `onair._MasterGrid.align`, so it has to mean what `pactor1`'s `invert` means
    # or every packet we transmit lands one cycle out of phase with the station
    # that just told us its phase. Encoder and decoder agreed on their own private
    # convention for shrike's whole on-air history and no round trip could see it.
    #
    # W4DNA retransmits its link-setup packet on three consecutive cycles here and
    # the peer answers each one 110 ms later, inside the same cycle -- where the
    # two directions share the shift (pactor1-data-packets.md sec 7). So the packet
    # decoder and the control-signal decoder are reading ONE physical fact three
    # times, in alternating polarity, and they must return the same number for it.
    pairs = []
    for pkt_t, cs_t in ((7.616, 8.693), (8.865, 9.943), (10.114, 11.194)):
        seg = a[int((pkt_t - 0.05) * rxfront.FS):int((pkt_t + 1.01) * rxfront.FS)]
        frames = p1rx.decode_p1_packets(seg)
        cs = p1rx.decode_control_signal(a, cs_t, 0.120)
        if len(frames) == 1 and cs is not None and cs.errors == 0:
            pairs.append((pkt_t, int(frames[0].inverted), cs.sense))
    check("all three W4DNA cycles give a packet and a zero-error answer",
          len(pairs) == 3, f"{len(pairs)} of 3")
    off = [p for p in pairs if p[1] != p[2]]
    check("the answer's shift sense IS the packet's, in the cycle they share",
          pairs and not off, f"{off}")
    # ...and the three do not all read the same, so this cannot pass on a
    # constant: a reader stuck in either convention fails it in every cycle.
    check("...over three cycles of alternating polarity",
          {p[1] for p in pairs} == {0, 1}, f"{[p[1] for p in pairs]}")

    # ...and the front end surfaces them as `cs` events with zero errors.
    evs = [e for e in rxfront.decode_events(a) if e.kind == "cs"]
    check("decode_events reports both as control signals", len(evs) >= 2,
          f"{len(evs)} cs events")
    check("every reported control signal claims zero errors",
          all("0 bit errors" in e.text for e in evs),
          "; ".join(e.text[:40] for e in evs))

    # -- the systematic 3-5 bit errors, and what they were ------------------
    #
    # Strong, on-frequency, correctly detected, unambiguously real bursts used to
    # read at 3-5 errors: far too few for noise against a distance-8 code, far too
    # many for a correct read. The cause was that the twelve-bit window began at
    # the BURST DETECTOR'S run start, and a 5 ms energy threshold is not a bit
    # grid -- measured against the alignments that decode, over 41 bursts from
    # four stations, it sits 5 to 36 ms early. Half a bit of that samples every
    # transition instead of every bit centre, which is exactly a handful of errors
    # in a balanced word.
    #
    # `cs_bits` now POSITIONS the window on the signal, blind to the codewords.
    # Two gateways recorded with our own transmitter off, so every burst in them
    # is theirs and none of it is ours:
    #
    #     WS8EOC   captures/provoke2/silent.wav        1 of 32 bursts ->  14
    #     W6IDS    captures/w6ids_silent/silent.wav    0 of 37        ->  16
    #
    # The count is the weaker half of this. The evidence is that a station's
    # bursts now AGREE: each recording decodes to one codeword and never to a
    # second, across a 1.25 s raster the station holds for tens of cycles. A
    # misaligned reader cannot produce that -- it produced one decode in 69 bursts
    # and had no agreement to show.
    caps = ARCHIVE / "captures"
    for name, floor in (("provoke2", 12), ("w6ids_silent", 13)):
        wav = caps / name / "silent.wav"
        if not wav.exists():
            print(f"  [SKIP] {wav} absent")
            continue
        c = rxfront.load_wav(str(wav))
        step = int(1.2 * rxfront.FS)
        words = []
        for k in range(0, c.size - step, step):
            seg = c[k:k + step]
            for s, d in rxfront._p1_cs_bursts(seg):
                r = p1rx.decode_control_signal(seg, s, d)
                if r is not None and r[1] == 0:
                    words.append(r[0])
        check(f"{name}: the gateway's bursts decode at zero errors",
              len(words) >= floor, f"{len(words)} bursts, floor {floor}")
        check(f"{name}: ...and they all name the SAME codeword",
              len(set(words)) == 1 and words[0] == pactor1.CS_SPEED,
              f"{sorted(set(words))}")

    # -- the refusal, and the burst it used to be taken on -------------------
    #
    # oracle_pactor3_dl6maa opens with a whole PACTOR-1 connect exchange before
    # its PACTOR-3 data: two 960 ms call packets at 2.120-3.110 and 3.370-4.360,
    # each followed 30 ms later by a short burst, at t=3.140 and t=4.395 -- 1.255 s
    # apart, the protocol's own raster. So those two bursts are the far end
    # ANSWERING the call, and CS1 is precisely what a connect is answered with.
    #
    # t=3.140 is 1400/1600 Hz two-tone with transitions at 10 and 20 ms, six ones
    # and six zeros, and reads CS1 at zero errors over a 9.7 ms span of grid
    # offsets. This test used to assert that it was NOT a codeword, on the
    # strength of a decoder that could not read it. Nothing about it was foreign.
    #
    # t=4.395 is: same tones, same 10/20 ms transitions, and no alignment anywhere
    # in +/- three bits brings it below one error. That is what a burst which is
    # not one of the four looks like, and it is the anchor the refusal belongs on.
    dl = CORPUS / "oracle_pactor3_dl6maa.wav"
    if dl.exists():
        b = rxfront.load_wav(str(dl))
        r = p1rx.decode_control_signal(b, 3.140, 0.140)
        check("the connect answer in the DL6MAA capture decodes as CS1",
              r is not None and r[:2] == (pactor1.CS_ACK_A, 0), f"got {r}")
        # t=4.395 IS A CODEWORD, and for as long as the table held four words
        # nothing could say so: it is 0x59A, six from every control signal in every
        # reading, and this asserted only that it was not one of the four. It sits
        # in the answer slot of the caller's announcement packet and the caller's
        # next burst is PACTOR-3. What it is for is not settled here; that it is
        # not noise is.
        r = p1rx.decode_control_signal(b, 4.395, 0.135)
        check("the burst at t=4.395 is 0x59A, not a control signal",
              r is not None and r[:2] == (pactor1.CS_59A, 0) and r.unassigned,
              f"got {r}")

    # The refusal, swept. One hand-picked position proves almost nothing: the
    # detector offers dozens of candidates across the corpus and it only takes
    # one of them landing on a codeword to invent a link. So run every candidate
    # the front end raises on signals that are definitively NOT PACTOR-1 control
    # signals, and require that none of them decodes at zero errors.
    #
    # `decode_control_signal` always returns a nearest codeword -- that is what
    # the equidistant code is for -- so the test is on the ERROR COUNT. Checking
    # `is not None` here passes trivially and measures nothing.
    #
    # oracle_pactor3_dl6maa is NOT in this list, and used to be. It carries a real
    # PACTOR-1 connect exchange, answer and all, so a decode in it is a decode and
    # counting one as a false accept is how a genuine CS1 came to be recorded as a
    # foreign burst above.
    step = int(1.2 * rxfront.FS)
    swept = accepted = 0
    for name in ("neg_noise_chatham", "neg_ft8_local", "neg_ft8_australia",
                 "neg_ft8_weak_remote", "neg_narrow_200hz", "neg_wide_560hz",
                 "edge_wspr_australia", "edge_winlink_oceania",
                 "edge_clipped_9pct", "edge_long_80m", "edge_ardop_suspect",
                 "pos_vara_netherlands", "qso_vara_ko2f",
                 "oracle_pactor2_ref", "watch_pactor3_maryland"):
        f = CORPUS / f"{name}.wav"
        if not f.exists():
            continue
        b = rxfront.load_wav(str(f))
        for k in range(0, min(b.size, int(60 * rxfront.FS)) - step, step):
            seg = b[k:k + step]
            for s, d in rxfront._p1_cs_bursts(seg):
                swept += 1
                r = p1rx.decode_control_signal(seg, s, d)
                if r is not None and r[1] == 0:
                    accepted += 1
    check("no foreign signal decodes as a control signal", accepted == 0,
          f"{accepted} false accepts in {swept} candidates")
    check("the sweep actually exercised the decoder", swept >= 20,
          f"only {swept} candidates raised")

    # How long to keep listening after a reply starts. This is a LATENCY budget,
    # not a margin to pad: every millisecond spent here comes out of the 1.25 s
    # cycle our acknowledgement has to land inside, and shrike has already been
    # measured answering outside it. One copy of the codeword decodes, so the
    # figure is set by the cliff and not by the peer's full 405 ms reply.
    #
    # Both directions are asserted. The floor stops the wait being trimmed below
    # what a decode needs; the sensitivity check stops the floor passing for the
    # wrong reason, by requiring that a shorter capture still fails. If that ever
    # starts passing, this test has gone blind.
    #
    # The cliff moved when the window started being positioned rather than
    # assumed. Swept in 5 ms steps on both anchors, 115 ms of post-onset audio
    # fails and 120 ms decodes -- the cliff is now exactly one codeword, where it
    # used to sit 20 ms above one. The reader wants the word and nothing else.
    from hfmodem.shrike import onair  # noqa: E402
    for t0, dur in ((8.69, 0.115), (11.19, 0.120)):
        enough = a[:int((t0 + onair.P1_BURST_S) * rxfront.FS)]
        r = p1rx.decode_control_signal(enough, t0, dur)
        check(f"P1_BURST_S={onair.P1_BURST_S:.2f}s decodes the burst at t={t0}s",
              r is not None and r[1] == 0, f"got {r}")
        short = a[:int((t0 + 0.115) * rxfront.FS)]
        rs = p1rx.decode_control_signal(short, t0, dur)
        check(f"115 ms of post-onset audio is NOT enough at t={t0}s "
              f"(the test can still fail)",
              rs is None or rs[1] > 0, f"got {rs}")

    # -- acquisition: the sliding correlator, and what confines it ----------
    # `acquire_control_signal` searches a WINDOW where `decode_control_signal`
    # reads one burst. That is the reference implementation's split and it is not
    # arbitrary: a running link already knows its grid, an acquiring one does not.
    #
    # The measurement that used to justify it -- 10% of W6IDS's bursts for the
    # single read against 36% for the search -- was really measuring the burst
    # start being used as a bit grid, and the two are now level (16 of 37 against
    # 15). The split stands on the architecture instead.
    #
    # It must stay confined. Only CS1 and CS4 are legal answers to a call, and a
    # match must be exact -- those two constraints are what stop 224 trials per
    # burst from finding a codeword in noise.
    from hfmodem.shrike import p1rx as _p1  # noqa: E402
    acq = _p1.acquire_control_signal(a, 8.69, 0.115)
    check("acquisition finds the anchor burst", acq is not None, f"got {acq}")
    if acq is not None:
        check("...and it is one of the two legal connect answers",
              acq[0] in (pactor1.CS_ACK_A, pactor1.CS_SPEED),
              f"returned index {acq[0]}")

    # The confinement itself, asserted rather than trusted. CS3 (changeover) and
    # CS2 are not legal answers to a call, so a clean one must NOT be acquirable
    # however perfectly it matches -- otherwise the search is choosing from four
    # codewords instead of two and its false-accept rate doubles.
    for idx, name in ((pactor1.CS_CHANGEOVER, "CS3"), (pactor1.CS_ACK_B, "CS2")):
        clean = np.concatenate([quiet, pactor1.control_signal(idx), quiet])
        check(f"{name} is not acquirable -- it is not a legal connect answer",
              _p1.acquire_control_signal(clean, 0.5, 0.120) is None,
              f"got {_p1.acquire_control_signal(clean, 0.5, 0.120)}")
    # ...while the two that ARE legal must be found, or the restriction is just
    # a decoder that cannot decode.
    for idx, name in ((pactor1.CS_ACK_A, "CS1"), (pactor1.CS_SPEED, "CS4")):
        clean = np.concatenate([quiet, pactor1.control_signal(idx), quiet])
        got = _p1.acquire_control_signal(clean, 0.5, 0.120)
        check(f"{name} IS acquirable", got is not None and got[0] == idx,
              f"got {got}")

    # -- what a gateway actually answered, and what it only repeated ---------
    #
    # "Every gateway we have ever called answers CS4" was read off the two silent
    # captures, and neither one can carry a connect answer: their metadata gives
    # `recording_starts_at_sample == samples_at_silent`, so the recording begins
    # at the moment we stopped calling, ten calls in. Every CS4 in them is an
    # in-session repeat 25 s downstream of an answer nobody recorded. Asserted
    # rather than remembered, because the reading that grew out of it -- that CS4
    # is a refusal and shrike has never connected to anything -- was wrong, and
    # would have been caught here.
    import json
    for name in ("provoke2", "w6ids_silent"):
        meta = caps / name / "silent.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        check(f"{name}: the recording starts where the calling stops",
              m["recording_starts_at_sample"] == m["samples_at_silent"]
              and len(m["calls"]) > 0,
              f"{len(m['calls'])} calls, rec@{m['recording_starts_at_sample']} "
              f"silent@{m['samples_at_silent']}")

    # `captures/witness.wav` is the one recording that holds an answer: a
    # third-party KiwiSDR 52 km away heard both stations, with no transmit/receive
    # blanking of its own. WS8EOC answered our call **CS1** at zero errors and only
    # then went to CS4, which it held to timeout, 24 s of that after our last packet
    # ended, before identifying in CW.
    #
    # The train below is BOTH transmitters: it heard us too, and our own fading
    # packets decode as codewords at zero errors. That is why this gate asserts an
    # ORDER and not a census. Counting codewords passes for a decoder that reads
    # the session backwards, and both codewords being present passes for one that
    # reads them interleaved. What no misreading produces -- and what neither
    # station's fragments can fake, since ours stop first -- is a clean partition
    # in time: an answer, then a different codeword held to timeout.
    w = caps / "witness.wav"
    if not w.exists():
        print(f"  [SKIP] {w.name} absent")
    else:
        c = rxfront.load_wav(str(w))
        step = int(1.2 * rxfront.FS)
        train = []
        for k in range(0, c.size - step, step):
            seg = c[k:k + step]
            for s, d in rxfront._p1_cs_bursts(seg):
                r = p1rx.decode_control_signal(seg, s, d)
                if r is not None and r[1] == 0:
                    train.append((k / rxfront.FS + s, r[0]))
        answers = [t for t, i in train if i == pactor1.CS_ACK_A]
        repeats = [t for t, i in train if i == pactor1.CS_SPEED]
        check("witness: the gateway's train is CS1 and CS4 and nothing else",
              len(answers) + len(repeats) == len(train) and len(train) >= 25,
              f"{len(train)} decodes, {sorted({i for _, i in train})}")
        check("witness: it ANSWERED CS1 -- the call was accepted",
              len(answers) >= 4, f"{len(answers)} CS1")
        check("witness: ...and every CS4 comes after every CS1",
              answers and repeats and max(answers) < min(repeats),
              f"CS1 to {max(answers):.2f}s, CS4 from {min(repeats):.2f}s")
        check("witness: it then held CS4 to timeout",
              len(repeats) >= 20 and max(repeats) - min(repeats) > 30,
              f"{len(repeats)} CS4 over "
              f"{max(repeats) - min(repeats):.1f}s")

    # -- the data phase, and the read that could not see it ------------------
    #
    # WS8EOC answered every cycle of the 2026-07-30 link and shrike decoded none
    # of them, so the gateway never saw a packet acknowledged and the session
    # stalled with both stations behaving correctly. The 24 receive windows of
    # that session are the anchor here, and two separate faults are pinned:
    #
    #   * the twelve bits sit within 5 ms of `ANCHORED_AT`, which is where the
    #     grid puts them -- not 20-45 ms away, which is where `cs_bits`' detector
    #     bracket was free to wander and did in 22 of the 24;
    #   * the peer's two tones arrived 2-19 dB apart, median 9, so `space > mark`
    #     read ten or eleven of every twelve slots as mark. `_cs_decide` splits at
    #     the median instead, which needs no knowledge of either level.
    #
    # The train is the evidence, exactly as it is for `witness.wav` above: a
    # partition in time, CS1 while the gateway was still asking for the packet
    # and CS4 once it gave up on it, with nothing else in between. A reader
    # picking codewords out of noise does not produce that ordering.
    ANCHORED_AT = {                       # ms into each window, measured
        2: 94.2, 3: 93.8, 4: 93.8, 5: 92.4, 6: 96.6, 7: 96.5, 8: 93.1,
        9: 93.9, 10: 94.8, 11: 101.6, 15: 94.5, 16: 94.5, 17: 1052.6,
        18: 1047.9, 19: 1048.3, 20: 72.2, 21: 88.3, 22: 86.9, 23: 88.7,
        24: 1046.3,
    }
    onair = evidence.CORPUS / "pactor-ws8eoc-20260730" / "onair-0730-2036"
    if not (onair / "rx_02.wav").exists():
        print(f"  [SKIP] {onair} absent")
    else:
        train, stray, train_sense = [], [], []
        for k in sorted(ANCHORED_AT):
            seg = rxfront.load_wav(str(onair / f"rx_{k:02d}.wav"))
            t0 = ANCHORED_AT[k] / 1000
            r = p1rx.cs_anchored(seg, t0)
            if r is not None and r[1] == 0:
                train.append((k, r[0]))
                train_sense.append((k, r.index, r.sense))
            r = p1rx.decode_control_signal(seg, t0, spec.P1_CS_S)
            if r is not None and r[1] == 0:
                stray.append((k, r[0]))
        check("ws8eoc: the data-phase answers decode at the anchor",
              len(train) >= 12, f"{len(train)} of {len(ANCHORED_AT)}")
        # The detector-width bracket does not merely miss them. Where it does
        # reach zero errors it reaches a DIFFERENT codeword -- CS3, two bits
        # along from the CS4 that is really there, which is what a window free to
        # slip a whole bit finds in a code whose words are shifts of each other.
        # A break-in reported to the state machine is worse than a missed
        # acknowledgement, so this is pinned as an inequality and not a count.
        agreed = set(stray) & set(train)
        check("ws8eoc: the detector-width read never names the right codeword",
              not agreed, f"agreed on {sorted(agreed)}")
        check("ws8eoc: ...and what it does read at zero errors is the ghost",
              all(i == pactor1.CS_CHANGEOVER for _, i in stray),
              f"{sorted({i for _, i in stray})}")
        # THE SHIFT, MEASURED ON A REAL GATEWAY. Each of these windows is one
        # 1.25 s cycle, so a shift that is a property of the cycle must invert
        # from one to the next -- and skipped windows must advance it by their
        # own count, not be skipped with them. All fourteen zero-error answers
        # satisfy sense == window & 1, across gaps of 1, 2, 3 and 5
        # windows, with no exception. That is docs/protocols/pactor/pactor1-data-packets.md §7
        # confirmed on a second station and a second session -- and the CS4 run
        # confirms the sharp half of it, that a REPEATED codeword inverts like
        # everything else rather than holding its sense.
        # The epoch is this capture's window numbering and nothing more; the
        # claim is the PARITY. It moved by one when `nearest_cs` stopped
        # reporting a sense of its own and started reporting the frame's, which
        # flipped all fourteen together and left the relation untouched.
        senses = [(k, sense) for k, _, sense in train_sense]
        odd = [(k, s) for k, s in senses if s != (k & 1)]
        check("ws8eoc: the answers' shift follows the CYCLE, with no exception",
              senses and not odd, f"{len(odd)} of {len(senses)} off parity: {odd}")
        held = [(a, b) for (a, sa), (b, sb) in zip(senses, senses[1:])
                if sb != (sa + b - a) & 1]
        check("ws8eoc: ...including across the windows that did not decode",
              not held, f"{held}")
        rep = [(k, s) for k, i, s in train_sense if i == pactor1.CS_SPEED]
        check("ws8eoc: ...and a REPEATED CS4 inverts like anything else",
              len({s for _, s in rep}) == 2, f"{rep}")
        acks = [k for k, i in train if i == pactor1.CS_ACK_A]
        reps = [k for k, i in train if i == pactor1.CS_SPEED]
        check("ws8eoc: the train is CS1 and CS4 and nothing else",
              len(acks) + len(reps) == len(train),
              f"{sorted({i for _, i in train})}")
        check("ws8eoc: ...and every CS4 comes after every CS1",
              acks and reps and max(acks) < min(reps),
              f"CS1 in {acks}, CS4 from {min(reps) if reps else '-'}")
        # And the other half: the same reader over audio from the same station
        # with no control signal in it. `CS_SPLIT_MIN` is what stands between a
        # median split -- which always returns SOME weight-6 word, and 10 of the
        # 924 of those are in the table -- and a receiver that acknowledges noise.
        #
        # The count is 10 rather than 8 because the two unassigned words are
        # weight-6 as well -- 1.5x the chance of landing on a word in the table --
        # and `cs_anchored` answers that by making an unassigned read clear
        # `EYE_MIN` on top of the split. The one read of these 560 that lands on
        # 0x6A9 scores 0.19 there against 0.56 to 0.93 for every real burst of the
        # word in the corpus, so the wider table costs this sweep nothing.
        quiet = accepted = spare = 0
        for k in (17, 18, 19, 24):
            seg = rxfront.load_wav(str(onair / f"rx_{k:02d}.wav"))
            for t in np.arange(0.15, 0.85, 0.005):
                quiet += 1
                r = p1rx.cs_anchored(seg, float(t))
                if r is not None and r[1] == 0:
                    spare += r.unassigned
                    accepted += not r.unassigned
        check("ws8eoc: and it invents no control signal in the quiet before them",
              accepted == 0, f"{accepted} accepts in {quiet} reads")
        check("ws8eoc: ...and none of the two unassigned words either",
              spare == 0, f"{spare} in {quiet} reads")
        check("ws8eoc: the quiet sweep actually ran", quiet >= 400, f"{quiet}")

    print("\nALL PASS" if ok else "\nFAILED")
    # Nothing ran. Individual fixtures `continue` when absent, so a file
    # whose corpus directory exists but whose recordings do not would
    # otherwise report a green pass having asserted nothing — which is the
    # exact difference between "we checked" and "we could not".
    if CHECKS == 0:
        print("  [SKIP] nothing was checked — no fixture was present")
        return 2
    return 0 if ok else 1


def test_the_search_looks_only_where_an_answer_is_due() -> None:
    """The connect search gets the turnaround band, never the window it sits in.

    Its accept is twelve bits against two codewords in two senses with no other
    gate, so the only thing bounding its false-accept rate is the width of what it
    is handed. What it should be handed is not "the audio this cycle happened to
    leave" but "where an answer to our own call could physically be": no earlier
    than the rig comes out of transmit, no later than a codeword we could still
    hear out before we key again, and measured from OUR CARRIER DROPPING.

    Three shapes of window, and the rule has to give the right answer to all three
    from one arithmetic:

      * a keyed cycle, which leaves 0.24 s clear -- the band is 47 ms of it;
      * a hushed cycle, 1.25 s of capture with no transmission of ours in front
        of it -- searched not at all;
      * a slot handed back, where `_regrid` extends the window past a boundary it
        could not key on. That one is why the flag this used to take is not
        enough: the window is a second long and NOT hushed, and one of the three
        phantom links of 2026-08-06 came off a codeword 833 ms into one.

    THE SETTLE IS AN INPUT TO ALL OF IT, which is the reason `_rig_settle` below
    exists as a separate scene: this one is the ft891, and on the ft891 the
    arithmetic closes.
    """
    from hfmodem.shrike import onair  # noqa: E402

    FS = rxfront.FS
    d_max_n = _d_max_n(0.04)
    keyed_n, cycle_n = int(0.241 * FS), int(1.248 * FS)

    t0, span = onair._acquisition_window(keyed_n, 0, d_max_n)
    assert t0 == onair.TR_SWITCH_S, t0
    assert 0.04 < span < 0.06, span
    # ...and it holds the answer a real gateway gives. WS8EOC's bursts start at
    # 87-134 ms with a median of 96 across the seven sessions of 2026-07-30
    # (`onair.PEER_TURNAROUND_S`), measured off the energy detector rather than
    # through this band -- the band cannot be its own evidence. It reaches the
    # median and it does not reach the tail; that is a property of the schedule
    # and not a tuning choice, and `_rig_settle` is where it is charged.
    lo_d, mid_d, _ = onair.PEER_TURNAROUND_S
    assert t0 <= lo_d and t0 + span >= mid_d, (t0, span, onair.PEER_TURNAROUND_S)

    # A hush puts our last carrier a whole cycle back, and carries the band off
    # the front of the window without needing to be told about hushes. Asserted as
    # a SPAN and not as an accept rate: a rate over a search that was never called
    # is 0.0 by construction, which is a sentence about this test and not about
    # the code.
    assert onair._acquisition_window(cycle_n, cycle_n, d_max_n)[1] == 0.0
    # A slot handed back keeps the band. The window is four times as long and
    # buys the search nothing.
    wide_t0, wide = onair._acquisition_window(int(1.047 * FS), 0, d_max_n)
    assert wide_t0 == t0 and (wide_t0 + wide) * FS <= d_max_n, (wide_t0, wide)
    assert wide < 0.12, wide

    # ...and what the rule is worth, charged to the PRODUCTION search rather than
    # asserted. The comparison is against the span the loop used to hand it: the
    # whole window, from 4 ms in.
    #
    # The draw counts are not decoration. At 200 the keyed comparison failed on
    # one seed in twenty, which is a test that reports the draw rather than the
    # code; at 1500 the worst ratio over the same twenty seeds is 0.60 against the
    # 0.75 asserted. The handed-back pair is a tenfold gap and needs no such care.
    rate = {}
    for name, n, since, draws in (("keyed", keyed_n, 0, 1500),
                                  ("handed back", cycle_n, 0, 200)):
        was = (onair.ACQUIRE_T0,
               n / FS - onair.ACQUIRE_T0 - onair.ACQUIRE_TAIL_S)
        for what, (a, b) in (("was", was),
                             ("now", onair._acquisition_window(n, since, d_max_n))):
            rng = np.random.default_rng(20260806)
            accepts = sum(
                p1rx.acquire_control_signal(
                    rng.standard_normal(n).astype(np.float32) * 0.05,
                    a, spec.P1_CS_S, span=b) is not None
                for _ in range(draws)) if b > 0 else 0
            rate[f"{name} {what}"] = accepts / draws
    print(f"  gaussian false accepts: {rate}")
    assert rate["handed back was"] > 0.40, rate
    assert rate["handed back now"] < 0.15, rate
    assert rate["keyed now"] < 0.75 * rate["keyed was"], rate


def _d_max_n(settle: float) -> int:
    """The latest turnaround a caller on this rig could still hear out, samples.

    Production's own, called rather than copied. This function used to restate the
    expression, and a bench that restates the arithmetic it is checking cannot
    disagree with it: replacing production's `_d_max_n` with `round(cycle * FS)` --
    search the whole cycle, the very thing the acquisition band exists to prevent --
    changed nothing in any of the 145 tests here.
    """
    from hfmodem.shrike import onair  # noqa: E402

    return onair._d_max_n(spec.CYCLE_SHORT_S, settle)


def test_the_settle_decides_whether_a_peer_can_be_heard_at_all() -> None:
    """Every rig in the table, and the band its PTT settle leaves.

    THE TWO ARITHMETICS HAVE TO AGREE. `_budget` decides at startup whether a
    schedule closes and refuses to key if it does not; `_acquisition_window`
    decides each cycle where to look. They differed by `ACQUIRE_TAIL_S` -- the 21
    ms the codeword search reads past the top of its band -- and the g90 lived in
    the gap: `_budget` passed it with a 15 ms band, `_acquisition_window` then
    handed it none, and the session printed "RX (nothing decoded)" every cycle.
    That line is indistinguishable from an empty frequency, which is why this is
    asserted as a pair rather than one at a time.

    The numbers below are the rig table's, not a chosen 8 ms: the old scene here
    passed `settle=0.008`, which is no rig, and so covered none of them.
    """
    from hfmodem.core.rigs import RIGS  # noqa: E402
    from hfmodem.shrike import onair  # noqa: E402

    FS = rxfront.FS
    _, mid_d, _ = onair.PEER_TURNAROUND_S
    seen = []
    for rig, cfg in sorted(RIGS.items()):
        settle = cfg["settle"]
        fits, line = onair._budget(spec.CYCLE_SHORT_S, 0.15, settle)
        keyed_n = int((spec.CYCLE_SHORT_S - settle - spec.P1_PACKET_S) * FS)
        t0, span = onair._acquisition_window(keyed_n, 0, _d_max_n(settle))
        print(f"  {rig}: settle {settle:.3f}, budget {'fits' if fits else 'REFUSES'}"
              f", band [{t0 * 1e3:.0f}, {(t0 + span) * 1e3:.0f}] ms")
        # THE PAIR. A schedule the budget passes must leave a band to search, and
        # one it refuses must not be searched behind its back.
        assert fits == (span > 0), (rig, fits, span, line)
        # ...and a rig that cannot reach the median answer is told so in the line
        # the operator reads, rather than in a cycle of silence.
        assert (span > 0 and t0 + span >= mid_d) == ("SHORT OF THE MEDIAN" not in line), \
            (rig, line)
        seen.append((rig, fits))
    assert len(seen) >= 3, seen
    # The table as it stands: one rig can call and hear an answer, two cannot.
    # Stated because it is a fact about this station's hardware and it is the
    # reason a g90 session has never connected.
    assert dict(seen) == {"ft891": True, "g90": False, "x6100": False}, seen


def test_an_empty_acquisition_band_is_said_to_be_empty() -> None:
    """The x6100's 0.400 settle drives `_d_max_n` NEGATIVE -- there is no `d`
    at which an answer fits -- and the budget line printed that as "holds an
    answer whole for d 55--230 ms": a phantom range an operator at arm time
    reads as 55-230, over a band that does not exist. A band that is empty is
    stated as the fact it is, in the one line the operator reads before the
    refusal."""
    from hfmodem.core.rigs import RIGS  # noqa: E402
    from hfmodem.shrike import onair  # noqa: E402

    fits, line = onair._budget(spec.CYCLE_SHORT_S, onair.TX_OFFSET_S,
                               RIGS["x6100"]["settle"])
    assert not fits
    assert "EMPTY" in line, line
    assert "whole for d" not in line, ("a negative band top is being printed "
                                       "as a range: " + line)
    # The g90's band exists and cannot be read -- a different failure, and it
    # keeps its different words.
    fits, line = onair._budget(spec.CYCLE_SHORT_S, onair.TX_OFFSET_S,
                               RIGS["g90"]["settle"])
    assert not fits and "NONE of which is long enough to read" in line, line


@pytest.mark.parametrize("rig,refuses", [("g90", True), ("x6100", True),
                                         ("ft891", False)])
def test_a_schedule_that_does_not_close_refuses_to_key(rig, refuses, tmp_path,
                                                       monkeypatch) -> None:
    """The refusal itself, not the verdict behind it.

    `_budget` returning False is asserted above; that it stops the transmitter was
    not. Neutering the guard -- `if False and not fits and args.transmit:` -- left
    all 145 tests here green, and the behaviour it removes is the one a commit
    message headlines: shrike will not key on a rig whose PTT settle closes the
    acquisition band, because the way a schedule that does not close fails is by
    transmitting over the far end while reporting that it stayed quiet.

    Driven through the real argv, and both sides of it: a rig the budget refuses
    must never reach `ota.Rig`, and one it passes must, or the case above would be
    satisfied by a station that never keys for anybody.
    """
    from hfmodem.shrike import onair  # noqa: E402

    class _ReachedTheRig(Exception):
        pass

    def _no_rig(*a, **k):
        raise _ReachedTheRig

    monkeypatch.setattr(onair.ota, "Rig", _no_rig)
    # --serial names a real character device: the arm gate ahead of `ota.Rig`
    # refuses a path that is not there, and this test is about the BUDGET
    # refusal, which must be the one that fires for g90/x6100 and must not
    # for the ft891.
    monkeypatch.setattr(sys, "argv",
                        ["onair", "--transmit", "--rig", rig, "--dial", "7101500",
                         "--dxcall", "N0DX", "--serial", "/dev/null",
                         "--outdir", str(tmp_path)])
    if refuses:
        with pytest.raises(SystemExit) as e:
            onair.main()
        assert "not keying" in str(e.value), e.value
    else:
        with pytest.raises(_ReachedTheRig):
            onair.main()


def _offer(ev, pairs):
    """Offer candidates at given CYCLES, one search per cycle in between.

    `_ConnectEvidence`'s span is a count of searches, and the loop searches every
    cycle it called in, so a call train with no hush in it makes the two the same
    number -- which is what these cases are written in.
    """
    fired, at = [], 0
    for cycle, d in pairs:
        for _ in range(cycle - at):
            ev.note_search(0.075)
        at = cycle
        if ev.offer(cycle, d):
            fired.append(cycle)
    return fired


def _search(seg, since_tx=0):
    """One cycle's connect search, exactly as `shrike.onair` runs it."""
    from hfmodem.shrike import onair  # noqa: E402

    t0, span = onair._acquisition_window(seg.size, since_tx, _d_max_n(0.04))
    if span <= 0:
        return None
    got = p1rx.acquire_control_signal(seg, t0, spec.P1_CS_S, span=span)
    return None if got is None else t0 + got[1]


def test_one_codeword_is_not_a_station() -> None:
    """Three accepts at one turnaround is an answer; one accept is a coin.

    THE POPULATION IS THE POINT. A false-accept rate taken over gaussian noise
    understates what a receiver meets on the air, so the floor is measured over
    real off-air energy addressed to nobody: 31996 cycle-windows, 11.3 hours from
    1005 recordings, all of rf-corpus and this station's own ARDOP captures. The
    production search accepts in 2.7% of them; the longest run of ADJACENT accepts
    agreeing on a turnaround to even 20 ms is TWO; and this rule fires on none of
    the 416 recordings that hold an accept at all. Real stations are the other way
    round: a peer answers every cycle at its own hardware turnaround until the
    caller's first data packet decodes, so its accepts pile up on one offset.

    That sweep is `working/pactor/analysis/p1cs/turnaround.py` and it needs the
    whole corpus. What is asserted HERE is the part that travels with the repo:
    the rule on offsets, and the six 2026-07-30 sessions committed under
    `captures/`. Those are fixtures, not conveniences -- a run that cannot find
    them fails rather than passing quietly, which is how the positive controls
    came to be pointed at a directory that does not exist and nobody noticed.
    The six sessions are 9.5 MB -- the 4.8 GB is the rest of `captures/`, which
    is gitignored and lives in rf-corpus -- so every checkout has them, and a
    checkout missing any of them is a broken tree that fails naming it. They do
    NOT cross the publication boundary -- the wav deny spares only the besra
    fixtures -- so from the distribution the whole set is absent by design, and
    that one reading skips, naming what did not cross, the way `archive.py`
    draws the same line for `working/`.
    """
    from hfmodem.shrike import onair  # noqa: E402

    # -- the rule itself, on offsets rather than on audio -------------------
    scatter = onair._ConnectEvidence()
    assert not _offer(scatter, ((1, 0.058), (2, 0.091), (3, 0.135))), \
        f"three accepts scattered across the band connected: {scatter.candidates}"
    station = onair._ConnectEvidence()
    fired = _offer(station, ((1, 0.0925), (3, 0.0938), (5, 0.0912)))
    assert fired == [5], f"a station answering at one turnaround did not: {fired}"
    # THE TWO BOUNDS, PLANTED. Widening either is what a later reader will be
    # tempted into, so each has a case that goes red the moment it is: `stale`
    # dies if `SPAN` grows, `scatter` if `TOL_S` does.
    stale = onair._ConnectEvidence()
    assert not _offer(stale, ((1, 0.0925), (12, 0.0938), (25, 0.0912))), \
        "three agreeing accepts spread over a whole call connected"
    # `stale` dies once `SPAN` reaches 25, and the bound stands at 12 -- the width
    # the docstring prices at one false link over the 416 negative recordings, and
    # the width WS8EOC's three answers of 2026-09-11 actually arrived in (searches
    # 5, 7 and 13 of `onair-0911-2332`). A twenty-four-cycle spread is a call's
    # worth of coincidence; nine searches is this station's own 4-call/6-hush
    # rotation, and the case below is that peer rather than a hypothesis.
    wide = onair._ConnectEvidence()
    assert _offer(wide, ((5, 0.098), (7, 0.100), (13, 0.099))) == [13], \
        "WS8EOC's three answers spread over nine searches did not connect"
    sparse = onair._ConnectEvidence()
    # `onair-0730-2036` -- a real gateway, answered sparsely. The span has to
    # hold it: measured on the negatives, nothing inside twelve cycles fires.
    assert _offer(sparse, ((3, 0.1), (6, 0.11), (8, 0.0975))) == [], \
        "an 11 ms spread is not one turnaround"
    sparse = onair._ConnectEvidence()
    assert _offer(sparse, ((3, 0.1), (6, 0.0988), (8, 0.0975))) == [8], \
        "a sparse real answer was refused"

    # ANY THREE, NOT THE LAST THREE. One chance accept landing between a peer's
    # answers must not undo them, and taking the last three in the span is exactly
    # a rule that lets it: the same four answers connect on their own and used to
    # go unrecognised with a single 400 ms accept dropped in the middle. Measured
    # over the corpus, 23% of eight-cycle spans of occupied HF hold a spurious
    # accept, so this is the common case and not the corner.
    peer = [(1, 0.0925), (2, 0.0930), (4, 0.0921), (5, 0.0928)]
    clean = onair._ConnectEvidence()
    assert _offer(clean, peer) == [4, 5], \
        f"a peer at one turnaround did not connect: {clean.candidates}"
    fouled = onair._ConnectEvidence()
    assert _offer(fouled, sorted(peer + [(3, 0.400)])) == [4, 5], \
        (f"one accept on noise defeated four agreeing answers: "
         f"{fouled.candidates}")

    # -- and against the recordings, through the production search ----------
    #
    # Every one of these is committed under `captures/`, so `want` is the cycle
    # the rule closes in and not merely whether it does. Two of the six are real
    # WS8EOC sessions the search never read three answers in: the rule declines
    # them, which is the behaviour being asserted rather than a shortfall.
    controls = (
        ("WS8EOC 2026-07-30 2036", SESSIONS / "onair-0730-2036", 7),
        ("WS8EOC 2026-07-30 2037", SESSIONS / "onair-0730-2037", None),
        ("WS8EOC 2026-07-30 2039", SESSIONS / "onair-0730-2039", 5),
        ("WS8EOC 2026-07-30 2047", SESSIONS / "onair-0730-2047", 4),
        ("WS8EOC 2026-07-30 2049", SESSIONS / "onair-0730-2049", 5),
        ("WS8EOC 2026-07-30 2050", SESSIONS / "onair-0730-2050", None))
    if not any(any(path.glob("rx_*.wav")) for _, path, _ in controls):
        pytest.skip(f"the six 2026-07-30 sessions under {SESSIONS} are absent "
                    "together -- committed to the source tree, they do not "
                    "cross the publication boundary; run from a checkout")
    seen = 0
    for name, path, want in controls:
        wavs = sorted(path.glob("rx_*.wav"))
        assert wavs, f"{name}: {path} holds no rx_*.wav -- fixture missing"
        seen += 1
        ev = onair._ConnectEvidence()
        at = None
        for wav in wavs:
            cyc = int(wav.stem.split("_")[1])
            ev.note_search(0.075)
            d = _search(rxfront.load_wav(str(wav)))
            if d is not None and ev.offer(cyc, d) and at is None:
                at = cyc
        print(f"  {name}: {len(ev.candidates)} candidate(s) "
              f"{[(c, round(d * 1e3)) for c, d, _ in ev.candidates]}, "
              f"{'connect at cycle ' + str(at) if at else 'no connect'}")
        assert at == want, (
            f"{name}: expected {'cycle ' + str(want) if want else 'no connect'}, "
            f"got {at} from {ev.candidates}")
    assert seen == 6, seen

    # The three calls of 2026-08-06, when they are here: gitignored, kept in
    # rf-corpus, and the reason any of this exists. Optional and SAID to be
    # optional -- a fixture that may be absent is not the same object as one that
    # must be present, and the two used to be the same loop.
    for name in ("onair-pactor-0806", "onair-pactor-w6ids-0806",
                 "onair-pactor-w9otr-0806"):
        wavs = sorted((SESSIONS / name).glob("rx_*.wav"))
        if not wavs:
            print(f"  [absent, not required] {name}")
            continue
        ev, at = onair._ConnectEvidence(), None
        for wav in wavs:
            cyc = int(wav.stem.split("_")[1])
            ev.note_search(0.075)
            d = _search(rxfront.load_wav(str(wav)))
            if d is not None and ev.offer(cyc, d) and at is None:
                at = cyc
        print(f"  {name}: {len(ev.candidates)} candidate(s), "
              f"{'CONNECT at ' + str(at) if at else 'no connect'}")
        assert at is None, f"{name}: a phantom link came back, at cycle {at}"


def test_the_candidate_count_is_printed_beside_its_own_null() -> None:
    """A count of candidates is not a measurement of the channel on its own.

    `acquire_control_signal` bandpasses 1200-1800 Hz, hard-thresholds the
    instantaneous frequency and compares twelve bits against two codewords in two
    senses at every alignment across the band. There is NO ENERGY TEST anywhere in
    it, so it accepts on nothing at 4/4096 per alignment and the same audio scaled
    by a factor of 100000 gives the identical answer. `connect candidates: 2` reads
    to an operator as "the channel answered twice"; it is very close to none, and
    the line has to say so without waiting for a statistics lesson.

    Both halves are asserted here: that the null the summary prints is the rate
    the search really has, and that the summary prints it.
    """
    from hfmodem.shrike import onair  # noqa: E402

    # -- the null is the search's own, measured against the search --------------
    rng = np.random.default_rng(20260820)
    span = 0.125
    noise = [rng.standard_normal(int(0.40 * p1rx.FS)) * 0.05 for _ in range(1200)]
    accepts = {g: {i for i, x in enumerate(noise)
                   if p1rx.acquire_control_signal(
                       (x * g).astype(np.float32), 0.05, spec.P1_CS_S,
                       span=span) is not None}
               for g in (1.0, 0.001, 100.0)}
    assert accepts[0.001] == accepts[1.0] == accepts[100.0], \
        "the search is amplitude-blind, and a gain change moved its accepts"
    got = len(accepts[1.0]) / len(noise)
    want = p1rx.acquire_null(span)
    assert 0.75 < got / want < 1.35, (
        f"the search accepts on noise in {got:.2%} of {len(noise)} trials and "
        f"`acquire_null` predicts {want:.2%} -- the summary prints the second "
        f"where the operator will read it as the first")

    # -- and the summary prints it beside the count ----------------------------
    ev = onair._ConnectEvidence()
    for _ in range(24):
        ev.note_search(span)
    empty = ev.report()
    assert "24 cycles searched" in empty, empty
    assert f"about {24 * want:.1f}" in empty, empty
    # WHAT WAS SEARCHED. The first cycle of a hush is searched too -- the band
    # belonging to the last call of a run lands in the cycle after it -- so
    # "any cycle we called in" understated the population the null is over.
    assert "cycle we called in" not in empty, empty

    ev.offer(3, 0.095)
    ev.offer(19, 0.069)
    two = ev.report()
    assert two.startswith(f"connect candidates: 2 of 24 cycles searched, "
                          f"chance alone gives about {24 * want:.1f}"), two
    assert "none corroborated" in two, two

    # A session whose windows never held the band searched nothing, and a null
    # over no cycles is not "about 0.0" -- it is not a figure at all.
    assert onair._ConnectEvidence().report() == (
        "connect candidates: no cycle was searched -- no window held the band "
        "an answer to our call was due in")


def test_the_connect_budget_outlasts_the_evidence_rule() -> None:
    """A call must still be running in the cycle its third answer arrives.

    Two counters bound a connect and they were set independently: the ARQ gives
    up after `max_connect_retries` cycles with no sign of the peer, and
    `_ConnectEvidence` will not call a codeword a station until three of them
    agree. A gateway that answers sparsely makes the second slower than the first
    -- WS8EOC answered `onair-0730-2036` on cycles 3, 5 and 7 -- and a cycle the
    peer WAS heard in is charged before it is forgiven, because `note_peer_heard`
    is set after that cycle's tick and consumed by the next one. So reaching
    cycle 7 costs five retries, and the budget was four.

    The failure is silent in the worst way: the session aborts one cycle before
    the evidence it was collecting would have closed, and reports that nobody
    answered.
    """
    from hfmodem.shrike.arq import PactorArq, State  # noqa: E402

    class _IO:
        def connect_burst(self, *a): pass
        def send_packet(self, *a, **k): pass
        def send_cs(self, *a): pass
        def connected(self, *a): pass
        def disconnected(self): pass
        def deliver(self, blob): pass
        def buffer(self, n): pass
        def log(self, m): pass

    fsm = PactorArq(_IO())
    fsm.on_host_connect("W9SSJ", "WS8EOC")
    reached = 0
    for cycle in range(1, 8):
        fsm.on_cycle()
        if fsm.state != State.CONNECTING:
            break
        reached = cycle
        if cycle in (3, 5, 7):
            fsm.note_peer_heard()          # a candidate, not yet corroborated
    assert reached == 7, (
        f"the call was abandoned in cycle {reached + 1}, before the third answer "
        f"of a peer on cycles 3/5/7 could be read (state {fsm.state}, budget "
        f"{fsm.cfg.max_connect_retries})")

    # THE OTHER END OF THE SAME COUNTER, which nothing held. Raising the budget only
    # ever makes the case above happier -- at 20 it is just as green -- and what it
    # buys is 25 seconds of this station transmitting at a frequency nobody is
    # answering on, every attempt. A call that hears nothing at all has to stop.
    silent = PactorArq(_IO())
    silent.on_host_connect("W9SSJ", "WS8EOC")
    ran = 0
    for cycle in range(1, 20):
        silent.on_cycle()
        if silent.state != State.CONNECTING:
            break
        ran = cycle
    assert ran <= 8, (
        f"a call nobody answered ran {ran} cycles -- {ran * 1.25:.0f} s of keying "
        f"at silence (budget {silent.cfg.max_connect_retries})")


def test_main() -> None:
    rc = main()
    if rc == 2:
        pytest.skip(f"no fixtures under {CORPUS} -- set HFMODEM_CORPUS to the "
                    "off-air recordings")
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
