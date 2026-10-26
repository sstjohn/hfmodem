<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-2 mail implementation review — 2026-09-20

`tools/console pactor ... --p2-mail` now selects a PACTOR-2 ceiling through the
existing mail and radio paths. It is an experimental approximation of a PTC-II
without PACTOR-3, not a claim of complete SCS interoperability.

## Intended behavior and evidence

SCS's [PTC-II 4.0 manual](https://www.rigpix.com/textandimaging/scs_ptcii_v4_manual.pdf),
§6.64, defines `MYLevel 2` as a level-II controller that switches to PACTOR-II
when the peer supports it. The implementation retains P1 setup and fallback,
and enforces a maximum protocol of P2 in grant handling, upgrade selection and
received packet/control events. No new console mode is introduced.

| Component | Current evidence / limitation |
| --- | --- |
| P2 modulation, FEC, CRC, markers | Implemented in `shrike/pactor2.py`; decoding in `p2rx.py`. |
| SL1–3 transmit, short and long cycles | Independently decoded by SCS's own decoder with payload recovery in the September 17 grading. |
| P2 SL1 entry waveform | SCS's decoder recovered it both cold and behind P1 setup. This establishes waveform readability, not successful negotiation. |
| SL4 | Self-decoding works in the newer grading, but no independent positive at SL4 exists. The ARQ ladder caps adaptation at SL3. Older comments attributing this solely to our receiver are stale. |
| Control signals and timing | Implemented and exercised in synthetic two-way audio tests; still a hypothesis without an independent responding peer. SCS's decoder cannot grade this reverse channel. |
| Connected ARQ | Existing audio tests cover payload, speed changes, long cycles, changeover and QRT between two copies of our implementation. |
| P1-to-P2 negotiation | `01` announcement / `0x6A9` response is the local capability hypothesis. The corpus search found established P2 traffic but no captured P1-to-P2 completion. |
| Winlink mail | Existing B2F client is wired through P2. No real-peer P2 mail transfer was established in this review. |

The central records: no captured session shows a PACTOR-2 entry, and the two
recorded PACTOR-1-to-PACTOR-3 completions skip Level 2 in one step; SCS's decoder
independently decoded every PACTOR-2 waveform this station can key except SL4
and the control signals, for which it has no decoder at all, so no
transmitter defect was found; the 1990 text-declared capability digit is dead
on modern air, and gateway status-bit correlation remains a live hypothesis
rather than a settled reading; and no arm flown here has ever logged a live
PACTOR-2 upgrade.
The older `--offer-pactor2` tries P3 first and then P2; it does not implement a
level-2 ceiling. Likewise `--p1-6a9 --p3-entry p2sl1` selects a P2-shaped entry
inside a P3 session. The new preset enters the actual P2 protocol and ladder.

## Preset

```sh
tools/console pactor fly KB5LZK --band 40 --p2-mail --sense --dry-run
```

Use `--execute` for a contact, `--send FILE|delivery` to upload as well as fetch,
and `--long-cycle` to permit extended packets. These examples specify command
syntax, not a verified P2-serving gateway. Some channels explicitly refuse
levels below P3.

The modem preset enables mail, requests eventual changeover, announces `01`,
and accepts `0x6A9` as experimental permission to enter P2 at SL1. `0x59A`
does not enter P3. An ordinary P1 acknowledgement accepts setup in P1 and
releases the initial changeover hold; subsequent P1 traffic clears the
capability bits. P2 timeout/fallback retains the ceiling. Reconnecting resets
contact state while preserving that policy. Grants cannot reopen teardown.

Console provenance records the ceiling, response word and mail settings.
Conflicting generation presets are rejected before opening hardware. P1/P2
occupancy uses 1200–1800 Hz. The existing power preset's `p3_watts` field is
also applied to the P2 PSK phase; `p1_watts` remains the setup/fallback phase.

## Remaining interoperability work

The decisive missing evidence is a cooperating SCS PTC-II at `MYLevel 2`, or a
continuous recording covering its P1 call, announcement, response, P2 entry,
controls, payload exchange and disconnect. Validate the `0x6A9` interpretation
and control placement there before treating a timeout as a waveform defect or
claiming P2 mail works. Separately obtain a real SL4 positive before lifting
the existing SL3 cap.

This review and implementation used offline checks only; no RF was transmitted.

## Validation of this change

667 pytest cases passed across the console, P2 policy/audio/QSO/entry tests,
grant reconnect/teardown/changeover tests and session reporting. The standalone
arm-default and phase-power suites also passed. The console dry-run and shell
syntax check passed.

One existing test,
`test_the_pactor_campaign_line_is_the_one_the_branch_point_flew`, was excluded
from the final run after reproducing its failure with `HEAD`'s unchanged
`consolelib/pactor.py`. Its expected P3 command omits the existing
`--p3-max-try` and `--p3-max-down` arguments. That unrelated fixture is unchanged.

The new audio test injects the hypothesized grant, then carries P2 payload
through the production modulation/receive path between two local instances.
It verifies the new routing without independently verifying the grant's meaning.

## WS8EOC follow-up: repeated `0x6A9`

The operator's 2026-09-20 23:25Z contact on 40 m decoded fourteen zero-error
`0x6A9` replies after the `0x11` announcement.
The first entered native P2; four SL1 bursts went out, but subsequent replies
were incorrectly omitted from the grant-retry accounting. The modem called
that silence and fell back after four cycles. No P2 acknowledgement, greeting,
or mail was decoded.

The follow-up fix credits the selected response word (`0x6A9` for this preset,
`0x59A` normally). Granted native P2 now uses the existing finite
`ENTRY_GRANT_CYCLES` budget, independently of the flag that selects P3's
special entry waveform. Repeated grants do not acknowledge bytes or confirm
the upgrade; genuinely unanswered cycles still reach the silence limit.

The teardown report now counts actual native P2 keyings and explicitly declines
to apply the P3 entry-position diagnostic to them. It no longer says that no
entry was keyed when P2 bursts were transmitted.

`test_p2_granted_retry.py` replays the recorded reply sequence as decoded events,
checks the finite retry limit, wrong-word and silence controls, P2 confirmation,
state reset, and the P2 keying report. The correction was tested offline; it
does not establish successful entry or mail delivery against WS8EOC.

## Offline corrections after the September 20 WS8EOC attempts

Native P2 now holds queued mail and the handover request until a P2 response
confirms entry. This state is separate from the special P3 entry renderer.
Repeated FSK grants retain the existing finite retry budget and P1 fallback.
The P2 header's low bit follows status counter parity, as observed in 30 matched
CRC-valid stock HB9AK frames, including retransmissions.

After an emitted P2 packet, the reply reader checks the P2 predicted instant
relative to that emission's phase reference. It preserves the separate P1
reply clock for repeated grants. It does not search across the listening
window. The 0.880/3.360-second P2 response offsets and CS modulation remain
hypotheses pending an independent responding-peer recording.

TX descriptions now contain the exact field, marker index and carrier
arrangement. Actual P2 transmissions also save the leveled, trimmed DAC waveform
and its phase-lead/timing metadata. These are references, not RF measurements.

Offline timing candidates at 0 and +18.75 ms were checked against recorded
sample indices and hashes. The second offset comes from P3 evidence and is
**not established for P2**. Those artifacts were initially prepared without changing production timing.
After the 19:49 attempt, the user requested the +18.75 ms candidate as the
`--p2-mail` default. The preset now delays native P2 phase by 900 samples at
48 kHz throughout the P2 session, including after confirmation. P1 setup and
fallback, ordinary P3, and the P3 `p2sl1` compatibility renderer keep their
existing timing. The finite entry retry budget is unchanged. Console provenance
records `p2_phase_delay_ms=18.75`; TX descriptions and DAC metadata also record
the offset. P2 reply prediction follows the actual shifted emission.
This remains an experimental timing hypothesis, not a confirmed interoperability
fix. No contact was transmitted while implementing it.
