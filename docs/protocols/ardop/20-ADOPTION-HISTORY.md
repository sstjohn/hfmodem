# How ARDOP Became a First-Class Winlink Mode

*Method: this account is built from public Winlink/ARSFI pages, the ARDOP specification and
host-interface documents, the Pat and BPQ32 community records, and the ardopcf fork's own docs
(`ardopcf/docs/About_Ardop.md`, `Motivation.md`). Every claim is cited. Items that could
not be confirmed from a primary source are marked **UNVERIFIED** with the reason.*

The point of this file is the **process**, not the DSP: it is the precedent for how a clean-sheet
open mode ("sabir") would petition ARSFI for the same standing.

---

## 1. Who proposed it, under what umbrella, and why

ARDOP (Amateur Radio Digital Open Protocol) was created by **Rick Muething, KN6KB** — the same
author who wrote its predecessor **WINMOR**
(`ardopcf/docs/About_Ardop.md`;
[WINMOR is Deprecated](https://winlink.org/content/winmor_deprecated)). It was run as a
**joint development effort under the Amateur Radio Safety Foundation,
Inc. (ARSFI)** — a non-profit public benefit corporation — with the **Winlink Development Team**
([ARDOP Overview, Rev 0.4, 15 Mar 2015](https://winlink.org/content/ardop_overview)). The overview
names Muething alongside Rob Hernandez KM6LBU, Phil Sherrod W4PHS, Steve Waterman K4CJX, Tom Lafleur
KA6IQA, Lor Kutchins W3QA and others; the public launch note also credits **John Wiseman G8BPQ**,
Matthew Pitts N8OHU, and Neil Hughes as contributors
([A New Open Protocol is Coming, 13 Jun 2015](https://winlink.org/content/new_open_protocol_coming)).

**Stated motivation.** Against **WINMOR**, ARDOP was to be "superior speed, robustness, and multiple
bandwidth options" and — decisively — **multi-platform**: "the design is open, and the software
implementations will be open-sourced," targeting Windows, Linux, macOS, iOS and Android, in software
or hardware ([New Open Protocol](https://winlink.org/content/new_open_protocol_coming)). Against
proprietary **PACTOR** (SCS hardware TNCs) and — later — **VARA** (closed freemium software), the
open license *was* the pitch: an open-source mode the network would not have to buy or take on
trust. The ardopcf fork still frames its existence exactly this way — keeping an open,
multi-platform mode alive so "Winlink gateway operators continue to support Ardop"
(`ardopcf/docs/Motivation.md`).

## 2. Timeline

| When | Milestone | Source |
|---|---|---|
| 15 Mar 2015 | ARDOP Overview Rev 0.4 published (design goals, dev team) | [ardop_overview](https://winlink.org/content/ardop_overview) |
| 13 Jun 2015 | Public announcement; alpha testing underway, beta expected "early Fall 2015" | [new_open_protocol_coming](https://winlink.org/content/new_open_protocol_coming) |
| ~end 2016 | "Near the end of beta testing for the ARDOP_Win TNC"; being integrated into Winlink Express + RMS Trimode; "2017 will be the year Winlink moves to ARDOP" | [ardop_news](https://winlink.org/content/ardop_news) |
| 2016–2017 | ARDOP (and VARA) offered as beta in the shipping client/gateway, after first proving out via BPQ32 | [ardop_and_vara_now_beta_testing](https://winlink.org/content/ardop_and_vara_now_beta_testing_winlink_software) |
| 27 Nov 2017 | ARDOP Specification dated build (the copy bundled with ardopcf) | `ardopcf/docs/refs/ARDOP_Specification_20171127.pdf` |
| 10 Jul 2020 | ARSFI Board **deprecates WINMOR**; asks sysops to move to ARDOP, VARA HF, PACTOR 3/4 | [winmor_deprecated](https://winlink.org/content/winmor_deprecated) |
| 1 Apr 2024 | Community fork **ardopcf** begins (v2.0.3.2.1, renamed v1.0.4.1.1 on 26 Apr 2024) | [changelog.md](https://github.com/pflarue/ardop/blob/master/changelog.md) |

**ARDOP 1 vs 2/3.** Winlink runs **ARDOP v1 only**. **ARDOP v2 "appears to be abandoned"**
([Pat wiki ARDOP](https://github.com/la5nta/pat/wiki/ARDOP), citing a 2019 issue), and a later
**ardop3** (adding Viterbi coding for weak-signal gains) also **stalled**
([bpq32 groups.io thread](https://groups.io/g/bpq32/topic/74339642) — page returned HTTP 402 to the
fetcher; summary is from the search index, treat the ardop3/Viterbi detail as **UNVERIFIED** pending
a direct read). **UNVERIFIED — "went closed source":** the public record says v2/v3 were *abandoned
/ never released*, not specifically that a finished v2 was withheld as closed source. Do not assert
the stronger "closed" claim without a primary source.

## 3. What had to exist before the network would carry it (each is a gate)

1. **A working TNC/modem.** `ARDOP_Win` (Muething, Windows) shipped as a component of Winlink Express;
   `ardopc`/`piardopc` (John Wiseman G8BPQ, C, multi-platform) began as a **translation of
   ARDOP_Win into C** (`ardopcf/docs/About_Ardop.md`). *Gate: a
   reference implementation the Winlink Team could bundle and a portable one for gateways/Linux.*
2. **Host-interface conformance.** ARDOP had to speak the Winlink **TNC host interface** so an
   unmodified client could drive it — Muething published a *Host Interface Spec for the ARDOP TNC*
   ([winlink.org PDF](https://winlink.org/sites/default/files/downloads/ardop_tnc_host_mode_interface_spec.pdf);
   the command set survives in ardopcf's
   `ardopcf/docs/Host_Interface_Commands.md`). *Gate: the
   modem is a black box the existing client/gateway plumbing can open, listen, connect, disconnect,
   and stream data through without bespoke code.*
3. **Gateway (RMS) support.** RMS Trimode had to add ARDOP to the protocols it will listen for and
   scan (see §4). *Gate: on-air availability — a client can select ARDOP but reach nothing until
   gateways answer it.*
4. **Client dropdown entry.** ARDOP had to appear as a selectable session type in Winlink Express
   ([ardop_and_vara_now_beta_testing](https://winlink.org/content/ardop_and_vara_now_beta_testing_winlink_software)).
   *Gate: routine reachability for ordinary users, not just testers.*
   **UNVERIFIED — exact GA date:** the specific Winlink Express release/date that first exposed ARDOP
   as a non-beta dropdown item is not pinned in a primary source; the beta-integration note is
   undated and "2017 will be the year" is aspirational.

## 4. What the gateway software specifically had to do

The server-side artifact is **RMS Trimode** — the sysop program that lets one station serve
several client protocols on shared radio/antenna/PTT
([RMS Trimode](https://winlink.org/content/rms_trimode)). Adding ARDOP meant:

- A new **listen/scan mode** selectable by checkbox alongside the others — `P3/4`, `P1/2`,
  `W` (WINMOR), **`A` (ARDOP)**, `Vara`, `Rp` (Robust Packet) — able to scan one or all protocols,
  and across multiple frequencies by time of day.
- **Launching and arbitrating the ARDOP TNC**: the WINMOR and ARDOP modems are software delivered
  *with* the RMS Trimode install (VARA, by contrast, is a separate download), and once a session
  starts in one protocol Trimode blocks the others until it completes
  ([RMS Trimode](https://winlink.org/content/rms_trimode)).

So the gateway gate was concrete and small: **ship the modem in the installer and add one entry to
the protocol scan list** — plus the message-routing side (RMS Relay) handling ARDOP sessions like
any other. **UNVERIFIED — RMS Relay specifics:** the exact RMS Relay changes are not documented in a
primary source I could reach.

## 5. How long, and what nearly stopped it

**Proposal (2015) → routine on-air use (2017) ≈ two years** to move from Rev 0.4 to a mode shipping
in both the flagship client and the standard gateway
([ardop_news](https://winlink.org/content/ardop_news);
[beta note](https://winlink.org/content/ardop_and_vara_now_beta_testing_winlink_software)). WINMOR's
formal deprecation followed in 2020, five years after proposal
([winmor_deprecated](https://winlink.org/content/winmor_deprecated)).

What put it at risk was **developer bandwidth.** The protocol's momentum
tracked one author: after v1 shipped, Muething's **ARDOP 2 and ARDOP 3 efforts stalled and were
abandoned** ([Pat wiki](https://github.com/la5nta/pat/wiki/ARDOP);
[bpq32 thread](https://groups.io/g/bpq32/topic/74339642)), and upstream `ardopc` went quiet. ARDOP
survived because it was **open and multi-platform**, which let the community catch it: Peter LaRue
**AI7YN** (formerly KG4JJA) forked ardopc into **ardopcf** on **1 Apr 2024**, fixing the "dreaded
decoding issue" on recent Linux kernels and resuming maintenance — announced to pat-users on
2 Apr 2024 ([changelog](https://github.com/pflarue/ardop/blob/master/changelog.md);
[pat-users thread](https://groups.google.com/g/pat-users/c/QGfUaTDuUrA)).

## 6. The lesson for sabir

To make the same request of ARSFI succeed, a clean-sheet open mode should arrive with the gates
already cleared, in the order the network cares about:

1. **Published, versioned, open specification** — ARDOP led with a dated overview and a formal spec.
   Sabir needs a stable, citable document, not just code.
2. **A modem that speaks the Winlink TNC host interface** — conform to the existing host-command
   contract so an unmodified client drives it as a black box. It is what lets the
   client-dropdown and gateway integration be trivial.
3. **A portable, permissively-licensed reference implementation** — ARDOP's C `ardopc` (multi-OS)
   is what made gateway and Raspberry-Pi deployment possible. Sabir should ship the Linux/portable
   modem *first*, not a Windows-only binary.
4. **A trivial gateway integration path** — target RMS Trimode's model: modem in the installer, one
   entry in the protocol scan list. Do not require sysops to run bespoke software.
5. **A concrete value proposition versus the incumbents** — ARDOP offered *open + multi-platform* vs
   WINMOR and vs proprietary PACTOR/VARA. Sabir needs a crisp "why carry this too" (openness
   plus a measured throughput/robustness story), phrased for gateway operators who bear the cost.
6. **More than one maintainer, and stay open.** ARDOP's near-death was single-author bandwidth; its
   rescue was that anyone *could* fork it. Sabir should plan for a maintainer community and an
   AGPL-class license from day one, so a quiet spell is survivable rather than fatal.
