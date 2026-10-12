# 2G ALE (MIL-STD-188-141B appendix A)

`hfmodem.bateleur.ale` receives and generates 2G ALE words for monitoring
and operator-directed calls and links. The operator can enable scanning to
retune the rig through a configured channel list. Host commands and events
are in [ALE-HOST.md](ALE-HOST.md).

## Interoperability evidence

| Part | Evidence retained in the tree | Open comparison |
|---|---|---|
| Clear calls, sounds, responses and AMD | Coding vectors and exchanges between this transmitter and receiver | An earlier LinuxALE 0.0.2 run reportedly read calls, sounds and responses. Its WAV, output and command were not retained; that result cannot be reproduced from the tree. |
| DTM | Local block, combining and ARQ tests | CRC coverage does not reproduce figure A-48; unverified against a foreign implementation. |
| DBM | Local interleaving, block and ARQ tests | Unverified against a foreign implementation. |
| Group, net, AnyCall and AllCall | Local multi-station exchanges | Foreign timing and group preamble recordings needed. |
| Linking protection AL-1/AL-2 | B.5.6.4 cipher examples, seed example and local exchanges | Word numbering, DBM pair numbering, clear authentication and time exchange remain unverified against a foreign implementation. |
| Selcall | Local character and waveform tests | Earlier recording results and the open M.493 comparison are described in [SELCALL.md](SELCALL.md). |

The transmitter sends each tone's three bits most significant first. Local
round trips test that choice; the unrecorded LinuxALE result does not provide
a retained interoperability fixture. The capabilities report follows figure
A-45's four DATA words after CMD, rather than the five named in
A.5.6.6.2.2. A captured capabilities reply with decoded fields would settle
that discrepancy.

Earlier performance tables quoted 100 handshakes per point and link times
of 13.0–13.3 seconds. Their harness and raw results were not retained.
The executable tests under `tests/bateleur/ale` exercise selected simulated
channels; they do not reproduce those tables.

## DTM and DBM

DTM uses 21-bit data fields with alternating DATA/REP preambles. This
implementation checks the message and stuffing bits in those fields,
excluding the CMD DTM and each three-bit preamble. It uses polynomial
0x1021, preset 0xFFFF, final complement, most significant bit first. ASCII
stuffing is DEL; binary stuffing is ones. A bare CMD CRC elsewhere covers
whole 24-bit words.

This reading is **unverified against a foreign implementation**. In the
[public MIL-STD-188-141B text](https://www.udxf.nl/MIL-STD-188-141B.pdf),
figure A-48 (printed page 175) labels the reconstructed example CRC 0x2C5B.
For `THE QUICK BROWN FOX!` plus seven DEL stuffing bits, this code computes
0xDB2A; with seven zero stuffing bits it computes 0x5452. Neither matches.
The figure is unresolved evidence, not an illustrative value to discard.
A foreign DTM recording must retain the complete CMD DTM, every DATA/REP
word including stuffing, CMD CRC and the ACK/NAK exchange, with the original
message bytes and receiver output. A repeated damaged block would also
settle combining behavior.

DTM NULL replies currently use KD4=1 for affirmative participation; DTM ARQ
uses KD4 as ACK/NAK rather than data type. These readings also need that
foreign exchange. Messages end in a BASIC block; received partial messages
are also delivered on a new message sequence, NULL, or link end.

DBM encodes 12-bit groups as Golay rows, interleaves by column, and places
the CRC in the final 16 bits of the padded data field. A foreign BASIC and
EXTENDED exchange with known bytes, depth, stuffing and ACK/NAK would settle
its wire format. Protected DBM additionally needs both endpoints' test key,
UTC and channel frequency to verify pair numbering.

Outbound DTM/DBM queues hold at most 16 messages and 65,536 bytes. Blocks
are sized to `[rig] max_key_s`, including the full source and destination
addresses. If the smallest block does not fit, the command reports an
error. A complete call, response, sound or AMD frame that exceeds the limit
is refused with its duration; it is never cut into incomplete ALE frames.
The configured channel count, address lengths and sounding duration must
therefore fit the selected limit. Retries stop after seven attempts; a
block exchange, including requested pauses, is capped at five minutes.

Reception retains DTM/DBM recognition and combining. The receiver bounds
announced blocks by its configured DTM word and DBM depth limits and drops
an unfinished block after its expected receive window. Nonfinite samples
are replaced with zero. A steady single tone does not extend ALE activity
indefinitely, and a signal-detection episode has a finite scan hold.

## Linking protection

AL-1 and AL-2 encrypt ALE words under a shared 56-bit key before Golay
coding. `[protocols.ale] lp_level` is zero by default; 1 selects AL-1 and 2
selects AL-2. `lp_key` is fourteen hexadecimal digits. Startup reports the
selected level without printing the key.

The operator's chosen `part97` station profile excludes linking protection,
regardless of channel. Under `unregulated`, startup and each transmit attempt
warn when the dial lies in an amateur band-plan range. The check uses the
same HF ranges as the logbook, including its broad 5.30–5.41 MHz 60 m range.
These are configured profile limits.

The implementation uses date, time, word number and frequency in the seed.
AL-1 uses 60-second protection intervals; AL-2 uses two seconds. The
receiver searches the current and adjacent intervals and retains protected
traffic recognition. Its candidate ranking and one-word lookahead are local
receive choices. Single-word protected detections are discarded because
they are too ambiguous under the candidate seeds.

`TIME` and `TIMECAST` exercise protected and clear time acquisition. A
foreign recording spanning an interval boundary, with known key, UTC,
frequency, clear authentication word and DBM payload, is needed to check
the numbering and time exchange. AL-3, AL-4, passive time acquisition and
the second-transmission restriction within one AL-2 interval are not
implemented.

## On a station

    [protocols.ale]
    enabled    = true
    host       = "ale"
    port       = 8330
    addresses  = ["N0CALL"]
    channels   = [7102000, 10145500, 14109000]   # dial frequencies in Hz
    scan_rate  = 2.0
    activity_s = 30.0
    sound_s    = 1800.0
    nets       = [{address = "NET1", member = "N0CALL", slot = 2}]

Host connections use the [normal host opening](HOST-API.md#host-connections).
`sound_s` is zero to disable periodic sounding, or at least 60 seconds.
`activity_s` must be finite and nonnegative. With `conventions = "hflink"`,
the HFLINK preset supplies the activity and sounding intervals; explicit
`activity_s` and `sound_s` entries must be removed. `base` and `pilot_channels` require that
preset. The operator supplies the channel list.

ALE holds scanning while another lane reports an active link, checks that
a requested tune succeeded, and checks the dial again before sending on a
channel. On stopping, it restores the configured station centre. Tune
failures name the dial frequency and the rig error. This coordination is
within this station process; a separate process has no shared ownership
protocol here. It requires separate coordination by the operator.

Other lanes' transmissions update ALE link activity. The arbiter reserves
an active lane's link against unrelated transmit errands. Individual
responses are limited to one per caller per minute, with at most 64 recent
callers. Repeated AllCalls do not refresh an established link's activity.

The receiver currently ends an ALE frame after an empty redundant-word
slot. The response budget is 5 Tw, about 653 ms; the empty slot consumes
392 ms. Software timing tests exclude hardware PTT, device buffering and
rig settling. A synchronized recording of the incoming final tone, PTT and
outgoing first tone is required to verify turnaround on a station. No
physical turnaround measurement is established here.

Net response slots whose earlier members are unknown assume one-word
addresses. A net with longer earlier addresses needs an explicitly agreed
slot plan. Group scanning calls alternate THRU/REP and leading calls use
TO/REP; these choices need a foreign group-call recording with known member
order and addresses.

## Monitoring directory

`ale.directory` reads the community ALE user list and local CSV files of
`kind,key,agency,description`. Configure `directory = ["default", "ours.csv"]`
to load them in the background; `WHO` and first-heard events show matches.
`hfmodem ale-list` lists entries, and its `fetch`, `who`, `freq`, `agency`,
`channels` and `acronym` subcommands query or refresh them. Sources are
identified on entries. A malformed refresh retains the last parseable
cache; a successful refresh replaces it atomically.

Directory, DTM/DBM and protected-traffic monitoring remain available.
Channel lists are not restricted to transmit profile segments at load;
each transmission still passes through the station's emission checks.
