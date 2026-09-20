# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the passive gateway survey must not get wrong before it aims a rig slot.

`tools/gwsurvey.py` decides which channel the next transmission goes to. Three
defects are pinned here, all found by review after the verdicts had already been
acted on:

  * **A decode attributed without its direction.** A callsign in a frame is
    either the station that KEYED it or the station it was ADDRESSED TO, and the
    survey pooled them by searching the whole decode line for the listed call.
    So a third party dialling WS8EOC graded WORTH CALLING — a rig slot aimed at a
    station nothing had been heard from, which is the exact failure the tool
    exists to prevent — while a gateway ANSWERING somebody on its own channel,
    the strongest evidence the runbook recognises, was demoted to CHANNEL ACTIVE.
  * **A substring where a callsign was meant.** `KW8MWX` contains `W8MW`, so a
    stranger's connect request graded W8MW worth calling; and `K5DAT-13` is never
    a substring of anything the monitors emit, because the survey hands them the
    base call, so that gateway could not be credited with its own answer.
  * **A gateway's answer that no line survived to carry.** `vara_monitor` writes
    the kind as a sixteen-column field and one space; the survey split on two
    spaces or more, so `connect-response` — sixteen characters exactly — matched
    nothing and was dropped before any of the above ran. Two perfect 15/15
    answers graded SILENT.

The third one is the reason this file's fixtures are what they are. The old ones
called themselves real monitor output and were not: the spacing was wrong in the
lines that mattered, and so were besra's frame names (`ConReq.500` for
`ConReq500M`), its session ids and its frame types. Nothing caught that because
no test ever called `parse_vara` — the fixtures went straight into `attribute`,
past the parser that was dropping them. Every line below is now output from the
monitor itself, and `_tier` runs the parsers the survey runs. Provenance is on
each block.

The occupancy score is the fourth defect: `core.busy` sized its FFT in samples
against 48 kHz, so on the KiwiSDR's 11999 Hz stream every frame ran 341 ms
instead of 85 and averaged across the keying gaps `burst` measures. That one is
pinned in `tests/kestrel/test_channel_busy.py`, beside the calibration it belongs
to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO / "tools")]
G = pytest.importorskip("gwsurvey", reason="tools/ is not beside the package")

# --------------------------------------------------------------------------- #
# `python3 tools/vara_monitor.py --wav <rendered channel> --calls WM4RB,WS8EOC,
# K5DAT,KC9GHZ`, over CR / connect-response bursts synthesised by
# `vara_mfsk.synth_burst` and a `vara_ofdm.link_setup_tx` over. The callsigns are
# chosen for the scenarios below; the LINES are the monitor's.
CR_TO_WM4RB = "[    1.98] CR               calling WM4RB  (31/31 payload, 8/10 preamble)"
CR_TO_WS8EOC = "[    5.23] CR               calling WS8EOC  (31/31 payload, 8/10 preamble)"
ANSWER_K5DAT = "[    8.48] connect-response to K5DAT  (15/15 payload, 8/8 preamble)"
ANSWER_WS8EOC = "[   10.96] connect-response to WS8EOC  (15/15 payload, 8/8 preamble)"
LINK_SETUP = "[   13.48] link-setup       from KC9GHZ  (CRC ok)"

# The same channel through a receiver tuned 47 Hz off, which is the state the
# monitor reports the offset from — nothing else about the decode changes.
CR_OFF_FREQUENCY = ("[    1.98] CR               calling WM4RB, -47 Hz off frequency"
                    "  (31/31 payload, 8/10 preamble)")

# Real off-air lines, `working/gwsurvey/run-20260804-2347Z/K5DAT-13_7104.7k.vara.txt`:
# an answer the scanner could not put a callsign to, and the shapes around it.
ANSWER_UNNAMED = ("[   35.13] connect-response to a station not on the candidate "
                  "list  (3/15 payload, 7/8 preamble)")
VARA_SHAPES = "\n".join([
    "[   25.26] wideband         undecoded (DATA/high-SL)  (no CRC lock)",
    "[   30.31] DBPSK burst      tokens unresolved — within the 32-45 col token "
    "vocabulary  (0.43s, +4.4 dB, ~40col)",
    "[   31.34] unknown            (2.09s, +8.1 dB, ~49sym)",
    "[   41.07] connected-ack    (carries no callsign)  (preamble x56)",
])

# `python3 -m hfmodem.shrike.monitor <wav>` over `pactor1.connect_signal("WS8EOC")`.
P1_CONNECT = "   0.00  CONNECT  ###CONNECT: [Normal Call: WS8EOC]"

# `python3 -m hfmodem.besra.monitor <wav>` over frames rendered by besra's own
# `phy.modulator.render_frame`. The frame names, type bytes and session ids are
# the monitor's; the callsigns are chosen for the scenarios.
CONREQ_KW8MWX = "    0.24  0x32 ConReq500M       sess 0xFF  KW8MWX > N8XYZ"
CONREQ_TO_W8MW = "    2.99  0x32 ConReq500M       sess 0xFF  KC3OWM > W8MW"
ID_W8MW = "    5.73  0x30 IDFrame          sess 0xFF  W8MW  EN72"

# Real off-air ARDOP, the same monitor over
# `tests/besra/fixtures/offair_ke8lva_greeting.wav`: a Winlink gateway sending its
# own greeting ("RMS Trimode 1.4.2." in the hex), naming nobody.
KE8LVA_GREETING = ("    1.85  0x51 4PSK.500.100.O   sess 0x51  128 B  "
                   "524d53205472696d6f646520312e342e…")

# Real off-air, `working/gwsurvey/run-20260804-2347Z/KB8AY_3586.5k.kestrel500.txt`
# and `.besra.txt`.
BW500_SHAPE = "   42.49  BW500 data    CRC FAIL  self-consistency 0.793"
BESRA_SHAPE = "   90.43  0x55 16QAM.500.100.O  sess 0x00  !CRC/RS FAIL"


def _tier(call, **logs):
    """The tier `call`'s channel earns from these MONITOR LOGS and nothing else.

    Through the parsers, because that is where the survey's answer is decided:
    a line the parser drops never reaches `attribute` to be attributed at all.
    """
    text = {"shrike": "", "vara": "", "besra": "", "kestrel500": ""} | logs
    s_dec, s_cs, _ = G.parse_shrike(text["shrike"])
    decoded = {"shrike": s_dec,
               "vara": G.parse_vara(text["vara"])[0],
               "besra": G.parse_besra(text["besra"])[0],
               "kestrel500": G.parse_kestrel500(text["kestrel500"])[0]}
    heard, called, other = G.attribute(call, decoded)
    return G.verdict(heard, called, other, n_cs=len(s_cs), n_shape=0, n_busy=0)


# --------------------------------------------------------------------------- #
# The parse, which is in front of everything else.
def test_a_gateways_answer_survives_the_parse():
    """`connect-response` fills the monitor's sixteen-column kind field exactly,
    so one space follows it and a split on two-or-more dropped the line."""
    dec, shape = G.parse_vara(ANSWER_K5DAT)
    assert dec == [ANSWER_K5DAT], f"the answer was parsed as {dec or shape}"


def test_the_answer_a_survey_acted_on_grades_worth_calling():
    """Two perfect answers on K5DAT's own channel, which is what the runbook
    calls the strongest evidence there is. It graded SILENT."""
    log = "\n".join([ANSWER_K5DAT, ANSWER_K5DAT.replace("8.48", "9.99")])
    assert _tier("K5DAT-13", vara=log) == "worth calling"


def test_an_off_frequency_call_is_still_a_named_decode():
    """A receiver 47 Hz off does not make a 31/31 decode anonymous. Anchoring the
    callsign on the end of the info field alone demoted it to structure."""
    dec, _ = G.parse_vara(CR_OFF_FREQUENCY)
    assert dec == [CR_OFF_FREQUENCY]
    assert G.who("vara", CR_OFF_FREQUENCY) == ("", "WM4RB")


def test_an_answer_with_no_callsign_on_it_is_structure():
    """The other side of the same rule: the scanner reports what it could not
    name, and an answer it could not name identifies nobody."""
    dec, shape = G.parse_vara(ANSWER_UNNAMED)
    assert (dec, shape) == ([], [ANSWER_UNNAMED])


def test_everything_the_monitor_classifies_without_naming_is_structure():
    dec, shape = G.parse_vara(VARA_SHAPES)
    assert dec == [] and len(shape) == 4


def test_the_survey_reads_the_line_the_monitor_actually_writes(capsys):
    """The coupling this file exists to hold, made with the producer rather than
    a typed copy of it: `vara_monitor.emit` writes the line, `parse_vara` reads
    it, and a change to either format string breaks this instead of a survey."""
    vm = pytest.importorskip("vara_monitor", reason="tools/ is not beside the package")
    for kind, info, quality in [("connect-response", "to K5DAT", "15/15 payload"),
                                ("CR", "calling WM4RB", "31/31 payload"),
                                ("link-setup", "from KC9GHZ", "CRC ok"),
                                ("one-hot burst", "one peak per column", "98col")]:
        vm.emit(12.34, vm.Result(kind, info, quality), vm.Session())
        line = capsys.readouterr().out.splitlines()[0]
        m = G._VARA_LINE.match(line)
        assert m, f"the survey could not read the monitor's own {kind!r} line"
        assert m.group(1).strip() == kind, f"{kind!r} was read as {m.group(1)!r}"


def test_a_kind_with_a_space_in_it_is_read_whole():
    """`one-hot burst` and `DATA over` are single kinds, not a kind and an info
    field. Splitting on whitespace read `one-hot` and threw the rest away."""
    ln = "[   35.14] one-hot burst    one peak per column  (1.05s, +23.4 dB, 98col)"
    assert G._VARA_LINE.match(ln).group(1).strip() == "one-hot burst"


# --------------------------------------------------------------------------- #
# Direction: whose transmission the decode was.
def test_a_third_party_working_the_channel_is_not_the_gateway():
    """The scenario the by-callsign rule was written for, and still right."""
    assert _tier("WS8EOC", vara=CR_TO_WM4RB) == "channel active"


def test_a_station_calling_the_gateway_is_not_the_gateway_transmitting():
    """Somebody dialling WS8EOC says the channel is in use. It says nothing
    about whether WS8EOC is awake, which is the question being asked."""
    assert _tier("WS8EOC", vara=CR_TO_WS8EOC) == "channel active"


def test_a_gateway_answering_on_its_own_channel_is_worth_calling():
    """The connect-response is the RESPONDER's burst [spec 05 §5.3 step 2], and
    the responder is the station that was called — so this one names the
    transmitter."""
    assert _tier("WS8EOC", vara=ANSWER_WS8EOC) == "worth calling"


def test_the_listed_ssid_does_not_hide_the_gateways_own_answer():
    """`K5DAT-13` is what Winlink lists; `K5DAT` is all any monitor can report,
    because the survey hands them the base call."""
    assert _tier("K5DAT-13", vara=ANSWER_K5DAT) == "worth calling"


def test_a_longer_callsign_containing_the_gateways_is_a_different_station():
    assert _tier("W8MW", besra=CONREQ_KW8MWX) == "channel active"
    assert G.who("besra", CONREQ_KW8MWX) == ("KW8MWX", "N8XYZ")


def test_every_monitors_direction():
    assert G.who("vara", CR_TO_WM4RB) == ("", "WM4RB")
    assert G.who("vara", ANSWER_K5DAT) == ("K5DAT", "")
    assert G.who("vara", LINK_SETUP) == ("KC9GHZ", "")
    assert G.who("shrike", P1_CONNECT) == ("", "WS8EOC")
    assert G.who("besra", CONREQ_TO_W8MW) == ("KC3OWM", "W8MW")
    assert G.who("besra", ID_W8MW) == ("W8MW", "")
    assert G.who("kestrel500", BW500_SHAPE) == ("", "")


def test_a_pactor_connect_names_the_station_being_called():
    """A PACTOR-1 link-setup carries the address of the station it is calling —
    `ptc.on_rx_connect` acts on one 'naming us' — so it is not that station's
    own transmission."""
    assert _tier("WS8EOC", shrike=P1_CONNECT) == "channel active"


def test_the_gateway_identifying_itself_is_worth_calling():
    assert _tier("W8MW", besra=ID_W8MW) == "worth calling"


def test_an_ardop_gateway_transmitting_cannot_be_attributed():
    """The gap `who` states, held as a fact rather than left as a surprise.

    An ARDOP data frame carries a session id and no callsign, so KE8LVA's own
    greeting — a gateway demonstrably up and talking — grades CHANNEL ACTIVE.
    Closing it needs the session id the ConAck shares with the ConReq that
    provoked it, and no capture here holds that pair.
    """
    assert G.who("besra", KE8LVA_GREETING) == ("", "")
    assert _tier("KE8LVA", besra=KE8LVA_GREETING) == "channel active"


# --------------------------------------------------------------------------- #
def test_frames_that_failed_their_check_are_structure_not_decodes():
    assert G.parse_besra(BESRA_SHAPE) == ([], [BESRA_SHAPE])
    assert G.parse_kestrel500(BW500_SHAPE) == ([], [BW500_SHAPE])


def test_no_decode_falls_through_to_the_energy_tiers():
    assert G.verdict([], [], [], n_cs=0, n_shape=0, n_busy=0) == "silent"
    assert G.verdict([], [], [], n_cs=1, n_shape=0, n_busy=0) == "energy only"
    assert G.verdict([], [], [], n_cs=0, n_shape=0, n_busy=3) == "energy only"


def test_the_dial_sits_below_the_published_centre():
    """Winlink publishes the channel CENTRE. Getting this backwards is silent
    and puts the operator 3 kHz off, and the report's own prose had it inverted
    while the arithmetic was right."""
    ch = G.Channel("W9SSJ", 7101.5, 0.0, "EN63", 1.0)
    assert ch.dial_khz == pytest.approx(7100.0)


def test_the_report_says_which_sense_each_decode_had(tmp_path):
    """Three headings, never one list. A reader deciding where to point a rig has
    to be able to tell 'the gateway keyed this' from 'somebody called it'."""
    decoded = {"shrike": [], "besra": [], "kestrel500": [],
               "vara": [ANSWER_WS8EOC, CR_TO_WS8EOC, CR_TO_WM4RB]}
    heard, called, other = G.attribute("WS8EOC", decoded)
    out = tmp_path / "out.md"
    G.write_report(out, [{
        "call": "WS8EOC", "freq_khz": 7101.5, "band": 40, "modes": ["Pactor"],
        "miles": 132.0, "grid": "EN72QQ", "hours_since_status": 1.0,
        "verdict": G.verdict(heard, called, other, 0, 0, 3),
        "observed_utc": "2026-08-06 03:57:28Z", "duration_s": 180.0,
        "receiver": "kiwi", "wav": "WS8EOC_7101.5k.wav",
        "occupancy": {"windows": 3, "judged": 3, "busy": 3,
                      "peak_margin_db": 19.7, "median_dbfs": -35.8},
        "occupant": None, "decoded": decoded, "cs_only": [],
        "structure": {}, "decoded_heard": heard,
        "decoded_called": called, "decoded_elsewhere": other,
    }], {"when": "now", "receivers": "kiwi", "seconds": 180.0,
         "rundir": "run", "pending": []})
    text = out.read_text()
    assert "**Verdict: WORTH CALLING**" in text
    assert "KEYED BY WS8EOC" in text and ANSWER_WS8EOC.strip() in text
    assert "ADDRESSED TO WS8EOC" in text and CR_TO_WS8EOC.strip() in text
    assert "naming SOMEBODY ELSE" in text and CR_TO_WM4RB.strip() in text


def test_the_report_states_the_dial_convention_the_right_way_round(tmp_path):
    out = tmp_path / "out.md"
    G.write_report(out, [], {"when": "now", "receivers": "kiwi", "seconds": 90.0,
                             "rundir": "run", "pending": []})
    assert "dial tuned 1500 Hz below the published centre" in out.read_text()
