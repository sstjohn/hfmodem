# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The per-modem confidence policy, tested on canned modem events.

``normalize`` is the whole adapter contract creance depends on, and it is pure —
no numpy, no modem import — so it is tested directly here. The audio ``main``
loops that call the real decoders need the off-air rf-corpus recordings, which
CI is not assumed to have, so they run behind a skip.
"""

import json
import os
from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem.tests import evidence
from creance.monitor import default_specs, device_source
from creance.monitor.activity import CONFIRMED, TENTATIVE
from creance.monitor.runners import besra_runner as besra
from creance.monitor.runners import capture_runner as capture
from creance.monitor.runners import kestrel_runner as kestrel
from creance.monitor.runners import p4_runner as p4
from creance.monitor.runners import shrike_runner as shrike


# -- shrike ------------------------------------------------------------------

def test_shrike_header_and_cs_and_connect_are_confirmed():
    assert shrike.normalize(1.0, "packet", "HEADER ...")["grade"] == CONFIRMED
    assert shrike.normalize(1.0, "cs", "ACK")["grade"] == CONFIRMED
    conn = shrike.normalize(1.0, "connect", "###CONNECT", station="W1AW")
    assert conn["grade"] == CONFIRMED
    assert conn["protocol"] == "PACTOR-1" and conn["station"] == "W1AW"


def test_shrike_connect_naming_no_callsign_is_tentative():
    """A track's instance read ``KO`` off a VARA session: the connect frame has
    no CRC, so only a callsign makes it a decode."""
    assert shrike.normalize(1.9, "connect", "###CONNECT", station="KO")["grade"] == TENTATIVE
    assert shrike.normalize(1.9, "connect", "###CONNECT", station="E")["grade"] == TENTATIVE
    assert shrike.normalize(0.1, "connect", "###CONNECT",
                            station="KE5YTA")["grade"] == CONFIRMED


def test_shrike_bare_pilot_and_fsk_are_tentative():
    assert shrike.normalize(1.0, "detect", "pilot")["grade"] == TENTATIVE
    assert shrike.normalize(1.0, "fsk", "FSK present")["grade"] == TENTATIVE


def test_shrike_cs_protocol_follows_the_modem_not_the_kind():
    """A control signal alone does not make a link PACTOR-3.

    Both protocols have control signals and they are different waveforms:
    PACTOR-1's are four 12-bit codewords in FSK, PACTOR-3's six 20-bit codewords
    in DBPSK on tones 5 and 12. Mapping the event KIND to a protocol made a
    PACTOR-1-only exchange with a real gateway report as "PACTOR-3 active (link
    decoded)" — a confident claim about a protocol that was never on the air.
    shrike says which one it read; the adapter must not overrule it.
    """
    # What the decoder reported wins over the kind, both ways.
    p1 = shrike.normalize(1.0, "cs", "REQ", protocol="PACTOR-1")
    assert p1["protocol"] == "PACTOR-1"
    assert p1["grade"] == CONFIRMED           # still a certain decode
    assert shrike.normalize(1.0, "cs", "ACK",
                            protocol="PACTOR-3")["protocol"] == "PACTOR-3"
    # Fallback for a shrike predating Event.protocol: sniff the text, which was
    # the only place the truth was ever written down.
    assert shrike.normalize(1.0, "cs",
                            "REQ  (0 bit errors, PACTOR-1)")["protocol"] == "PACTOR-1"
    assert shrike.normalize(1.0, "cs", "ACK")["protocol"] == "PACTOR-3"


def test_shrike_protocol_split_p3_vs_p1():
    assert shrike.normalize(1.0, "packet", "")["protocol"] == "PACTOR-3"
    assert shrike.normalize(1.0, "fsk", "")["protocol"] == "PACTOR-1"


def test_shrike_uses_human_tag_labels():
    assert shrike.normalize(1.0, "packet", "")["kind"] == "HEADER"
    # Two tags deliberately disagree with their kind, because the kinds are a
    # published interface and the tags are what an operator reads: `p1reply` is a
    # 1400/1600 burst that nothing decoded and nothing attributed, and `detect`'s
    # only emitter is a PACTOR-2 carrier pair.
    assert shrike.normalize(1.0, "p1reply", "")["kind"] == "P1-BURST"
    assert shrike.normalize(1.0, "detect", "")["kind"] == "P2-PAIR"


def test_shrike_names_a_pactor3_codeword_by_its_number():
    """PACTOR-1's control signals keep the `CS` tag and name themselves in the
    detail; a PACTOR-3 codeword carries its number, which is what an operator
    hunting for PACTOR-3 reads first."""
    assert shrike.normalize(1.0, "cs", "REQ", protocol="PACTOR-3", cs=1)["kind"] == "CS2"
    assert shrike.normalize(1.0, "cs", "CS1/ack", protocol="PACTOR-1", cs=0)["kind"] == "CS"
    assert shrike.normalize(1.0, "cs", "ACK", protocol="PACTOR-3", cs=0,
                            grade=TENTATIVE)["grade"] == TENTATIVE


def test_shrike_detect_takes_its_protocol_from_the_decoder():
    """`detect` is a PACTOR-2 carrier pair and used to be graded PACTOR-3 by a
    lookup default — a field nobody set reading as a confident wrong answer."""
    assert shrike.normalize(1.0, "detect", "",
                            protocol="PACTOR-2")["protocol"] == "PACTOR-2"
    assert shrike.normalize(1.0, "detect", "")["protocol"] == "UNKNOWN"


# -- kestrel -----------------------------------------------------------------

def test_kestrel_matched_gateway_is_confirmed_and_named():
    got = kestrel.normalize(2.0, "CR", "calling W1AW", "12/13", gateway="W1AW")
    assert got["grade"] == CONFIRMED
    assert got["station"] == "W1AW" and got["role"] == "gateway"


def test_kestrel_resolved_caller_is_confirmed():
    got = kestrel.normalize(2.0, "link-setup", "from K1ABC", "CRC ok",
                            caller="K1ABC")
    assert got["grade"] == CONFIRMED
    assert got["role"] == "caller" and got["station"] == "K1ABC"


def test_kestrel_data_over_is_confirmed_link_activity():
    assert kestrel.normalize(2.0, "DATA over", "512 payload bytes",
                             "CRC ok")["grade"] == CONFIRMED


def test_kestrel_unattributed_handshake_is_tentative():
    got = kestrel.normalize(2.0, "CR", "calling (dest not in gateway list)",
                            "best 3/13")
    assert got["grade"] == TENTATIVE
    assert got["station"] == "" and got["protocol"] == "VARA"


def test_kestrel_unknown_burst_names_no_protocol():
    got = kestrel.normalize(2.0, "unknown", "", "0.4s")
    assert got["grade"] == TENTATIVE and got["protocol"] == "UNKNOWN"


def test_kestrel_one_hot_burst_names_no_protocol():
    """The shape a rec3 short frame has, and so does every other keyed narrowband
    carrier. Graded VARA by the fallback it put 30 VARA lines on five corpus
    recordings holding no VARA at all, the PACTOR-2 oracle among them."""
    got = kestrel.normalize(2.0, "one-hot burst", "one peak per column", "0.9s")
    assert got["grade"] == TENTATIVE and got["protocol"] == "UNKNOWN"


def test_kestrel_dbpsk_burst_names_no_protocol():
    """A differential-BPSK collapse on VARA's control sub-bands with no token
    pattern resolved. It fires 45 times across seven regression-corpus recordings
    that hold no VARA — 80 m band noise, the PACTOR-1 captures, the PACTOR-3
    oracle — against 14 on the three VARA fixtures, so the shape may not carry the
    protocol any more than `one-hot burst` may."""
    got = kestrel.normalize(2.0, "DBPSK burst", "tokens unresolved — longer than "
                            "the 32-45 col token vocabulary", "2.97s, ~278col")
    assert got["grade"] == TENTATIVE and got["protocol"] == "UNKNOWN"


def test_kestrel_control_token_is_tentative_vara():
    got = kestrel.normalize(2.0, "connected-ack-token", "", "BW2300 h1 q0.9")
    assert got["grade"] == TENTATIVE and got["protocol"] == "VARA"


# -- besra -------------------------------------------------------------------

def test_besra_conreq_is_confirmed_and_names_the_caller():
    got = besra.normalize(1.2, "ConReq2000M", True, "> K4PAR-2", "KC3OWM")
    assert got["grade"] == CONFIRMED and got["protocol"] == "ARDOP"
    assert got["station"] == "KC3OWM" and got["role"] == "caller"


def test_besra_data_and_id_frames_are_confirmed():
    assert besra.normalize(3.0, "4PSK.500.100.E", True, "128 B")["grade"] == CONFIRMED
    ident = besra.normalize(3.0, "IDFrame", True, "IO91", "M7TFF")
    assert ident["grade"] == CONFIRMED and ident["role"] == ""


def test_besra_failed_integrity_is_not_a_detection():
    """A frame whose RS/CRC did not validate proved nothing, so it is dropped
    rather than downgraded. ARDOP names a frame with ten 4FSK symbols, which
    VARA and PACTOR energy reaches often enough that besra needs a per-frame-class
    quality floor to hold it off; a failed body check behind such a header is the
    signature of exactly that, not of a weak ARDOP station."""
    assert besra.normalize(1.0, "ConReq500M", False, "> GB7RDG-15", "M7TFF") is None
    assert besra.normalize(1.0, "4FSK.2000.600.E", False, "512 B") is None
    assert besra.normalize(1.0, "ConAck500", False) is None


def test_besra_bare_control_frames_are_tentative():
    """DATAACK/DATANAK/ConAck and friends carry no callsign and no payload CRC,
    so the frame-type header is their whole proof. They are real signal, unproven
    identity — never a confirmed ARDOP link."""
    for name in ("DATAACK", "DATANAK", "IDLE", "DISC", "ConRejBusy", "PingAck"):
        assert besra.normalize(1.0, name, True)["grade"] == TENTATIVE
    assert besra.normalize(1.0, "ConAck500", True,
                           "leader 240 ms")["grade"] == TENTATIVE


# -- every runner, started for real ------------------------------------------
#
# ``normalize`` is pure and every test above it runs without importing a modem,
# which is why they all passed while one of the three ``main`` loops could not
# start at all: the failing import is inside ``main``, and nothing here had ever
# executed one. These two do. They need no corpus — a runner handed immediate EOF
# still performs every import it owns and flushes what it holds — so they run
# everywhere rather than behind the corpus skip the audio tests sit behind.

def _starts(spec) -> subprocess.CompletedProcess:
    """Run one runner to completion on an empty stream, from the tree its spec
    names. ``cwd`` is load-bearing: it is what a runner resolves its own inputs
    against, and it is the difference between this checkout and an installed
    distribution."""
    return subprocess.run([spec.interpreter, spec.runner, *spec.args],
                          cwd=spec.cwd, input=b"", capture_output=True,
                          timeout=300)


@pytest.mark.parametrize("bare", [False, True], ids=["source-tree", "bare-tree"])
@pytest.mark.parametrize("name", ["kestrel", "shrike", "besra", "p4"])
def test_a_runner_reported_available_starts(tmp_path, name, bare):
    """The contract the whole aggregator rests on: a monitor creance says it can
    run must reach the point of reading audio. Where that breaks, a zero from
    this modem is indistinguishable from a quiet band.

    Held against both trees a runner is ever launched in. ``cwd`` is what a
    runner resolves its own inputs against, so the two are different questions:
    the source checkout has the field tools beside it, and an empty directory is
    the smallest honest model of an installed distribution, which does not.
    """
    spec = next(s for s in default_specs(tmp_path if bare else None)
                if s.name == name)
    if not spec.available():
        pytest.skip(f"{name} reports itself unavailable here, which is its own "
                    "honest answer — what it costs is recorded below")
    done = _starts(spec)
    assert done.returncode == 0, done.stderr.decode()[-2000:]


def test_bare_tree_has_the_shipped_runners(tmp_path, monkeypatch):
    """Kestrel, shrike, besra and lanner ship their decode paths.

    P4 requires the corpus collector from the source tree, so a PYTHONPATH that
    carries tools/ is no longer a bare tree.
    """
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and not (Path(p) / "corpuslib").is_dir()))
    can = {s.name for s in default_specs(tmp_path) if s.available()}
    assert can == {"kestrel", "shrike", "besra", "lanner"}


# -- besra over real audio ---------------------------------------------------

BESRA = next(s for s in default_specs() if s.name == "besra")
#: The off-air corpus is the one input that is genuinely outside the
#: distribution — a sibling checkout by default, CREANCE_CORPUS_DIR anywhere.
#: `BESRA.cwd` is the tree under test, which in a worktree holds the code and
#: none of the recordings — six tests skipped there for that reason alone.
CORPUS = Path(os.environ.get("CREANCE_CORPUS_DIR", evidence.CORPUS))
FIXTURES = CORPUS / "regress" / "fixtures"

live = pytest.mark.skipif(
    not (BESRA.available() and Path(BESRA.interpreter).exists()
         and FIXTURES.is_dir()),
    reason="besra's interpreter or the rf-corpus fixtures are not here")


def _pcm(path: Path) -> bytes:
    from hfhost.audio import WavReplaySource

    return b"".join(p for _, p in WavReplaySource(path, realtime=False).frames())


def _run(spec, pcm: bytes) -> list[dict]:
    """Run the real runner over audio, driven exactly as ``ModemMonitor`` drives
    it: 48 kHz s16le on stdin, detection JSON out."""
    done = subprocess.run([spec.interpreter, spec.runner, *spec.args], input=pcm,
                          cwd=spec.cwd, stdout=subprocess.PIPE, timeout=900,
                          check=True)
    return [json.loads(ln) for ln in done.stdout.splitlines() if ln.strip()]


def _detections(fixture: str, spec=BESRA) -> list[dict]:
    return _run(spec, _pcm(FIXTURES / fixture))


def _retuned(pcm: bytes, hz: float, pad_s: float = 0.5) -> bytes:
    """The recording as a sender ``hz`` off tune would have put it on the air:
    the whole passband slid, single-sideband, with quiet either side."""
    import numpy as np
    from scipy.signal import hilbert

    x = np.frombuffer(pcm, "<i2").astype(float)
    pad = np.zeros(int(pad_s * 48000))
    x = np.concatenate([pad, x, pad])
    x = (hilbert(x) * np.exp(2j * np.pi * hz * np.arange(x.size) / 48000)).real
    return np.clip(x, -32767, 32767).astype("<i2").tobytes()


@live
def test_besra_runner_decodes_the_ardop_oracle():
    """The ardopcf reference ConReq500M, whose content is known by construction.
    Both callsigns must survive the 48 kHz stream and the decimation back to the
    12 kHz besra decodes at."""
    got = _detections("oracle_ardop_conreq.wav")
    assert [d["kind"] for d in got] == ["ConReq500M"]
    assert got[0]["grade"] == CONFIRMED and got[0]["protocol"] == "ARDOP"
    assert got[0]["station"] == "M7TFF" and "GB7RDG-15" in got[0]["detail"]


@live
@pytest.mark.parametrize("fixture", ["pos_vara_session.wav",
                                     "oracle_pactor3_dl6maa.wav",
                                     "pos_p1cs_ws8eoc.wav",
                                     "neg_ft8_local.wav",
                                     "neg_noise_chatham.wav"])
def test_besra_runner_is_silent_on_non_ardop_audio(fixture):
    """The failure that matters. ARDOP's frame-type header is ten 4FSK symbols
    and VARA/PACTOR energy has minted phantom acks off it before, so a runner
    that cries ARDOP over another modem's traffic is worse than no runner."""
    assert _detections(fixture) == []


# -- p4 ----------------------------------------------------------------------

def test_p4_only_a_crc_valid_chirp_or_a_bank_word_is_confirmed():
    assert p4.normalize(1.0, "P4-CHIRP-CRC", "", 8.0)["grade"] == CONFIRMED
    assert p4.normalize(1.0, "P4-WORD", "", grade=CONFIRMED)["grade"] == CONFIRMED
    assert p4.normalize(1.0, "P4-ROBUST-CS", "")["grade"] == TENTATIVE
    assert p4.normalize(1.0, "P4-CHIRP-HEADER", "", 8.0)["grade"] == TENTATIVE
    spread = p4.normalize(1.0, "P4-SPREAD", "SF-16 spread 0.356")
    assert spread["grade"] == TENTATIVE and spread["offset_hz"] is None
    assert spread["protocol"] == "PACTOR-4"


# -- offsets over real audio -------------------------------------------------
#
# Each runner's offset is its own decoder's measurement, and each is held here
# against a recording that says what the answer is: a station the corpus already
# names as off tune, a fixture shifted onto the grid by its own construction, or
# the fixture slid by a known amount.

KESTREL = next(s for s in default_specs() if s.name == "kestrel")
SHRIKE = next(s for s in default_specs() if s.name == "shrike")
P4 = next(s for s in default_specs() if s.name == "p4")
#: 2026-07-26: our eight connect requests to KB9MMT, its answers, and a third
#: station calling somebody at BW500 two tone bins low at 100.81 s.
KB9MMT_0726 = CORPUS / "offair" / "kb9mmt_reply_20260726" / "rig_rx.wav"


def _live(spec, *paths: Path):
    missing = [p for p in paths if not p.exists()]
    return pytest.mark.skipif(
        not (spec.available() and FIXTURES.is_dir()) or bool(missing),
        reason=f"{spec.name} or its recordings are not here: {missing}")


@_live(KESTREL, KB9MMT_0726)
@pytest.mark.skipif(not (Path(KESTREL.cwd) / "winlink-vara-gateways.csv").exists(),
                    reason="no cached Winlink gateway list to attribute against")
def test_kestrel_reads_the_scanners_shift_onto_the_offset():
    """The handshake scanner finds a preamble at each of seven tone shifts, and the
    shift is the sender's tuning error to the 23.4 Hz tone grid."""
    crs = [d for d in _run(KESTREL, _pcm(KB9MMT_0726)) if d["kind"] == "CR"]
    ours = [d for d in crs if d["station"] == "KB9MMT"]
    third = [d for d in crs if abs(d["t"] - 100.81) < 0.05]
    assert len(ours) == 8 and all(d["offset_hz"] == 0.0 for d in ours)
    assert len(third) == 1 and third[0]["offset_hz"] == -46.9, third


@_live(BESRA)
@pytest.mark.parametrize("hz", [0.0, 37.0, -120.0])
def test_besra_reads_the_leader_offset(hz):
    """The ardopcf ConReq500M is on tune by construction; slid, it reads the slide
    to the leader estimate's own tolerance."""
    got = _run(BESRA, _retuned(_pcm(FIXTURES / "oracle_ardop_conreq.wav"), hz))
    assert [d["kind"] for d in got] == ["ConReq500M"]
    assert abs(got[0]["offset_hz"] - hz) <= 3.0, got[0]


def _pair_offset(x) -> float:
    """The PACTOR-1 pair's offset measured here, independently of the runner:
    each tone's peak within 100 Hz of 1400 and 1600 Hz, off a Hann-windowed
    spectrum zero-padded to 0.2 Hz bins, and the mean of the two deviations."""
    import numpy as np

    n = 1 << max(18, int(np.ceil(np.log2(len(x)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    f = np.fft.rfftfreq(n, 1 / 48000)
    near = [(f > hz - 100) & (f < hz + 100) for hz in (1400.0, 1600.0)]
    return float(np.mean([f[m][np.argmax(spec[m])] - hz
                          for m, hz in zip(near, (1400.0, 1600.0))]))


@_live(SHRIKE)
def test_shrike_fits_the_pactor1_pair():
    """Each reading agrees with the pair measured here on the same audio, to the
    runner's 5 Hz fit grid: over the whole KE5YTA link, and over WS8EOC's CS1
    burst alone. Every PACTOR-1 line of the link carries one figure, because it
    is a property of the station and not of the frame."""
    import numpy as np

    pcm = _pcm(FIXTURES / "pos_pactor1_local.wav")
    local = [d for d in _run(SHRIKE, pcm) if d["protocol"] == "PACTOR-1"]
    assert any(d["kind"] == "CONNECT" and d["station"] == "KE5YTA" for d in local)
    [offset] = {d["offset_hz"] for d in local}
    assert abs(offset - _pair_offset(np.frombuffer(pcm, "<i2").astype(float))) <= 5.0

    pcm = _pcm(FIXTURES / "pos_p1cs_ws8eoc.wav")
    (cs1,) = [d for d in _run(SHRIKE, pcm) if d["kind"] == "CS"]
    burst = np.frombuffer(pcm, "<i2").astype(float)[
        int(cs1["t"] * 48000):int((cs1["t"] + shrike.FSK_SPAN_S["cs"]) * 48000)]
    assert cs1["protocol"] == "PACTOR-1"
    assert abs(cs1["offset_hz"] - _pair_offset(burst)) <= 5.0


@_live(SHRIKE)
def test_shrike_reads_the_pactor2_marker_pair_and_leaves_pactor3_unplaced():
    """HB9AK was shifted onto 1400/1600 Hz when the fixture was cut, so the truth is
    zero. The marker's bin is a 25 Hz grid and a marker can arm one bin over, so
    every reading is within a bin and most are on it."""
    p2 = [d["offset_hz"] for d in _detections("pos_p2_sl3_hb9ak_055156.wav", SHRIKE)
          if d["protocol"] == "PACTOR-2"]
    assert len(p2) >= 9 and all(abs(o) <= 25.0 for o in p2), p2
    assert p2.count(0.0) > len(p2) / 2, p2
    p3 = [d for d in _detections("pos_pactor3_sl6.wav", SHRIKE)
          if d["protocol"] == "PACTOR-3"]
    assert p3 and all(d["offset_hz"] is None for d in p3
                      if "acquisition RX" not in d["detail"])


#: 2026-09-23, W9SSJ and WS8EOC in PACTOR-3 at SL1 on 80 m, as a KiwiSDR near
#: WS8EOC heard both of them: 12 kHz USB audio, the link 16 Hz low.
WS8EOC_KIWI_0923 = Path(SHRIKE.cwd) / "captures" / "onair-0923-1918" / "witness.wav"


@_live(SHRIKE, WS8EOC_KIWI_0923)
def test_shrike_names_pactor3_control_signals_a_kiwi_heard():
    """rxfront's pilot-pointed read named one control signal and one header
    here. The acquisition names both stations' codewords on their cycle and
    says the two answer instants are one link."""
    got = [d for d in _run(SHRIKE, _pcm(WS8EOC_KIWI_0923)) if d["protocol"] == "PACTOR-3"]
    cs = [d for d in got if d["kind"].startswith("CS") and d["grade"] == CONFIRMED]
    assert len(cs) >= 20 and {"CS1", "CS3", "CS4"} <= {d["kind"] for d in cs}
    assert all(-18.0 <= d["offset_hz"] <= -14.0 for d in cs if d["offset_hz"] is not None)
    assert sum(d["kind"] == "HEADER" for d in got) >= 8
    assert [d["kind"] for d in got].count("SESSION") >= 1


@_live(SHRIKE, WS8EOC_KIWI_0923)
def test_shrike_reads_pactor3_headers_off_the_track_grid():
    """A track's channel is centred on a 100 Hz grid, so its link can sit 50 Hz
    off where the header search gives out at 30. Slid 56 Hz up, the link reads
    +40 Hz in the instance; the codewords place it and the headers follow."""
    track = replace(SHRIKE, args=("--centre", "1500", "--bw", "2800", "--shape", "wide"))
    got = [d for d in _run(track, _retuned(_pcm(WS8EOC_KIWI_0923), 56.0))
           if d.get("protocol") == "PACTOR-3"]
    assert sum(d["kind"] == "HEADER" for d in got) >= 8
    assert all(35.0 <= d["offset_hz"] <= 45.0 for d in got
               if d["kind"].startswith("CS") and d["offset_hz"] is not None)


@_live(SHRIKE)
@pytest.mark.parametrize("fixture", ["pos_vara_dbpsk.wav", "pos_vara_session.wav"])
def test_shrike_says_no_pactor3_codeword_over_vara(fixture):
    """VARA's PSK spells control codewords, nine across these two, at +31 and
    +52 Hz. Their comb is flat and no three keep a cycle."""
    assert not [d for d in _detections(fixture, SHRIKE) if d["protocol"] == "PACTOR-3"]


def _dragon_chirp(index: int, hz: float) -> bytes:
    """One of the four recorded Dragon chirps, as `corpuslib/test_watch` prepares
    them: lifted by the 242.4 Hz its README says the recording sits low, then by
    ``hz`` more, and set 6 s into a stream so it crosses a block boundary."""
    import numpy as np
    from hfmodem import tests
    from scipy.signal import hilbert, resample_poly

    with np.load(Path(tests.__file__).parent / "shrike/fixtures/sigidwiki/p4-chirps/audio.npz") as z:
        fs, x = int(z["fs"]), z["audio"][index].astype(float)
    x = (hilbert(x) * np.exp(2j * np.pi * (242.4 + hz) * np.arange(x.size) / fs)).real
    x = np.r_[np.zeros(6 * 48000), resample_poly(x, 48000, fs), np.zeros(4800)]
    return (x / np.abs(x).max() * 0.5 * 32767).astype("<i2").tobytes()


@_live(P4)
@pytest.mark.parametrize("index", range(4))
def test_p4_reads_the_chirp_offset(index):
    """Lifted by the 242.4 Hz the fixture's note says the recording sits low,
    the chirp's truth is 0 Hz, and the dual-lane header search reads it within
    what its 150 ms header resolves, 1 / 0.15 s = 6.7 Hz, plus one step of its
    4 Hz grid. Slid by a known amount, the reading follows the slide to the
    grid."""
    base = _run(P4, _dragon_chirp(index, 0.0))
    assert [(d["kind"], d["grade"]) for d in base] == [("P4-CHIRP-CRC", CONFIRMED)]
    assert abs(base[0]["t"] - 6.0) < 0.01
    assert abs(base[0]["offset_hz"]) <= 1 / 0.15 + 4.0
    for hz in (20.0, -33.0):
        (got,) = _run(P4, _dragon_chirp(index, hz))
        assert abs(got["offset_hz"] - (base[0]["offset_hz"] + hz)) <= 4.0, (hz, got)


PACTORIV_AUDIO1 = CORPUS / "sigidwiki" / "PACTORIVaudio1.mp3"


@_live(P4, PACTORIV_AUDIO1)
def test_p4_names_the_callers_word_and_the_answer():
    """The caller keys its robust short packet three times and the bank names
    each header CALLER20. The answerer's three bursts are the 27-symbol answer
    words the bank has held since cc8d67740 (X27) and 68c16261d (V27, U27)."""
    got = _run(P4, _pcm(PACTORIV_AUDIO1))
    words = [(round(d["t"]), d["detail"].split()[0]) for d in got
             if d["kind"] == "P4-WORD" and d["grade"] == CONFIRMED]
    assert words == [(5, "p4-word:CALLER20"), (6, "p4-word:CALLER20"),
                     (7, "p4-word:CALLER20"), (9, "p4-word:V27"),
                     (10, "p4-word:X27"), (12, "p4-word:U27")]
    assert not any(d["kind"] == "P4-SPREAD" for d in got)


@_live(P4)
@pytest.mark.parametrize("fixture", ["pos_vara_session.wav", "pos_pactor1_local.wav",
                                     "pos_pactor3_sl6.wav", "neg_noise_chatham.wav"])
def test_p4_is_silent_on_other_modes(fixture):
    assert _detections(fixture, P4) == []


# -- capture -----------------------------------------------------------------

_FAKE_SD = '''
import threading, time

BLOCK = 1200

def query_devices(index=None):
    devs = [{"name": "MacBook Pro Microphone", "max_input_channels": 1,
             "max_output_channels": 0},
            {"name": "USB Audio Device", "max_input_channels": 2,
             "max_output_channels": 2}]
    return devs if index is None else devs[index]

class RawInputStream:
    def __init__(self, device=None, channels=1, samplerate=None, dtype=None,
                 callback=None):
        self.device, self.callback = device, callback
        self.stop = threading.Event()

    def _run(self):
        n = 0
        while not self.stop.is_set():
            self.callback(bytes([n % 256]) * (2 * BLOCK), BLOCK, None, None)
            n += 1
            time.sleep(0.01)

    def __enter__(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
'''


def _capture(tmp_path, device):
    """Supply the inherited claim and separate status pipe required by U7."""
    import fcntl

    (tmp_path / "sounddevice.py").write_text(_FAKE_SD)
    (tmp_path / "sitecustomize.py").write_text(
        "from hfmodem.core import claim\n"
        "claim.process_started = lambda pid: 'Sat_Oct_17_12:00:00_2026'\n")
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'rig.claim.cleanup').touch()
    fd = os.open(state / 'rig.claim', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.pwrite(fd, f'{os.getpid()} Sat_Oct_17_12:00:00_2026 shell test 123 fixture\n'.encode(), 0)
    os.lseek(fd, 123, os.SEEK_SET)
    reader, writer = os.pipe()
    status = fcntl.fcntl(writer, fcntl.F_DUPFD_CLOEXEC, 10)
    os.close(writer)
    package = Path(capture.__file__).resolve().parents[4] / 'hfmodem'
    env = dict(os.environ, PYTHONPATH=f'{tmp_path}:{package}',
               HFMODEM_COEXIST_HARNESS='u7', HFMODEM_STATE_DIR=str(state),
               HFMODEM_CAPTURE_ACK_FD=str(status))
    runner = str(Path(capture.__file__))
    wrapper = ('import os, runpy, sys; os.dup2(int(sys.argv[1]), 9); '
               'sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0], run_name="__main__")')
    try:
        proc = subprocess.Popen([sys.executable, '-c', wrapper, str(fd), runner, device], env=env,
                                pass_fds=(fd, status), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        os.close(fd)
        os.close(status)
    with os.fdopen(reader, 'rb') as acknowledgement:
        line = acknowledgement.readline()
    if device == 'USB Audio':
        assert line == b'CLAIM ADMITTED\n', proc.stderr.read()
    return proc


def test_the_capture_runner_writes_the_card_straight_through(tmp_path):
    proc = _capture(tmp_path, "USB Audio")
    got = proc.stdout.read(4800)
    proc.terminate()
    proc.wait(timeout=5)

    assert len(got) == 4800
    assert set(got) < set(range(4))          # the fake's first blocks, in order


def test_the_capture_runner_refuses_a_device_it_cannot_name(tmp_path):
    """`find_device` is what every transmit path resolves with, and it refuses an
    ambiguous or absent name rather than picking. A sense reading the wrong card
    is a channel verdict about a channel nobody is on."""
    proc = _capture(tmp_path, "Nothing Here")
    assert proc.wait(timeout=15) != 0
    assert b"no in audio device" in proc.stderr.read()


def test_the_device_source_spawns_that_runner_under_the_runners_interpreter():
    src = device_source("USB Audio Device")
    assert src.cmd[0] == sys.executable
    assert Path(src.cmd[1]).name == "capture_runner.py"
    assert src.cmd[2] == "USB Audio Device"
