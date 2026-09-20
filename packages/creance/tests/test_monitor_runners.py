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
from creance.monitor.runners import shrike_runner as shrike


# -- shrike ------------------------------------------------------------------

def test_shrike_header_and_cs_and_connect_are_confirmed():
    assert shrike.normalize(1.0, "packet", "HEADER ...")["grade"] == CONFIRMED
    assert shrike.normalize(1.0, "cs", "ACK")["grade"] == CONFIRMED
    conn = shrike.normalize(1.0, "connect", "###CONNECT", station="W1AW")
    assert conn["grade"] == CONFIRMED
    assert conn["protocol"] == "PACTOR-1" and conn["station"] == "W1AW"


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
@pytest.mark.parametrize("name", ["kestrel", "shrike", "besra"])
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


def test_a_bare_tree_hears_two_of_the_three_modems(tmp_path):
    """What an installed distribution can actually listen to, written down.

    This is a standing report of a gap rather than a design: shrike and besra
    drive decode paths that live in the shipped package, and kestrel's segmenter
    does not ship, so a distribution has no VARA ear. It is asserted rather than
    left to be discovered because the alternative is the test above passing by
    skipping everything. When kestrel's monitor moves into the package this fails
    and is the notice that it did.
    """
    can = {s.name for s in default_specs(tmp_path) if s.available()}
    assert can == {"shrike", "besra"}


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


def _detections(fixture: str) -> list[dict]:
    """Run the real runner over one corpus recording, driven exactly as
    ``ModemMonitor`` drives it: 48 kHz s16le on stdin, detection JSON out."""
    from hfhost.audio import WavReplaySource

    pcm = b"".join(p for _, p in
                   WavReplaySource(FIXTURES / fixture, realtime=False).frames())
    done = subprocess.run([BESRA.interpreter, BESRA.runner], input=pcm,
                          cwd=BESRA.cwd, stdout=subprocess.PIPE, timeout=300,
                          check=True)
    return [json.loads(ln) for ln in done.stdout.splitlines() if ln.strip()]


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
    """The capture runner as the driver spawns it, over a PortAudio that is not
    one — the card is the one thing a test cannot supply."""
    (tmp_path / "sounddevice.py").write_text(_FAKE_SD)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    runner = str(Path(capture.__file__))
    return subprocess.Popen([sys.executable, runner, device], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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
