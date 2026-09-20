# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The on-air mail argv, end to end against stand-ins — the path the radio runs.

``onair.sh vara-mail`` drives ``tools/kestrel_connect.py`` with a rigctld, a real
audio device, ``--arm`` and the mail flags. Every mail test before this one ran
the tool's ``--dry-run``, and ``--dry-run`` skips the entire rig stage by design
— so the first time the identify/arm/tune wiring ever executed was at the
transmitter, where a daemon whose radio had stopped answering CAT took the
attempt down as an IndexError traceback (2026-08-03: one rig slot spent,
nothing sent, nothing learned). These tests run the launcher's own argv shape
through the tool as a subprocess, against
:class:`~hfmodem.tests.kestrel.fake_rigctld.FakeRigctld` and the fake audio
device, and hold two lines: the healthy path reaches the air and ends honestly,
and every way the run can fail before the air is a refusal that names its stage
— never a traceback, and never after something keyed.

The gateway list here reproduces the station actually called that day: K5DAT-13
listens at VARA 500 on 40 m and plain VARA on 20 m. Reachability is decided by
the channel called, not the station — the 20 m listing must not vouch for a
40 m call kestrel's BW2300 connect-request can never raise.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import corpora
from .fake_rigctld import serving
from hfmodem.core.occupied import FILTER_HZ
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_TOOLS = corpora.TOOLS
_FAKE_AUDIO = Path(__file__).resolve().parent / "fakeaudio"

pytestmark = pytest.mark.skipif(not (_TOOLS / "kestrel_connect.py").exists(),
                                reason="tools/ not present (installed-wheel run)")


@pytest.fixture
def rigctld():
    yield from serving()


_GATEWAYS = ("Callsign,Frequency,Mode\n"
             'K5DAT-13,"7,104.700 KHz",VARA 500\n'
             'K5DAT-13,"14,108.000 KHz",VARA\n')


#: The keying line these runs name. ``--arm`` refuses to start without one, because
#: the last-resort unkey has nothing to pull down otherwise; the gate stats the path
#: and never opens it, so a character device nothing is holding is what a radioless
#: rehearsal wants. The launcher keys through this line as well (``--line-ptt``) and
#: these runs key through the daemon: /dev/null takes the ioctl and moves no pin, and
#: what is rehearsed here is the mail path rather than the keying path.
PTT_LINE = "/dev/null"


def _mail_argv(tmp_path: Path, port: int, freq: str = "14106.5") -> list[str]:
    body = tmp_path / "body.txt"
    body.write_text("mail for the far end\n")
    gw = tmp_path / "gateways.csv"
    gw.write_text(_GATEWAYS)
    return [sys.executable, str(_TOOLS / "kestrel_connect.py"),
            "--gateway", "K5DAT-13", "--mycall", "W9SSJ", "--bw", "2300",
            "--tx-device", "fake", "--rx-device", "fake",
            "--rigctld", f"127.0.0.1:{port}", "--freq", freq,
            "--arm", "--expect-model", "FT-891", "--ptt-device", PTT_LINE,
            "--gateways", str(gw),
            "--mail-send", str(body), "--mail-to", "SMTP:test@example.net",
            "--mail-fetch", "--mail-out", str(tmp_path / "mail"),
            "--listen-first", "0", "--timeout", "5", "--no-record"]


def _run(argv: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(_FAKE_AUDIO), str(corpora.PKG_ROOT)]),
           "KESTREL_FAKE_AUDIO":
               json.dumps({"log": str(tmp_path / "audio-events.jsonl")})}
    return subprocess.run(argv, capture_output=True, text=True, timeout=180,
                          env=env, cwd=str(tmp_path))


#: Below this, a play carried nothing: `fakeaudio` scores a block of zeros at
#: -240 dBFS, and the quietest thing the tool actually modulates is far above it.
SILENT_DBFS = -100.0


def _plays(tmp_path: Path) -> list[dict]:
    """Every write the fake device took, transmissions and the warm-up alike."""
    log = tmp_path / "audio-events.jsonl"
    rows = ([json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
            if log.exists() else [])
    return [e for e in rows if e["event"] == "play"]


def _emitted(tmp_path: Path) -> list[np.ndarray]:
    """Every transmission the fake device carried, as the samples it carried.

    ``T 1`` counts key-ups, not waveforms: with ``hs.originate`` replaced by a bare
    key up and down, the healthy-path test below still passed. The device parks each
    transmission beside its event log, which is the only thing here that knows what
    went out.

    An armed session also writes silence to the transmit device before it keys
    anything, to pay that device's first stream open outside a keyed window
    [`AudioVaraIO._warm_transmit_path`]. It reaches the card and it is not a
    transmission -- nothing modulated, nothing keyed, nothing on the band -- so it
    is not what this returns. That it really is silent is
    :func:`test_the_warm_up_the_armed_session_writes_carries_nothing`.
    """
    return [np.fromfile(e["audio"], dtype=np.float32).astype(float)
            for e in _plays(tmp_path) if e["dbfs"] > SILENT_DBFS]


def test_the_launcher_and_these_tests_drive_the_same_argv():
    """The flags rehearsed here are the flags ``onair.sh vara-mail`` sends; a
    launcher that grows or renames one must drag this file with it, or the
    rehearsal quietly stops covering the command the operator runs."""
    launcher = _TOOLS / "onair.sh"
    assert launcher.exists(), "onair.sh is the command this file claims to rehearse"
    # Anchored at a line start, because a case label is one. This used to split on
    # the bare string and a prose mention of "vara-mail)" inside another verb's
    # comment sent it off to read the `ardop)` block, where it duly reported that
    # vara-mail had stopped passing kestrel_connect.py.
    body = launcher.read_text()
    assert body.count("\nvara-mail)") == 1, "the vara-mail verb is not where it was"
    # Ended at the verb's own `;;`, which is the one at the case's indentation.
    # The optional bandwidth reads its argument with an inner `case`, and a bare
    # `;;` closes each of those arms: split on the string and the block stopped
    # four lines in, before the command it exists to check.
    block = body.split("\nvara-mail)")[1].split("\n  ;;")[0]
    for flag in ("kestrel_connect.py", "--gateway", "--mycall", "--bw",
                 "--tx-device", "--rx-device", "--rigctld", "--freq", "--arm",
                 "--expect-model", "--ptt-device", "--gateways"):
        assert flag in block, f"onair.sh vara-mail no longer passes {flag}"
    # The mail flags travel by way of the gate, which sorts --force out of them
    # and hands the rest back as KEY_ARGS. Where they start is the argument after
    # the bandwidth positional, which is 4 or 5 depending on whether one was given.
    assert '"${@:$rest}"' in block, "the mail flags no longer reach the gate"
    assert '"${KEY_ARGS[@]}"' in block, "the mail flags no longer reach the tool"


def test_the_onair_mail_argv_runs_every_stage_to_an_honest_no_connect(
        rigctld, tmp_path):
    """The healthy path, radiolessly: identify, arm, tune with the QSY read
    back, key the connect-request through the daemon, and end with PTT down,
    an honest NOT connected, and a mail summary that names its stage.

    What went out under the key is demodulated back, because a run that keys and
    transmits nothing looks identical from the daemon's side.
    """
    srv = rigctld()
    r = _run(_mail_argv(tmp_path, srv.port), tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 1, r.stdout + r.stderr
    out = r.stdout
    assert "model='FT-891'" in out
    assert "arming" in out
    assert "tuned 14106.5 kHz dial" in out
    assert srv.freq == 14106500 and srv.mode == "PKTUSB"
    assert "originating: W9SSJ -> K5DAT-13" in out
    assert srv.seen("T 1") >= 1, "the connect-request never keyed"

    bursts = _emitted(tmp_path)
    assert bursts, "the transmitter was keyed and no audio went out under it"
    n_sym = len(VF.CR.preamble) + VF.CR.n_payload
    for x in bursts:
        at = MK.lock_preamble(x, VF.CR)
        assert at is not None, (
            f"{len(x) / 48000:.3f} s went on the air carrying no connect-request "
            f"preamble")
        call, m, nn = VF.best_match(MK.demod_tones(x[at:], n_sym),
                                    ["K5DAT-13"], VF.CR)
        assert (call, m) == ("K5DAT-13", nn), (
            f"the burst reads {m}/{nn} payload tones for {call} — it is not a "
            f"connect-request to the station the argv named")

    assert srv.ptt == 0, "PTT was left up"
    assert "NOT connected" in out
    assert "mail: nothing moved" in out


def test_the_warm_up_the_armed_session_writes_carries_nothing(rigctld, tmp_path):
    """What :func:`_emitted` leaves out, checked rather than assumed.

    An armed session writes one block of silence to the transmit device before it
    keys anything, so that device's first stream open is paid outside a keyed
    window [`AudioVaraIO._warm_transmit_path`]. It is the one thing that reaches
    the card and is not a transmission, and the whole of what makes that true is
    that it carries nothing and lands before the first key-up. Both are read off
    the device's own log and the daemon's own clock, so a warm-up that grew
    modulation or drifted inside the keyed region fails here rather than going
    quietly out over the air.
    """
    srv = rigctld()
    _run(_mail_argv(tmp_path, srv.port), tmp_path)

    silent = [e for e in _plays(tmp_path) if e["dbfs"] <= SILENT_DBFS]
    assert len(silent) == 1, (
        f"{len(silent)} silent write(s) reached the transmit device — the warm-up "
        f"is one block, and anything else silent on the card is a burst that lost "
        f"its modulation on the way out")
    keyed = srv.stamped("T 1")
    assert keyed, "the run never keyed, so there is no first key-up to be ahead of"
    assert silent[0]["t"] < keyed[0], (
        f"the warm-up was written {silent[0]['t'] - keyed[0]:.3f} s after the "
        f"transmitter first came up — being outside the keyed window is the whole "
        f"of why it is allowed to reach the card at all")


def test_a_radio_that_stops_answering_cat_is_a_named_refusal(rigctld, tmp_path):
    """The 2026-08-03 failure: the daemon completes every connection, takes
    every command, and answers none of them. That must read as the rig stage
    refusing by name — not as an IndexError out of the frequency parse."""
    srv = rigctld(answer=False)
    r = _run(_mail_argv(tmp_path, srv.port), tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "RIG NOT ANSWERING" in r.stdout
    assert "not answering CAT" in r.stdout
    assert srv.seen("T 1") == 0 and srv.ptt == 0


def test_no_daemon_at_all_is_a_named_refusal(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    r = _run(_mail_argv(tmp_path, port), tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "CANNOT REACH rigctld" in r.stdout


def test_a_qsy_the_rig_did_not_take_is_never_keyed_over(rigctld, tmp_path):
    """``F`` refused in band leaves the rig on the old frequency; the readback
    must catch it, because a call on the wrong dial is silent in both
    directions and burns the slot just as thoroughly as a crash."""
    srv = rigctld(refuse=("F",))
    r = _run(_mail_argv(tmp_path, srv.port), tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "TUNE DID NOT TAKE" in r.stdout
    assert srv.seen("T 1") == 0


def test_a_filter_too_narrow_for_the_emission_is_never_keyed_through(
        rigctld, tmp_path):
    """The half of the readback nothing did. A rig whose widest filter is 500 Hz
    answers a 3000 Hz request with 500 and reports PKTUSB on the right dial, so
    every check this tool made passed while BW2300 — 610-2405 Hz, 1810 Hz about a
    1500 Hz centre — went out through a filter passing 1250-1750. The mode string is not
    evidence about the width, and the width is what carries the signal.
    """
    srv = rigctld(passbands=(500,))
    r = _run(_mail_argv(tmp_path, srv.port), tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "TUNE DID NOT TAKE" in r.stdout
    assert "filter is 500 Hz" in r.stdout and "1810 Hz" in r.stdout, r.stdout
    assert srv.seen("T 1") == 0


def test_the_tuned_line_states_the_filter_the_rig_selected(rigctld, tmp_path):
    """Not the one we asked for. A rig picks off a ladder — the FT-891's widest
    PKTUSB filter is 2.4 kHz — so 2400 is the correct answer to a 3000 Hz request
    and wide enough for BW2300's 1640. The line used to read "2700 Hz", a number
    nothing had asked for and nothing had confirmed."""
    srv = rigctld()
    r = _run(_mail_argv(tmp_path, srv.port), tmp_path)
    assert f"M PKTUSB {FILTER_HZ}" in srv.commands, srv.commands
    assert "tuned 14106.5 kHz dial, PKTUSB / 2400 Hz" in r.stdout, r.stdout


def test_a_mail_file_that_cannot_load_is_refused_before_the_rig_is_touched(
        rigctld, tmp_path):
    srv = rigctld()
    argv = _mail_argv(tmp_path, srv.port)
    argv[argv.index("--mail-send") + 1] = str(tmp_path / "absent.txt")
    r = _run(argv, tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "MAIL REFUSED before anything keys" in r.stdout
    assert srv.commands == [], "the rig was touched for a run that could never send"


def test_a_body_file_with_nowhere_to_go_is_refused_the_same_way(rigctld, tmp_path):
    srv = rigctld()
    argv = _mail_argv(tmp_path, srv.port)
    i = argv.index("--mail-to")
    del argv[i:i + 2]
    r = _run(argv, tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "MAIL REFUSED before anything keys" in r.stdout
    assert srv.commands == []


def test_arming_without_a_keying_line_is_refused_before_the_rig_is_touched(
        rigctld, tmp_path):
    """This exact argv, armed, is what ran on 2026-08-10 with no line named behind
    it: rigctld took the unkey and never answered, the last-resort RTS drop had
    nothing to pull down, and the transmitter stayed up with no modulation until the
    operator was told to unkey by hand. Every armed run in this file was built the
    same way and not one of them noticed, which is the hole this closes from the
    rehearsal's side.

    The refusal is at argument parsing, so it costs the fake daemon nothing — an
    armed run that cannot unkey must not get as far as identifying a rig.
    """
    srv = rigctld()
    argv = _mail_argv(tmp_path, srv.port)
    i = argv.index("--ptt-device")
    del argv[i:i + 2]
    r = _run(argv, tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "--arm requires --ptt-device" in r.stdout, r.stdout
    assert srv.commands == [], f"the rig was touched anyway: {srv.commands}"
    assert srv.seen("T 1") == 0 and srv.ptt == 0


def test_a_keying_line_with_no_device_behind_it_is_refused_the_same_way(
        rigctld, tmp_path):
    """Named is not the same as there. A path that exists and is not a character
    device is the case an existence check waves through, and the fallback needs a
    line rather than a file: ``drop_rts`` on one fails as an ioctl, in the seconds
    after rigctld has stopped answering."""
    srv = rigctld()
    argv = _mail_argv(tmp_path, srv.port)
    line = tmp_path / "usbserial-XXXXB1"
    line.write_text("")
    argv[argv.index("--ptt-device") + 1] = str(line)
    r = _run(argv, tmp_path)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSING TO ARM" in r.stdout and str(line) in r.stdout, r.stdout
    assert srv.commands == [], f"the rig was touched anyway: {srv.commands}"
    assert srv.ptt == 0


def test_the_channel_actually_called_decides_reachability(rigctld, tmp_path):
    """The exact call from 2026-08-03: K5DAT-13 on the 40 m channel it lists
    as VARA 500 only. The per-station check saw the 20 m plain-VARA listing
    and said nothing, and the attempt would have spent its whole window on a
    station that cannot bring a session up with kestrel at that frequency.

    Every BW500 waveform the connect needs exists since 2026-08-15 — request,
    response and link-setup, all bench-confirmed against a real VARA — and the
    refusal still stands, because this tool cannot drive the session either way:
    ``vara_arq._NSYM`` reads a received burst by symbol count and the two
    bandwidths share theirs, so calling here would key BW2300 audio at it. See
    ``test_tools_cli.test_the_connect_tool_offers_only_the_bandwidth_it_speaks``."""
    srv = rigctld()
    r = _run(_mail_argv(tmp_path, srv.port, freq="7103.2"), tmp_path)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "only as VARA 500" in r.stdout
    assert "NOT calling a channel that cannot answer" in r.stdout
    assert srv.commands == [], "refused after the rig was already touched"


def test_force_overrides_the_gateway_list_but_still_says_so(rigctld, tmp_path):
    """The list can be stale and the operator outranks it — but the refusal's
    reason is still said out loud before the first key-up."""
    srv = rigctld()
    r = _run(_mail_argv(tmp_path, srv.port, freq="7103.2") + ["--force"],
             tmp_path)
    assert "only as VARA 500" in r.stdout
    assert "originating: W9SSJ -> K5DAT-13" in r.stdout
