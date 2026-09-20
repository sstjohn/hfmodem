# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The transmit seam: CAT that tunes, a keying line that keys, and neither
doing the other's job.

sabir could render a waveform and decode one but had no way to put either on the
air. This is that path, and it is deliberately the smallest thing that can be
trusted with a transmitter -- which now means it contains no keying logic at
all. `core.ptt.LineKeyer` is the whole of it: one ioctl on a descriptor this
object holds from `arm()` to `hand_back()`, read back both ways, with the
retries, the escalation wordings and the last-resort drop living where the other
three modems already found them. sabir carried its own copy of that ladder --
its own retry count, its own backoff, its own stuck-transmitter wording -- and a
fourth copy is how the same defect comes to be fixed four times on four
different days.

**Why one `rigctl` process per command, when the flock's settled advice is a
persistent connection.** That advice is right and it is about ARQ: a one-shot
cost ~900 ms on this FT-891 against 0.07 ms down a long-lived pipe, that lands
on the *unkey*, and on PACTOR the peer answers ~0.29 s in, inside that slot. Nothing here keys through rigctl, so no
one-shot is ever on the near side of an unkey; what CAT does here is set the
dial, read it back, and set the mode, three times in a session, at arm time.
Measured against ``rigctl -m 1 -``: interactive pipe mode echoes each command,
emits blank lines between them, and answers ``m`` with *two* lines (mode, then
passband). A reader taking one line per command desynchronises on the first
``m`` and from then on validates every command against the previous command's
reply. A one-shot cannot desynchronise: the contract is the process exit status,
and a bad port exits 2 (verified).

**The dial is set and read back in one invocation.** Two invocations is not the
same measurement -- anything between them can move the VFO, and on this station
something does. besra's `qsy` has the same shape for the same reason.

**Nothing keys without a checked emission.** The rig holds the station's
regulatory profile and `key()` takes an `Emission`, so a path that cannot say
what it is about to put on the air cannot put anything on the air. `core.rig`
paid for that shape: an earlier design there had a perfectly good gate that
*nothing ever called*, and a gate no transmit path consults is documentation.
Making it an argument means forgetting is a type error rather than an omission --
and this module was the fifth private copy of that rig and the only one keying
past no gate at all.

**What is bounded, and by what.** `transmit` is besra's `tune_atu` shape, which
is the shape that stopped sticking finals: the audio device is opened once
*before* PTT, the rig's own settle comes between the key and the first sample, a
watchdog force-unkeys through the keyer's independent path if the play overruns,
and the key comes down in `finally`. Four bounds, none of which depends on the
others, and the rig's own TX time-out timer under all of them.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import replace

import numpy as np

from hfmodem.core import levels
from hfmodem.core.audio import WARM_S, play_drained
from hfmodem.core.occupied import FILTER_HZ
from hfmodem.core.ptt import LineKeyer
from hfmodem.core.regulatory import Control, Emission, NotPermitted, Profile
from hfmodem.core.rigs import RIGS
from hfmodem.sabir.phy.rate import FS

#: hamlib's "NET rigctl": a rigctld client. On a shared station the serial port
#: is not sabir's to take -- rigctld already holds it, so a direct
#: `rigctl -r /dev/...` gets "device busy" -- and the daemon is also the only
#: thing on such a station resembling an arbiter between the agents sharing one
#: transceiver. It tunes and it never keys; `tools/lib/ports.sh` starts it
#: `ptt_type=None` so the keying line has exactly one owner.
NET_MODEL = 2
DEFAULT_RIGCTLD = "127.0.0.1:4532"

_CAT_TIMEOUT_S = 8.0


def rigctl_path() -> str | None:
    """`rigctl` off PATH, or None. No shipped directory default -- a
    home-directory path here was one station's, and what PATH does not carry the
    launcher supplies (`tools/lib/ports.sh`)."""
    return shutil.which("rigctl")


#: Read before anything is armed, so a station with no rigctl finds out there
#: rather than at a key-down.
NO_RIGCTL = (
    "rigctl not found on PATH, and it is what tunes this rig. Install hamlib "
    "(`brew install hamlib`, or your package manager); for a build of your own, "
    "name its bin directory as HAMLIB_BIN and the launchers in tools/ will put "
    "it on PATH for everything they start. Check with: command -v rigctl")


class CatError(RuntimeError):
    """The radio could not be reached over CAT. Always fatal before a key-up."""


class Rig:
    """One radio: hamlib for frequency and mode, `LineKeyer` for the key.

    The split is the design. rigctld owns the CAT port exclusively and answers
    `t` with ENAVAIL because it has no PTT of its own; this object owns the
    keying line and never asks a daemon to pull it. On 2026-08-13 this station's
    daemon answered every command until the first key-down of a session under RF
    and then answered nothing -- twelve unkeys out of twelve -- while the line
    read low instantly, eight of eight. A transmit path that has to work while
    the antenna radiates must not travel the link that fails only then.
    """

    def __init__(self, radio: str, ptt_device: str, *, profile: Profile,
                 control: Control, serial: str | None = None,
                 rigctld: str | None = None, log=print) -> None:
        if radio not in RIGS:
            raise ValueError(f"unknown radio {radio!r}; known: {', '.join(RIGS)}")
        if not (serial or rigctld):
            raise ValueError("name --serial or --rigctld: nothing here can tune "
                             "a radio it has no way to talk to")
        spec = RIGS[radio]
        self.radio = radio
        self.model, self.baud = spec["model"], spec["baud"]
        self.mode = spec["mode"]
        self.profile, self.control = profile, control
        #: The pause between key-down and the first sample, per rig because its
        #: cause is per rig; `core.rigs` holds the value and the reason.
        self.settle = spec["settle"]
        self.serial, self.rigctld = serial, rigctld
        self.ptt_device = ptt_device
        self.dial_hz: int | None = None
        self._log = log
        self._keyer = LineKeyer(ptt_device, log, alarm=log)
        self.retired = False

    # -- CAT: frequency and mode, and never PTT ----------------------------

    def _base(self) -> list[str]:
        exe = rigctl_path()
        if exe is None:
            raise CatError(NO_RIGCTL)
        if self.rigctld:
            return [exe, "-m", str(NET_MODEL), "-r", self.rigctld]
        return [exe, "-m", str(self.model), "-r", self.serial,
                "-s", str(self.baud)]

    def _run(self, *args: str) -> tuple[int, str]:
        try:
            p = subprocess.run(self._base() + list(args), capture_output=True,
                               text=True, timeout=_CAT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return 124, ""
        return p.returncode, (p.stdout or "").strip()

    def identify(self) -> str:
        """The model name hamlib reports, or ``"?"``.

        Through rigctld the daemon is already bound to a rig and `\\dump_caps`
        describes the netrigctl backend rather than the radio, so "?" is an
        unreadable model and not a mismatch. It verifies nothing either: the
        dial readback is what decides whether this process may key.
        """
        rc, out = self._run("\\dump_caps")
        for line in out.splitlines():
            if "Model name:" in line:
                return line.split(":", 1)[1].strip()
        return "?"

    def transmitting(self) -> bool | None:
        """What the radio says about its own PTT, or None if it will not say.

        None is the ordinary answer here and is not a failure: a daemon started
        `ptt_type=None` -- the only configuration this station accepts, so that
        the keying line has one owner -- has no PTT to report. Only a `True` is
        an answer, and it means somebody else is holding this transmitter.
        """
        rc, out = self._run("t")
        first = out.splitlines()[0].strip() if out else ""
        if rc != 0 or first[:1] not in "01":
            return None
        return first.startswith("1")

    def qsy(self, dial_hz: int) -> int | None:
        """Set the dial and read it back -- one invocation -- or None when the
        rig will not say.

        The set and its readback travel together because a second invocation is
        a second moment, and the dial does not stay where you left it: another
        agent sweeping moves it. Mode is set separately and afterwards, since
        the FT-891 keeps a mode per band and a band jump restores whatever that
        band was last left in.
        """
        rc, out = self._run("F", str(int(dial_hz)), "f")
        got = out.split()
        if rc != 0 or not got or not got[-1].lstrip("-").isdigit():
            return None
        self.dial_hz = int(got[-1])
        return self.dial_hz

    def set_mode(self, passband: int = FILTER_HZ) -> None:
        self._run("M", self.mode, str(int(passband)))

    # -- the key -----------------------------------------------------------

    def arm(self) -> None:
        """Open the keying line. `RtsPtt.open` deasserts before it does anything
        else, so this is the one moment a wrong port is cheap."""
        self._keyer.arm()

    def key(self, emission: Emission, note: str = "") -> bool:
        """Bring the transmitter up for one burst; the verdict is the line's own
        readback.

        The emission is checked against the station's profile before the line
        moves, and against the dial the radio *read back* rather than the one it
        was asked for -- a rig that landed somewhere else is transmitting
        somewhere else. `NotPermitted` names the rule and the numbers and is not
        caught here: a refusal the caller can mistake for a line that would not
        confirm is a refusal nobody acts on.

        What is sabir's here is the retire: a transmitter nobody has confirmed
        is down is not one this rig may key again. A retired rig refuses key-up
        and never key-down -- after a signal-time unkey a racing thread must not
        put the line back up in the window before the interpreter exits.
        """
        if self._keyer is None:                  # handed back; the port is closed
            self._log("PTT up refused: the keying line has been handed back")
            return False
        if self.retired:
            self._log("PTT up refused: this rig is retired")
            return False
        if self.dial_hz is None:
            raise NotPermitted(
                "the dial has not been read back, so there is nothing to place "
                "this emission on. Nothing keys on a frequency this process has "
                "not verified.")
        self.profile.check(replace(emission, dial_hz=self.dial_hz),
                           control=self.control)
        took = self._keyer.key(True, note)
        if self._keyer.must_retire:
            self.retire()
        return took

    def unkey(self, why: str = "") -> bool:
        """Bring the key down. No emission and no gate: a transmitter coming
        down is not something any regulator refuses, and a check between a
        caller and the key coming down is a check that can hold it up."""
        if why:
            self._log(f"unkey: {why}")
        if self._keyer is None:
            return True                          # nothing is up to take down
        took = self._keyer.key(False)
        if self._keyer.must_retire:
            self.retire()
        return took

    def retire(self) -> None:
        self.retired = True

    def hand_back(self) -> None:
        """Deassert, read back, close. The line is not reopened -- a reopen
        raises RTS, and that is a key-down -- so this drops the keyer rather
        than keeping one whose port is gone, and a second call is a no-op
        instead of a second verdict on a line nobody holds."""
        keyer, self._keyer = self._keyer, None
        if keyer is not None:
            keyer.hand_back()


# -- audio ----------------------------------------------------------------
#: This was 0.1, a placeholder for the measurement nobody had taken -- 15.6 dB
#: under the rest of the station, on the one modem whose whole job is to be heard
#: by a receiver it will never hear back. The measurement exists now and is the
#: station's: `core.levels.TX_DRIVE`.
#:
#: At the meter: a beacon is the hottest thing this station emits at
#: a given PEAK. It is near-constant-envelope (3.2-4.9 dB crest) and runs
#: unbroken for a minute or more, where an ARQ burst is a second or two at half
#: duty -- so the same number that leaves an ARQ modem comfortable is a full-duty
#: carrier here. `--gain` is the per-run override.
DEFAULT_TX_GAIN = levels.TX_DRIVE


def levelled(audio: np.ndarray, gain: float = DEFAULT_TX_GAIN) -> np.ndarray:
    """Peak-normalised to `gain`, which is what sets RF power on a data
    interface -- not the rig's power setting."""
    return levels.at_drive(audio, gain)


def transmit(rig: Rig, audio: np.ndarray, emission: Emission, *, device=None,
             gain: float = DEFAULT_TX_GAIN, max_key_s: float | None = None,
             note: str = "") -> float:
    """One bounded transmission. Returns the seconds of audio that went out.

    `emission` is what this array will occupy on the air, and it is required for
    the same reason `key()` requires it: there is no way to reach the key from
    here without describing what goes out of it.

    besra's `tune_atu` shape, and every part of it is load-bearing: the device is
    opened before PTT so the open is not paid under a live carrier and a device
    that will not open refuses before the key -- every time, which is why this
    warms itself rather than calling `core.audio.warm_output`, whose open is once
    per process; an unattended beacon session keys many times over and `WARM_S` of
    unkeyed silence in front of each buys the check again. The rig's settle sits
    between the key and the first sample, a watchdog force-unkeys
    through the keyer's own path if the play overruns, and the key comes down in
    `finally` whatever happened in between. A key-up the line would not confirm
    returns before any audio is played -- `LineKeyer.key` has already deasserted
    to be sure -- because playing into a rig that may not be keyed is how a
    slot is spent proving nothing.
    """
    block = levelled(audio, gain)
    seconds = block.size / FS
    bound = max_key_s if max_key_s is not None else seconds + 5.0
    if seconds >= bound:
        raise ValueError(
            f"{seconds:.1f} s of audio against a {bound:.1f} s key ceiling -- "
            "raise --max-key deliberately or send less")

    play_drained(np.zeros(int(WARM_S * FS), dtype=np.float32), FS, device)

    done = threading.Event()

    def watchdog() -> None:
        if not done.wait(bound):
            rig.unkey(f"watchdog: keyed past {bound:.1f} s")
            rig.retire()

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        if not rig.key(emission, note):
            return 0.0
        time.sleep(rig.settle)
        play_drained(block, FS, device)
        return seconds
    finally:
        rig.unkey()
        done.set()


def tone(seconds: float, hz: float = 1500.0) -> np.ndarray:
    """A steady tone at the signal centre, for the two things a short burst
    cannot do: confirm audio is reaching the rig at all (the power meter moves
    or it does not), and set drive against ALC. Constant envelope, so what the
    meter reads is the level the waveform will actually run at."""
    n = int(seconds * FS)
    x = np.sin(2 * np.pi * hz * np.arange(n) / FS)
    ramp = int(0.05 * FS)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return x
