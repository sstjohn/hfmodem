# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Entry point for besra's ARDOP host server (``besra-modem``).

Runs the two-socket host interface over a `ModemCore`:

  * ``--modem loopback`` (default) — the dialect face, no waveform;
  * ``--modem besra`` — the real PHY+ARQ `BesraModem` joined to a radioless echo
    peer, so a client (Pat, Winlink Express) can dial it end to end with no radio;
  * ``--radio ft891 --serial … --audio-in … --audio-out …`` — the real modem on a
    real rig: PTT/CAT over hamlib and audio over the sound card, so a client can
    connect to a live ARDOP station or RMS gateway.

With ``--call GATEWAY`` and mail flags it is the whole station instead: no host
ports, the modem's own observer feeds a Winlink B2F session (`station.mail`'s
`ArdopMail`), and the process connects, runs the exchange, disconnects and
exits. Same flags as ``shrike.onair`` — one shape for mail on the air.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from . import protocol as P
from .modem_core import LoopbackModem
from .server import HostServer
from ...core import band, config, levels, rates
from ...core.ptt import arming_refusal
from ...winlink import CLIENT_SID

#: Beside every other on-air recording this station makes.
RECORD_DIR = Path(__file__).resolve().parents[5] / "logs" / "onair"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="besra ARDOP host server")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; ardopcf defaults to 0.0.0.0)")
    ap.add_argument("--control-port", type=int, default=P.DEFAULT_CONTROL_PORT,
                    help=f"command port (default {P.DEFAULT_CONTROL_PORT}; data port is +1)")
    ap.add_argument("--data-port", type=int, default=None,
                    help="data port (default: control-port + 1)")
    ap.add_argument("--bandwidth", type=int, default=500,
                    choices=(200, 500, 1000, 2000), help="session bandwidth (Hz)")
    ap.add_argument("--modem", choices=("loopback", "besra"), default="loopback",
                    help="loopback: the dialect face, no waveform. besra: the real "
                         "PHY+ARQ modem + a radioless echo peer.")
    ap.add_argument("--peer-call", default="BESRA-1",
                    help="the echo peer's callsign to ARQCALL (--modem besra)")

    rg = ap.add_argument_group("real radio (overrides --modem)")
    rg.add_argument("--radio", choices=("ft891", "g90", "x6100"),
                    help="drive this rig: audio over the sound card, PTT/CAT over hamlib")
    rg.add_argument("--serial", help="rig CAT serial device (e.g. /dev/ttyUSB0)")
    rg.add_argument("--rigctl", default="rigctl",
                    help="hamlib rigctl to drive (default: on PATH; pass a full path "
                         "for a from-source build)")
    rg.add_argument("--ptt-device", metavar="DEV",
                    help="serial line whose RTS is the PTT. Without --line-ptt it "
                         "is the emergency unkey's route only: the hamlib one-shot "
                         "that unkey falls back on travels the link RF disrupts, "
                         "and this does not")
    rg.add_argument("--line-ptt", action="store_true",
                    help="key that line directly, holding it for the session, and "
                         "leave hamlib to set frequency and mode. On this station "
                         "rigctld stopped answering on the first key-down of a "
                         "session under RF; a transmit path that has to work while "
                         "the antenna radiates must not travel that link. Start the "
                         "daemon with ptt_type=None so nothing else owns the line")
    rg.add_argument("--rigctld", metavar="HOST:PORT",
                    help="drive a shared rigctld (netrigctl) instead of opening the CAT "
                         "port directly — the station rule; e.g. localhost:4532")
    rg.add_argument("--audio-in", help="capture (receiver) audio device")
    rg.add_argument("--audio-out", help="playback (transmit) audio device")
    rg.add_argument("--tx-drive", type=float, default=None,
                    help="soundcard peak for TX (0..1); the audio level sets RF "
                         "power on a data interface. Raise against a watched "
                         "meter. Defaults to [audio] tx_drive from the station "
                         f"file named by {config.STATION_ENV}, else "
                         f"{levels.TX_DRIVE}")
    rg.add_argument("--dial", type=int, help="set the rig dial frequency (Hz) at startup")
    rg.add_argument("--channel", type=int,
                    help="tune to this channel CENTRE (Hz); dial = centre − 1500")
    rg.add_argument("--list-audio", action="store_true", help="list audio devices and exit")
    rg.add_argument("--record", type=Path, default=RECORD_DIR, metavar="DIR",
                    help=f"directory for the receive recording (default {RECORD_DIR})")
    rg.add_argument("--no-record", dest="record", action="store_const", const=None,
                    help="do not record — an on-air attempt nobody recorded cannot be "
                         "diagnosed afterwards, so this is rarely what you want")

    mg = ap.add_argument_group(
        "winlink mail (one exchange over the live link, then exit)")
    mg.add_argument("--call", metavar="CALLSIGN",
                    help="connect to this gateway and run the attached mail "
                         "exchange instead of serving the host ports")
    mg.add_argument("--transmit", action="store_true",
                    help="ARM the rig for a --call mail run (licensed op only); "
                         "without it a --radio mail run refuses to start. The "
                         "server modes are unaffected — there a host client "
                         "initiates, not this process.")
    mg.add_argument("--mycall", help="our callsign (required with --call)")
    mg.add_argument("--mail-send", action="append", metavar="FILE",
                    help="Winlink mail: a rendered .b2f message, or a body file "
                         "with --mail-to (repeatable)")
    mg.add_argument("--mail-fetch", action="store_true",
                    help="Winlink mail: collect whatever the gateway holds")
    mg.add_argument("--mail-to", default="",
                    help="recipient for a --mail-send body file")
    mg.add_argument("--mail-subject", default="")
    mg.add_argument("--mail-password", default="",
                    help=f"secure-login answer to the gateway's ;PQ: challenge. "
                         f"Spelled out here it is in this process's argv, which "
                         f"every `ps` can read: prefer --mail-password-file or "
                         f"${config.PASSWORD_ENV}")
    mg.add_argument("--mail-password-file", default="",
                    help="read the ;PQ: answer from this file instead, so it "
                         "never reaches argv (default: "
                         f"${config.PASSWORD_ENV})")
    mg.add_argument("--mail-sid", default="",
                    help="the client type this station announces: the name and "
                         "version half of the B2F SID, no brackets and no "
                         "capability letters. Defaults to [station] client_sid "
                         f"from the station file named by {config.STATION_ENV}, "
                         f"else {CLIENT_SID}")
    mg.add_argument("--mail-out", default="logs/mail",
                    help="where received messages are written")
    mg.add_argument("--mail-timeout", type=float, default=900.0,
                    help="seconds for the whole exchange, connect included "
                         "(default 900)")

    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    ap = parser()
    args = ap.parse_args()
    args.mail_password = config.mail_password(args.mail_password, args.mail_password_file)

    if args.list_audio:
        from ...core.devices import list_devices
        list_devices()
        return 0

    if (args.mail_send or args.mail_fetch) and not args.call:
        ap.error("--mail-send/--mail-fetch need --call GATEWAY")
    if args.line_ptt and not args.ptt_device:
        ap.error("--line-ptt needs --ptt-device DEV — the line to key")
    if args.call:
        if not args.mycall:
            ap.error("--call needs --mycall")
        if not args.mail_send and not args.mail_fetch:
            ap.error("--call: nothing to do — give --mail-send FILE, "
                     "--mail-fetch, or both")
        # A --call run initiates on its own — unlike the server modes, where a
        # host client keys through us — so arming it is an explicit act, the
        # same rule every other keying tool at this station follows.
        if args.radio and not args.transmit:
            ap.error("--call with --radio keys the transmitter; pass --transmit")
        # Load every outbound message now, while a bad path or a body file with
        # no --mail-to costs a usage line — not after the radio link is up,
        # where the same mistake used to come back as a traceback with the rig
        # already tuned.
        from ...winlink import load_outbound
        from ...winlink.message import MessageError
        for p in args.mail_send or []:
            try:
                load_outbound(p, args.mycall, to=args.mail_to,
                              subject=args.mail_subject)
            except (OSError, MessageError) as e:
                ap.error(f"--mail-send {p}: {e}")

    if args.radio and (args.serial or args.rigctld) and (
            refusal := arming_refusal(args.ptt_device)):
        ap.error(refusal)

    link = None
    if args.radio:
        modem, link = _radio_modem(args)
    elif args.modem == "besra":
        from ..sim.echo import besra_with_echo_peer
        modem = besra_with_echo_peer(args.peer_call, bandwidth=args.bandwidth)
    else:
        modem = LoopbackModem(bandwidth=args.bandwidth)

    try:
        if args.call:
            if not args.radio:
                # kestrel_connect prints the same banner for its --dry-run: a
                # radioless mail run's log is otherwise line-for-line what a
                # real on-air failure prints, and that log is what an
                # operator carries back as the finding.
                print(f"  DRY RUN — no radio attached (--modem {args.modem}), "
                      "nothing keyed", flush=True)
            return _run_mail(args, modem)
        server = HostServer(
            modem, host=args.host, control_port=args.control_port,
            data_port=args.data_port, quiet=args.quiet,
        )
        server.serve_forever()
        return 0
    finally:
        if link is not None:
            # Where shrike and kestrel say it: with the capture still up, and on
            # the one path both a mail run and a served session leave by.
            print(rates.host_report(), flush=True)
            link.close()


def _mail_observer():
    """The live link's observer for a mail run: `station.mail.ArdopMail`'s
    routing — only ARQ payload feeds the session — with the link edges said out
    loud and remembered, because an unattended mail run is diagnosed from its
    stdout. Built by a factory so the `station` import stays out of module load
    (station.mail imports this package's `modem_core`)."""
    from ...station.mail import ArdopMail

    class _MailSession(ArdopMail):
        def __init__(self, client):
            super().__init__(client)
            self.down = threading.Event()
            self.no_link = threading.Event()

        def modem_newstate(self, state: str) -> None:
            print(f"  [{state}]", flush=True)

        def modem_connected(self, remote: str, bw: int) -> None:
            print(f"  CONNECTED {remote} @ {bw} Hz", flush=True)
            super().modem_connected(remote, bw)

        def modem_disconnected(self) -> None:
            self.down.set()
            print("  link down", flush=True)

        def modem_status(self, text: str) -> None:
            # The session's own connect verdict, relayed verbatim on the host
            # protocol; "FAILED!" is how it says the retry budget is spent.
            print(f"  {text}", flush=True)
            if "FAILED" in text:
                self.no_link.set()

        def modem_fault(self, text: str) -> None:
            print(f"  FAULT {text}", flush=True)

    return _MailSession


def _run_mail(args, modem) -> int:
    """One Winlink B2F exchange over the live ARQ link, then exit.

    The modem's observer is the mail client's transport: `modem_connected` is
    `link_up`, delivered ARQ payload feeds the session in arrival order, and
    everything the session answers goes out through `modem.transmit`, chunked
    and acknowledged by the link's own law. ARDOP's turn law needs no help from
    here — an IRS with data queued BREAKs — so this loop only watches the clock
    and tears the link down when the exchange is over.
    """
    from ...winlink import (B2FSession, MailClient, load_outbound, summarize,
                            write_inbox)

    outbox = [load_outbound(p, args.mycall, to=args.mail_to,
                            subject=args.mail_subject)
              for p in (args.mail_send or [])]
    session = B2FSession(args.mycall, role="calling", target=args.call,
                         password=args.mail_password, outbox=outbox,
                         client_sid=config.client_sid(args.mail_sid))
    client = MailClient(session, modem.transmit)
    obs = _mail_observer()(client)
    modem.set_mycall(args.mycall.upper())
    modem.start(obs)
    deadline = time.monotonic() + args.mail_timeout
    # What became of the LINK, in the vocabulary shrike's session summary
    # already uses; `summarize` below says what became of the exchange.
    ended = "the run was cut short before the link was decided"
    try:
        print(f"mail: {args.mycall.upper()} -> {args.call.upper()} — connecting "
              f"as {session.sid}", flush=True)
        modem.connect(args.call.upper())
        while (time.monotonic() < deadline and not modem.connected
               and not obs.no_link.is_set()):
            time.sleep(0.2)
        if not modem.connected:
            ended = ("no link — nothing answered the connect request"
                     if obs.no_link.is_set() else
                     "no link — out of time before anything answered")
            print("mail: NO LINK", flush=True)
            # The session may still be mid-attempt, keying connect-requests on
            # its own timers; abort ends that before the pump is torn down, so
            # the last PTT edge is the session's own down rather than whatever
            # state the teardown caught it in.
            modem.abort()
            return 1
        while time.monotonic() < deadline and not client.done and modem.connected:
            time.sleep(0.2)
        # A failed session is finished too — `_fail` sets the same _DONE state a
        # clean FQ does — so `done` alone cannot say which happened. On
        # 2026-08-14 the KY4RY run printed "exchange complete — closing" for a
        # session that had already failed on the gateway's banner, nine seconds
        # ahead of the "stage failed" its teardown summary carried, and the two
        # lines together read at the rig as a contradiction nobody could
        # account for. The failure is the more specific answer, so it goes first.
        if session.failure:
            ended = "the exchange failed and the link was closed"
            print("mail: exchange failed — closing the link", flush=True)
        elif client.done:
            ended = "the exchange finished and the link was closed"
            print("mail: exchange complete — closing", flush=True)
        elif modem.connected:
            ended = (f"out of time after {args.mail_timeout:.0f} s and the link "
                     f"was closed")
            print("mail: out of time — closing the link", flush=True)
        else:
            ended = "the link went down mid-exchange"
        modem.disconnect()
        end = time.monotonic() + 30.0
        while time.monotonic() < end and modem.connected and not obs.down.is_set():
            time.sleep(0.2)
        ok = session.done and not session.failure
        if len(session.sent_mids) != len(outbox):
            ok = False               # a message left behind is not a clean run
        return 0 if ok else 1
    finally:
        modem.stop()
        print(f"session ended: {ended}", flush=True)
        print(summarize(session), flush=True)
        for p in write_inbox(session, args.mail_out):
            print(f"mail: wrote {p}", flush=True)


def _radio_modem(args):
    """A threaded BesraModem bound to a real rig via RadioLink. Tunes the rig if a
    dial/channel is given and starts the receive stream."""
    from ..arq.modem import BesraModem
    from ...core.rigs import RIGS
    from ..radio import Rig, RadioLink, center_to_dial

    import logging
    import signal

    # On a real rig, surface what besra decodes and its PTT/watchdog diagnostics —
    # the first on-air sessions were undiagnosable because the log was silent.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    modem = BesraModem(bandwidth=args.bandwidth, threaded=True)
    dial = center_to_dial(args.channel) if args.channel is not None else args.dial
    rig = (Rig.named(args.radio, args.serial, rigctl=args.rigctl,
                     rigctld=args.rigctld, ptt_device=args.ptt_device,
                     line_ptt=args.line_ptt)
           if (args.serial or args.rigctld) else None)
    if rig is not None:
        # Positive identification before arming — keying the wrong radio is the
        # expensive mistake — and a SIGTERM/SIGHUP unkey, since atexit does not
        # run on either and every other unkey here hangs off a `finally`, which
        # a signal never reaches. A session killed mid-over on 2026-08-04 kept
        # transmitting until the operator cut power; the ordinary ways an
        # unattended run is stopped (timeout(1), a supervisor, a dropped
        # terminal) are exactly the ways that left it keyed.
        # Through rigctld the daemon is already bound to a specific rig and \dump_caps
        # reports the netrigctl backend, not the model, so identify returns "?" — that
        # is not a mismatch, only an unreadable model, and must not raise a false alarm.
        # It verifies nothing either: on 2026-08-13 this branch said "ok" and keyed,
        # four times, on whatever dial the previous run left. The QSY gate below is
        # what decides whether this process may key.
        model = rig.identify()
        digits = "".join(c for c in args.radio if c.isdigit())    # ft891 -> 891
        if model == "?":
            print(f"besra on {args.radio}: model via rigctld not readable "
                  f"(daemon owns the rig) — unverified; the dial readback "
                  f"decides", flush=True)
        elif digits and digits not in model:
            # For the server, a warning: the host client decides whether to key.
            # A --call mail run initiates by itself the moment it starts, so
            # there a mismatch is a refusal — keying the wrong radio is the
            # expensive mistake, and nobody is watching an unattended run.
            if getattr(args, "call", None):
                raise SystemExit(
                    f"REFUSING TO START: rig reports {model!r}, not the "
                    f"--radio {args.radio} this mail run would key")
            print(f"!! WARNING: connected rig {model!r} does not match --radio {args.radio}",
                  flush=True)
        else:
            print(f"besra on {args.radio}: rig reports {model!r}", flush=True)

        def _unkey_and_exit(signum, _frame):
            # Retire before unkeying, so a racing transmit thread cannot put
            # the line back up in the window before the interpreter exits; and
            # exit nonzero, because a run killed mid-over did not succeed.
            rig.retire()
            rig.unkey(signal.Signals(signum).name)
            raise SystemExit(1)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _unkey_and_exit)

        # A rig with no dial named is the 2026-08-13 failure with the gate
        # removed: nothing to ask for, so nothing to verify, and the modem keys
        # on whatever the last run left. `tools/onair.sh` passes --channel on
        # both ARDOP verbs, so naming it is what the working configuration
        # already does.
        if dial is None:
            raise SystemExit(
                f"NOT KEYING: --radio {args.radio} with no --channel or --dial "
                f"-- the dial would be whatever the rig was last left on, and "
                f"nothing here would know")
        # The QSY gate, in shrike's shape: set the dial, read it back, and
        # refuse to key when the rig does not report the frequency that was
        # asked for. On 2026-08-13 four attempts went out on whatever dial
        # the previous run left — across the PACTOR channels — while the log
        # printed the dial besra *wanted*; the operator caught it at the
        # rig, twice, and nothing in any log would have. An unreadable
        # answer refuses too, where shrike proceeds unverified: an
        # unreadable rig is not a verified dial, and unlike the model read
        # above — which netrigctl legitimately hides — the frequency read
        # passes through the daemon, so "nothing" here means nobody is
        # talking to the radio. Transmitting on an unintended frequency is
        # the licensee's regulatory problem, not just a wasted attempt, so
        # this refusal is unconditional: there is no flag past it.
        got = rig.qsy(dial)
        if got is None or abs(got - dial) > band.QSY_TOLERANCE_HZ:
            raise SystemExit(
                f"QSY FAILED: asked {dial} Hz, rig reports "
                f"{got if got is not None else 'nothing readable'} "
                f"-- not keying")
        # Mode AFTER frequency: the FT-891 keeps a mode per band, so a band
        # jump can restore whatever the new band was last left in.
        rig.set_mode(RIGS[args.radio]["mode"])
    wav = recording_path(args.record, dial) if args.record else None
    # Resolved here rather than in `main`, because this is where the transmitter
    # is built and `main` is not its only caller.
    drive = config.tx_drive(args.tx_drive)
    link = RadioLink(modem, in_device=args.audio_in, out_device=args.audio_out, rig=rig,
                     record=wav, drive=drive)
    link.start()
    if not args.quiet:
        print(f"besra on {args.radio}: dial {dial or 'unset'}, audio in={args.audio_in} "
              f"out={args.audio_out}, TX drive {drive:.2f} peak", flush=True)
        print(f"  recording receive audio to {wav}" if wav else
              "  NOT recording receive audio", flush=True)
    return modem, link


def recording_path(directory: Path, dial: int | None) -> Path:
    """A timestamped WAV naming the frequency it was heard on. UTC, because the
    only reason to open one of these is to line it up against a log."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(directory) / f"{stamp}-besra-{dial or 'nodial'}.wav"


if __name__ == "__main__":
    raise SystemExit(main())
