# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One command, several verbs.

    hfmodem devices                     what sound cards this machine has
    hfmodem config   FILE               load a station file and print what it means
    hfmodem rig      FILE [--arm]       talk to the radio; --arm proves the PTT line
    hfmodem rig      FILE --preflight   measure the station before a session
    hfmodem station  FILE               run the station
    hfmodem version

Every verb that can key a transmitter refuses unless the station file says
`transmit = true`, and says so rather than doing nothing quietly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hfhost.config import ConfigError

from hfmodem import __version__


def _config(path: str):
    from hfmodem.core import config
    return config.load(path)


def cmd_devices(args) -> int:
    from hfmodem.core import devices
    try:
        devices.list_devices()
    except Exception as exc:                        # noqa: BLE001
        print(f"cannot enumerate audio devices: {exc}", file=sys.stderr)
        return 1
    return 0


def _ptt_verdict(port: str) -> str:
    """What a stat can prove about the keying line, without opening it —
    opening an RTS keying line asserts RTS, which is a key-down. Report
    rather than refuse: the refusal lives at the arm gate."""
    from hfmodem.core.ptt import PttError, require_char_device
    try:
        require_char_device(port)
    except PttError as exc:
        return str(exc).removeprefix(f"PTT port {port}").lstrip(" :") or str(exc)
    return "a character device"


def cmd_config(args) -> int:
    from hfmodem.winlink import sid_line

    cfg = _config(args.file)
    st, rig, audio = cfg.station, cfg.rig, cfg.audio
    print(f"station    {st.mycall or '(no callsign)'} {st.grid}")
    print(f"winlink    announcing as {sid_line(st.client_sid)}")
    print(f"control    {cfg.control.value}")
    print(f"rules      {cfg.profile.name}"
          + (f" ({cfg.profile.licence.value})" if hasattr(cfg.profile, "licence") else "")
          + (f" — {cfg.profile.because}" if hasattr(cfg.profile, "because") else ""))
    print(f"transmit   {'enabled' if st.transmit else 'DISABLED'}")
    print(f"rig        {rig.model} via {rig.cat} at {rig.host}:{rig.port}")
    print(f"centre     {rig.centre_hz / 1e6:.6f} MHz  ->  dial {rig.dial_hz / 1e6:.6f} MHz")
    where = (f"{rig.ptt.port} ({_ptt_verdict(rig.ptt.port)})" if rig.ptt.port
             else "(unset)")
    print(f"ptt        {rig.ptt.backend} on {where}, settle {rig.ptt.settle_s} s")
    print(f"audio      in {audio.input or '(unset)'} / out {audio.output or '(unset)'}"
          f" @ {audio.rate} Hz, gain {audio.input_gain}, tx drive {audio.tx_drive}")
    enabled = [n for n, p in cfg.protocols.items() if p.enabled]
    print(f"protocols  {', '.join(enabled) if enabled else '(none enabled)'}")
    print(f"listens    {'yes — this station is automatically controlled' if cfg.listens else 'no'}")
    return 0


def cmd_rig(args) -> int:
    from hfmodem.core import ptt as pttmod
    from hfmodem.core.rig import Cat, Rig, RigError, unkey_on_signal

    cfg = _config(args.file)
    if (args.arm or args.preflight) and not cfg.station.transmit:
        what = ("proving the PTT line" if args.arm
                else "measuring what the rig does when it is keyed")
        print(f"transmit = false in this station file: {what} would "
              "key the radio, so it is refused. Set it deliberately.",
              file=sys.stderr)
        return 2

    if not cfg.rig.ptt.port:
        print("[rig.ptt] port is unset; nothing to key.", file=sys.stderr)
        return 2

    if args.preflight:
        from hfmodem.station import preflight
        return preflight.run(cfg)

    cat = Cat(cfg.rig.host, cfg.rig.port)
    line = pttmod.RtsPtt(cfg.rig.ptt.port)

    rig = Rig(model=cfg.rig.model, cat=cat, ptt=line, profile=cfg.profile,
              control=cfg.control, mycall=cfg.station.mycall,
              transmit=cfg.station.transmit, max_key_s=cfg.rig.max_key_s)
    with unkey_on_signal(lambda: rig):
        try:
            line.open()
            print(rig.arm(prove_ptt=args.arm))
        except (RigError, pttmod.PttError) as exc:
            print(f"arm failed: {exc}", file=sys.stderr)
            return 1
        finally:
            rig.close()
    return 0


def cmd_station(args) -> int:
    from hfmodem.station.process import Station

    cfg = _config(args.file)
    station = Station(cfg)
    try:
        return station.run()
    except KeyboardInterrupt:
        return 0


def cmd_mail(args) -> int:
    from hfmodem.station import mail
    return mail.run(args)


def cmd_version(args) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hfmodem", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="verb", required=True)

    sub.add_parser("devices", help="list sound cards").set_defaults(fn=cmd_devices)

    c = sub.add_parser("config", help="load a station file and print what it means")
    c.add_argument("file", type=Path)
    c.set_defaults(fn=cmd_config)

    r = sub.add_parser("rig", help="talk to the radio")
    r.add_argument("file", type=Path)
    r.add_argument("--arm", action="store_true",
                   help="prove the PTT line — keys the radio for ~0.2 s with no "
                        "modulation, so nothing identifiable (or audible) goes out")
    r.add_argument("--preflight", action="store_true",
                   help="measure the station: capture levels, lost samples, the ADC/DAC "
                        "offset, PTT actuation and T/R recovery. Keys with no "
                        "modulation, so nothing goes into the antenna")
    r.set_defaults(fn=cmd_rig)

    s = sub.add_parser("station", help="run the station")
    s.add_argument("file", type=Path)
    s.set_defaults(fn=cmd_station)

    m = sub.add_parser(
        "mail",
        help="run a Winlink B2F exchange over one of this station's modems",
        description="Rehearse a mail exchange over the chosen modem's own "
                    "byte path — everything but the RF — and print the "
                    "on-air command that carries the same exchange to the "
                    "rig. This verb itself cannot key a transmitter.")
    m.add_argument("--gateway", required=True, help="the RMS callsign to work")
    m.add_argument("--protocol", required=True,
                   choices=["pactor", "ardop", "vara"])
    m.add_argument("--mycall", required=True)
    m.add_argument("--freq", type=int, default=0,
                   help="published channel centre Hz (Winlink lists centres; "
                        "the dial is 1500 below)")
    m.add_argument("--bandwidth", type=int, default=500,
                   help="ARDOP channel bandwidth Hz — it belongs to the channel "
                        "like the centre does, and the printed on-air command "
                        "carries it (default 500)")
    m.add_argument("--send", action="append", metavar="FILE",
                   help="a message to send: a rendered .b2f message, or a "
                        "body file with --to (repeatable)")
    m.add_argument("--to", default="", help="recipient for a --send body file")
    m.add_argument("--subject", default="", help="subject for a --send body file")
    m.add_argument("--fetch", action="store_true",
                   help="collect waiting mail (the rehearsal gateway holds one)")
    m.add_argument("--password", default="",
                   help="Winlink secure-login password for the ;PQ: challenge. "
                        "Spelled out here it is in this process's argv, which "
                        "every `ps` can read: prefer --password-file or "
                        "$WINLINK_PASSWORD")
    m.add_argument("--password-file", default="",
                   help="read the ;PQ: password from this file instead, so it "
                        "never reaches argv (default: $WINLINK_PASSWORD)")
    m.add_argument("--sid", default="",
                   help="the client type this station announces: the name and "
                        "version half of the B2F SID, no brackets and no "
                        "capability letters. Defaults to [station] client_sid")
    m.add_argument("--out", default="logs/mail",
                   help="where received messages are written")
    m.set_defaults(fn=cmd_mail)

    sub.add_parser("version").set_defaults(fn=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as exc:
        # The station file is the operator's, and every one of these errors names
        # the key it is about. A traceback over the top of that says the tool
        # broke rather than the file.
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
