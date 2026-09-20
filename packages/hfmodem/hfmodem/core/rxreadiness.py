# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fail-closed startup mode/nominal-width evidence, not full RX calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import fcntl
import hashlib
import math
import os
from pathlib import Path
import stat
import time

from .occupied import FILTER_HZ, occupied_hz

# Manufacturer FT-891 CAT reference p17, DATA/RTTY/PSK width settings.
FT891_DATA_WIDTHS = frozenset((50, 100, 150, 200, 250, 300, 350, 400, 450,
                             500, 800, 1200, 1400, 1700, 2000, 2400, 3000))


class ReceiverReadbackError(RuntimeError):
    """A missing or failed observation must never become the requested value."""


@dataclass(frozen=True)
class RadioMode:
    mode: str
    passband_hz: int
    raw_response: str


def parse_frequency_readback(stdout, returncode, stderr=""):
    value = stdout.strip()
    if returncode != 0 or "RPRT -" in stderr or not value.isdecimal() or int(value) <= 0:
        raise ReceiverReadbackError("frequency readback failed or is ambiguous")
    return int(value)


def assessment_profile(args):
    values = {k: v for k, v in vars(args).items()
              if not k.startswith("_") and k not in ("rx_assessment", "rx_claim_fd")}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()


def assessment_sources():
    """Critical-source binding; launcher qualification binds the full source set."""
    package = Path(__file__).resolve().parents[1]
    names = ("core/rxreadiness.py", "shrike/ota.py", "shrike/onair.py", "shrike/live.py",
             "shrike/rxfront.py", "shrike/p3acquire.py", "shrike/ptc.py", "shrike/arq.py")
    return {n: hashlib.sha256((package / n).read_bytes()).hexdigest() for n in names}


READBACK_SCHEMAS = ("radio-startup-readback-1", "radio-assessed-readback-1")


def session_readback_path(output_dir: Path, stamp: str) -> Path:
    """This session's evidence file, written beside an earlier one, never over it.

    Two arms can share one output directory, and the second still has to record
    what its own radio answered; the mere presence of a record is not a fault.
    What remains a refusal is a file at the canonical name that this module did
    not write: unreadable output cannot be told apart from a failed readback,
    and nothing here overwrites either.
    """
    path = output_dir / "radio-readback.json"
    if not path.exists():
        return path
    try:
        prior = json.loads(path.read_text())
    except (OSError, ValueError):
        prior = None
    if not isinstance(prior, dict) or prior.get("schema") not in READBACK_SCHEMAS:
        raise ReceiverReadbackError(f"{path} is not this station's readback evidence")
    return output_dir / f"radio-readback-{stamp}.json"


def verify_inherited_claim(serial, fd):
    """Validate the station's existing inherited flock, not an environment claim."""
    if type(fd) is not int or fd < 3:
        raise ReceiverReadbackError("missing inherited station claim descriptor")
    path = Path("/tmp") / ("hfmodem-rig-" + Path(serial).name + ".claim")
    owned, named = os.fstat(fd), path.stat()
    if (not stat.S_ISREG(owned.st_mode) or owned.st_uid != os.getuid()
            or (owned.st_dev, owned.st_ino) != (named.st_dev, named.st_ino)):
        raise ReceiverReadbackError("station claim descriptor does not name this CAT claim")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {"device": owned.st_dev, "inode": owned.st_ino, "cat": serial}


def read_assessment(args, serial, *, clock=None):
    clock = clock or time.monotonic
    claim = verify_inherited_claim(serial, args.rx_claim_fd)
    path = Path(args.rx_assessment)
    if path.stat().st_size > 65536:
        raise ReceiverReadbackError("oversized RX assessment")
    record = json.loads(path.read_text())
    if (not isinstance(record, dict) or record.get("schema") != "rx-assessment-1"
            or record.get("assessment_complete") is not True):
        raise ReceiverReadbackError("missing or malformed RX assessment")
    now = clock()
    issued, expires = record.get("issued_monotonic"), record.get("expires_monotonic")
    if (type(issued) not in (int, float) or type(expires) not in (int, float)
            or not math.isfinite(issued) or not math.isfinite(expires)
            or not issued <= now < expires or not 0 < expires-issued <= 900):
        raise ReceiverReadbackError("expired or invalid RX assessment interval")
    if (record.get("claim") != claim or record.get("profile_sha256") != assessment_profile(args)
            or record.get("sources") != assessment_sources()):
        raise ReceiverReadbackError("RX assessment ownership, profile or source changed")
    actual = record.get("actual")
    if (not isinstance(actual, dict) or type(actual.get("frequency_hz")) is not int
            or type(actual.get("passband_hz")) is not int or not isinstance(actual.get("mode"), str)):
        raise ReceiverReadbackError("malformed assessed radio tuple")
    return record


def verify_assessed_receiver_startup(rig, *, args, serial, expected_mode, model, dial, output_dir):
    """Reobserve the assessed tuple before RF, with no post-CCA setters."""
    try:
        record = read_assessment(args, serial)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        with session_readback_path(output_dir, stamp).open("x") as handle:
            frequency = rig.get_frequency_readback()
            actual = rig.get_mode()
            observed = {"frequency_hz": frequency, "mode": actual.mode,
                        "passband_hz": actual.passband_hz}
            if frequency != dial or observed != record["actual"]:
                raise ReceiverReadbackError("radio drifted after RX/CCA assessment; not retuning")
            width = validate_mode_width(actual, expected_mode=expected_mode, model=model,
                         span_hz=occupied_hz("pactor", "1" if args.pactor1_only else ""))
            if read_assessment(args, serial) != record:
                raise ReceiverReadbackError("RX assessment changed during readback")
            report = {"schema": "radio-assessed-readback-1", "status": "readback_verified_only",
                      "actual": observed, "assessment": record, "width": width,
                      "setters_after_assessment": 0, "rf_authorized": False,
                      "full_receiver_verified": False}
            json.dump(report, handle, indent=2)
            handle.write("\n")
        return report
    except BaseException as exc:
        try:
            rig._close()
        except BaseException as cleanup_error:
            exc.add_note(f"Owned CAT cleanup also failed: {cleanup_error}")
        if isinstance(exc, Exception):
            raise SystemExit(f"NOT KEYING: assessed receiver state unverified: {exc}") from exc
        raise


def parse_mode_readback(stdout: str, returncode: int, stderr: str = "") -> RadioMode:
    if returncode != 0 or "RPRT -" in stdout or "RPRT -" in stderr:
        raise ReceiverReadbackError("mode readback failed or is unsupported")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if (len(lines) != 2 or not lines[0] or any(c.isspace() for c in lines[0])
            or not lines[1].isdecimal() or int(lines[1]) <= 0):
        raise ReceiverReadbackError("radio supplied no unambiguous mode/passband readback")
    return RadioMode(lines[0], int(lines[1]), stdout)


def validate_mode_width(actual: RadioMode, *, expected_mode: str, model: int,
                        span_hz: tuple[float, float], centre_hz: float = 1500) -> dict:
    lo, hi = span_hz
    if (not all(math.isfinite(x) for x in (lo, hi, centre_hz))
            or not 0 <= lo < centre_hz < hi):
        raise ReceiverReadbackError("invalid occupied-span contract")
    if actual.mode != expected_mode:
        raise ReceiverReadbackError(f"requested {expected_mode}, radio reports {actual.mode}")
    if model == 1036 and expected_mode == "PKTUSB" and actual.passband_hz not in FT891_DATA_WIDTHS:
        raise ReceiverReadbackError("undocumented FT-891 DATA passband readback")
    minimum = 2 * max(centre_hz - lo, hi - centre_hz)
    if actual.passband_hz < minimum:
        raise ReceiverReadbackError(
            f"radio reports {actual.passband_hz} Hz width; profile requires at least {minimum:g} Hz")
    return {"occupied_hz": [lo, hi], "nominal_centre_hz": centre_hz,
            "minimum_nominal_width_hz": minimum,
            "width_source": "Hamlib get_mode response",
            "filter_position_verified": False, "full_receiver_verified": False}


def verify_receiver_startup(rig, *, expected_mode: str, model: int,
                            pactor1_only: bool, output_dir: Path) -> dict:
    """Request, observe and gate before first RF; preserve exact startup evidence.

    Mode/nominal-width only. IF shift, DATA-menu cuts, DSP and analog/codec
    calibration remain unknown. P1-only uses its enforced narrower emission;
    other supported rigs are not forced through FT-891-specific width steps.
    """
    output_dir = Path(output_dir)
    now = datetime.now(timezone.utc)
    report = {"schema": "radio-startup-readback-1", "status": "pending",
              "utc": now.isoformat(), "model": model,
              "requested": {"mode": expected_mode, "passband_hz": FILTER_HZ},
              "actual": None, "profile": "pactor1-only" if pactor1_only else "pactor-upgrade",
              "calibration": {"DATA_OUT": None, "codec_input_scalar": None,
                              "AF_gain": None, "RF_gain": None, "AGC": None,
                              "IF_shift": None, "DATA_filters": None, "DSP": None},
              "full_receiver_verified": False, "rf_authorized": False}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = session_readback_path(output_dir, now.strftime("%Y%m%dT%H%M%S.%fZ"))
        with path.open("x") as handle:
            try:
                if rig.set_mode(expected_mode, passband=FILTER_HZ) is not True:
                    raise ReceiverReadbackError("mode request did not enter the CAT transport")
                actual = rig.get_mode()
                report["actual"] = asdict(actual)
                report.update(validate_mode_width(actual, expected_mode=expected_mode,
                    model=model, span_hz=occupied_hz("pactor", "1" if pactor1_only else "")))
                report["status"] = "mode_width_verified_only"
            except BaseException as exc:
                report["status"] = "refused"
                report["reason"] = str(exc)
                try:
                    json.dump(report, handle, indent=2)
                    handle.write("\n")
                except BaseException as metadata_error:
                    exc.add_note(f"Receiver refusal metadata also failed: {metadata_error}")
                raise
            else:
                json.dump(report, handle, indent=2)
                handle.write("\n")
    except BaseException as exc:
        try:
            rig._close()
        except BaseException as cleanup_error:
            exc.add_note(f"Owned CAT cleanup also failed: {cleanup_error}")
        if isinstance(exc, Exception):
            raise SystemExit(f"NOT KEYING: receiver mode/passband unverified: {exc}") from exc
        raise
    print(f"receiver readback: {report['actual']['mode']} {report['actual']['passband_hz']} Hz "
          f"(requested {FILTER_HZ}); nominal width verified, IF/DSP/calibration unknown")
    return report
