# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Prepare bounded current-protocol datagrams and verify independent recordings.

This module never opens a radio or a network connection. Successful decoding
proves the captured bytes, not the provenance of the recording or an ARQ link.
"""
from __future__ import annotations

import argparse
import json
import secrets
from hashlib import sha256, shake_256
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from hfmodem.sabir import offair
from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig
from hfmodem.sabir.arq.modem import LinkModem
from hfmodem.sabir.arq.profiles import PROFILES
from hfmodem.sabir.frame.datagram import Fragment, Reassembler, fragment_object
from hfmodem.sabir.phy.modem import GEARS
from hfmodem.sabir.phy.rate import CENTER_HZ, FS

FORMAT = "sabir-recording-proof-1"
MAX_SECONDS = 90.0
MAX_PACKET_SECONDS = 25.0
PROOF_PROFILES = tuple(name for name, p in PROFILES.items() if not p.floor)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*.py"))}


def _asset(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("manifest asset must be inside its directory")
    return path


def prepare(output_dir, *, profiles=("workhorse",), callsign: str,
            token: str | None = None, seed: int | None = None,
            payload_bytes: int = 256) -> Path:
    """Write manifest, exact expected payloads and an identified transmit WAV.

    The default token is fresh randomness. A seed makes local tests reproducible;
    both the token and seed are recorded. Repeated profiles have distinct IDs.
    """
    from hfmodem.sabir.onair import _identified

    callsign = callsign.upper()
    wire.station_bytes(callsign)
    profiles = (profiles,) if isinstance(profiles, str) else tuple(profiles)
    if not profiles or len(profiles) > 8 or any(p not in PROOF_PROFILES for p in profiles):
        raise ValueError("choose 1..8 OFDM DATA profiles")
    if type(payload_bytes) is not int or not 64 <= payload_bytes <= 2048:
        raise ValueError("payload size must be 64..2048 bytes")
    if token is None:
        token = secrets.token_hex(16) if seed is None else sha256(f"seed:{seed}".encode()).hexdigest()[:32]
    if not isinstance(token, str) or not 1 <= len(token) <= 64 or not token.isascii():
        raise ValueError("token must contain 1..64 ASCII characters")
    root = Path(output_dir).resolve()
    if (root / "manifest.json").exists():
        raise ValueError("proof manifest already exists; use a fresh directory")
    frames, trials = [], []
    elapsed = 1.0
    for index, name in enumerate(profiles):
        label = f"{token}/{index}/{name}".encode()
        mid = sha256(b"message-id/" + label).digest()[:16]
        prefix = b"SABIR-PROOF\0" + mid
        payload = prefix + shake_256(b"payload/" + label).digest(payload_bytes - len(prefix))
        record = fragment_object(payload, callsign, fragment_bytes=payload_bytes,
                                 parity=False, message_id=mid)[0].pack()
        modem = LinkModem(ArqConfig(callsign=callsign))
        modem.send_datagram(record, name)
        audio = offair.to_real(modem.outbox[0])
        seconds = len(audio) / FS
        if seconds > MAX_PACKET_SECONDS:
            raise ValueError(f"{name} packet exceeds {MAX_PACKET_SECONDS:g} seconds; reduce payload")
        audio /= max(float(np.max(np.abs(audio))), 1e-12)
        frames.extend([audio, np.zeros(FS)])
        trials.append({"index": index, "profile": name, "profile_id": PROFILES[name].id,
                       "message_id": mid.hex(), "payload_file": f"payload-{index}.bin",
                       "payload_sha256": sha256(payload).hexdigest(), "payload_bytes": len(payload),
                       "datagram_bytes": len(record), "packet_seconds": seconds,
                       "reference_start_s": elapsed, "_payload": payload})
        elapsed += seconds + 1.0
    audio = _identified(np.concatenate([np.zeros(FS), *frames]), callsign)
    if len(audio) / FS > MAX_SECONDS:
        raise ValueError(f"identified burst exceeds {MAX_SECONDS:g} seconds; select fewer profiles")
    # Account for the complete nominal filter transition, not just carrier endpoints.
    spans = [(GEARS[p].passband_hz[0] - GEARS[p].transition_hz,
              GEARS[p].passband_hz[1] + GEARS[p].transition_hz) for p in profiles]
    lo = min(1250.0, *(s[0] for s in spans))
    hi = max(1750.0, *(s[1] for s in spans))
    root.mkdir(parents=True, exist_ok=True)
    waveform = root / "proof.wav"
    offair.wav_write(str(waveform), audio)
    for trial in trials:
        (root / trial["payload_file"]).write_bytes(trial.pop("_payload"))
    manifest = {"format": FORMAT, "callsign": callsign, "token": token, "seed": seed,
                "waveform": waveform.name, "waveform_sha256": _sha(waveform),
                "sample_rate_hz": FS, "nominal_center_hz": CENTER_HZ,
                "transmit_seconds": len(audio) / FS, "audio_span_hz": [lo, hi],
                "source_hashes": source_hashes(), "trials": trials,
                "scope": "one-way datagram reception; no ARQ or RF provenance assertion"}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def validate_manifest(manifest_path, *, callsign: str | None = None,
                      require_current_source: bool = True) -> dict:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported proof manifest")
    if callsign is not None and manifest["callsign"] != callsign.upper():
        raise ValueError("proof callsign differs from the transmitting station")
    if require_current_source and manifest.get("source_hashes") != source_hashes():
        raise ValueError("Sabir sources changed after preparation; prepare a fresh proof")
    if _sha(_asset(path.parent, manifest["waveform"])) != manifest["waveform_sha256"]:
        raise ValueError("prepared waveform hash mismatch")
    if not 0 < manifest["transmit_seconds"] <= MAX_SECONDS:
        raise ValueError("invalid proof transmission duration")
    if not isinstance(manifest.get("trials"), list) or not 1 <= len(manifest["trials"]) <= 8:
        raise ValueError("manifest must contain 1..8 trials")
    for trial in manifest["trials"]:
        payload = _asset(path.parent, trial["payload_file"]).read_bytes()
        if len(payload) != trial["payload_bytes"] or sha256(payload).hexdigest() != trial["payload_sha256"]:
            raise ValueError("expected payload hash mismatch")
        if trial["profile"] not in PROOF_PROFILES or PROFILES[trial["profile"]].id != trial["profile_id"]:
            raise ValueError("invalid proof profile")
    return manifest


def load_burst(manifest_path, callsign: str | None = None):
    """Validate the reviewed artifact and return (Burst, manifest), without I/O devices."""
    from hfmodem.sabir.onair import Burst

    manifest = validate_manifest(manifest_path, callsign=callsign)
    wav = _asset(Path(manifest_path).resolve().parent, manifest["waveform"])
    rate, raw = wavfile.read(wav)
    if rate != FS or raw.ndim != 1 or len(raw) / FS != manifest["transmit_seconds"]:
        raise ValueError("prepared waveform geometry mismatch")
    audio = offair.wav_read(str(wav))
    if not np.all(np.isfinite(audio)) or np.max(np.abs(audio)) > 1.00001:
        raise ValueError("invalid prepared audio amplitude")
    return Burst(audio, "Sabir current-protocol DATAGRAM proof", *manifest["audio_span_hz"]), manifest


class _Receiver(LinkModem):
    def __init__(self):
        super().__init__(ArqConfig())
        self.records = []
        self.profile_id = None

    def _datagram(self, hdr, body):
        self.profile_id = hdr.gear
        super()._datagram(hdr, body)

    def on_datagram(self, record):
        self.records.append((self.profile_id, bytes(record)))


def verify(manifest_path, capture_path) -> dict:
    """Blindly segment a complete recording and run the ordinary modem decoder.

    Expected payload bytes are used only after decoding, for exact comparison.
    Capture timestamps and measured header profiles come from receive processing.
    """
    from hfmodem.sabir.monitor import segment

    manifest_path, capture_path = Path(manifest_path).resolve(), Path(capture_path).resolve()
    manifest = validate_manifest(manifest_path, require_current_source=False)
    rate, raw = wavfile.read(capture_path)
    audio = offair.wav_read(str(capture_path))
    if not np.all(np.isfinite(audio)):
        raise ValueError("capture contains non-finite samples")
    receiver = _Receiver()
    observed, decode_errors = [], []
    for at, burst in segment(audio):
        before = len(receiver.records)
        try:
            receiver.on_air(offair.to_analytic(burst))
        except (ValueError, IndexError) as exc:
            decode_errors.append({"t_s": at / FS, "error": str(exc)})
        for profile_id, record in receiver.records[before:]:
            try:
                fragment = Fragment.unpack(record)
                result = Reassembler().accept(record)
            except (ValueError, UnicodeError):
                continue
            if result is not None:
                _, payload = result
                observed.append({"profile_id": profile_id, "message_id": fragment.message_id.hex(),
                                 "source": fragment.source, "destination": fragment.destination,
                                 "payload_sha256": sha256(payload).hexdigest(), "_payload": payload,
                                 "t_s": at / FS})
    trials = []
    for expected in manifest["trials"]:
        payload = _asset(manifest_path.parent, expected["payload_file"]).read_bytes()
        matches = [r for r in observed if r["message_id"] == expected["message_id"]
                   and r["profile_id"] == expected["profile_id"]
                   and r["source"] == manifest["callsign"] and r["destination"] == "*"
                   and r["_payload"] == payload and r["payload_sha256"] == expected["payload_sha256"]]
        trials.append({"profile": expected["profile"], "profile_id": expected["profile_id"],
                       "message_id": expected["message_id"], "payload_sha256": expected["payload_sha256"],
                       "ok": bool(matches), "byte_exact": bool(matches),
                       "received_copies": len(matches), "received_at_s": [r["t_s"] for r in matches]})
    for result in observed:
        result.pop("_payload")
    capture_hash = _sha(capture_path)
    ok = bool(trials) and all(t["ok"] for t in trials)
    return {"ok": ok, "byte_exact": ok, "evidence_scope": "offline_decode", "radio_proof": False,
            "capture_is_reference": capture_hash == manifest["waveform_sha256"],
            "capture": str(capture_path), "capture_sha256": capture_hash,
            "capture_sample_rate_hz": int(rate), "capture_seconds": len(raw) / rate,
            "manifest_sha256": _sha(manifest_path),
            "transmit_source_hashes": manifest["source_hashes"],
            "decode_source_hashes": source_hashes(),
            "source_matches": manifest["source_hashes"] == source_hashes(),
            "trials": trials, "observed": observed,
            "decode_errors": decode_errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("output_dir")
    make.add_argument("--callsign", required=True)
    make.add_argument("--profiles", nargs="+", choices=PROOF_PROFILES, default=["workhorse"])
    make.add_argument("--bytes", type=int, dest="payload_bytes", default=256)
    make.add_argument("--seed", type=int)
    make.add_argument("--token")
    check = commands.add_parser("verify")
    check.add_argument("manifest")
    check.add_argument("capture")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        print(prepare(**args))
        return 0
    report = verify(args["manifest"], args["capture"])
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
