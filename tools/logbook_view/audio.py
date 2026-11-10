# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The recordings an arm made, each with what produced it.

Kinds, by writer:

    rig tap      tools/txwitness.py `record`, started by consolelib/station.py
                 `start_tap` on the rig's monitor line-in: both sides of the
                 exchange as the rig heard them, our transmission included.
    sdr witness  tools/kiwi_witness.py, started by onair.sh `--witness`: a remote
                 KiwiSDR tuned to the channel. witness.json names the receiver.
    rig rx       our own receiver's input on the data jack: shrike's session
                 stream.wav (PACTOR), kestrel_connect.py's and besra's
                 logs/onair recordings (VARA, ARDOP). Muted while we key.
    sense        rig.sh's channel-sense window, kept when it refused or waited.

rx_NN/hold_NN/tx_NN clips are cut from stream.wav or rendered for one burst, so
they are not listed; stream.wav holds them. A transcript names each recording as
it was written, a .wav; many have since been compressed to .flac beside it.
"""
from __future__ import annotations

import csv
import json
import re
import wave
from pathlib import Path

KINDS = ('rig tap', 'sdr witness', 'rig rx', 'submitted tx', 'sense')
# The lines a transcript names a recording by, and the writer each line belongs to.
NAMED = (
    ('sdr witness', re.compile(r'witness: (?:kiwi_witness pid \d+ -> )?(\S+/witness\.wav)')),
    ('rig rx', re.compile(r'^session stream: (\S+\.(?:wav|flac))', re.M)),
    ('rig rx', re.compile(r'(?:recording receive audio to|recorded \d+s of receive audio to) (\S+\.(?:wav|flac))')),
    ('sense', re.compile(r'the window this rests on is kept: (\S+\.wav)')),
)


def named(text: str) -> list[tuple[str, str]]:
    """(kind, path as written) for every recording the transcript names, first mention first."""
    found = sorted((m.start(), kind, m[1]) for kind, pattern in NAMED for m in pattern.finditer(text))
    return list(dict.fromkeys((kind, path) for _, kind, path in found))


def on_disk(path: Path) -> Path | None:
    """The recording a transcript names, or the FLAC it was compressed to."""
    from hfmodem.core import wav
    from hfmodem.core.capture_identity import validate_artifact
    from hfmodem.core.capture_identity import IdentityError
    try:
        base = wav.recording_base(path)
    except (OSError, ValueError, IdentityError):
        return None
    from hfhost.recordings import open_segment
    try:
        parts = wav.segments(base)
    except (FileNotFoundError, ValueError):
        return None
    try:
        for part in parts:
            with open_segment(part) as stream:
                validate_artifact(Path(stream.name))
    except (OSError, ValueError, IdentityError):
        return None
    return wav.stored(base) if len(parts) == 1 and parts[0].stem == base.stem else base


def kiwis(root: Path) -> dict[str, str]:
    """Each KiwiSDR the witness picks from, by host:port, as a place and a grid."""
    path = root / 'working' / 'kiwi-candidates.csv'
    if not path.is_file():
        return {}
    with path.open(newline='') as f:
        return {f"{r['hostname']}:{r['port']}": f"{r['name_location'].split('|')[-1].strip()} ({r['grid']})"
                for r in csv.DictReader(f)}


def sidecar(path: Path) -> dict:
    side = path.with_suffix('.json')
    try:
        return json.loads(side.read_text()) if side.is_file() else {}
    except (OSError, ValueError):
        return {}


def duration_s(path: Path, side: dict) -> float | None:
    if seconds := side.get('duration_s') or side.get('seconds'):
        return round(seconds, 1)
    from hfmodem.core import wav
    try:
        lengths = [_file_duration(part) for part in wav.segments(path)]
    except (OSError, ValueError):
        return None
    return round(sum(lengths), 1) if all(s is not None for s in lengths) else None


def _file_duration(path):
    from hfhost.recordings import audio_info
    try:
        rate, samples = audio_info(path)
        return samples / rate if rate > 0 else None
    except (OSError, EOFError, ValueError, wave.Error):
        return None


def label(kind: str, side: dict, receivers: dict[str, str]) -> str:
    if kind == 'sdr witness':
        where = receivers.get(side.get('receiver', ''))
        return f'KiwiSDR, {where}' if where else f"KiwiSDR {side.get('receiver', 'receiver not recorded')}"
    return {'rig tap': 'Rig monitor tap, both sides', 'rig rx': 'Our receiver',
            'submitted tx': 'Submitted TX', 'sense': 'Channel check before keying'}[kind]
