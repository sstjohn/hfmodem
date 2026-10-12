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

KINDS = ('rig tap', 'sdr witness', 'rig rx', 'sense')
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
    selected = next((p for p in (path, path.with_suffix('.flac')) if p.is_file()), None)
    if selected is not None:
        from hfmodem.core.capture_identity import validate_artifact
        validate_artifact(selected)
    return selected


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
    try:
        if path.suffix == '.flac':
            # STREAMINFO, the first metadata block: 20 bits of sample rate, 8 of layout, 36 of samples.
            with path.open('rb') as f:
                head = f.read(26)
            info = int.from_bytes(head[18:26], 'big')
            rate, samples = info >> 44, info & (1 << 36) - 1
            return round(samples / rate, 1) if head[:4] == b'fLaC' and rate and samples else None
        with wave.open(str(path)) as w:
            return round(w.getnframes() / w.getframerate(), 1)
    except (OSError, EOFError, wave.Error):
        return None


def label(kind: str, side: dict, receivers: dict[str, str]) -> str:
    if kind == 'sdr witness':
        where = receivers.get(side.get('receiver', ''))
        return f'KiwiSDR, {where}' if where else f"KiwiSDR {side.get('receiver', 'receiver not recorded')}"
    return {'rig tap': 'Rig monitor tap, both sides', 'rig rx': 'Our receiver',
            'sense': 'Channel check before keying'}[kind]
