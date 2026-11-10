# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Logical recording discovery and file replacement retries, without DSP imports."""
from contextlib import contextmanager
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import re
import struct


def metadata(path):
    try:
        side = json.loads(Path(path).with_suffix('.json').read_text())
    except (FileNotFoundError, ValueError):
        return None
    if (isinstance(side, dict) and isinstance(side.get('segments'), list)
            and side['segments'] and all(isinstance(name, str) for name in side['segments'])
            and isinstance(side.get('started_utc'), str) and side['started_utc']):
        return side
    return None


def base(path):
    path = Path(path)
    match = re.fullmatch(r'(.+)-\d{8}T\d{4}Z(?:-s\d{20})?', path.stem)
    if match:
        stems = [match[1], re.sub(r'-l[0-9a-f]+$', '', match[1])]
        for stem in dict.fromkeys(stems):
            candidate = path.with_name(stem + '.wav')
            side = metadata(candidate)
            if side and path.with_suffix('.wav').name in {
                    Path(name).with_suffix('.wav').name for name in side['segments']}:
                return candidate
    return path


def stored(path):
    path = Path(path)
    for candidate in (path, path.with_suffix('.wav'), path.with_suffix('.flac')):
        try:
            candidate.stat()
        except FileNotFoundError as exc:
            if exc.filename != os.fspath(candidate):
                raise
            continue
        return candidate
    return path


def open_segment(path):
    """The open descriptor survives storage replacing the file."""
    for attempt in range(2):
        candidate = stored(path)
        try:
            return candidate.open('rb', buffering=0)
        except FileNotFoundError:
            if attempt:
                raise


@contextmanager
def wave_source(stream, *, decode=None):
    """Decode an already open FLAC, preserving its original WAV header."""
    if Path(stream.name).suffix != '.flac':
        yield stream
        return
    exe = shutil.which('flac')
    if decode is None and not exe:
        raise FileNotFoundError('flac is not installed (macOS: brew install flac; Debian/Ubuntu: apt install flac)')
    # FLAC's foreign-metadata restore requires a named seekable input.
    with tempfile.TemporaryDirectory() as tmp:
        encoded, decoded = Path(tmp) / 'input.flac', Path(tmp) / 'decoded.wav'
        with encoded.open('wb') as out:
            shutil.copyfileobj(stream, out)
        if decode is None:
            subprocess.run([exe, '-s', '-d', '--keep-foreign-metadata-if-present',
                            '-o', str(decoded), str(encoded)], check=True, capture_output=True)
        else:
            decode(encoded, decoded)
        with decoded.open('rb', buffering=0) as out:
            yield out


def audio_info(path):
    with open_segment(path) as stream:
        rate, count = stream_info(stream)
        if rate <= 0:
            raise ValueError('Invalid recording sample rate')
        if not count and Path(stream.name).suffix == '.flac':
            try:
                with wave_source(stream) as decoded:
                    result = stream_info(decoded)
                    if not result[1]:
                        raise ValueError('Recording length unavailable')
                    return result
            except subprocess.CalledProcessError as exc:
                raise ValueError('Recording length unavailable') from exc
        return rate, count


def require_frames(actual, promised, name, *, error=ValueError):
    """Recordings shorter than the promised frame count raise."""
    if actual < promised:
        raise error(f'{name}: recording ended early ({actual} of {promised} frames)')


def stream_info(stream):
    """Rate and promised frames, leaving the open recording at its beginning."""
    try:
        header = stream.read(12)
        if header[:4] == b'fLaC':
            info = header[8:] + stream.read(14)
            packed = int.from_bytes(info[10:18], 'big')
            return packed >> 44, packed & ((1 << 36) - 1)
        if header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError(f'{stream.name}: invalid recording header')
        rate = align = frames = None
        while chunk := stream.read(8):
            tag, size = chunk[:4], struct.unpack('<I', chunk[4:])[0]
            if tag == b'fmt ':
                fmt = stream.read(size)
                rate = struct.unpack_from('<I', fmt, 4)[0]
                align = struct.unpack_from('<H', fmt, 12)[0]
                if size & 1:
                    stream.read(1)
            elif tag == b'data':
                frames = size
                if not align:
                    raise ValueError(f'{stream.name}: incomplete recording header')
                offset = stream.tell()
                actual = stream.seek(0, 2) - offset
                require_frames(actual // align, size // align, stream.name)
                break
            else:
                stream.seek(size + (size & 1), 1)
        if rate is None or not align or frames is None:
            raise ValueError(f'{stream.name}: incomplete recording header')
        return rate, frames // align
    except struct.error as exc:
        raise ValueError(f'{stream.name}: incomplete recording header') from exc
    finally:
        stream.seek(0)


def sample_offset(path):
    match = re.search(r'-\d{8}T\d{4}Z-s(\d{20})(?:\.pending)?\.(?:wav|flac|json)$', Path(path).name)
    return int(match[1]) if match else None


def offset_segments(path):
    """Map absolute source offsets to files for an offset-named family."""
    path = Path(path)
    side = metadata(path)
    if side is None or not any(sample_offset(name) is not None for name in side['segments']):
        return None
    result = {}
    pattern = re.compile(re.escape(path.stem) + r'(?:-l[0-9a-f]+)?-\d{8}T\d{4}Z-s\d{20}\.(wav|flac)')
    for name in side['segments']:
        first = sample_offset(name)
        if not pattern.fullmatch(name) or first is None or first in result:
            raise ValueError('Recording sidecar names an invalid segment')
        result[first] = stored(path.parent / name)
    return dict(sorted(result.items()))


def _policy(path):
    path = Path(path).absolute().with_suffix('.wav')
    for root in path.parents:
        try:
            index = json.loads((root / 'recording.json').read_text())
        except (OSError, ValueError):
            continue
        refs = [(name, lane, None) for name, lane in index.get('lanes', {}).items()]
        refs += [(name, ref, cid) for cid, contact in index.get('contacts', {}).items()
                 for name, ref in contact.get('lanes', {}).items()]
        for name, ref, cid in refs:
            if not ref.get('source') or (root / ref['source']).absolute().with_suffix('.wav') != path:
                continue
            rows = [row for row in index.get('segments', [])
                    if row.get('lane') == name and row.get('contact_id') == cid
                    and all(row.get(key) == ref.get(key) for key in ('source_id', 'index_id'))]
            return root, index, ref, sorted(rows, key=lambda r: r['actual'][0])
    return None


def intervals(path, *, start=None, stop=None):
    """Expected absolute half-open bounds and paths, without reading PCM."""
    policy = _policy(path)
    if policy is not None and policy[3]:
        root, _, _, rows = policy
        return [(tuple(row['actual']), stored(root / row['source'])) for row in rows
                if (start is None or row['actual'][1]>start) and (stop is None or row['actual'][0]<stop)]
    offsets = offset_segments(path)
    if offsets is not None:
        capacity = (metadata(path) or {}).get('segment_samples')
        return [((first, first + audio_info(part)[1]), part) for first, part in offsets.items()
                if (stop is None or first<stop) and
                (start is None or capacity is None or first+capacity>start)]
    parts, capacity = indexed(path)
    result, end = [], 0
    for i, part in parts.items():
        first = i * capacity if capacity is not None else end
        if capacity is not None and ((stop is not None and first>=stop)
                                     or (start is not None and first+capacity<=start)):
            continue
        if capacity is None:
            try:
                side = json.loads(Path(part).with_suffix('.json').read_text())
                # A window saved off the stream's clock records its end as null.
                if side.get('start_stream_sample') is not None:
                    first = side['start_stream_sample']
                elif side.get('end_stream_sample') is not None and 'samples' in side:
                    first = side['end_stream_sample'] - side['samples']
            except (OSError, ValueError):
                pass
        end = first + audio_info(part)[1]
        result.append(((first, end), part))
    return result


def status(path):
    """Policy availability and expected coverage survive moved audio files."""
    policy = _policy(path)
    if policy is None:
        try:
            spans = intervals(path)
            count = sum(hi - lo for (lo, hi), _ in spans)
            return dict(state='available', actual=[spans[0][0][0], spans[-1][0][1]],
                        expected_samples=count, available_samples=count,
                        segments=[dict(source=str(p), actual=list(bounds)) for bounds, p in spans])
        except (OSError, ValueError):
            return dict(state='unavailable', actual=None, expected_samples=None,
                        available_samples=0, segments=[])
    root, index, ref, rows = policy
    available_samples, segments = 0, []
    for row in rows:
        item = dict(row)
        try:
            rate, frames = audio_info(root / row['source'])
            if rate != row['rate'] or frames != row['samples']:
                raise ValueError('Recording length or rate mismatch')
            item['available'] = True
            available_samples += frames
        except (OSError, ValueError):
            item['available'] = False
        segments.append(item)
    actual = ref.get('actual', [ref.get('origin'), ref.get('position')])
    expected = sum(r['samples'] for r in rows)
    state = ('disabled' if index.get('policy') == 'off' else
             'unavailable' if ref.get('state') in ('failed', 'disabled') else
             'contact-only' if index.get('policy') == 'contacts' and 'kind' not in ref else
             'unavailable' if available_samples < expected else
             'pending' if not rows or any(r.get('state') != 'finalized' for r in rows) else
             'available')
    return dict(state=state, actual=actual, expected_samples=expected,
                available_samples=available_samples, gaps=ref.get('gaps', []), segments=segments)


def indexed(path, capacity=3600 * 48000):
    """Return segment indices without closing holes; legacy files use index zero."""
    path = Path(path)
    policy = _policy(path)
    if policy is not None and policy[3]:
        root, _, ref, rows = policy
        side = metadata(path) or {}
        capacity = side.get('segment_samples', capacity * ref['rate'] // 48000)
        return {i: stored(root / row['source']) for i, row in enumerate(rows)}, capacity
    legacy = stored(path)
    if legacy.is_file():
        return {0: legacy}, None
    side = metadata(path)
    if side is None:
        return {0: legacy}, None
    offsets = offset_segments(path)
    if offsets is not None:
        return dict(enumerate(offsets.values())), side.get('segment_samples', capacity)
    pattern = re.compile(re.escape(path.stem) + r'-(\d{8}T\d{4}Z)\.(wav|flac)')
    found = {path.parent / Path(name).with_suffix('.wav').name for name in side['segments']}
    capacity = side.get('segment_samples', capacity)
    start = datetime.fromisoformat(side['started_utc'].replace('Z', '+00:00'))
    rate = side.get('samplerate') or audio_info(min(found))[0]
    result = {}
    for part in sorted(found):
        match = pattern.fullmatch(part.name)
        if match is None:
            raise ValueError('Recording sidecar names an invalid segment')
        stamp = datetime.strptime(match[1], '%Y%m%dT%H%MZ').replace(tzinfo=timezone.utc)
        # Names have minute precision; the recording epoch can include seconds.
        index = round((stamp.timestamp() - int(start.timestamp() // 60) * 60) * rate / capacity)
        expected = datetime.fromtimestamp(start.timestamp() + index * capacity / rate, timezone.utc)
        if index < 0 or expected.strftime('%Y%m%dT%H%MZ') != pattern.fullmatch(part.name)[1] or index in result:
            raise ValueError('Recording segments have inconsistent timestamps')
        result[index] = stored(part)
    return result, capacity


def segments(path, capacity=3600 * 48000):
    parts, capacity = indexed(path, capacity)
    offsets = offset_segments(path)
    if offsets is not None:
        end = rate = None
        for first, part in offsets.items():
            current_rate, frames = audio_info(part)
            if end is not None and (first != end or rate != current_rate):
                raise ValueError('Contact recording is not yet completely written (gap or rate change)')
            end, rate = first + frames, current_rate
    if capacity is not None and offset_segments(path) is None:
        if set(parts) != set(range(max(parts) + 1)):
            raise ValueError('Contact recording is not yet completely written (missing segment)')
        rate = None
        for index, part in parts.items():
            try:
                current_rate, frames = audio_info(part)
            except FileNotFoundError:
                raise ValueError('Contact recording is not yet completely written (missing segment)') from None
            if rate is not None and current_rate != rate:
                raise ValueError('Recording segments have different sample rates')
            rate = current_rate
            if index < max(parts) and frames != capacity:
                raise ValueError('Contact recording is not yet completely written (short segment)')
    return [parts[index] for index in sorted(parts)]


def refuse_occupied(path):
    path = Path(path)
    pattern = re.compile(r'\.?' + re.escape(path.stem) + r'-\d{8}T\d{4}Z(?:-s\d{20})?(?:\.pending)?\.(wav|flac)')
    if any(p.exists() or p.is_symlink() for p in (path, path.with_suffix('.flac'))) or (
            path.parent.exists() and any(pattern.fullmatch(p.name) for p in path.parent.iterdir())):
        raise FileExistsError(f'Recording family already exists: {path}')



def size_bytes(path):
    total = 0
    for part in segments(path):
        with open_segment(part) as stream:
            total += os.fstat(stream.fileno()).st_size
    return total


def available(path):
    try:
        path = base(path)
        policy = _policy(path)
        if policy is not None and (policy[1].get('policy') == 'off'
                                   or policy[2].get('state') in ('disabled', 'failed')):
            return False
        parts = [p for _, p in intervals(path)] if policy is not None and policy[3] else segments(path)
        for part in parts:
            with open_segment(part):
                pass
        return True
    except (OSError, ValueError):
        return False


if __name__ == '__main__':
    import sys
    sys.exit(0 if available(sys.argv[1]) else 1)
