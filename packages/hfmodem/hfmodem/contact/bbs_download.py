# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Single-message VarAC file codec, verified against KQ4QWH 2026-10-25.

Offer integer and MD5 cover the gzip bytes; SFB carries their base64 encoding.
Only this observed gzip variant is supported. Limits are local policy.
"""
import base64
import binascii
import hashlib
import os
from pathlib import Path
import tempfile
import zlib

MAX_COMPRESSED = 32768
MAX_OUTPUT = 1048576
DOWNLOAD_SECONDS = 1800

from .filetransfer import RawTransferCapture


class DownloadCapture(RawTransferCapture):
    """Known file transfers allow 30 minutes; opaque experiments remain 120 s."""
    max_seconds = DOWNLOAD_SECONDS


def validate_offer(offer):
    if not offer.wire_name.endswith('.VarAC_gzip'):
        raise ValueError('Only the observed VarAC gzip file format is supported')
    if not 0 < offer.declared_integer <= MAX_COMPRESSED:
        raise ValueError('File exceeds the 32 KiB compressed download limit')


def decode_file(offer, message):
    validate_offer(offer)
    if not message.startswith('<SFB>'):
        raise ValueError('Expected an SFB file message')
    encoded = message[5:]
    if len(encoded) != 4 * ((offer.declared_integer + 2) // 3):
        raise ValueError('File encoded length does not match offer')
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Invalid file base64') from error
    if len(compressed) != offer.declared_integer:
        raise ValueError('File compressed length does not match offer')
    if hashlib.md5(compressed).hexdigest().upper() != offer.digest_text.upper():
        raise ValueError('File checksum does not match offer')
    decoder = zlib.decompressobj(31)
    try:
        data = decoder.decompress(compressed, MAX_OUTPUT + 1)
    except zlib.error as error:
        raise ValueError('Invalid gzip file') from error
    if len(data) > MAX_OUTPUT or decoder.unconsumed_tail:
        raise ValueError('File exceeds the 1 MiB expanded download limit')
    if not decoder.eof or decoder.unused_data:
        raise ValueError('Incomplete gzip file or unexpected trailing data')
    return data


def save_file(root, name, data):
    """Publish only a completed file, in a unique locally generated directory."""
    # Reuse the offer parser's flat-filename checks, even for offline callers.
    from .filetransfer import parse_offer
    parse_offer(f'<SF:{name}|0|{"0" * 32}>')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='bbs-', dir=root))
    temporary = directory / ('.partial-download' if name == '.partial' else '.partial')
    destination = directory / name
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
