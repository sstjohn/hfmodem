# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded reachable link waveforms, prepared outside the key window.

Source inspection never lends or commits bytes. Admission only looks up a key;
a late source update is prepared on the next cycle after a cold-miss refusal.
"""
from . import p4breakintx, p4chirp, p4shorttx, p4robustlongtx, p4robusttx
from .p4observation import Kind


class Waveforms(dict):
    def __init__(self, word_render):
        super().__init__()
        self.word_render = word_render


def cache_key(kind, word=None, value=None, header='A', payload=b'', status=0,
              repeat=False, chirp_slot=0, level=None):
    key = kind, word, value, header, payload, status
    if kind == Kind.LONG:
        return key+(level, repeat)
    return key+(repeat, chirp_slot % 2) if kind == Kind.CHIRP else key


def prefix(source, size):
    """The lender's escaped-byte boundary, without changing its ownership."""
    end = min(size, len(source))
    i = 0
    while i < end:
        width = 2 if source[i] == 0x1c else 1
        if i+width > end:
            return source[:i]
        i += width
    return source[:end]


def reachable_keys(machine):
    kinds = {Kind.BODY}
    if machine.data_kind in (Kind.BODY, Kind.NORMAL):
        kinds.add(Kind.NORMAL)
    if machine.data_kind in (Kind.CHIRP, Kind.LONG):
        kinds.add(machine.data_kind)
    packets = [p for p in (machine._next, machine._data_packet) if p is not None]
    kinds.update(p.kind for p in packets if p.kind in (Kind.BODY, Kind.NORMAL, Kind.CHIRP, Kind.LONG))
    inflight = machine.outbox.inflight
    queued = bytes(machine.outbox.arq._outbuf)
    keys = set()
    for kind in kinds:
        level = machine.long_level if kind == Kind.LONG else None
        size = ({2: 40, 3: 69, 4: 141}[level] if level is not None else
                {Kind.BODY: 31, Kind.NORMAL: 40, Kind.CHIRP: 22}[kind])
        recut = prefix(inflight+queued, size)
        fields = {prefix(inflight, size), prefix(queued, size), recut}
        if not inflight:
            fields.add(prefix(queued[len(recut):], size))
        fields.update(p.payload for p in packets if p.kind == kind)
        for field in fields-{b''}:
            statuses = (*range(4), *range(0x40, 0x44), *range(0x60, 0x64)) if kind == Kind.LONG else (
                range(0x60, 0x64) if kind == Kind.CHIRP else (
                (*range(4), 0x40) if kind == Kind.BODY else range(4)))
            variants = ((False, 0), (False, 1), (True, 0), (True, 1)) if kind == Kind.CHIRP else (
                ((False, 0), (True, 0)) if kind == Kind.LONG else ((False, 0),))
            headers = ("A", "B") if kind == Kind.NORMAL else ("A",)
            for status in statuses:
                for repeat, slot in variants:
                    for header in headers:
                        keys.add(cache_key(kind, header=header, payload=field, status=status,
                                           repeat=repeat, chirp_slot=slot, level=level))
    for p in packets:
        if p.payload:
            keys.add(cache_key(p.kind, p.word, p.value, p.header, p.payload,
                               p.status, p.repeat, p.chirp_slot, p.level))
    for value in machine.ladder:
        for header in ("A", "B"):
            keys.add(cache_key(Kind.SHORT, value=value, header=header))
    return keys


def prepare(cache, machine, *, word_render='off'):
    """Fill/prune the exact reachable matrix. Call only before cycle reads."""
    if getattr(cache, 'word_render', word_render) != word_render:
        raise ValueError('P4 waveform cache render mode mismatch')
    keys = reachable_keys(machine)
    for key in sorted(keys, key=repr):
        if key in cache:
            continue
        kind, _, value, header, field, status, *variant = key
        if kind == Kind.CHIRP:
            repeat, slot = variant
            cache[key] = p4chirp.entry_packet(field, status, repeat=repeat,
                                            rank0_upper=bool(slot))
        elif kind == Kind.LONG:
            level, repeat = variant
            cache[key] = p4robustlongtx.robust_long(field, status, level,
                                                  repeat=repeat, fs=machine.fs)
        elif kind == Kind.SHORT:
            cache[key] = p4robusttx.robust_short(value, header, fs=machine.fs)
        elif kind == Kind.BODY:
            cache[key] = p4breakintx.changeover_packet(field, status, word_render=word_render)
        elif kind == Kind.NORMAL:
            cache[key] = p4shorttx.short_packet(field, status, header=header)
    for key in list(cache):
        if key[4] and key not in keys:
            del cache[key]
