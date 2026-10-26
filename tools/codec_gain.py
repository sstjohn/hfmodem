#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Read/set the input gain of a named CoreAudio device (the rig codec).

AppleScript's ``set volume input volume`` only touches the *default* input, which
here is the built-in microphone — the rig codec has to be addressed directly. The
high speed levels carry data in the amplitude of a QAM constellation, so a
clipped capture is undecodable; this is what keeps the recording in range.

    python3 tools/codec_gain.py                     # show
    python3 tools/codec_gain.py --set 0.4           # set scalar gain 0..1
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import struct
import json
import math

_ca = _cf = None


def _load_audio():
    global _ca, _cf
    if _ca is None:
        _ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))
        _cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))


kAudioObjectSystemObject = 1


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32),
                ("mScope", ctypes.c_uint32),
                ("mElement", ctypes.c_uint32)]


SCOPE_GLOBAL = _fourcc("glob")
SCOPE_INPUT = _fourcc("inpt")
SCOPE_OUTPUT = _fourcc("outp")          # the codec's playback side = the rig's TX drive
PROP_DEVICES = _fourcc("dev#")
PROP_NAME = _fourcc("lnam")
PROP_VOLUME = _fourcc("volm")
PROP_STREAMS = _fourcc("stm#")


def _get(obj: int, addr: _Addr, size: int, buf):
    _load_audio()
    sz = ctypes.c_uint32(size)
    err = _ca.AudioObjectGetPropertyData(ctypes.c_uint32(obj), ctypes.byref(addr),
                                         0, None, ctypes.byref(sz), buf)
    return err, sz.value


def _cfstr_to_py(ref) -> str:
    _cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
    p = _cf.CFStringGetCStringPtr(ref, 0x08000100)          # kCFStringEncodingUTF8
    if p:
        return p.decode()
    buf = ctypes.create_string_buffer(256)
    _cf.CFStringGetCString(ref, buf, 256, 0x08000100)
    return buf.value.decode()


def devices() -> list[tuple[int, str]]:
    _load_audio()
    addr = _Addr(PROP_DEVICES, SCOPE_GLOBAL, 0)
    sz = ctypes.c_uint32(0)
    err = _ca.AudioObjectGetPropertyDataSize(ctypes.c_uint32(kAudioObjectSystemObject),
                                       ctypes.byref(addr), 0, None, ctypes.byref(sz))
    if err:
        raise OSError(f"device list query OSStatus {err}")
    n = sz.value // 4
    arr = (ctypes.c_uint32 * n)()
    err, _ = _get(kAudioObjectSystemObject, addr, sz.value, arr)
    if err:
        raise OSError(f"device list read OSStatus {err}")
    out = []
    for did in arr:
        naddr = _Addr(PROP_NAME, SCOPE_GLOBAL, 0)
        ref = ctypes.c_void_p()
        err, _ = _get(did, naddr, ctypes.sizeof(ref), ctypes.byref(ref))
        if err:
            raise OSError(f"device name query OSStatus {err}")
        out.append((int(did), _cfstr_to_py(ref) if not err and ref else f"<{did}>"))
    return out


def streams(did: int, scope: int) -> int:
    _load_audio()
    addr = _Addr(PROP_STREAMS, scope, 0)
    sz = ctypes.c_uint32(0)
    err = _ca.AudioObjectGetPropertyDataSize(ctypes.c_uint32(did), ctypes.byref(addr), 0, None,
                                       ctypes.byref(sz))
    if err:
        raise OSError(f"stream query OSStatus {err}")
    return sz.value // 4


def find(name: str, scope: int = SCOPE_INPUT) -> int:
    """A codec can list as two devices of one name, capture and playback; take the
    one that carries `scope`, else the first match."""
    matches = [did for did, dname in devices() if name.lower() in dname.lower()]
    if not matches:
        raise SystemExit(f"no audio device matching {name!r}")
    return next((did for did in matches if streams(did, scope)), matches[0])


def get_gain(did: int, scope: int = SCOPE_INPUT):
    """Volume scalar for `scope`, trying the master element then channels 1/2."""
    for elem in (0, 1, 2):
        addr = _Addr(PROP_VOLUME, scope, elem)
        val = ctypes.c_float()
        err, _ = _get(did, addr, ctypes.sizeof(val), ctypes.byref(val))
        if err == 0:
            return elem, float(val.value)
    return None, None


def set_gain(did: int, gain: float, scope: int = SCOPE_INPUT) -> bool:
    _load_audio()
    ok = False
    for elem in (0, 1, 2):
        addr = _Addr(PROP_VOLUME, scope, elem)
        val = ctypes.c_float(gain)
        err = _ca.AudioObjectSetPropertyData(ctypes.c_uint32(did), ctypes.byref(addr),
                                             0, None, ctypes.sizeof(val),
                                             ctypes.byref(val))
        if err == 0:
            ok = True
    return ok


def capability(device: str, scope_name="input", requested=None) -> dict:
    """Query presence and writability before classifying an absent control.

    This is opt-in; legacy find/get/set and CLI formatting remain separate.
    A readable but read-only scalar is supported with writable=False.
    """
    result = dict(status="error", device=device, scope=scope_name, value=None,
                  writable=None, error=None)
    if requested is not None:
        result['requested'] = requested
    try:
        if scope_name not in ("input", "output"):
            raise ValueError("scope must be input or output")
        if not device:
            raise ValueError("audio device is unresolved")
        if requested is not None and (not math.isfinite(requested) or not 0 <= requested <= 1):
            raise ValueError("gain must be finite and in [0, 1]")
        scope = SCOPE_INPUT if scope_name == "input" else SCOPE_OUTPUT
        _load_audio()
        matches = [(did, name) for did, name in devices() if device.lower() in name.lower()]
        selected = [(did, name) for did, name in matches if streams(did, scope)]
        if len(selected) != 1:
            raise ValueError(f"audio device lookup {'ambiguous' if len(selected) > 1 else 'failed'}: {device!r}")
        did, name = selected[0]
        result.update(device=name, device_id=did)
        controls = []
        for elem in (0, 1, 2):
            addr = _Addr(PROP_VOLUME, scope, elem)
            if not _ca.AudioObjectHasProperty(ctypes.c_uint32(did), ctypes.byref(addr)):
                continue
            writable = ctypes.c_ubyte()
            err = _ca.AudioObjectIsPropertySettable(ctypes.c_uint32(did), ctypes.byref(addr),
                                                    ctypes.byref(writable))
            if err:
                raise OSError(f"volume writability query OSStatus {err}")
            val = ctypes.c_float()
            err, size = _get(did, addr, ctypes.sizeof(val), ctypes.byref(val))
            if err or size != ctypes.sizeof(val) or not math.isfinite(val.value) or not 0 <= val.value <= 1:
                raise OSError(f"volume read query OSStatus {err}, size {size}")
            controls.append((elem, bool(writable.value), float(val.value)))
        if not controls:
            result.update(status="unsupported", writable=False)
            return result
        elem, writable, value = next((c for c in controls if c[1]), controls[0])
        result.update(status="supported", element=elem, writable=writable, value=value)
        if requested is not None:
            if not writable:
                result.update(status="unsupported", error="volume is read-only")
                return result
            for elem, writable, _ in controls:
                if not writable:
                    continue
                addr = _Addr(PROP_VOLUME, scope, elem)
                val = ctypes.c_float(requested)
                err = _ca.AudioObjectSetPropertyData(ctypes.c_uint32(did), ctypes.byref(addr),
                    0, None, ctypes.sizeof(val), ctypes.byref(val))
                if err:
                    raise OSError(f"volume write rejected: OSStatus {err}")
            observed = capability(device, scope_name)
            if observed['status'] != 'supported':
                raise OSError(f"volume readback failed: {observed['error']}")
            result['value'] = observed['value']
        return result
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result.update(status="error", error=str(exc))
        return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", type=float, help="gain scalar 0..1")
    ap.add_argument("--scope", choices=("input", "output"), default="input",
                    help="input = RX capture level; output = TX drive into the rig "
                         "(set output low enough that the rig's ALC barely moves)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", action="store_true", help="structured capability and optional write result")
    from hfmodem.core.rigs import RIGS, launcher_defaults
    ap.add_argument("--rig", choices=tuple(RIGS))
    args = ap.parse_args()

    if args.list:
        for did, name in devices():
            print(f"  {did:5d}  {name}")
        return 0

    launcher_defaults(args, audio_fields={"device": args.scope},
                      legacy={"device": "USB Audio Device"})

    if args.json:
        result = capability(args.device, args.scope, args.set)
        print(json.dumps(result))
        return 0 if result['status'] == 'supported' else 1

    scope = SCOPE_INPUT if args.scope == "input" else SCOPE_OUTPUT
    did = find(args.device, scope)
    elem, cur = get_gain(did, scope)
    if cur is None:
        print(f"{args.device}: device exposes NO settable {args.scope} volume "
              f"(set it on the radio instead)")
        return 1
    print(f"{args.device}: {args.scope} gain = {cur:.3f} (element {elem})")
    if args.set is not None:
        if not set_gain(did, args.set, scope):
            print("  set FAILED — device rejected the write")
            return 1
        _, now = get_gain(did, scope)
        print(f"  set -> {now:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
