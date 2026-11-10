# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Mains to the transceiver, on the station's own smartplug.

WHY THE STATION HAS A POWER SWITCH AT ALL. The ATU hunting a match throws enough
RFI to take the rig's CAT interface down with it, and only a mains cycle brings
the interface back. The tuner keeps its stored matches across the cut, so nothing
has to be re-found.

AND THE HUNT CANNOT BE PREDICTED OR AVOIDED. The tuner recalls a stored match per
segment, but it starts a fresh hunt whenever the recall lands above about 1.8:1
-- and what moves the SWR is the end-fed wire moving, so a breeze that stretches
it or ice on it is enough. Any channel, any time, including one the tuner has
worked a hundred times. This is a standing condition of the station and is the
reason the plug is on the mains in the first place; the recoveries it logs are
also the only record anything here keeps of when the antenna moved.

THE PLUG IS A KP115 SPEAKING TP-LINK'S LOCAL PROTOCOL: obfuscated JSON on 9999,
an XOR autokey from 171, a four-byte length ahead of it on TCP and none on UDP.
No credential and no cloud are in the path, and the whole client is below.

ITS ADDRESS IS DHCP AND IS NEVER WRITTEN DOWN. Every lookup ends at a device that
answered with this alias and this MAC; the cached address is a shortcut past the
broadcast and not a claim about who is there, so it is checked like any other.

TWO MEASUREMENTS OFF THIS PLUG SHAPE EVERYTHING BELOW. The meter lags the relay:
the first reading after the relay opens still carries the old wattage and takes a
couple of seconds to fall, so a check that reads once and immediately reports a
load that is already dead. And switch-on draws 1101 mA of inrush against 306 mA
(21 W) receiving, so a cycle holds the load at zero rather than flipping the
relay twice in a row.

WHAT THE METER CAN AND CANNOT WITNESS. It sees a carrier: a 50 W transmission is
an unmistakable step above the 21 W the rig draws receiving, and that is a real
reading with no CAT traffic behind it. It cannot see a KEY. A rig keyed with no
audio drive makes no RF and draws about what it draws receiving, so a quiet meter
is not evidence that the key is up and must never be read as one. `cycle` refuses
a load that is making RF -- hot-switching the finals is how a radio dies -- and
that refusal is a refusal only. Nothing here clears a radio to be switched.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
from .operator import data_root, settings

ALIAS = None
MAC = None
PORT = 9999
KEY = 171

ZERO_W = 0.5        # the meter reads a hair above nothing with the relay open
IDLE_W = 21.0       # the transceiver receiving, per the plug's own meter
CARRIER_W = 40.0    # above this the rig is making RF; below it says nothing at all
FELL_S = 8.0        # how long the load is given to read zero once the relay opens
DOWN_S = 5.0        # how long it is held there before power comes back
SETTLE_S = 6.0      # how long switch-on is given to fall from inrush towards idle
POLL_S = 0.5


def obfuscate(text):
    key, out = KEY, bytearray()
    for byte in text.encode():
        key ^= byte
        out.append(key)
    return bytes(out)


def clarify(data):
    key, out = KEY, bytearray()
    for byte in data:
        out.append(key ^ byte)
        key = byte
    return out.decode(errors='replace')


@dataclass(frozen=True)
class Reading:
    """One look at the plug: the relay's own state and what the meter made of it."""
    relay: int
    watts: float
    volts: float
    milliamps: int

    def spell(self):
        return (f'relay {"on" if self.relay else "off"}, {self.watts:.1f} W, '
                f'{self.volts:.1f} V, {self.milliamps} mA')


@dataclass
class Plug:
    address: str

    def ask(self, payload, timeout=5.0):
        body = obfuscate(json.dumps(payload))
        with socket.create_connection((self.address, PORT), timeout) as link:
            link.sendall(struct.pack('>I', len(body)) + body)
            buffer = b''
            while len(buffer) < 4 or len(buffer) - 4 < struct.unpack('>I', buffer[:4])[0]:
                chunk = link.recv(4096)
                if not chunk:
                    raise OSError(f'the plug at {self.address} closed the connection mid-answer')
                buffer += chunk
        return json.loads(clarify(buffer[4:]))

    def sysinfo(self):
        return self.ask({'system': {'get_sysinfo': {}}})['system']['get_sysinfo']

    def reading(self):
        meter = self.ask({'system': {'get_sysinfo': {}},
                          'emeter': {'get_realtime': {}}})
        live = meter['emeter']['get_realtime']
        return Reading(relay=meter['system']['get_sysinfo']['relay_state'],
                       watts=live['power_mw'] / 1000.0, volts=live['voltage_mv'] / 1000.0,
                       milliamps=live['current_ma'])

    def switch(self, state):
        answer = self.ask({'system': {'set_relay_state': {'state': int(state)}}})
        failed = answer['system']['set_relay_state'].get('err_code')
        if failed:
            raise OSError(f'the plug at {self.address} refused the relay ({answer})')
        return self.reading()

    def until(self, settled, ceiling_s):
        """Read the meter until it agrees with the relay, and say how long that took.

        The meter lags, so every reading before the last one is evidence of the
        lag and not of the load: what is returned is the first reading that
        satisfied the caller, or the last one taken when the ceiling ran out.
        """
        started = time.monotonic()
        while True:
            seen, spent = self.reading(), time.monotonic() - started
            if settled(seen) or spent >= ceiling_s:
                return seen, spent
            time.sleep(POLL_S)


def named(info):
    return info.get('alias'), info.get('mac') or info.get('mic_mac')


def identified(info, alias=ALIAS, mac=MAC):
    seen_alias, seen_mac = named(info)
    return bool(mac and (seen_mac or '').upper() == mac.upper()
                and (not alias or seen_alias == alias))


def broadcast(timeout=3.0, alias=ALIAS, mac=MAC):
    """Every plug on the segment that answers to this alias or MAC, nearest first."""
    probe = obfuscate(json.dumps({'system': {'get_sysinfo': {}}}))
    found = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        sock.sendto(probe, ('255.255.255.255', PORT))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data, (address, _) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                info = json.loads(clarify(data))['system']['get_sysinfo']
            except (ValueError, KeyError):
                continue
            if identified(info, alias, mac) and address not in found:
                found.append(address)
    return found


def find(alias=ALIAS, mac=MAC, cache=None):
    """The plug, by who it says it is and never by where it was last seen.

    The cached address is tried first because it is one round trip against the
    three seconds a broadcast waits out -- but what makes it the plug is the
    alias and the MAC in its own answer, so a lease that moved to a neighbour's
    lamp falls through to the broadcast rather than switching the lamp.
    """
    global ZERO_W, IDLE_W, CARRIER_W
    configured = settings().get('power', {})
    mac = mac or configured.get('mac')
    alias = alias or configured.get('alias')
    import math
    import re
    if not isinstance(mac, str) or not re.fullmatch(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', mac):
        raise ValueError('Power control needs an explicit power.mac in operator.toml; nothing contacted.')
    values = [configured.get(k) for k in ('zero_w', 'idle_w', 'carrier_w')]
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            or not 0 <= values[0] < values[1] < values[2]):
        raise ValueError('Configure measured power limits: 0 <= zero_w < idle_w < carrier_w; nothing contacted.')
    ZERO_W, IDLE_W, CARRIER_W = values
    cache = cache or data_root() / '.plug-address'
    remembered = cache.read_text().strip() if cache.is_file() else ''
    if remembered:
        plug = Plug(remembered)
        try:
            if identified(plug.sysinfo(), alias, mac):
                return plug
        except (OSError, ValueError, KeyError):
            pass
    for address in broadcast(alias=alias, mac=mac):
        plug = Plug(address)
        if not identified(plug.sysinfo(), alias, mac):
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(address + '\n')
        return plug
    raise OSError(f'No smartplug answering to {alias!r} / {mac} on this segment. '
                  'It speaks the legacy local protocol on 9999 and a broadcast found '
                  'nothing; check it is powered and on this network.')


def status(plug=None):
    return (plug or find()).reading()


def on(plug=None):
    """Close the relay and read until the load has fallen out of inrush towards idle.

    A load settling to idle is a rig that powered up, which is worth having as
    corroboration -- but it is not a rig that is operable, and nothing here
    pretends otherwise. What answers that is a dial query.
    """
    plug = plug or find()
    plug.switch(1)
    return plug.until(lambda read: ZERO_W < read.watts <= CARRIER_W, SETTLE_S)


def off(plug=None):
    """Open the relay and read until the load actually reads zero."""
    plug = plug or find()
    plug.switch(0)
    seen, spent = plug.until(lambda read: read.watts <= ZERO_W, FELL_S)
    return seen, spent


def cycle(plug=None, hold_s=DOWN_S):
    """Mains off, proved dead, held down, and back on. Yields what each step measured.

    IT REFUSES A RADIO THAT IS MAKING RF, and that is the whole of what the meter
    can say. Cutting mains to a transmitting radio hot-switches the finals, and a
    carrier is a fifth again of what the rig draws receiving, so the meter catches
    one. It cannot catch a key: a rig keyed with no drive makes no RF and draws
    what receive draws. Nothing below reads a quiet meter as permission to switch
    -- what makes the recovery path safe is that the arm it recovers has already
    died before keying anything.

    POWER COMES BACK EVEN WHEN THE LOAD NEVER READ ZERO. A meter that still shows
    watts after the relay opened means the relay did not open or the rig is fed
    from somewhere else -- neither is improved by leaving the station dark, so the
    relay is restored and the caller is told what the meter said.
    """
    plug = plug or find()
    before = plug.reading()
    if before.watts > CARRIER_W:
        raise RuntimeError(f'the rig is drawing {before.watts:.1f} W against {IDLE_W:.0f} W '
                           'receiving, so it is putting out a carrier -- and cutting mains '
                           'to a transmitting radio hot-switches the finals. Nothing '
                           'switched.')
    yield f'  power: {before.spell()} before the cut'
    try:
        dead, fell_s = off(plug)
        yield (f'  power: mains off -- {spell(dead, fell_s)}'
               + ('' if dead.watts <= ZERO_W else
                  ' -- THE LOAD NEVER READ ZERO, so the cut did not reach the rig'))
        if dead.watts <= ZERO_W:
            time.sleep(hold_s)
            yield f'  power: held down {hold_s:.0f}s with the load at zero'
        live, settle_s = on(plug)
    finally:
        plug.switch(1)      # whatever failed above, the station is not left dark
    yield f'  power: mains on -- {spell(live, settle_s)}'
    if dead.watts > ZERO_W:
        raise RuntimeError(f'the load still read {dead.watts:.1f} W {fell_s:.1f}s after the '
                           'relay opened, so the rig was never off. Power is back on.')


def spell(reading, spent=None):
    return reading.spell() + ('' if spent is None else f' after {spent:.1f}s')


def report(action):
    """`console power <action>`: one line a step, and the plug's own measurements in it."""
    plug = find()
    if action == 'status':
        print(f'{plug.address}  {status(plug).spell()}')
        return 0
    if action == 'cycle':
        for line in cycle(plug):
            print(line, flush=True)
        return 0
    seen, spent = (on if action == 'on' else off)(plug)
    print(f'{plug.address}  {spell(seen, spent)}')
    return 0
