# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Operator-visible P3 traffic. Observations only; never drives the ARQ.

TX retries count emitted copies, excluding refused slots. RX repeats count
decoded copies, not transmissions inferred through an unreadable interval.
Control replies have their own counter so they cannot reset packet retries.
"""
from . import spec


class TrafficLog:
    def __init__(self):
        self._copies = {}
        self._direction = None
        self.levels = {}
        self.requested_rx_level = None
        self._request_reported = False
        self.generations = {"RX": 0, "TX": 0}

    def _copy(self, stream, identity):
        old, count = self._copies.get(stream, (None, 0))
        count = count + 1 if identity == old else 1
        self._copies[stream] = (identity, count)
        return count

    def packet(self, direction, sl, status, payload, *, kind="DATA",
               long_cycle=False, crc_ok=True, information=None, field=None):
        payload = bytes(payload)
        # A direction change starts a new packet-number space. A control reply
        # does not change direction. Speed changes may retransmit the same data.
        if crc_ok and self._direction != direction:
            self._copies.clear()
            self._direction = direction
            if direction == "TX":
                self.requested_rx_level = None
        copy = self._copy(direction, (kind, status, payload)) if crc_ok else None
        if copy == 1:
            self.generations[direction] += 1
        counter = (f"attempt={copy} retry={copy-1}" if direction == "TX" else
                   f"copy={copy} repeat={copy-1}") if copy else "unvalidated"
        geometry = ("CHANGEOVER fixed 2-carrier" if kind == "CHANGEOVER" else
                    f"SL{sl} {kind}")
        cycle = "long/3.75s" if long_cycle else "short/1.25s"
        crc = ("CRC-ENCODED" if direction == "TX" else "CRC-VALID") if crc_ok else "CRC-FAILED"
        lines = [f"    [traffic] {direction} P3 {geometry} {cycle} "
                 f"seq={status & 3} type={(status >> 2) & 7} "
                 f"status=0x{status:02x} {counter} [{crc}]",
                 f"    [traffic] {direction} payload {len(payload)}B "
                 f"hex={payload.hex()} bytes={payload!r}"]
        if field is not None:
            lines.append(f"    [traffic] {direction} field "
                         f"(data+fill+status+CRC, before FEC) hex={field.hex()}")
        elif information is not None:
            lines.append(f"    [traffic] {direction} information "
                         f"(data+fill+status, before CRC) hex={information.hex()}")
        # Changeover has fixed geometry; its synthetic SL1 does not prove the
        # ordinary data sender changed level.
        if crc_ok and kind != "CHANGEOVER":
            previous = self.levels.get(direction)
            self.levels[direction] = sl
            if previous != sl:
                old = "unknown" if previous is None else f"SL{previous}"
                lines.append(f"    [speed] {direction} data {old} -> SL{sl} observed")
            if direction == "RX" and self.requested_rx_level is not None:
                target = self.requested_rx_level
                if sl >= target:
                    lines.append(f"    [speed] requested RX SL{target}; "
                                 f"received SL{sl}: speed-up confirmed")
                    self.requested_rx_level = None
                elif not self._request_reported:
                    lines.append(f"    [speed] requested RX SL{target}; "
                                 f"next decoded data is SL{sl}: speed-up unconfirmed")
                    self._request_reported = True
        return lines

    @staticmethod
    def control_meaning(direction, ci, arq=None):
        if (direction == "RX" and ci in (3, 4) and arq is not None
                and getattr(arq.io, "p3_tx_gear_hold", False)
                and getattr(arq.io, "protocol", None) == spec.Protocol.PACTOR3
                and arq.tx_seq is not None
                and getattr(arq, "_p3_tx_gear_command", None) == ci):
            return f"HELD {spec.CS_NAMES[ci]} / REPEAT seq={arq.tx_seq} (trial)"
        if ci not in (0, 1):
            return spec.CS_NAMES[ci]
        if arq is None:
            return "ACK/REPEAT (packet counter unknown)"
        if direction == "TX":
            seq = arq.rx_seq
            if getattr(arq, "_rx_seen", False) and ci == (seq & 1):
                return f"ACK seq={seq}"
            return f"REPEAT/await seq={(seq + 1) % 4}"
        seq = arq.tx_seq
        if seq is None:
            return "ACK/REPEAT (no outstanding TX packet)"
        return f"{'ACK' if ci == (seq & 1) else 'REPEAT'} seq={seq}"

    def terminal(self, header_bit):
        copy = self._copy("TX terminal", header_bit)
        return [f"    [traffic] TX P3 TERMINAL marker VH{header_bit} "
                f"attempt={copy} retry={copy-1} fixed 2-carrier 100Bd "
                "payload=0B (marker has no byte field or CRC)"]

    def control(self, direction, ci, arq=None):
        word = spec.CONTROL_SIGNALS[ci]
        bits = "".join(str((word >> bit) & 1) for bit in range(20))
        meaning = self.control_meaning(direction, ci, arq)
        # Same codeword can ACK seq 1 and seq 3. The accepted-packet generation
        # separates those replies even when intermediate replies did not key.
        copy = self._copy(direction + " control",
                          (ci, meaning, self.generations["RX" if direction == "TX" else "TX"]))
        decoded = "decoded-" if direction == "RX" else ""
        lines = [f"    [traffic] {direction} P3 CS{ci+1} {meaning} "
                 f"reply-copy={copy} reply-repeat={copy-1} "
                 f"100Bd DBPSK ch5+ch12 {decoded}word=0x{word:05X} "
                 f"{decoded}bits-lsb-first={bits} payload=0B"]
        if ci == 3:
            if direction == "TX":
                sl = self.levels.get("RX")
                self.requested_rx_level = sl + 1 if sl is not None and sl < 6 else None
                self._request_reported = False
                target = (f"SL{sl} -> SL{sl+1}" if self.requested_rx_level
                          else "next level (current RX level unknown or maximum)")
                lines.append(f"    [speed] TX CS4 requests peer data {target}; "
                             "awaiting a decoded packet at the new level")
            elif meaning.startswith("HELD "):
                lines.append("    [speed] RX held CS4: trial retains the pending "
                             "TX packet and speed; no additional bytes acknowledged")
            else:
                lines.append("    [speed] RX CS4 asks us to increase TX data speed; "
                             "subsequent TX packet logs show the emitted level")
        return lines


def for_host(host):
    """Share TX/RX observations without confusing the ARQ's TX level with RX."""
    log = getattr(host, "_p3_traffic_log", None)
    if log is None:
        log = TrafficLog()
        host._p3_traffic_log = log
    return log


def received(host, event):
    if event.protocol != "PACTOR-3":
        return
    log = for_host(host)
    if event.packet is not None:
        sl, status, payload, ok = event.packet
        lines = log.packet("RX", sl, status, payload,
                           kind="CHANGEOVER" if event.breakin else "DATA",
                           long_cycle=event.cycle_long, crc_ok=ok,
                           information=event.information)
    elif event.cs is not None:
        lines = log.control("RX", event.cs, host.arq)
    else:
        return
    emit(lines)


def emit(lines):
    print("\n".join(lines), flush=True)
