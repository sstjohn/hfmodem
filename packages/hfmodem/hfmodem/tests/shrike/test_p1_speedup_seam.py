# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The CS4 speed-up consumes an ACK position, including after a changeover.

hf-pactor tx_100_cs1 -> tx_100_200_cs2 confirms the first 200 Bd block on
CS1; CS2 declines the trial without accepting that block. The opposite phase
is its mirror. Real K0NTS/KB5LZK PCM reached this branch on 2026-09-10.
"""
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, pactor1, ptc, rxfront
from hfmodem.shrike.arq import IRS, ISS, State
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_p1peer import ACCEPT_FCS, _fcs, peer_read

PAYLOAD = bytes(range(65, 165))
SPEEDUP_PATH = Path(__file__).with_name("fixtures") / "p1-speedup.json"

class Peer:
    def __init__(self):
        self.frames = []

    def send_p1_packet(self, payload, baud, count, **kwargs):
        packet = pactor1.data_packet(payload, baud, count, **kwargs)
        self.frames.append((count, baud, packet[0], payload))

    def send_p1_breakin(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, "BK", payload))

    def __getattr__(self, name):
        return lambda *a, **kw: None


def after_breakin(peer=None, payload=PAYLOAD):
    peer = Peer() if peer is None else peer
    host = ptc.PtcHost(peer=peer, mycall="W9SSJ")
    host.stay_in_pactor1 = True
    host.arq.on_host_listen(True)
    host.arq.on_rx_connect("K0NTS", "W9SSJ")
    host.arq.on_host_data(payload)
    host.arq.on_host_breakin()
    host.on_rx_event(rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                                  packet=(0, 0, b"prompt\r", True)))
    host.arq.on_cycle()
    assert peer.frames[-1] == (0, 100, "BK", payload[:7])
    return host, peer


def answer(host, cs):
    host.on_rx_event(rxfront.Event(0, "cs", "", protocol="PACTOR-1", cs=cs))
    host.arq.on_cycle()


@pytest.mark.skipif(not SPEEDUP_PATH.exists(),
                    reason=f"the recorded speed-up answers {SPEEDUP_PATH.name} "
                           "are not installed")
@pytest.mark.parametrize("station", ["k0nts", "kb5lzk"])
def test_recorded_speedup_ack_advances_packet_two(station):
    host, peer = after_breakin()
    rows = json.loads(SPEEDUP_PATH.read_text())
    for row in (r for r in rows if r["station"] == station):
        ev = onair._SessionRx._p1_cs(recorded_pcm(row), row["anchor"])
        assert ev is not None and ev.cs == row["cs"]
        host.on_rx_event(ev)
        host.arq.on_cycle()
    # BK carried bytes0..6, #1 carried7..14, #2 carried15..34. Its real
    # recorded ACK must send the NEXT twenty bytes, not retransmit #2.
    assert peer.frames[-1] == (3, 200, 0x55, PAYLOAD[35:55])
    assert host._hispeed_tries == 0


@pytest.mark.parametrize("prior", [pactor1.CS_ACK_A, pactor1.CS_ACK_B])
def test_speedup_ack_and_decline_are_opposite_phases(prior):
    for accepted in (True, False):
        host, peer = after_breakin()
        answer(host, pactor1.CS_ACK_A)
        if prior == pactor1.CS_ACK_B:
            answer(host, prior)
        answer(host, pactor1.CS_SPEED)
        trial = peer.frames[-1]
        answer(host, prior if accepted else prior ^ 1)
        got = peer.frames[-1]
        if accepted:
            assert got[0] == (trial[0] + 1) % 4 and got[1] == 200
            assert got[3] != trial[3]
        else:
            assert got[0] == trial[0] and got[1] == 100
            assert got[3] == trial[3][:8]
            assert got[2] == trial[2]


def test_header_phase_persists_across_new_packets_and_retries():
    host, peer = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    first = peer.frames[-1]
    answer(host, pactor1.CS_ACK_A)  # repeat of first ordinary packet
    assert peer.frames[-1] == first
    for cs in (pactor1.CS_ACK_B, pactor1.CS_ACK_A, pactor1.CS_ACK_B):
        answer(host, cs)
    assert [f[:3] for f in peer.frames[-3:]] == [
        (2, 100, 0xAA), (3, 100, 0x55), (0, 100, 0xAA)]


def test_speeddown_reject_resets_header_then_alternates():
    host, peer = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    answer(host, pactor1.CS_SPEED)
    answer(host, pactor1.CS_ACK_A)  # first 200 Bd packet accepted
    answer(host, pactor1.CS_ACK_B)  # second 200 Bd packet accepted
    assert peer.frames[-1][:3] == (0, 200, 0xAA)
    answer(host, pactor1.CS_SPEED)  # true speed-down, counter0 retained
    assert peer.frames[-1][:3] == (0, 100, 0x55)
    answer(host, pactor1.CS_ACK_A)
    assert peer.frames[-1][:3] == (1, 100, 0xAA)


def test_new_contact_resets_header_phase():
    host, peer = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    answer(host, pactor1.CS_SPEED)
    assert host._p1_header_inverted and host._p1_speedup_pending
    host.arq.on_host_abort()
    assert not host._p1_header_inverted and not host._p1_speedup_pending
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    answer(host, pactor1.CS_SPEED)  # a fresh CS4 connect answer, not speed-up
    assert peer.frames[-1][:3] == (1, 100, 0xAA)


def test_a_cs4_on_a_torn_down_link_keys_nothing_and_claims_nothing():
    """`_end_link` leaves exactly the state that reads the next codeword as an
    offer: `p1_baud` 100, `_prev_rx_cs` None, `_p1_speedup_armed` True. This seam
    runs on every decoded control signal, and an arm goes on observing after
    teardown, so a stray CS4 off the channel used to write
    `link rises to 200 Bd` under a `** LINK DOWN **` -- ve1yz-observe2 line 352,
    kb5lzk-observe line 216, and the 2026-09-16 sense arm at line 280. Nothing is
    keyed either way; what it costs is a transcript that asserts a speed change
    the session never made.
    """
    host, _ = after_breakin()
    host.arq.on_host_disconnect()
    for _ in range(40):
        if host.arq.state not in (State.CONNECTED, State.DISCONNECTING):
            break
        host.arq.on_cycle()
    assert host.arq.state is not State.CONNECTED and host.p1_baud == 100
    assert host._p1_speedup_armed and host._prev_rx_cs is None
    said = []
    host.log = said.append
    answer(host, pactor1.CS_SPEED)
    assert host.p1_baud == 100
    assert not any("200 Bd" in line for line in said), said


def test_changeover_ends_the_speed_trial():
    host, _ = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    answer(host, pactor1.CS_SPEED)
    assert host._p1_speedup_pending
    answer(host, pactor1.CS_CHANGEOVER)
    assert not host._p1_speedup_pending and host._hispeed_tries == 0
    assert host.p1_baud == 200


@pytest.mark.parametrize("crc_ok", [True, False])
def test_changeover_packet_ends_trial_only_if_accepted(crc_ok):
    host, _ = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    answer(host, pactor1.CS_SPEED)
    host.on_rx_event(rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                                  breakin=True,
                                  packet=(0, 0, b"new peer turn", crc_ok)))
    assert host.arq.role == (IRS if crc_ok else ISS)
    assert host._p1_speedup_pending is not crc_ok
    assert host.p1_baud == 200


@pytest.mark.parametrize("protocol", [ptc.Protocol.PACTOR2, ptc.Protocol.PACTOR3])
def test_leaving_pactor1_ends_speed_trial_without_resetting_header(protocol):
    host, _ = after_breakin()
    answer(host, pactor1.CS_ACK_A)
    answer(host, pactor1.CS_SPEED)
    host.protocol = protocol
    assert not host._p1_speedup_pending and host._hispeed_tries == 0
    host.protocol = ptc.Protocol.PACTOR1
    assert host._p1_header_inverted  # this is still the same sending turn


class AudioPeer(onair.RadioTx):
    """Keep the real renderer; replace only the final audio/hardware sink."""

    def __init__(self, outdir):
        super().__init__(transmit=False, outdir=outdir)
        self.frames = []
        self.data_audio = []

    def _tx(self, audio, what, **kwargs):
        if what.startswith(("P1 pkt", "P1 BREAK-IN")):
            self.data_audio.append((onair._trim_silence(audio), self._sent_invert))

    def send_p1_packet(self, payload, baud, count, **kwargs):
        result = super().send_p1_packet(payload, baud, count, **kwargs)
        self.frames.append((count, baud, kwargs["header"], payload))
        return result

    def send_p1_breakin(self, payload, baud, count, **kwargs):
        result = super().send_p1_breakin(payload, baud, count, **kwargs)
        self.frames.append((count, baud, "BK", payload))
        return result


def wire_packet(peer):
    """Fixed-origin, fixed-sense tone decisions and an independent bitwise CRC."""
    count, baud, head, _ = peer.frames[-1]
    audio, shift = peer.data_audio[-1]
    assert len(audio) == 46080  # 960 ms at either baud, including changeover
    raw = peer_read(audio, shift=shift, baud=baud)["bytes"]
    nhead = (2 if baud == 100 else 3) if head == "BK" else 1
    assert _fcs(raw[nhead:]) == ACCEPT_FCS
    assert raw[-3] & 3 == count
    if head != "BK":
        assert raw[0] == head
    return raw[nhead:-3].rstrip(bytes([pactor1.IDLE]))


def test_sustained_200_baud_and_second_changeover_render_without_byte_loss(tmp_path):
    payload = bytes(65 + i % 26 for i in range(250))
    host, peer = after_breakin(AudioPeer(tmp_path), payload)
    received = bytearray(wire_packet(peer))
    answer(host, pactor1.CS_ACK_A)
    received.extend(wire_packet(peer))
    answer(host, pactor1.CS_SPEED)
    # An independent peer follows the transmitted counter, including the first
    # 200 Bd ACK after CS4 and the wrap from counter3 to counter0.
    for seq in (2, 3, 0, 1, 2, 3):
        assert peer.frames[-1][:3] == (seq, 200, 0x55 if seq & 1 else 0xAA)
        received.extend(wire_packet(peer))
        answer(host, seq & 1)
    received.extend(wire_packet(peer))  # the peer's BK acknowledges this block
    host.on_rx_event(rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                                  breakin=True,
                                  packet=(0, 0, b"peer took turn", True)))
    assert host.arq.role == IRS
    host.arq.on_host_breakin()
    host.on_rx_event(rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                                  packet=(0, 1, b"peer second packet", True)))
    host.arq.on_cycle()
    assert peer.frames[-1][:3] == (0, 200, "BK")
    received.extend(wire_packet(peer))
    answer(host, pactor1.CS_ACK_A)
    assert peer.frames[-1][:3] == (1, 200, 0x55)
    first = wire_packet(peer)
    answer(host, pactor1.CS_ACK_A)  # lost ACK: repeat keeps count, header, field
    assert peer.frames[-1][:3] == (1, 200, 0x55)
    assert wire_packet(peer) == first
    received.extend(first)
    answer(host, pactor1.CS_ACK_B)
    assert peer.frames[-1][:3] == (2, 200, 0xAA)
    received.extend(wire_packet(peer))
    assert len(received) == 213
    assert received == payload[:213]
