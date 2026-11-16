# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.shrike import arq, compress, spec


# The binary tail contains escape bytes, so a text-only prefix cannot prove
# preservation of the wire suffix when retries give way to a goodbye.
PENDING_HOST = bytes.fromhex(
    "3953534a20504143544f5220746573742076696120575338454f4320323032362d30392d"
    "323420303334347a00300002fac74289020000ecfd7e1c7c6d67137dc5f2783a3daebeff"
    "8771c2cce5ba1c7ed00f8e160dfef2a5b6eeb6fab2bf02ad1a346765dada5e02ff2b817e"
    "e7fb100a7841325f4a73f7f9a557c723bf21d68a1b957de838b0e75fffd900fe3751163f"
    "677f4fd733e728e1bc342fbe0b19b32343fba12c1c7cda5f85edaee6f3e2f6fbcdba56b8"
    "c6e5debe2489b8176d4d7882cf735249cb21674416f0c0ae768e4c38947ce030e817118a"
    "f82594a2648811cd0c8063e18fbe6115b86911671551e0c99e6950d81f701c7e22a503c9"
    "197152471b1c7de09fa6b1f16f3fbf807a7e2f6ccb090c363869f41b1234159287df3ab3"
    "0bbea2d122e2278d96400465df3fc98402e3e169f17401aeb11aff446e483dbefdf7138b"
    "bbf7009e5dccfde9b593b310668c848ab3c0487bfe7ed9ef2fdf096ebf38e9e8b7e748fd"
    "07ff69ce490e6b2e8d56bb1c7e9e5d5434c67d8b3ccb1c7eb3135bc91c7d7bc1259d30de"
    "d5eb913cbb61bb9b50185db86ce51057dcd1513343ffe7b6d6f6b999f4e8c807c19e34ea"
    "887f16619adb910677a25d6bdbba9971c4fa75d5692670e725dc2f818ccbd189b9397b8b"
    "66610931e1e9aa4bec34acf2982fc31271d4b11c7dda1c7cd77e39e17a196496cb0c73ba"
    "c9e4f20936468f3111766ecd4523e8a0e92a0525891afe16c2fdf1ca47183b04d34c04c8"
)
PENDING_WIRE = compress.transparent(PENDING_HOST)


class PacketOutput(arq.ArqIO):
    protocol = spec.Protocol.PACTOR3
    _uninvited_entry = False

    @property
    def uninvited_entry(self) -> bool:
        return self._uninvited_entry

    def __init__(self):
        self.slot = 114
        self.packets = []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((self.slot, sl, bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse else len(payload)


def _waiting_for_slot_123():
    output = PacketOutput()
    link = arq.PactorArq(output)
    link.role = arq.ISS
    link._enter_connected()
    link._sl = 1
    link._next_seq = 1
    link.on_host_data(PENDING_HOST)
    for slot in range(114, 123):
        output.slot = slot
        link.on_cycle()
        assert output.packets[-1] == (slot, 1, b"9SSJ ", 0x21, False)
    assert link._inflight.retries == link.cfg.max_retries == 8
    output.slot = 123
    return link, output


@pytest.mark.parametrize("refuse", [False, True])
def test_exhausted_iss_offers_goodbye_in_its_current_slot(refuse):
    link, output = _waiting_for_slot_123()
    output.refuse = refuse
    before = len(output.packets)
    link.on_cycle()
    assert output.packets[before:] == [(123, 1, b"", 0x9a, False)]
    assert link._qrt_pending and link._budget_goodbye
    assert link.state == arq.State.DISCONNECTING
    assert link.said_goodbye is not refuse
    output.refuse = False
    output.slot = 124
    link.on_cycle()
    assert output.packets[before + 1:] == [(124, 1, b"", 0x9a, False)]


def test_exhausted_iss_retains_unconfirmed_field_and_suffix():
    link, output = _waiting_for_slot_123()
    link.on_cycle()
    retained = link.transfer_unconfirmed
    assert retained is not None
    assert (retained["counter"], retained["status"], retained["sl"]) == (1, 0x21, 1)
    assert retained["field"] == b"9SSJ "
    assert retained["suffix"] == PENDING_WIRE[5:]
    assert link.tx_acked_bytes == 0
    for slot in range(124, 124 + arq.GOODBYE_PLACE_TICKS):
        output.slot = slot
        link.on_cycle()
    assert retained["field"] + retained["suffix"] == PENDING_WIRE
    assert retained["host_unconfirmed"] == PENDING_HOST
    assert link.unconfirmed_transfers == [retained]


@pytest.mark.parametrize("kind,retained", [
    ("entry", False), ("entry-data", False), ("uninvited-entry", False),
    ("goodbye", False), ("breakin", True), ("p1", False), ("p2", False),
])
def test_nonordinary_retry_exhaustion_keeps_its_existing_close_path(kind, retained):
    output = PacketOutput()
    link = arq.PactorArq(output)
    link.role = arq.ISS
    link._enter_connected()
    if kind in ("entry", "entry-data"):
        link.entry_pending = True
        if kind == "entry-data":
            link.cfg.entry_ladder = ("data",)
    elif kind == "uninvited-entry":
        output._uninvited_entry = True
    elif kind == "goodbye":
        link.on_host_disconnect()
    elif kind in ("p1", "p2"):
        output.protocol = (spec.Protocol.PACTOR1 if kind == "p1"
                           else spec.Protocol.PACTOR2)
    link._start_next_packet(breakin=kind == "breakin")
    link._inflight.retries = link.cfg.max_retries
    before = list(output.packets)
    link._on_nak(speed_down=False, answered=False)
    assert output.packets == before
    assert len(link.unconfirmed_transfers) == int(retained)


def test_receiving_giveup_does_not_add_an_ordinary_data_snapshot():
    output = PacketOutput()
    link = arq.PactorArq(output)
    link.role = arq.IRS
    link._enter_connected()
    link.on_host_data(b"waiting for our turn")
    link._give_up("receive budget exhausted")
    assert link._qrt_pending and link._breakin_pending
    assert link.unconfirmed_transfers == []
