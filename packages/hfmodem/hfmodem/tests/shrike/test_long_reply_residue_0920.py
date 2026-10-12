"""VE3KPG 22:42: a delayed long CRC must not strand replies inside the packet."""
from types import SimpleNamespace
import pytest
from hfmodem.shrike import onair, spec


def scene():
    g = onair._MasterGrid(90297, 60000, 4320, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(True)
    g.note_p3_packet(10666816, 158160, 180000, identity=(False, 1, 0x22, b'example'))
    return g


@pytest.mark.parametrize('slot,expected', [(180,182),(181,182),(182,182),(184,185)])
def test_late_long_decode_selects_peer_reply_residue(slot, expected):
    g = scene()
    assert g.p3_reply_slot(slot) == expected
    assert g.p3_reply_slot(expected) == expected
    # Whole-slot correction preserves the precise pulse/audio epoch.
    assert g.anchor == 90297
    phase, width, period, _ = g._p3_peer
    gap = (g.boundary(expected) - phase) % period
    assert width < gap < period - 10080


def test_wrong_residue_latch_is_rechecked_on_corrected_slot():
    g = scene()
    g._p3_reply_phase_invalid = True
    aims = []
    tx = SimpleNamespace(p3_reply_shift=lambda slot: None,
                         aim=lambda grid,slot: aims.append(slot))
    assert 'remains invalid' in g.p3_control_refusal(181)
    slot = onair._p3_place_reply(g,tx,181)
    assert slot == 182 and aims == [182]
    assert g.p3_control_refusal(slot) is None
    # No reset when no cycle-residue repair occurred.
    g._p3_reply_phase_invalid = True
    assert onair._p3_place_reply(g,tx,182) == 182
    assert 'remains invalid' in g.p3_control_refusal(182)


@pytest.mark.parametrize('kind', ['sending','short','stale','mismatch'])
def test_unrelated_or_stale_clocks_cannot_rephase(kind):
    g = scene()
    if kind == 'sending':g.sending=True
    if kind == 'short':g.regear(False)
    if kind == 'stale':g.cycles += onair.ONSET_MAX_CYCLES+1
    if kind == 'mismatch':g._p3_peer=(10666816,38880,60000,g.cycles)
    g._p3_reply_phase_invalid=True
    tx = SimpleNamespace(p3_reply_shift=lambda slot: None,aim=lambda *args: None)
    assert onair._p3_place_reply(g,tx,181)==181
    assert g._p3_reply_phase_invalid


def test_recorded_wrong_slot_recovers_through_actual_control_emission(tmp_path):
    from hfmodem.shrike.p3trial import ReplyClock
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, key_ack
    g = scene()
    tx = driver(g,tmp_path)
    tx.live.limit = 300 * onair.FS
    tx.reply_clock = ReplyClock(entry_phase=1, pulse_epoch=1831149)
    tx.aim(g,181)
    g._p3_reply_phase_invalid = True
    slot = onair._p3_place_reply(g,tx,181)
    assert slot == 182 and g.anchor == 90297
    key_ack(tx,g,slot)
    assert tx.n == 1 and not tx.refused
    assert tx.tx_audio_start == g.boundary(slot)
    assert g.key_refusal(tx.tx_key_up,tx.tx_end-tx.tx_key_up) is None
