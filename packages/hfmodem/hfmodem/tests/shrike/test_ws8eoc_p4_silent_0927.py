# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)


import numpy as np
import pytest

from hfmodem.tests.evidence import CAPTURES
from hfmodem.shrike import onair, p4breakintx
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


class Live(Stream):
    key_notice = 1652
    transmit = True

    def clamp_late(self, at):
        return max(0, self.sample_now() + self.key_notice - at)


def grid():
    return onair._MasterGrid(2634, 60000, 0, packet_n=46080,
                             cs_n=12000, d_max_n=4800)


def test_live_notice_is_not_spent_twice(monkeypatch):
    audio = np.zeros(3 * FS)
    y = round(.923 * FS)
    body = p4breakintx.changeover_packet(b'RMS Trimode 1.4.4.0\r\nW9SSJ has ', 0)
    audio[y:y + len(body)] = body
    live = Live(audio)
    tx = Tx(live)
    tx.p4_link_options.p4_link_timeout = 2.1
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    onair._run_p4_link(tx, live, grid())
    sent = tx.p4_probe.emissions[1:]
    assert len(sent) == 1, tx.p4_probe.link_events
    assert sent[0]['word'] == 'X27'
    assert abs(sent[0]['audio_start'] - (y + round(.920 * FS))) <= 1


def hold(monkeypatch):
    path = CAPTURES / 'onair-0927-1547/stream.flac'
    if not path.exists():
        pytest.skip('local WS8EOC September 27 capture absent')
    audio = onair.session.load_wav(str(path), FS)
    live = Live(audio)
    live.pos = round(12.51 * FS)
    tx = Tx(live)
    tx.slot, tx.tx_end = 11, 588714
    tx.p4_probe.emissions.clear()
    tx.p4_probe.requested = False
    tx.p4_link_options.p4_data = 'body'
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    tx.sessrx.host = host
    start = round(12.10 * FS)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    heard = onair._p4_hold_word(tx, host, live, grid(), audio[start:live.pos], start, 588714)
    assert heard == 9
    return tx, live, heard


def test_local_hold_grant_passes_live_admission(monkeypatch):
    tx, live, heard = hold(monkeypatch)
    tx.p4_link_options.p4_link_timeout = 1.2
    onair._run_p4_probe(tx, live, grid(), heard=heard)
    first = next((e for e in tx.p4_probe.link_events if e['event'] == 'Keyed'), None)
    assert first is not None, tx.p4_probe.link_events
    assert first['kind'] == 'word' and first['word'] == 'X27'
    assert abs(first['at'] - 631897) <= 96
    assert tx.p4_probe.emissions[0]['word'] == 'X27'
    assert not any(e.get('reason') == 'deadline' for e in tx.p4_probe.link_events)


def test_local_refinement_keeps_each_body_in_its_own_cycle(monkeypatch, capsys):
    tx, live, heard = hold(monkeypatch)
    tx.p4_link_options.p4_link_timeout = 120
    admission = onair._keyable_slot
    monkeypatch.setattr(onair, '_keyable_slot',
                        lambda live, raster, slot, lead, **kw:
                        admission(live, raster, slot, 0, **kw))
    monkeypatch.setattr(tx, '_tx', lambda *a: setattr(tx, 'refused', True))
    receiver = tx.sessrx.p4
    onair._run_p4_probe(tx, live, grid(), heard=heard)
    unread = [e['at'] for e in tx.p4_probe.link_events if e['event'] == 'BREAKIN_UNREAD']
    assert not any(abs(at - target) <= 96 for at in unread for target in (1127734, 1607735))
    bodies = [o.stream_sample for o in receiver.observations if o.kind == 'body']
    for onset in np.arange(12.2445, 42.245 + .01, 1.25):
        assert any(abs(at / FS - onset) < .005 for at in bodies), (onset, bodies)
    assert '[p4 link] Refused admission' in capsys.readouterr().out
