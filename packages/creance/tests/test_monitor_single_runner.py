# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Generated transmissions through passive streaming adapters and stdin."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import resample_poly

from creance.monitor.runners import single_runner as R, lanner_runner as L
from creance.monitor.activity import Detection


def heard(rx, audio, rate):
    x = resample_poly(audio, 48000 // rate, 1) if rate != 48000 else audio
    out = []
    for at in range(0, len(x), 4800):
        out += rx.push(x[at:at + 4800])
    out += rx.flush()
    for row in out:
        Detection.from_json(row)
    return out


@pytest.mark.parametrize('standard,mode', [
    ('110', '1200S'), ('4285', '1200S'), ('4529', '600S'), ('4539', '3200S'),
    ('WBHF', '1200S'), ('39', '600/33/A10.8'), ('16', '1200'), ('FSK', '75W')])
def test_each_bateleur_receiver_decodes_its_waveform(standard, mode):
    from hfmodem.bateleur.modem import Mode
    sent = b'CQ W1AW TEST 73'
    args = R.parser().parse_args(['--protocol', 'test', '--standard', standard, '--mode', mode])
    rx = R.receiver(args)
    audio = np.concatenate([np.zeros(3000), Mode.parse(standard, mode).render(sent), np.zeros(9600)])
    rows = heard(rx, audio, 9600)
    # The raw 16-tone waveform pads this odd-length message to whole symbols.
    expected = sent.decode() + (r'\x00' if standard == '16' else '')
    assert ''.join(r.get('text', '') for r in rows) == expected
    assert any(r['kind'] == 'START' for r in rows)
    assert all(0 <= r['t'] <= len(audio) / 9600 for r in rows)


def test_async_serial_deframer_preserves_characters_across_chunks():
    from hfmodem.bateleur import charframe, mil110
    sent = b'HELLO 73'
    framed = charframe.pack(charframe.frame(sent, charframe.CharFormat.parse('7E1')), 'lsb')
    args = R.parser().parse_args(['--protocol', 'mil110', '--standard', '110', '--framing', '7E1'])
    rows = heard(R.receiver(args), mil110.transmit(framed, '1200S'), 9600)
    assert ''.join(r.get('text', '') for r in rows) == sent.decode()


def test_robust_packet_releases_crc_checked_frames():
    from hfmodem.ax25.frame import ui_frame
    from hfmodem.brubru import tx
    frame = ui_frame('W1AW', 'CQ', b'HELLO MONITOR')
    burst, = tx.transmit([frame])
    rows = heard(R.Packet('rpr'), burst.audio, 9600)
    assert len(rows) == 1
    assert frame.hex() in rows[0]['detail']
    assert rows[0]['grade'] == 'confirmed'


def test_ale_receives_calls_and_message_without_a_controller():
    from hfmodem.bateleur.ale import fsk, words as W
    x = fsk.modulate(W.call('JOE', 'SAM', scan_words=4, message=W.amd_words('HELLO')))
    rows = heard(R.Packet('ale'), x, 8000)
    assert any(r.get('station') == 'SAM' and 'HELLO' in r['detail'] for r in rows)


def test_selcall_receives_a_page_without_acknowledging():
    from hfmodem.bateleur.selcall import fsk, code
    x = fsk.call_audio(code.Call('page', '567890', '123456', text='HELLO'))
    rows = heard(R.Packet('selcall'), x, 8000)
    assert any(r.get('station') == '123456' and 'HELLO' in r['detail'] for r in rows)


def test_sabir_receives_a_beacon():
    from hfmodem.sabir import offair
    from hfmodem.sabir.arq import wire
    from hfmodem.sabir.floor.mfsk import FloorModem
    b = wire.Beacon.build(wire.capabilities(range(3, 8), wire.FASTCTL), 'W1AW', profile=1).pack()
    x = offair.to_real(FloorModem().transmit(b))
    rows = heard(R.Sabir(), np.r_[np.zeros(48000), x, np.zeros(48000)], 48000)
    assert any('W1AW' in r['detail'] and r['grade'] == 'confirmed' for r in rows)


def test_cw_receives_the_selected_audio_tone():
    from hfmodem.core import cwid
    args = R.parser().parse_args(['--protocol', 'cw', '--audio-centre', '1000'])
    x = np.r_[np.zeros(48000), cwid.audio('W1AW', 20, tone_hz=1000), np.zeros(48000)]
    x += np.random.default_rng(44).normal(0, .003, len(x))
    rows = heard(R.receiver(args), x, 48000)
    assert [r['text'] for r in rows] == ['W1AW']


def test_mod110_reports_channel_bits_once_and_does_not_claim_payload_decoding():
    from hfmodem.bateleur import mod110
    sent = np.random.default_rng(42).integers(0, 2, 300, dtype=np.uint8)
    x = np.r_[mod110.synthesize(sent), np.zeros(96000)]
    rows = heard(R.Mod110(), x, 9600)
    assert len(rows) == 1
    assert rows[0]['kind'] == 'CHANNEL-BITS'
    assert 'payload coding unknown' in rows[0]['detail']
    assert np.packbits(sent).tobytes().hex()[:40] in rows[0]['detail']


def test_runner_contract_handles_split_s16le_samples(monkeypatch):
    from hfmodem.bateleur import fsk
    class Split(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 9599))
    audio = resample_poly(fsk.transmit(b'HELLO', '75W'), 5, 1)
    raw = (audio * 20000).astype('<i2').tobytes()
    out = io.StringIO()
    monkeypatch.setattr(R.sys, 'stdin', SimpleNamespace(buffer=Split(raw)))
    monkeypatch.setattr(R.sys, 'stdout', out)
    assert R.main(['--protocol', 'fsk', '--standard', 'FSK', '--mode', '75W']) == 0
    assert ''.join(json.loads(s).get('text', '') for s in out.getvalue().splitlines()) == 'HELLO'


@pytest.mark.parametrize('protocol', ['RTTY', 'SITOR-B', 'NAVTEX'])
def test_lanner_single_protocol_never_runs_the_other_demodulator(monkeypatch, protocol):
    from hfmodem.lanner import rtty, sitor, navtex
    if protocol == 'RTTY':
        audio = rtty.modulate('CQ W1AW 73\n')
        monkeypatch.setattr(sitor, 'transmissions', lambda *a, **k: pytest.fail('SITOR decoder ran'))
    else:
        audio = navtex.modulate(navtex.Message('Q', 'A', 17, 'HELLO'), phasing_pairs=4)
        monkeypatch.setattr(rtty.Stream, 'feed', lambda *a, **k: pytest.fail('RTTY decoder ran'))
        monkeypatch.setattr(rtty.Stream, 'flush', lambda *a, **k: pytest.fail('RTTY decoder ran'))
    raw = (audio * 32767).astype('<i2').tobytes()
    out = io.StringIO()
    monkeypatch.setattr(L.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(L.sys, 'stdout', out)
    assert L.main(['--protocol', protocol, '--search-span', '2500']) == 0
    rows = [json.loads(s) for s in out.getvalue().splitlines()]
    assert rows
    assert {r['protocol'] for r in rows} <= ({'SITOR-B', 'NAVTEX'} if protocol == 'SITOR-B' else {protocol})


@pytest.mark.parametrize('keyed', ['hard', 'shaped'], indirect=True)
def test_rtty_keeps_listening_after_the_signal_moves_in_the_radio_passband(monkeypatch, keyed):
    x = np.r_[keyed('FIRST\n', mark_hz=1085, space_hz=915), np.zeros(3 * 48000),
              keyed('SECOND\n', mark_hz=3985, space_hz=3815)]
    out = io.StringIO()
    monkeypatch.setattr(L.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO((x * 32767).astype('<i2').tobytes())))
    monkeypatch.setattr(L.sys, 'stdout', out)
    assert L.main(['--protocol', 'RTTY', '--search-span', '2500']) == 0
    rows = [json.loads(s) for s in out.getvalue().splitlines()]
    assert ''.join(r['text'] for r in rows) == 'FIRST\nSECOND\n'
    assert rows[0]['centre_hz'] == pytest.approx(1000, abs=10)
    assert rows[-1]['centre_hz'] == pytest.approx(3900, abs=10)


@pytest.mark.parametrize('centre', [350, 3900])
def test_navtex_decodes_at_both_edges_of_a_four_khz_passband(monkeypatch, centre):
    from scipy.signal import hilbert
    from hfmodem.lanner import navtex
    x = navtex.modulate(navtex.Message('Q', 'A', 17, 'FOUR KHZ PASSBAND'), phasing_pairs=4)
    # Translate the generated 1700 Hz pair to the receiver audio frequency.
    x = (hilbert(x) * np.exp(2j * np.pi * (centre - 1700) * np.arange(len(x)) / 48000)).real
    out = io.StringIO()
    monkeypatch.setattr(L.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO((x * 32767).astype('<i2').tobytes())))
    monkeypatch.setattr(L.sys, 'stdout', out)
    assert L.main(['--protocol', 'NAVTEX', '--search-span', '2500']) == 0
    rows = [json.loads(s) for s in out.getvalue().splitlines()]
    assert len(rows) == 1
    assert rows[0]['text'] == 'FOUR KHZ PASSBAND'
    assert rows[0]['centre_hz'] == pytest.approx(centre, abs=10)


@pytest.mark.parametrize('standard,mode', [('FSK', '75V16D'), ('4529', '600S/2400')])
def test_wide_filter_bateleur_subcarriers_decode_without_transmit_limits(standard, mode):
    from hfmodem.bateleur import fsk, s4529
    audio = (fsk if standard == 'FSK' else s4529).transmit(b'WIDE FILTER', mode)
    args = R.parser().parse_args(['--protocol', 'test', '--standard', standard, '--mode', mode])
    rows = heard(R.receiver(args), audio, 9600)
    assert ''.join(r.get('text', '') for r in rows) == 'WIDE FILTER'
