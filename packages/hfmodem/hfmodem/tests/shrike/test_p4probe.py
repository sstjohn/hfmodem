# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 probes preserve full RX windows and cannot inherit P3 retry behaviour."""
from types import SimpleNamespace
import numpy as np
import pytest

from hfmodem.shrike import onair, p4probe

FS = onair.FS


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_connected_probe_retains_sample_timing(host):
    from hfmodem.core.audio import timestamp_limitation
    from hfmodem.shrike import p4breakintx
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    audio = np.zeros(3 * FS)
    start = round(.923 * FS)
    body = p4breakintx.changeover_packet(b'hello', 0)
    audio[start:start+len(body)] = body
    live = Stream(audio)
    live.timestamp_limitation = timestamp_limitation(host)
    live.lat_provenance = 'nominal' if live.timestamp_limitation else 'measured'
    live.driver_offset_samples = 1771 if live.timestamp_limitation else 756
    tx = Tx(live)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    rows = tx.p4_probe.emissions[1:]
    assert rows and rows[0]['word'] == 'X27'
    for row in rows:
        if live.timestamp_limitation:
            assert row['timestamp_limitation'] == live.timestamp_limitation
            assert row['timing_estimated'] is True
            assert row['driver_offset_samples'] == 1771
            assert 'nominal' in row['timing_basis']
        else:
            assert not {'timestamp_limitation', 'timing_estimated'} & row.keys()


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
@pytest.mark.parametrize('error', [False, True])
def test_print_query_retains_sample_timing(host, error, monkeypatch, capsys):
    from hfmodem.core.audio import timestamp_limitation
    limitation = timestamp_limitation(host)
    row = dict(index=1)
    if error:
        row['error'] = 'mock reader failure'
    else:
        monkeypatch.setattr(onair.p4oracle, 'lines', lambda row: ['reply at +118 ms'])
    if limitation:
        row.update(timestamp_limitation=limitation, driver_offset_provenance='nominal')
    onair._print_p4_query(row)
    output = capsys.readouterr().out
    if limitation:
        assert limitation in output and 'estimated' in output and 'nominal' in output
    else:
        expected = ('    [p4 query] 1 NOT READ: mock reader failure\n' if error
                    else 'reply at +118 ms\n')
        assert output == expected


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
@pytest.mark.parametrize('error', [False, True])
def test_query_retains_sample_timing(host, error, monkeypatch, capsys):
    from hfmodem.core.audio import timestamp_limitation
    limitation = timestamp_limitation(host)
    probe = p4probe.EntryProbe()
    emission = dict(audio_end=96000, status=0x61)
    if limitation:
        emission.update(timestamp_limitation=limitation, driver_offset_provenance='nominal')
    if error:
        def unread(*args, **kwargs):
            raise ValueError('mock reader failure')
        monkeypatch.setattr(onair.p4oracle, 'query', unread)
    row = onair._p4_query_row(probe, 1, emission, np.zeros(480), headers=False)
    assert row['chirp_end_sample'] == 96000 and row['chirp_end_s'] == 2.0
    assert bool(row.get('error')) == error
    onair._print_p4_query(row)
    output = capsys.readouterr().out
    if limitation:
        assert row['timestamp_limitation'] == limitation
        assert row['timing_estimated'] is True
        assert 'nominal' in row['timing_basis']
        assert limitation in output and 'estimated' in output and 'nominal' in output
    else:
        assert not {'timestamp_limitation', 'timing_estimated'} & row.keys()
        assert 'estimated' not in output and 'nominal' not in output


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(onair, '_load', lambda _: np.zeros(90 * FS, np.float32))
    live = onair._ReplayInput('synthetic')
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live = live
    tx.p4_probe = p4probe.EntryProbe()
    grid = onair._MasterGrid(0, 60000, 0, packet_n=46080, cs_n=5760, d_max_n=6240)
    tx.aim(grid, 0)
    # Any invocation would reproduce the expensive P3 pre-key path's bug.
    tx.sessrx = SimpleNamespace(bridge=lambda _: pytest.fail('P3 bridge during probe'))
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **k: pytest.fail('P3 scan'))
    monkeypatch.setattr(onair, '_scan_previous_window', lambda *a, **k: pytest.fail('P3 scan'))
    tx.send_p4_entry_packet(b'', 0x31)
    return tx, live, grid


@pytest.mark.parametrize('values,headers', [((7,38), ['A','A']), ((7,7,38), ['A','B','A'])])
def test_opening_robust_packets_are_consecutive(setup, values, headers):
    tx, live, grid = setup
    tx.p4_probe = p4probe.EntryProbe(values=values)
    tx.p4_probe.request(0, b'', 0)
    onair._run_p4_probe(tx, live, grid)
    emissions = tx.p4_probe.emissions
    assert [e['value'] for e in emissions] == list(values)
    assert [e['header'] for e in emissions] == headers
    assert [e['slot'] for e in emissions] == list(range(emissions[0]['slot'], emissions[0]['slot'] + len(values)))


def test_interrupt_restores_entry_receiver(setup, monkeypatch):
    tx, live, grid = setup
    receiver = tx.sessrx
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(tx, '_tx', interrupt)
    with pytest.raises(KeyboardInterrupt):
        onair._run_p4_probe(tx, live, grid)
    assert tx.sessrx is receiver and not tx.p4_probe.emitting


def test_stall_sends_the_opening_without_a_silent_window(setup):
    tx, live, grid = setup
    onair._run_p4_probe(tx, live, grid, stalled=12)
    assert [e['value'] for e in tx.p4_probe.emissions] == [7,38]
    assert tx.p4_probe.path == 'stall'
    assert not any(w['kind'] == 'silent listen' for w in tx.p4_probe.windows)


def test_opening_registers_short_iss_emissions(setup):
    from hfmodem.shrike.p4receiver import Kind, Role
    tx, live, grid = setup
    emissions = []
    tx.sessrx.p4 = SimpleNamespace(note_emission=emissions.append, cfo_hz=None,
                                 read_window=lambda *a, **kw: [])
    tx.sessrx.host = SimpleNamespace(arq=None)
    tx.p4_probe.request(0, b'', 0)
    onair._run_p4_probe(tx, live, grid)
    assert len(emissions) == 2
    assert all(e.kind == Kind.SHORT and e.role == Role.ISS for e in emissions)
    assert [e.stream_sample for e in emissions] == [e['audio_start'] for e in tx.p4_probe.emissions]
