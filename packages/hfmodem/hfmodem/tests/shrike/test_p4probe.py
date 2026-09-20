"""P4 probes preserve full RX windows and cannot inherit P3 retry behaviour."""
from types import SimpleNamespace
import numpy as np
import pytest

from hfmodem.shrike import onair, p4probe

FS = onair.FS


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


def test_grant_queues_no_immediate_transmission(setup):
    tx, live, _ = setup
    assert tx.p4_probe.requested and not tx.slots_used
    tx.send_p4_entry_packet(b'new', 0x32)
    assert tx.p4_probe.status == 0x31 and tx.p4_probe.payload == b''
    tx._tx(np.zeros(100), 'old ARQ retry')
    assert tx.refused and not tx.slots_used


def test_two_chirps_with_complete_receive_windows(setup, capsys):
    tx, live, grid = setup
    grid_before = (grid.protocol, grid.data_n, grid.slot_n)
    onair._run_p4_probe(tx, live, grid)
    probe = tx.p4_probe
    assert len(probe.emissions) == 2
    assert 'final receive window complete' in probe.reason
    for emitted in probe.emissions:
        window = next(w for w in probe.windows if w['start'] == emitted['audio_end'])
        assert window['end'] - window['start'] == 4 * FS
    assert probe.emissions[1]['audio_start'] - probe.emissions[0]['audio_end'] >= 4 * FS
    assert (grid.protocol, grid.data_n, grid.slot_n) == grid_before
    assert not probe.emitting and tx.sessrx is not None
    text = capsys.readouterr().out
    assert 'IS GONE' not in text and 'LATE TO THE KEY' not in text
    assert 'wrong rate' not in text and '106%' not in text


def test_guard_never_stands_down(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(grid, 'key_refusal', lambda *a, **k: 'recorded P1 reply conflict')
    tx.guard_drops = 100  # Previous refusals must not grant permission.
    onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emissions
    assert 'refused' in tx.p4_probe.reason
    assert tx.n == 1


def test_explicit_guard_override(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(grid, 'key_refusal', lambda *a, **k: 'recorded P1 reply conflict')
    tx.qrm_guard = False
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 2


@pytest.mark.parametrize('attempts', [1, 4])
def test_burst_cap(setup, attempts):
    tx, live, grid = setup
    tx.p4_probe.attempts = attempts
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == attempts


def test_deadline_reserves_receive_window(setup):
    tx, live, grid = setup
    tx.p4_probe.timeout = 10
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 1
    assert 'deadline' in tx.p4_probe.reason
    assert live.pos <= 10 * FS


def test_wall_clock_deadline_even_with_stalled_sample_clock(setup, monkeypatch):
    tx, live, grid = setup
    monkeypatch.setattr(p4probe.time, 'monotonic', lambda: tx.p4_probe.grant_wall + 61)
    onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emissions and 'deadline' in tx.p4_probe.reason


def test_interrupt_restores_state(setup, monkeypatch):
    tx, live, grid = setup
    def interrupt(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(tx, '_tx', interrupt)
    with pytest.raises(KeyboardInterrupt):
        onair._run_p4_probe(tx, live, grid)
    assert not tx.p4_probe.emitting and tx.sessrx is not None


def test_truncated_capture_stops(setup):
    tx, live, grid = setup
    live.audio = live.audio[:5 * FS]
    onair._run_p4_probe(tx, live, grid)
    assert len(tx.p4_probe.emissions) == 1
    assert 'capture ended' in tx.p4_probe.reason
