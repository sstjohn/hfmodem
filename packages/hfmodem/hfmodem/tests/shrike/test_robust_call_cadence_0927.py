"""A robust caller keys every cycle while the P4 reader is armed for the link.

onair-0927-1749 (WS8EOC) and -1755 (VE3KPG) called with `--connect robust` and
lost slots every cycle: SLOT n IS GONE by +190-260 ms, REGRID GAVE UP, six calls
in 26 s. The P1-connect arm 1744, same flags otherwise, kept every slot. The
probe arms the P4 reader from the start, and the frame scan ran it over every
call window. A P1 call leaves a 0.207 s window, under `p4oracle.word_scan`'s
0.24 s floor, so only the tail read ran (about 20 ms). The robust call is 80 ms
shorter, its window is 0.287 s, and the full word scan ran on it: 73 ms on the
1749 tape unloaded, against the 30 ms `PREKEY_RESERVE_S`. Each slot handed back
lengthened the next window, and the scan cost grows with it (157-234 ms on 1-1.5 s).

The bench charges those two reads their cost measured on the 1749 tape.
"""
import numpy as np
import pytest

from hfmodem.shrike import onair, p4oracle
from hfmodem.tests.shrike.test_armdefaults import _parsed
from hfmodem.tests.shrike.test_call_cadence import _Clock
from hfmodem.tests.shrike.test_grid import _Rig

SLOT_N = round(1.25 * onair.FS)
SCAN_S_PER_S = 0.2
TAIL_S = 0.020
ARM = ("--after-connect", "p1-robust", "--after-connect-cycles", "6",
       "--p1-robust-stall", "12", "--p1-robust-stall-then", "goodbye",
       "--p1-status-bits45", "2", "--no-p3-fallback", "--p3-traffic-sl", "1",
       "--p3-speed-up", "hold", "--p4-link", "--p4-probe-cadence", "consecutive",
       "--p4-probe-values", "7,38", "--p4-probe-headers", "A,A")


def _calls(tmp_path, monkeypatch, connect):
    clock, calls = _Clock(), []
    monkeypatch.setattr(onair.ota, 'Rig', lambda *a, **kw: _Rig())
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *a, **kw: None)
    monkeypatch.setattr(onair, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(onair, '_LiveInput', lambda *a, **kw: clock)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    scan, tail = p4oracle.word_scan, p4oracle.word_tail

    def charged_scan(audio, fs=onair.FS, *a, **kw):
        if len(audio) >= round(0.24 * fs):
            clock.spend(round(len(audio) * SCAN_S_PER_S))
        return scan(audio, fs, *a, **kw)

    def charged_tail(audio, fs=onair.FS, **kw):
        clock.spend(round(TAIL_S * fs))
        return tail(audio, fs, **kw)

    monkeypatch.setattr(p4oracle, 'word_scan', charged_scan)
    monkeypatch.setattr(p4oracle, 'word_tail', charged_tail)
    emit = onair.RadioTx._tx

    def transmit(tx, audio, what, *a, **kw):
        before = len(tx.keyed)
        emit(tx, audio, what, *a, **kw)
        if len(tx.keyed) > before and what.startswith('connect->'):
            calls.append(tx.tx_audio_start)

    monkeypatch.setattr(onair.RadioTx, '_tx', transmit)
    extra = ("--after-robust", "p4-robust") if connect == "robust" else ()
    onair.run(_parsed('--transmit', '--serial', '/dev/null', '--dial', '7100000',
                      '--outdir', str(tmp_path / 'capture'), '--hold', '2', '--retries', '8',
                      '--max-cycles', '10', '--connect', connect, *ARM, *extra))
    return calls


@pytest.mark.parametrize("connect", ["normal", "robust"])
def test_the_caller_keys_every_cycle_with_the_p4_reader_armed(
        tmp_path, monkeypatch, capsys, connect):
    calls = _calls(tmp_path, monkeypatch, connect)
    out = capsys.readouterr().out
    assert len(calls) == 9, out
    assert set(np.diff(calls)) == {SLOT_N}, np.diff(calls)
    assert "IS GONE" not in out and "REGRID GAVE UP" not in out
