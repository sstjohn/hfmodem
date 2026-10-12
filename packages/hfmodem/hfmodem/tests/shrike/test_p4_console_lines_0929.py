# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The lines a P4 link adds to its transcript so the console can read it as overs.

The decoded bytes of each CRC-valid frame (`[p4 frame]`), the counters that
stand between the link and its next state (`[p4 over]`), a capture loss the
moment the link sees it, the P1 stall's repeat requests and the silent
listen's entry ceiling. The console's reading of them is `tools/consolelib/view.py`.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, p4breakintx
from hfmodem.tests.shrike.test_p1_robust_stall_0926 import STALL, _stalled_arm
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

ROOT = Path(__file__).resolve().parents[5]
GREETING = b'RMS Trimode 1.4.4.0\r\nW9SSJ has '


def screen(text):
    path = require_input(ROOT / 'tools/consolelib/view.py')
    spec = importlib.util.spec_from_file_location('view', path)
    view = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view)
    shown = view.View()
    out = [s for line in text.splitlines(keepends=True) for s in shown.feed(line)]
    return ''.join(out + list(shown.close()))


class Keying(Tx):
    """The fake radio, printing its key the way `RadioTx._tx` does."""

    def _tx(self, audio, what):
        super()._tx(audio, what)
        print(f"  TX[{len(self.slots_used)}] {what}  ({len(audio) / FS:.1f}s)  -- keying")


def grid():
    return onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)


def greeting_link(capsys, stream=None):
    """WS8EOC-80 at 20:34 opened with this field three times over, a cycle apart."""
    audio = np.zeros(9 * FS)
    body = p4breakintx.changeover_packet(GREETING, 0)
    for k in range(3):
        at = round((0.923 + 1.25 * k) * FS)
        audio[at:at + len(body)] = body
    stream = stream or Stream(audio)
    stream.audio = audio
    tx = Keying(stream)
    tx.p4_link_options.p4_link_timeout = 8
    onair._run_p4_link(tx, stream, grid())
    return capsys.readouterr().out


def test_each_crc_valid_frame_prints_its_bytes(capsys):
    log = greeting_link(capsys)
    frames = re.findall(r"\[p4 frame\] body at ([\d.]+) status=0x00 data=([0-9a-f]+)", log)
    assert [at for at, _ in frames] == ['0.923', '2.173', '3.423']
    assert {bytes.fromhex(data) for _, data in frames} == {GREETING}
    shown = screen('    [p4 probe] path: robust packets on the P1 ACK\n' + log)
    assert '    0.92 ← Y27 + body c0 0x00 CRC ok "RMS Trimode 1.4.4.0␍␊W9SSJ has "\n' in shown
    assert '(repeat ×3, identical)' in shown


def test_every_cycle_closes_with_its_counters(capsys):
    log = greeting_link(capsys)
    decided = re.findall(r"\[p4 decision\] \S+ peer=([\d.]+)", log)
    closed = re.findall(r"\[p4 over\] at ([\d.]+)", log)
    # The ceiling can end the last cycle before it closes; the view renders it at LINK DOWN.
    assert closed == decided[:len(closed)] and len(decided) - len(closed) <= 1
    last = re.findall(r"\[p4 over\] at [\d.]+ (.*)", log)[-1]
    assert last.startswith('state=IRS_ANSWER expect=packet silent=2/16 unread=0 ')
    assert 'slot_accepted=false' in last and 'purpose=repeat-request' not in last


def test_2031_ladder_counts_each_short_against_its_budget(capsys):
    """WS8EOC-80 at 20:31: nothing answered the probe's v7, and the ladder spent its
    budget of eight on it -- the probe's short and seven of the link's."""
    stream = Stream(np.zeros(14 * FS))
    tx = Keying(stream)
    tx.p4_probe.emissions[0].update(value=7, header='A')
    tx.p4_probe.values = (7, 38)
    tx.p4_link_options.p4_link_timeout = 13
    onair._run_p4_link(tx, stream, grid())
    log = capsys.readouterr().out
    assert re.findall(r"try=(\d/8)", log) == [f'{n}/8' for n in range(2, 9)] + ['8/8']
    shown = screen('    [p4 probe] path: robust packets on the P1 ACK\n' + log).splitlines()
    ours = [line for line in shown if '→' in line]
    assert [re.sub(r", link \d+/13 s", "", line) for line in ours] == [
        f'{1.25 * k:8.2f} → short v7 (try {k + 1}/8)' for k in range(1, 8)]
    assert [line for line in shown if 'nothing' in line][-1] == \
        '    9.69 ← nothing (expected word, silent 8/16, link 10/13 s)'
    assert shown[-1] == ('         == LINK DOWN: retry budget exhausted -- 0 frames decoded, '
                         '7 keys sent, greeting lines read: 0; stuck at our short v7 ×7')


class Losing(Stream):
    """1259's input lost 456 samples mid-link; the arm said so only at teardown."""
    lost = 0

    def take_until(self, end):
        out = super().take_until(end)
        if self.pos >= 2 * FS and not self.lost:
            self.lost = 456
        return out


def test_capture_loss_is_said_in_the_cycle_it_happens(capsys):
    log = greeting_link(capsys, Losing(None))
    assert len(re.findall(r"!!!! CAPTURE LOSS: the input lost 456 samples \(10 ms\) by ([\d.]+) "
                          r"in the P4 link", log)) == 1
    assert re.search(r"\n +\d+\.\d\d !! stream lost 456 samples \(10 ms\)\n",
                     screen('    [p4 probe] path: robust packets on the P1 ACK\n' + log))


def test_stall_counts_its_repeat_requests_and_the_listen_its_ceiling(tmp_path, monkeypatch):
    _, _, log, _ = _stalled_arm(tmp_path, monkeypatch, STALL)
    assert re.findall(r"\[host\] first P1 packet repeat request (\d/\d) before the stall path", log) \
        == [f'{n}/{STALL}' for n in range(1, STALL)]
    ceilings = [float(left) for left in re.findall(
        r"\[p4 probe\] listened to [\d.]+; entry ceiling ([\d.]+) of 90 s left", log)]
    assert ceilings and ceilings == sorted(ceilings, reverse=True)
