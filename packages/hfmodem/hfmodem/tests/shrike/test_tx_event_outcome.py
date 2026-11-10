# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A burst's tx_event says what reached the DAC, not what was planned."""
import json
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_tx_moni_lag import callback_stream


class _Done:
    """The callback's completion flag, pumping the fake card while it is waited on."""

    def __init__(self, live, clock, arming=None):
        self.live, self.clock, self.arming, self.flag = live, clock, arming, False

    def clear(self):
        self.flag = False
        if self.arming is not None:
            # Between the deadline check and arming the callback.
            self.arming(self.live)

    def set(self):
        self.flag = True

    def wait(self, limit):
        while self.live._tx is not None:
            self.live._stream.pump()
        if not self.flag:
            # An unset wait returns once its timeout has passed.
            self.clock.now += limit
        return self.flag


def _completed(monkeypatch, tmp_path, capsys, *, arming=None, during=None, unkey_late_s=0.):
    live, clock, _ = callback_stream(monkeypatch)
    live._tx_done = _Done(live, clock, arming)
    # The interpreter stalls between the audio's last sample and the unkey.
    live._sleep_until = lambda t: setattr(clock, 'now', t + (unkey_late_s if t == live._tx_end_time else 0.))
    if during is not None:
        pump = live._stream.pump
        def pumped():
            pump()
            during(live)
        live._stream.pump = pumped
    tx = onair.RadioTx(transmit=True, outdir=tmp_path, settle=.04,
                       rig=SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None))
    tx.live = live
    audio = pactor1.connect_signal('WS8EOC')
    clock.now = (live._stream.n + 13000) / 48000
    failure = None
    try:
        tx._tx(audio, 'connect->WS8EOC')
    except SystemExit as exc:
        failure = exc
    events = [json.loads(line[len('tx_event '):]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('tx_event ') and '"completed"' in line]
    return events, tx, live, failure


def test_a_burst_on_time_is_sent_to_its_planned_end(monkeypatch, tmp_path, capsys):
    (event,), tx, _, _ = _completed(monkeypatch, tmp_path, capsys)
    assert event['outcome'] == 'sent'
    assert (event['audio_start_sample'], event['audio_end_sample']) == (tx.tx_audio_start, tx.tx_end)


def test_a_burst_armed_after_its_first_sample_is_late(monkeypatch, tmp_path, capsys):
    def stalled(live):
        for _ in range(200):
            live._stream.pump()
    (event,), tx, _, _ = _completed(monkeypatch, tmp_path, capsys, arming=stalled)
    assert event['outcome'] == 'late'
    assert event['audio_start_sample'] > tx.tx_audio_start
    assert event['audio_end_sample'] == tx.tx_end


def test_a_burst_the_watchdog_cut_is_logged_interrupted_where_it_stopped(monkeypatch, tmp_path, capsys):
    def cut(live):
        if live._tx is not None and live._stream.n > live._tx_at + 2048:
            live._tx = None
    (event,), tx, live, failure = _completed(monkeypatch, tmp_path, capsys, during=cut)
    assert failure is not None
    assert event['outcome'] == 'interrupted'
    planned = live.corrected_emission_start - live.geometry_n
    assert event['audio_start_sample'] == planned
    assert planned < event['audio_end_sample'] < planned + len(pactor1.connect_signal('WS8EOC'))


@pytest.mark.parametrize('late_s, outcome', [(0.05, 'sent'), (0.08, 'late'), (3.0, 'late')])
def test_a_carrier_held_past_its_audio_is_late(monkeypatch, tmp_path, capsys, late_s, outcome):
    (event,), tx, _, _ = _completed(monkeypatch, tmp_path, capsys, unkey_late_s=late_s)
    assert event['outcome'] == outcome
    assert event['audio_end_sample'] == tx.tx_end


def test_a_burst_unkeyed_during_its_final_drain_is_interrupted(monkeypatch, tmp_path, capsys):
    (event,), tx, _, _ = _completed(monkeypatch, tmp_path, capsys, unkey_late_s=-0.02)
    assert event['outcome'] == 'interrupted'
    assert event['audio_end_sample'] == tx.tx_end - round(0.02 * 48000)


def test_a_transport_without_emission_facts_reports_the_plan(monkeypatch, tmp_path, capsys):
    tx = onair.RadioTx(transmit=True, outdir=tmp_path, settle=.04,
                       rig=SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None))
    tx.live = SimpleNamespace(keyed_s=1.0)
    tx.tx_audio_start, tx.tx_end, tx.tx_key_up = 10, 20, 5
    tx._log_completed('connect', [0.] * 10, None, duplex=True)
    event = json.loads(capsys.readouterr().out.splitlines()[-1][len('tx_event '):])
    assert (event['outcome'], event['audio_start_sample'], event['audio_end_sample']) == ('sent', 10, 20)
