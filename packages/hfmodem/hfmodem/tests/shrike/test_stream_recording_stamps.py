# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Where the host's stamps carry no converter time, the stream is placed by count."""
import sys

import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_stream_recording import BLOCK, FS, _Card, _plant, _Status


class _Packets:
    """Callback-entry stamps: one 10 ms packet delivers several blocks back to back."""

    def __init__(self, n: int) -> None:
        self.inputBufferAdcTime = self.currentTime = self.outputBufferDacTime = (
            int((n + 1) * BLOCK / 480 + 1) * 0.010)


def _open(monkeypatch, audio, path, host):
    monkeypatch.setenv('HFMODEM_RECORDING', 'everything')

    class _FakeSd:
        @staticmethod
        def InputStream(callback=None, **_):
            card = _Card(audio, callback)
            card.device = 3
            return card

        @staticmethod
        def query_devices(device):
            assert device == 3
            return {'hostapi': 7}

        @staticmethod
        def query_hostapis(index):
            assert index == 7
            return {'name': host}

    monkeypatch.setitem(sys.modules, 'sounddevice', _FakeSd)
    return onair._LiveInput('fake', None, record=path)


def _deliver(card, blocks, status=None):
    for _ in range(blocks):
        blk = card.audio[card.n:card.n + BLOCK]
        card.callback(blk.reshape(-1, 1), blk.size, _Packets(card.n // BLOCK), status or _Status())
        card.n += blk.size


def test_wasapi_stamps_are_not_read_as_loss(tmp_path, monkeypatch):
    live = _open(monkeypatch, _plant(400 * BLOCK, 3), tmp_path / 'stream.wav', 'Windows WASAPI')
    card = live._stream
    _deliver(card, 300)
    lane = live._rx_lane
    assert live.timestamp_limitation
    assert lane.gaps == [] and lane.position == card.n
    live.close()
    assert 'uncertain' not in lane.row


def test_a_reported_overflow_is_still_a_gap(tmp_path, monkeypatch):
    live = _open(monkeypatch, _plant(400 * BLOCK, 4), tmp_path / 'stream.wav', 'Windows WASAPI')
    card = live._stream
    _deliver(card, 100)
    at = live._rx_lane.position
    over = _Status()
    over.input_overflow = True
    _deliver(card, 1, over)
    _deliver(card, 50)
    lane = live._rx_lane
    assert lane.gaps == [[at, at + 1]]
    live.close()
    assert lane.row['uncertain']['from_sample'] == at


@pytest.mark.parametrize('host', ['Core Audio', None])
def test_converter_stamps_are_still_fitted(tmp_path, monkeypatch, host):
    live = _open(monkeypatch, _plant(400 * BLOCK, 5), tmp_path / 'stream.wav', host)
    assert live.timestamp_limitation is None
    live.close()
