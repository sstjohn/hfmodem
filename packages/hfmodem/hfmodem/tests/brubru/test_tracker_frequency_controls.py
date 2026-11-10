# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Legacy FSK tone storage must not silently retune the RPR producer."""
import pytest

from hfmodem.brubru import waveform
from hfmodem.core import hostmode as hm
from .test_tracker import setup, request


def test_legacy_fsk_default_and_required_init_do_not_retune_rpr():
    host, _, _, settings = setup()
    host.feed(b'\x1bJHOST1\r')
    assert request(host, '%F').data == b'1700'
    assert request(host, '%F1500').code == hm.OK
    assert request(host, '%F').data == b'1500'
    assert not settings
    assert waveform.CENTRE == 1500


@pytest.mark.parametrize('unsupported', [300, 1499, 1501, 2400])
def test_rpr_center_reports_the_real_fixed_producer_and_refuses_retuning(unsupported):
    host, _, _, settings = setup()
    host.feed(b'\x1bJHOST1\r')
    assert request(host, '%L').data == str(int(waveform.CENTRE)).encode()
    assert request(host, '%L1500').code == hm.OK
    before = dict(host.values)
    assert request(host, f'%L{unsupported}').code == hm.FAIL
    assert host.values == before and not settings
