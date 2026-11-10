# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PTC-IIIusb 4.1 section 6.46's dummy query, not a Winlink agreement."""
from hfmodem.shrike.ptc import PtcHost
from hfmodem.core import hostmode
from .test_ptc_compatibility import Client


def test_exact_documented_ptc_iii_capability_query_without_setting_mutation():
    host = PtcHost()
    before = dict(host.settings)
    response = host.feed(b'lIcEnSe\r')
    assert b'\r\nLICENSE: 010000141714CB53 ABCDEFGHIJKL\r\n' in response
    assert host.settings == before and host.sent_total == 0 and host.contact_ready()
    assert b'LICENSE:' not in host.feed(b'LI\r')  # LI remains LIsten.


def test_license_does_not_accept_an_argument_or_store_a_key():
    host = PtcHost()
    before = dict(host.settings)
    for arg in (b'0', b'1', b'example-key'):
        assert b'*** ERROR:' in host.feed(b'LICENSE '+arg+b'\r')
        assert host.settings == before and host.sent_total == 0


def test_forwarded_argument_refusal_is_cached_without_mutation_or_fake_credit():
    host = PtcHost()
    client = Client(host)
    before = dict(host.settings)
    response = client.cmd('#LICENSE example-key')
    assert response.code == hostmode.FAIL
    assert client.retry() == response
    assert host.settings == before and host.sent_total == 0
