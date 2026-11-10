# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An accepted DATA window still needs its ACK after transport refusal."""
import pytest

from .receive_answer_helpers import key_answer
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.tests.kestrel.test_vara_giveup import _connected


@pytest.mark.parametrize("last", (False, True))
def test_ack_declined_by_transport_is_retried_without_requesting_missing_data(last):
    hs, io = _connected(transmits=False)
    hs.turn = VA._TURN_PEER
    key_answer(hs, last=last, owes_release=False)
    assert not io.sent
    assert hs._answer_owed == (VA._OWED_RELEASE if last else VA._OWED_OVER)
    assert not hs._owed_block
    assert hs._reacks == hs._since_progress == 0

    io.transmits = True
    assert hs._reack(recognized=True, query_start=hs._unkeyed_at + .1)
    assert len(io.sent) == 1
    assert (hs._reacks if last else hs._positive_reacks) == hs._since_progress == 1
    assert not hs._owed_block
    assert not any("tx NAK" in message for message in io.msgs)
