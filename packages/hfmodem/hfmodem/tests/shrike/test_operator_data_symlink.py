# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A worktree's ``operator-data`` is a symlink to the checkout's, and a registered
capture read through the link is an unregistered mirror to `capture_identity`."""

import pytest

from hfmodem.core import wav
from hfmodem.core.capture_identity import IdentityError
from hfmodem.tests.evidence import OPERATOR_DATA, require_input

CAPTURE = ('captures/onair-20261106T044355270112Z-24568757-0330-428d-b108-8ec4eabd9b6f'
           '/stream-l7278-20261106T0444Z-s00000000000000000129.flac')


def test_symlinked_operator_data_reads_through_the_resolved_root(tmp_path):
    require_input(OPERATOR_DATA / CAPTURE)
    link = tmp_path / 'operator-data'
    link.symlink_to(OPERATOR_DATA, target_is_directory=True)

    assert link.resolve() == OPERATOR_DATA
    assert wav.read(OPERATOR_DATA / CAPTURE, stop=4800).size == 4800
    with pytest.raises(IdentityError, match='unregistered root'):
        wav.read(link / CAPTURE, stop=4800)
