# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""BBS configuration refusals precede service acquisition; runtime errors fail."""
from types import SimpleNamespace

import pytest

from hfmodem.bbs.store import OwnershipError
from hfmodem.station.chat import Attachment, AttachmentGate
from hfmodem.station.sessions import Refused


def attachment(path):
    station = SimpleNamespace(lanes={'vara': SimpleNamespace(runtime=SimpleNamespace(attachments=AttachmentGate()))},
        arbiter=object(), rig=SimpleNamespace(line_state='low'), station_file=None,
        cfg=SimpleNamespace(bbs=SimpleNamespace(config=str(path))))
    fence = SimpleNamespace(check=lambda: None)
    return Attachment(SimpleNamespace(station=station, fence=fence, session=None))


def test_invalid_bbs_config_refuses_before_service_acquires_any_owner(tmp_path, monkeypatch):
    from hfmodem.bbs import service
    monkeypatch.setattr(service, 'open_service', lambda path: pytest.fail('invalid config reached Service'))
    item = attachment(tmp_path / 'missing.toml')
    with pytest.raises(Refused) as error:
        item.open()
    assert error.value.reason == 'bbs-config'
    item.release()


@pytest.mark.parametrize('failure,reason', [(OwnershipError('already owned'), 'bbs-owned'),
                                           (RuntimeError('partial worker startup failed'), None),
                                           (OSError('runtime resource failed'), None)])
def test_only_actual_owner_conflict_becomes_startup_refusal(tmp_path, monkeypatch, failure, reason):
    from hfmodem.bbs import service
    path = tmp_path / 'bbs.toml'
    path.write_text('[bbs]\nid = "fixture"\n')
    def fail(path): raise failure
    monkeypatch.setattr(service, 'open_service', fail)
    item = attachment(path)
    with pytest.raises(type(failure) if reason is None else Refused) as error:
        item.open()
    if reason is not None:
        assert error.value.reason == reason
    else:
        assert error.value is failure
    item.release()
