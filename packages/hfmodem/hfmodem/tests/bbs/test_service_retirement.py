# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""BBS teardown retains ownership until all private workers have retired."""
import threading

import pytest

from hfmodem.bbs.config import BBSConfig
from hfmodem.bbs.service import Service
from hfmodem.bbs.store import OwnershipError, Store


def make_service(tmp_path):
    export = tmp_path / 'files'
    export.mkdir()
    config = BBSConfig(state_root=tmp_path / 'state', export_root=export)
    return Service(config, games={}, config_path=tmp_path / 'bbs.toml')


def assert_owned(service):
    assert service.store is not None
    assert service.catalog._fd is not None
    with pytest.raises(OwnershipError):
        Store(service.config.state_root)


def test_close_cancels_queued_reload_and_retains_running_worker(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    started, release = threading.Event(), threading.Event()

    def load():
        started.set()
        assert release.wait(2)
        return service.config

    monkeypatch.setattr(service, '_load_policy', load)
    try:
        running = service.reload()
        assert started.wait(2)
        queued = service.reload()
        pool = service._reload_pool
        with pytest.raises(RuntimeError, match='owner lock retained'):
            service.close(timeout=0)
        assert queued.cancelled()
        assert not running.done()
        assert service._reload_pool is pool
        assert_owned(service)
        with pytest.raises(RuntimeError, match='closing'):
            service.reload()
    finally:
        release.set()
        service.close()
    assert all(not thread.is_alive() for thread in pool._threads)
    with Store(service.config.state_root):
        pass


def test_close_polls_pending_core_work_before_releasing_store(tmp_path):
    service = make_service(tmp_path)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(2)
        return None

    try:
        assert service.core._job('probe', work)
        assert started.wait(2)
        with pytest.raises(RuntimeError, match='shutdown still pending'):
            service.close(timeout=0)
        assert_owned(service)
        release.set()
        service.close()
        assert service.core.quiescent
        assert all(not thread.is_alive() for thread in service.core._executor._threads)
    finally:
        release.set()
        service.close()


@pytest.mark.parametrize('worker_kind', ['reload', 'core'])
def test_done_future_is_not_proof_of_worker_retirement(tmp_path, monkeypatch, worker_kind):
    service = make_service(tmp_path)
    started, finish_job = threading.Event(), threading.Event()
    callback_started, finish_callback = threading.Event(), threading.Event()

    def work():
        started.set()
        assert finish_job.wait(2)
        return service.config if worker_kind == 'reload' else None

    def callback(_):
        callback_started.set()
        assert finish_callback.wait(2)

    try:
        if worker_kind == 'reload':
            monkeypatch.setattr(service, '_load_policy', work)
            future = service.reload()
        else:
            service.core._job('probe', work)
            future = service.core._jobs[-1][2]
        assert started.wait(2)
        future.add_done_callback(callback)
        finish_job.set()
        assert callback_started.wait(2)
        assert future.done()
        service.core.tick()
        assert service.core.quiescent
        with pytest.raises(RuntimeError, match='worker retirement still pending'):
            service.close(timeout=0)
        assert service.core.quiescent
        assert_owned(service)
        finish_callback.set()
        service.close()
        assert service.store is None
    finally:
        finish_job.set()
        finish_callback.set()
        service.close()


def test_uncertain_cleanup_keeps_service_and_owner_lock(tmp_path):
    service = make_service(tmp_path)
    try:
        service.core._uncertain_cleanup = True
        with pytest.raises(RuntimeError, match='shutdown still pending'):
            service.close(timeout=0)
        assert_owned(service)
    finally:
        service.core._uncertain_cleanup = False
        service.close()


@pytest.mark.parametrize(('timeout', 'deadline', 'expected'), [
    (5, 102, 2), (20, None, 5), (1, 102, 1), (5, 99, 0),
])
def test_worker_join_uses_remaining_deadline_and_five_second_cap(
        tmp_path, monkeypatch, timeout, deadline, expected):
    from hfmodem.bbs import service as module
    service = make_service(tmp_path)
    waits = []

    class RetiredWorker:
        def join(self, timeout):
            waits.append(timeout)

        def is_alive(self):
            return False

    worker = RetiredWorker()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 100)
    service.core._executor._threads.add(worker)
    try:
        service.close(timeout=timeout, deadline=deadline)
    finally:
        service.core._executor._threads.remove(worker)
    assert waits == [expected]


def test_success_closes_catalog_then_store_after_worker_exit(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    service.core._executor.submit(lambda: None).result(timeout=2)
    order = []
    catalog_close, store_close = service.catalog.close, service.store.close

    def close_catalog():
        assert all(not worker.is_alive() for worker in service.core._executor._threads)
        assert service.store.db is not None
        order.append('catalog')
        catalog_close()

    def close_store():
        assert service.catalog._fd is None
        order.append('store')
        store_close()

    monkeypatch.setattr(service.catalog, 'close', close_catalog)
    monkeypatch.setattr(service.store, 'close', close_store)
    service.close()
    service.close()
    assert order == ['catalog', 'store']


def test_all_worker_joins_share_one_deadline(tmp_path, monkeypatch):
    from hfmodem.bbs import service as module
    service = make_service(tmp_path)
    now, waits = [100], []

    class RetiredWorker:
        def join(self, timeout):
            waits.append(timeout)
            now[0] += .25

        def is_alive(self):
            return False

    workers = {RetiredWorker(), RetiredWorker()}
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    service.core._executor._threads.update(workers)
    try:
        service.close(deadline=102)
    finally:
        service.core._executor._threads.difference_update(workers)
    assert waits == [2, 1.75]
