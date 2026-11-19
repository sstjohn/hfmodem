# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Shared service lifetime and catalog selection, independent of any modem."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from ..core.waits import join
from .config import load


def catalog_root(config, legacy_root=None):
    """A generic BBS has exactly one authoritative export root."""
    if legacy_root:
        legacy = Path(legacy_root).expanduser().resolve()
        if config.export_root is None or legacy != config.export_root:
            raise ValueError('legacy BBS directory conflicts with the shared BBS export_root')
    return config.export_root


class Service:
    def __init__(self, config, *, legacy_root=None, games=None, config_path=None):
        from hfmodem.contact.bbs import HostCatalog
        from .core import BBSCore
        from .store import Store

        self.config = config
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self._reload_pool = None
        self._reload_futures = []
        self._closing = False
        root = catalog_root(config, legacy_root)
        self.catalog = None
        self.store = None
        self.core = None
        try:
            self.store = Store(config.state_root)
            if games is None:
                from .games import configured_games
                games = configured_games(config)
            if root is not None:
                self.catalog = HostCatalog()
                self.catalog.enable(root)
            self.core = BBSCore(config, self.store, games=games, catalog=self.catalog)
        except BaseException:
            if self.store is not None:
                self.store.close()
            if self.catalog is not None:
                self.catalog.close()
            raise

    def reload(self):
        """Read policy off the polling thread; caller applies future.result() there.

        Live changes are limited to default/per-callsign limits. Identity,
        timezone, runtime and catalog changes require stopping the service.
        """
        if self._closing:
            raise RuntimeError('BBS service is closing')
        if self.config_path is None:
            raise ValueError('no configuration file associated with this service')
        if self._reload_pool is None:
            self._reload_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bbs-config')
        self._reload_futures = [future for future in self._reload_futures if not future.done()]
        future = self._reload_pool.submit(self._load_policy)
        self._reload_futures.append(future)
        return future

    def _load_policy(self):
        updated = load(self.config_path)
        for field in self.config.__dataclass_fields__:
            if field not in ('limits', 'callsigns') and getattr(updated, field) != getattr(self.config, field):
                raise ValueError(f'{field} cannot change during live policy reload')
        return updated

    def close(self, *, timeout=5, deadline=None):
        """Retire workers before releasing resources, within five seconds.

        ``deadline`` is an optional absolute monotonic deadline supplied by an
        enclosing teardown. A failed retirement keeps the Store owner lock;
        callers must retain this service and can retry close after workers exit.
        Like core polling and reload submission, close runs on the owner thread.
        """
        end = time.monotonic() + max(0, min(timeout, 5))
        if deadline is not None:
            end = min(end, deadline)
        self._closing = True
        if self._reload_pool is not None:
            for future in self._reload_futures:
                future.cancel()
            self._reload_pool.shutdown(wait=False, cancel_futures=True)
        if self.store is None:
            return
        self.core.shutdown()
        while not self.core.quiescent and time.monotonic() < end:
            self.core.tick()
            if not self.core.quiescent:
                time.sleep(min(.01, max(0, end - time.monotonic())))
        if not self.core.quiescent:
            # Never release the owner lock while a worker can mutate state.
            raise RuntimeError('BBS shutdown still pending; owner lock retained')
        self.core.shutdown()
        # Future completion alone precedes executor worker exit. shutdown(False)
        # queues the exit sentinel; join its tracked threads with the same budget.
        # ThreadPoolExecutor has no public bounded-join API. These are the actual
        # workers of our two private executors, not the process's other threads.
        for pool in (self._reload_pool, self.core._executor):
            if pool is None:
                continue
            for worker in tuple(pool._threads):
                if not join(worker, end):
                    raise RuntimeError('BBS worker retirement still pending; owner lock retained')
        self._reload_pool = None
        self._reload_futures.clear()
        if self.catalog is not None:
            self.catalog.close()
        self.store.close()
        self.store = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_service(path, *, legacy_root=None):
    return Service(load(path), legacy_root=legacy_root, config_path=path)
