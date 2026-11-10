# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Shared service lifetime and catalog selection, independent of any modem."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

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
        if self.config_path is None:
            raise ValueError('no configuration file associated with this service')
        if self._reload_pool is None:
            self._reload_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='bbs-config')
        return self._reload_pool.submit(self._load_policy)

    def _load_policy(self):
        updated = load(self.config_path)
        for field in self.config.__dataclass_fields__:
            if field not in ('limits', 'callsigns') and getattr(updated, field) != getattr(self.config, field):
                raise ValueError(f'{field} cannot change during live policy reload')
        return updated

    def close(self, *, timeout=5):
        if self._reload_pool is not None:
            self._reload_pool.shutdown(wait=False, cancel_futures=True)
            self._reload_pool = None
        if self.store is None:
            return
        self.core.shutdown()
        deadline = time.monotonic() + timeout
        while not self.core.quiescent and time.monotonic() < deadline:
            self.core.tick()
            time.sleep(.01)
        if not self.core.quiescent:
            # Never release the owner lock while a worker can mutate state.
            raise RuntimeError('BBS shutdown still pending; owner lock retained')
        self.core.shutdown()
        self.store.close()
        self.store = None
        if self.catalog is not None:
            self.catalog.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_service(path, *, legacy_root=None):
    return Service(load(path), legacy_root=legacy_root, config_path=path)
