# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One source of operator facts for every contact waveform."""


def facts_from(cfg):
    power = cfg.declared_power_w
    return dict(name=cfg.station.name, qth=cfg.station.qth, grid=cfg.station.grid,
                rig=cfg.selected_rig.upper(), antenna=cfg.station.antenna,
                power=f"{power:g} watts" if power is not None else "",
                weather=cfg.station.weather)
