# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One source of operator facts for every contact waveform."""


def facts_from(cfg):
    return dict(name=cfg.station.name, qth=cfg.station.qth, grid=cfg.station.grid,
                rig=cfg.selected_rig.upper(), antenna=cfg.station.antenna,
                power=f"{cfg.rig.power_w:g} watts" if cfg.rig.power_w is not None else "",
                weather=cfg.station.weather)
