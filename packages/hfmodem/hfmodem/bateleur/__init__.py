# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""bateleur: HF waveforms, linking and monitoring.

``mil110``, ``s4285`` and ``s4539`` provide serial waveforms; ``tone39`` and
``tone16`` provide parallel waveforms. ``spec``, ``fec``, ``interleave`` and
``rs16`` hold their tables and coding. ``phy``, ``patheq`` and ``blockeq``
provide pulse shaping and equalization; ``stream`` supplies the receivers.
``modem``, ``worker`` and ``host`` connect them to station lanes and clients.
``offair``, ``sim`` and ``measure`` render, decode and measure test signals.

``s5066`` supplies the link layer and SIS host interface. ``ale`` provides
2G ALE operation and monitoring, including directory lookup, DTM/DBM and
linking protection. ``selcall`` receives and generates HF selective calls.
See ``docs/protocols/bateleur`` for each component's evidence and limits.
"""
