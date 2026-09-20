# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir — an open HF data protocol, and this implementation of it.

The protocol is specified in ``docs/protocols/sabir/SPEC.md``, with normative
known-answer vectors in ``tests/sabir/test_kat.py``; the two waveform families,
the gear ladder, the ARQ layer and the host interface live in the subpackages
below.

``sabir.sim`` ships with the package rather than beside it: the host server is
sim-backed today (``sabir.host.run_server`` drives two endpoints over a
simulated channel) and stays that way until the M5 audio front end lands, so
the simulation code is a runtime dependency and not a research aside.
"""
