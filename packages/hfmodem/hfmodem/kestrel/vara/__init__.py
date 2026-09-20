# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel VARA-compatible connection interop (spec-derived).

Modules:
  * :mod:`vara_frames` — MFSK handshake-tone generators + recognizers (spec 04 §4.2).
  * :mod:`vara_mfsk`   — MFSK synth + tone demod (spec 04 §4.2.1).
  * :mod:`vara_ofdm`   — BW2300 OFDM link-setup TX + RX over the rec3 chain (spec 01/03/04).
  * :mod:`vara_arq`    — VARA-station handshake driver over the MFSK bursts (spec 05).
"""
