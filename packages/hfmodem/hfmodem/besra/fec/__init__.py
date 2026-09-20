# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Forward error correction for besra: shortened Reed-Solomon over GF(2^8)."""

from .rs import rs_parity, rs_correct

__all__ = ["rs_parity", "rs_correct"]
