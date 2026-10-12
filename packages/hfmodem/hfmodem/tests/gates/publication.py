# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Shared publication marker pattern."""
import re


MARKERS = re.compile(
    r"\[pro[v]:|\bPMON\b|\boracle\b|\bdecompi[l]|\bdisassembl|\bclean[- ]?room\b"
    r"|\bquarantin[e]\b|reverse[- ]engineer|\bIDA Pro\b|\bGhidra\b",
    re.IGNORECASE)
