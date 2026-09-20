# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The one packaged data file kestrel has left, read as a package resource.

Every constant table either modem needs is computed by `rx/tablegen.py` at the
point of use. The BW500 synthesis pulse is not: it was deconvolved out of
recorded audio, and while that deconvolution re-runs in seconds, the recording it
reads does not ship -- so the pulse stays a file under ``tx/assets/``.
It is package data, so an installed distribution has no source tree to resolve a
relative path against; ``importlib.resources`` finds it wherever the package
itself was installed.
"""
from __future__ import annotations

import io
from importlib.resources import files
from typing import BinaryIO


def open_binary(package: str, *parts: str) -> BinaryIO:
    """An asset as a seekable binary stream, for readers like ``np.load``."""
    node = files(package)
    for part in parts:
        node = node / part
    return io.BytesIO(node.read_bytes())
