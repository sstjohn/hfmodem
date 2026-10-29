# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os
from pathlib import Path


def installed_data():
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'hfmodem'
