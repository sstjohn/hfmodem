# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Contact callsign shape checks, shared independently of the waveform."""
import re

BASE_CALL = r'(?:[BFGIKMNRW]|[A-Z]{2}|[0-9][A-Z]|[A-Z][0-9])[0-9][A-Z]{1,4}'
CALL = re.compile(BASE_CALL)


def checked(value):
    if not isinstance(value, str):
        raise ValueError('A contact callsign must be text.')
    call = value.strip().upper()
    if not re.fullmatch(BASE_CALL + r'(?:/(?:P|M|MM|AM|[0-9]|' + BASE_CALL + '))?', call):
        raise ValueError(f'{value!r} is not a complete contact callsign; edit the first copy.')
    return call
