# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""hfmodem — four HF softmodems and the station that runs them.

Each protocol is a subpackage: `shrike` (PACTOR-1/2/3), `kestrel` (VARA),
`besra` (ARDOP) and `sabir`. They share `core` for everything that touches the
radio rather than the waveform, and `host` for the dialects an application
speaks to a modem.
"""

__version__ = "0.4.2"


from .bateleur.config import (validate as _validate_bateleur_config,
                              validate_ale as _validate_ale_config,
                              validate_selcall as _validate_selcall_config)
from .core.config_validation import PROTOCOL_VALIDATORS as _protocol_validators

_protocol_validators["bateleur"] = _validate_bateleur_config

_protocol_validators["ale"] = _validate_ale_config
_protocol_validators["selcall"] = _validate_selcall_config
