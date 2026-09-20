# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Winlink mail above the link: the B2 forwarding protocol, once, for every modem.

Three transports in this package — PACTOR, VARA, ARDOP — carry the same mail
session, so the mail layer lives here and not three times. A modem hands the
decoded link bytes to a :class:`B2FSession` and transmits what comes back;
nothing in here knows what a waveform is.

The protocol is open and published: the FBB forwarding protocol (f6fbb.org
documentation) with the Winlink B2F extensions (winlink.org/B2F), whose
compression source the Winlink team publishes. `lzhuf` is that compression,
`message` the message structure a transfer carries, `session` the exchange.
"""
from .client import (MailClient, MailExchange, load_outbound,
                     progress_to_stdout, summarize, write_inbox)
from .lzhuf import LzhufError, compress, decompress
from .message import Attachment, Message, MessageError, compose
from .session import (CLIENT_SID, B2FSession, mask_pr, secure_login_response,
                      sid_line)

__all__ = [
    "Attachment", "B2FSession", "CLIENT_SID", "LzhufError", "MailClient",
    "MailExchange", "Message", "MessageError", "compose", "compress",
    "decompress", "load_outbound", "mask_pr", "progress_to_stdout",
    "secure_login_response", "sid_line", "summarize", "write_inbox",
]
