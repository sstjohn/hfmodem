# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)


def choose_chat_ui(requested, *, textual_available, stdin_tty, controlling_tty,
                   term, inside_emacs, columns, rows):
    if requested == 'line':
        return 'line', None
    rich = (stdin_tty and controlling_tty and bool(term) and term.lower() not in ('dumb', 'unknown') and
            inside_emacs is None and columns >= 80 and rows >= 24)
    if requested == 'auto' and not rich:
        return 'line', None
    if not textual_available:
        return 'line', "Textual is unavailable; install with pip install 'hfmodem[tui]'; using line console."
    if not controlling_tty:
        return 'line', ('Textual needs /dev/tty; using line console.'
                        if requested == 'textual' else None)
    if requested == 'textual':
        return 'textual', None
    return ('textual' if rich else 'line'), None
