# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Claim and trap extracts; source parity is checked by test_cleanup_harness."""

WRAPPERS = {'autopilot.sh': {'claim': 'claim_rig autopilot || exit $?',
                  'cleanup': 'stop() {\n'
                             '  local rc=$?\n'
                             '  rm -f "$TARGETS" "$TARGETS".{raw,why}\n'
                             '  claim_exit "$rc"\n'
                             '}\n'
                             'trap stop EXIT\n'
                             "trap 'exit 130' INT\n"
                             "trap 'exit 143' TERM\n"
                             "trap 'exit 129' HUP\n"},
 'campaign.sh': {'claim': 'claim_rig campaign || exit $?', 'cleanup': ''}}
