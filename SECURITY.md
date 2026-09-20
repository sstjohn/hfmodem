# Security

hfmodem can key a radio transmitter, and it carries a Winlink secure-login
password. Defects in either category deserve a quiet report before a public one.

**In scope.** Anything that keys a transmitter the operator did not arm, holds
PTT past its watchdog, defeats the `transmit = false` interlock, or exposes a
credential such as the Winlink password. A decode-path crash on hostile audio is
an ordinary bug: report it in the open.

**How to report.** Open an issue at the project repository — the address is at
the end of `NOTICE`. If the report should not start in public, say only that in
the issue and ask for a private channel; one will be arranged from there. This
project publishes no maintainer address and has no security team.

**What to expect.** This is a volunteer project. Reports are read; there is no
response-time promise; fixes land in the open tree, with credit if you want it.
