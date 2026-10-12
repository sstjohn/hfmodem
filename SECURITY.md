# Security

hfmodem can key a radio transmitter, and it carries a Winlink secure-login
password. Defects in either category deserve a quiet report before a public one.

**In scope.** Anything that keys a transmitter the operator did not arm, holds
PTT past its watchdog, defeats the `transmit = false` interlock, or exposes a
credential such as the Winlink password. A decode-path crash on hostile audio is
an ordinary bug: report it in the open.

**How to report.** Email [hfmodem@ssj.lol](mailto:hfmodem@ssj.lol) for a private
report. Ordinary bugs can be reported in the project repository — the address
is at the end of `NOTICE`. This project has no security team.

**What to expect.** This is a volunteer project. Reports are read; there is no
response-time promise; fixes land in the open tree, with credit if you want it.
