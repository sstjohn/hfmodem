# Secret operating desk

This is a static, nonoperational operating desk demo. Its Chat, Contest and
Traffic workspaces use invented exercise data. Tuning, connecting, sending,
receiving and Stop controls simulate radio activity locally; they do not claim
hardware, open a modem connection or transmit. Advanced console tools produce
command previews and never execute them. Demo preferences and exercise state
are saved in this browser and can be reset.

The normal site has no visible invitation to this room. The shared cheat-code
loader opens it, and direct navigation remains possible. Pages use `noindex`;
this is an easter egg, not an access-control boundary. The static builder adds
the same experimental prerelease warning and project links as other pages.

Only reviewed shell assets and the public logbook adapter belong here. No
historical station snapshot, private transcripts, legacy replay assets or
additional recording copies are included. The logbook presents the six
approved public VARA, PACTOR-3 and ARDOP send/receive examples and links to their
existing inspectors. The VARA send demonstrates L9; the retained receive is
L4. ARDOP audio is from remote witnesses. Keep the public evidence limitations
and shared data-loss disclosure visible.

Keyboard controls, native dialogs, stable focus during updates, bounded
screen-reader announcements and configurable signal palettes are retained
from the reviewed frontend. WCAG 2.2 AA is the baseline, not a verified
conformance claim. Actual screen-reader, speech-control and physical switch
(including sip-and-puff) testing remain outstanding. See the site’s
[accessibility guidance](../../docs/ACCESSIBILITY.md) for the wider evaluation limits.

The command metadata preserves parser options and source revision hashes.
Machine-specific paths are replaced with relative example paths; the command
builder does not verify that those paths exist on a visitor’s machine.

## Public logbook data contract

`public-qsos.mjs` is the desktop’s read-only adapter for the curated public
logbook. It can also be loaded from a module script in this directory:

```js
import {loadPublicQsos} from './public-qsos.mjs';
const rows = await loadPublicQsos();
```

The adapter reads the existing public `../examples/logbook/entries.json` and
its six approved manifests. It requires exactly VARA send/receive, PACTOR-3
send/receive and ARDOP send/receive, with their reviewed peers and roles.
It does not import a historical station snapshot. No recordings, event maps,
transcripts, credentials or station coordinates are copied into this room.

Rows retain the public speed/mail labels and separate acknowledged/decoded
speed evidence, original mail counts, recording source kinds and full replay
limitations. `inspectorHref` points to the existing public inspector; all links
are relative to this directory. Use the labels as text, and show limitations
when a row is expanded. WW2MI's subsequent receive failure is retained in its
note and limits; delivery is counted only when the public session says so.
The data-loss disclosure stays linked through `dataLossHref`.

Build checks can call the pure `publicQsoRows(entries, manifestsByRoute)`
projection against approved files. Gate the six route/peer/role identities,
matching dates and bands, actual message counts, speed labels and limitations,
and resolving inspector links. Reject extra or duplicate routes, missing
manifests, mismatched roles and private references. A load failure should leave
an empty logbook with an error, never substitute invented or private records.
This adapter supplies recorded evidence, not live station availability.
