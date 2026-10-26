<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# VarAC BBS files

The VarAC terminal can download files and serve an explicitly selected directory.
Restart the terminal to load code/configuration changes. These settings do not
start a session or schedule beacons by themselves.

## Download

After connecting and completing QSY off the calling frequency:

```text
/bbs list
/bbs get 1
```

`get` requests the selected filename, accepts a matching supported offer, receives
the file, verifies the compressed byte count and MD5 and the GZIP stream, and
saves the original file before sending `<SFOK>`. The terminal prints its path in
the VarAC mailbox's `downloads/bbs-*/` directory under the operator-data root. Each
transfer has a unique directory; an existing download is never overwritten.
No file is automatically opened or executed.

The supported format is one ordinary length-prefixed `<SFB>` message containing
Base64-encoded GZIP. Limits are 32 KiB compressed, 1 MiB expanded, 64 KiB captured
transport data, and 30 minutes receiving. Unsupported formats, corrupt or
incomplete data, timeouts, and write errors never produce a success reply.
`/stop` and `/bye` cancel the operation and close the contact.

`/bbs request 1` (also `/bbs offer 1`) requests only the file offer, without
accepting it. `/bbs capture` explicitly accepts that offer for opaque protocol
research; it retains the original 64 KiB / 120 second bounds and does not claim
or acknowledge a download. Use `/bbs get` for normal downloading.

## Serve a directory

At startup:

```sh
tools/console vara chat --band 20 --bbs-directory /absolute/path/to/bulletins
```

Or add this key to the existing `[protocols.vara]` table in `station.toml`:

```toml
[protocols.vara]
bbs_directory = "bulletins"
```

Relative TOML paths are relative to the station file named by `HFMODEM_STATION`;
the console supplies that station file to the terminal. CLI paths are relative
to the current working directory and override the TOML setting. `--no-bbs`
disables hosting even when a directory is configured.

A usable export automatically adds service `B` to opening reports and the BBS bit
to advanced beacons. The default VMail flag is retained. Advanced beacons require
a six-character locator and the KISS connection; a plain legacy beacon cannot
carry service flags. Use `/beacon` to enable the existing beacon schedule and
leave `/auto on` to accept callers. Hosting does not depend on beaconing being
scheduled. BBS list/file requests are accepted only on an established link off
the calling frequency, after the opening exchange settles.

Only safe regular files directly in the directory are exported. Hidden files,
subdirectories, symbolic links and hard-linked files are excluded. The catalog
checks root/file identity and serves an immutable snapshot of each requested
file. Snapshot bytes remain independent of later edits; exported files are not
locked against concurrent local writers. Do not put private files in the export
directory: it is public to callers.
No upload or remote write operation is implemented.

The catalog defaults to 500 files, 1 MiB per original file, 16 MiB total, and at
most 2,000 directory entries. Exceeding a catalog limit refuses the export.
Files whose compressed encoding exceeds 32 KiB are refused at download request;
the directory reports original sizes, so compressibility determines eligibility.
A sender waits 60 seconds for readiness and up to 30 minutes for receipt. A modem
buffer reaching zero does not mean receipt: only the peer's `<SFOK>` confirms it.

## Evidence and validation

The 2026-10-25 KQ4QWH capture from this station's session contained 1,654 incoming
transport bytes [T: 1 session]:
`1649 ` followed by `<SFB>` and 1,644 Base64 characters. Decoding produced 1,232
GZIP bytes whose MD5 was `52809BA0CBC6068F3E8977F8D47D7DCC`, exactly matching the
offer, and the complete 2,516-byte welcome bulletin. Its SHA256 is
`44c36a9969d42021d608be4eee4b61db5ef7c9c8762e07405728d41ce45497b1`.
The capture did not send `<SFOK>`: the original session ran the research receiver.

The [regression fixture](../packages/hfmodem/hfmodem/tests/contact/fixtures/kq4qwh-bbs-1025.json) preserves the
received chunk boundaries and offer. Tests cover real-capture replay, client/server
application round trip, arbitrary fragmentation, trailing chat, checksum/GZIP
failures, expansion bounds, cancellation, disconnect, stale readiness, export
path restrictions, configuration, and non-overwriting save behavior. Live RF
interoperability of the new automatic receiver and sender remains to be tested.
