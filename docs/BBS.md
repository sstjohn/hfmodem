<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Classic bulletin board (BBS)

The station includes optional classic bulletin-board functionality: boards,
messages and a few text games, reachable over the radio entrances.
The BBS is not radio-qualified; the current radio entrance uses VarAC chat.

To enable an existing board configuration, add this to `station.toml`:

```toml
[bbs]
config = "bbs.toml"
```

The path is relative to `station.toml`. An omitted or empty setting leaves the
interactive service disabled unless an explicit BBS config is supplied.
The setting selects the board configuration; it does not install games.
