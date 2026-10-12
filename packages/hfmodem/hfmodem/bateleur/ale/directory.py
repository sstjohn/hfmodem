# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A directory of who is on the air with ALE: the monitoring community's
list of agencies, their frequencies and their stations' addresses, read at
run time — it is theirs, it is revised often, and it is not bundled.

The list's own layout (ominous-valve.com/ale-list.txt is the one it was
written against) is plain text in four parts:

1. Agencies and frequencies: an agency's name on a line of its own, lines
   of notes, then its frequencies in kHz, one to a tab-indented line, each
   with an optional note ("2001 clg 1226", "snd (LSB)").
2. Addresses: ``ADDRESS<tabs>AGENCY<tabs>DESCRIPTION``. In an address,
   ``#`` stands for a digit, ``x`` for a letter and ``-`` for what varies.
3. Notes, kept as text.
4. Acronyms: ``ACRONYM<tab>MEANING``.

``load`` fetches it (or reads a file), keeping a copy so that a station
without the network still has the last one; ``Directory.who`` says who an
address is, ``Directory.at`` who uses a frequency.

**More than one source.** ``load`` takes several, and merges them: every
entry keeps the name of the list it came from. Each source's format is
recognised from its text: the list above, or a plain CSV that
any other list can be turned into::

    kind,key,agency,description
    address,K4PWO,HFN,Pilot station
    freq,14109.0,HFN,Pilot channel
    acronym,HFN,,Global ALE High Frequency Network

"""
from __future__ import annotations

import csv
import io
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URL = "https://ominous-valve.com/ale-list.txt"


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "hfmodem" / "ale-list.txt"


@dataclass
class Agency:
    name: str
    notes: list[str] = field(default_factory=list)
    freqs: list[tuple[float, str]] = field(default_factory=list)   # (kHz, note)
    source: str = ""


@dataclass
class Address:
    pattern: str
    agency: str
    description: str
    regex: re.Pattern | None = None       # None: the pattern is the address
    source: str = ""

    def matches(self, address: str) -> bool:
        a = address.upper()
        return a == self.pattern if self.regex is None else bool(self.regex.fullmatch(a))

    def __str__(self) -> str:
        return f"{self.pattern}: {self.agency}" + (f" — {self.description}" if self.description else "")


@dataclass
class Directory:
    agencies: list[Agency] = field(default_factory=list)
    addresses: list[Address] = field(default_factory=list)
    acronyms: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    source: str = ""

    def who(self, address: str) -> list[Address]:
        """The entries an address matches: exact ones first, then patterns,
        the most specific (most literal characters) first."""
        a = address.strip().upper()
        exact = [e for e in self.addresses if e.regex is None and e.pattern == a]
        pats = [e for e in self.addresses if e.regex is not None and e.matches(a)]
        pats.sort(key=lambda e: -sum(c not in "#x-" for c in e.pattern))
        return exact + pats

    def at(self, khz: float, tolerance: float = 0.5) -> list[tuple[Agency, float, str]]:
        """Who lists a frequency within ``tolerance`` kHz: (agency, kHz, note)."""
        return [(ag, f, note) for ag in self.agencies for f, note in ag.freqs
                if abs(f - khz) <= tolerance]

    def agency(self, text: str) -> list[Agency]:
        t = text.lower()
        return [ag for ag in self.agencies
                if t in ag.name.lower() or any(t in n.lower() for n in ag.notes)]

    def stations_of(self, text: str) -> list[Address]:
        t = text.lower()
        return [e for e in self.addresses if t in e.agency.lower() or t in e.description.lower()]

    def expand(self, acronym: str) -> str | None:
        return self.acronyms.get(acronym) or self.acronyms.get(acronym.upper())

    def merge(self, other: "Directory") -> "Directory":
        """Another source's entries added to these (agencies of the same
        name from different sources stay apart: each says where it is from)."""
        self.agencies += other.agencies
        self.addresses += other.addresses
        for k, v in other.acronyms.items():
            self.acronyms.setdefault(k, v)
        self.notes = "\n\n".join(x for x in (self.notes, other.notes) if x)
        self.source = ", ".join(x for x in (self.source, other.source) if x)
        return self

    def summary(self) -> str:
        n_f = sum(len(a.freqs) for a in self.agencies)
        return (f"{len(self.agencies)} agencies, {n_f} frequencies, {len(self.addresses)} "
                f"addresses, {len(self.acronyms)} acronyms ({self.source or 'unknown source'})")


# -- reading the list -------------------------------------------------------------

_RULE = re.compile(r"^={20,}\s*$")
_PART = re.compile(r"^\s*([1-4])\s*:\s*(.*)$")
_FREQ = re.compile(r"^\t\s*(\d{3,5}(?:[.,][0-9Oo]+)?)\s*(.*)$")


def _pattern(p: str) -> re.Pattern | None:
    if not any(c in p for c in "#x-"):
        return None
    out = []
    for c in p:
        out.append(r"\d" if c == "#" else "[A-Z]" if c == "x" else ".*" if c == "-"
                   else re.escape(c))
    return re.compile("".join(out))


def _khz(s: str) -> float | None:
    s = s.replace(",", ".").replace("O", "0").replace("o", "0")
    try:
        return float(s)
    except ValueError:
        return None


def parse(text: str, source: str = "") -> Directory:
    """The list's text -> a Directory. What it cannot read, it leaves out."""
    lines = text.replace("\r\n", "\n").split("\n")
    # Where each part starts: a numbered title between two rules.
    part_at: dict[int, int] = {}
    for i, ln in enumerate(lines):
        m = _PART.match(ln)
        if m and ((i > 0 and _RULE.match(lines[i - 1])) or
                  (i + 1 < len(lines) and _RULE.match(lines[i + 1]))):
            part_at.setdefault(int(m.group(1)), i + 1)
    bounds = sorted(part_at.items(), key=lambda kv: kv[1])
    parts: dict[int, list[str]] = {}
    for k, (n, at) in enumerate(bounds):
        end = bounds[k + 1][1] - 1 if k + 1 < len(bounds) else len(lines)
        parts[n] = [ln for ln in lines[at:end] if not _RULE.match(ln)]
    d = Directory(source=source)
    d.agencies = _agencies(parts.get(1, []))
    d.addresses = _addresses(parts.get(2, []))
    for e in d.agencies + d.addresses:
        e.source = source
    d.notes = "\n".join(parts.get(3, [])).strip()
    for ln in parts.get(4, []):
        f = [x.strip() for x in re.split(r"\t+", ln.strip()) if x.strip()]
        if len(f) >= 2 and not ln.startswith((" ", "\t")):
            d.acronyms[f[0]] = " ".join(f[1:])
    return d


def _agencies(lines: list[str]) -> list[Agency]:
    """Blocks: a name, notes, frequencies. A block with no frequencies is
    the part's own preamble, and goes."""
    out: list[Agency] = []
    block: list[str] = []

    def close():
        if not block:
            return
        freqs, text = [], []
        for ln in block:
            m = _FREQ.match(ln)
            k = _khz(m.group(1)) if m else None
            if k is not None and 1500 <= k <= 30000:
                freqs.append((k, m.group(2).strip()))
            else:
                text.append(ln.strip())
        text = [t for t in text if t and t != "---"]
        if freqs and text:
            out.append(Agency(text[0], text[1:], freqs))
        block.clear()

    blanks = 0
    for ln in lines:
        if not ln.strip():
            # Agencies are two or more blank lines apart; a single blank
            # line is inside an agency's notes.
            blanks += 1
            if blanks == 2:
                close()
            continue
        blanks = 0
        if ln.strip() == "---" or (block and _FREQ.match(block[-1]) and not _FREQ.match(ln)):
            close()
            if ln.strip() == "---":
                continue
        block.append(ln)
    close()
    return out


def _addresses(lines: list[str]) -> list[Address]:
    out = []
    for ln in lines:
        if not ln.strip() or ln.startswith((" ", "\t")):
            continue
        f = [x.strip() for x in re.split(r"\t+", ln) if x.strip()]
        if len(f) < 2 or " " in f[0] or len(f[0]) > 15 or "=" in ln[:6]:
            continue                              # prose, or the legend
        pat = f[0]
        if not re.fullmatch(r"[A-Z0-9@?#x\-]+", pat):
            continue
        out.append(Address(pat, f[1], " ".join(f[2:]), _pattern(pat)))
    return out


# -- other formats ----------------------------------------------------------------------

def _is_csv(text: str) -> bool:
    head = text.lstrip().split("\n", 1)[0].lower().replace(" ", "")
    return head.startswith("kind,key,agency")


def parse_csv(text: str, source: str = "") -> Directory:
    """``kind,key,agency,description`` rows: kind is address, freq (key in
    kHz) or acronym."""
    d = Directory(source=source)
    by_name: dict[str, Agency] = {}
    for row in csv.DictReader(io.StringIO(text.lstrip())):
        kind = (row.get("kind") or "").strip().lower()
        key = (row.get("key") or "").strip()
        agency = (row.get("agency") or "").strip()
        desc = (row.get("description") or "").strip()
        if kind == "address" and key:
            pat = key.upper() if not any(c in key for c in "#x-") else key
            d.addresses.append(Address(pat, agency, desc, _pattern(pat), source))
        elif kind == "freq" and key:
            k = _khz(key)
            if k is not None:
                ag = by_name.setdefault(agency, Agency(agency or "(unnamed)", source=source))
                ag.freqs.append((k, desc))
        elif kind == "acronym" and key:
            d.acronyms[key] = desc or agency
    d.agencies = list(by_name.values())
    return d


def _is_list(text: str) -> bool:
    return bool(re.search(r"^\s*1\s*:\s*Agencies", text, re.M)) or \
        bool(re.search(r"^\s*2\s*:\s*Call ?Signs", text, re.M))


def parse_any(text: str, source: str = "") -> Directory:
    if _is_csv(text):
        return parse_csv(text, source)
    if _is_list(text):
        return parse(text, source)
    raise ValueError(f"{source or 'the text'}: not a directory format; expected an ALE list or directory CSV")


# -- getting it -----------------------------------------------------------------------

def _cache_for(url: str) -> Path:
    if url == DEFAULT_URL:
        return cache_path()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])[:120]
    return cache_path().parent / "directories" / safe


def load_one(source: str | None = None, *, refresh: bool = False,
             timeout: float = 30.0) -> Directory:
    """From a file, a URL, or (``None``) the default URL. A fetched list is
    kept in the cache, and a fetch that fails falls back to the copy kept
    there; ``refresh=False`` reads a kept copy without asking the network
    if there is one."""
    src = source or DEFAULT_URL
    if not src.startswith(("http://", "https://")):
        return parse_any(Path(src).read_text(encoding="utf-8", errors="replace"), src)
    cache = _cache_for(src)
    if not refresh and cache.exists():
        return parse_any(cache.read_text(encoding="utf-8", errors="replace"),
                         f"{src} (kept copy)")
    try:
        req = urllib.request.Request(src, headers={"User-Agent": "hfmodem"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
        d = parse_any(text, src)
    except (OSError, ValueError):
        if cache.exists():
            return parse_any(cache.read_text(encoding="utf-8", errors="replace"),
                             f"{src} (kept copy; fetch failed)")
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent,
                                     prefix=cache.name + ".", delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, cache)
    return d


def load(sources=None, *, refresh: bool = False, timeout: float = 30.0) -> Directory:
    """One source or several (files and URLs), merged; ``None`` is the
    default list."""
    if sources is None or isinstance(sources, str):
        return load_one(sources, refresh=refresh, timeout=timeout)
    d = Directory()
    for s in sources:
        d.merge(load_one(s, refresh=refresh, timeout=timeout))
    return d
