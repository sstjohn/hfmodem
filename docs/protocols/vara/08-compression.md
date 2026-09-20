# 08 — Payload compression (host bytes → pre-FEC frame)

Compression sits between the host data port (8301, raw bytes in) and the
CRC+FEC+OFDM frame builder ([`03`](03-coding.md), [`04`](04-frame-block-formats.md)).
The host writes **uncompressed** bytes; VARA compresses them; the compressed
container is what CRC+turbo+OFDM then carry.

**Registration.** VARA engages compression only on a registered build. An
unregistered build passes the payload through unchanged whatever `COMPRESSION` is
set to, so `BUFFER` clears identical byte counts either way
([`06` §6.1](06-speed-gearshift.md)).

## 8.1 Codec identity

VARA's modem-layer compression is **static, order-0 (memoryless) Huffman coding
over 8-bit byte symbols, with a per-message code table transmitted inline.**

| Attribute | Value |
|---|---|
| Family | static order-0 Huffman (byte alphabet) |
| NOT | adaptive / canonical Huffman; **no** LZ / dictionary / match stage (not deflate) |
| Table | per-message, built from that message's own byte frequencies, shipped inline |
| Preset dictionary | none |
| TEXT vs FILES | **one** modem-layer scheme; the container variant (HE0 vs HE3), not the codec, differs. `COMPRESSION TEXT/FILES/OFF` selects *whether/what* to compress, not a second algorithm |

> The `zlib/deflate` sometimes associated with "Winlink" is the **B2F
> application layer** (a different, higher layer), not VARA's port-8301 modem
> compression. Do not conflate.

## 8.2 Container wire format

Two containers. Header is 4 bytes.

- **HE0** = `48 45 30 0D` (`"HE0\r"`) + **raw uncompressed bytes** (passthrough).
- **HE3** = `48 45 33 0D` (`"HE3\r"`) + compressed payload, laid out:

| Field | Width | Notes |
|---|---|---|
| Header | 4 B | `"HE3\r"` |
| Parity | 1 B | **XOR (LRC)** of all *uncompressed* message bytes (the reference code labels it "CRC"; it is not CRC-8) |
| Message length | 4 B **little-endian** | decoded length in bytes |
| SymbolCount | 2 B **little-endian** | # distinct byte values = # Table-1 entries = # leaves (1..256) |
| Huffman Table 1 | SymbolCount × 2 B | per entry `[symbol_byte][code_length_bits]`, ordered by ascending byte value |
| Encoded body | var | concatenated prefix codes; final partial byte zero-padded |
| Huffman Table 2 | var | the codeword bit-strings, same symbol order as Table 1; lengths known from Table 1; final partial byte zero-padded |

> Consequence: Table 2 (needed to interpret the body) **trails** the body, so the
> whole HE3 message must be buffered before decoding.
> The byte offsets that follow from these widths — parity at 4, length 5–8,
> symcount 9–10, Table 1 at 11 — are those of the reference implementation; this
> document does not fix them independently against a pre-FEC frame.

## 8.3 Bit packing (the load-bearing detail)

**LSB-first** within each byte, for BOTH the encoded body and Table 2's codeword
bits: `bitValue[8] = {1,2,4,8,16,32,64,128}`, fill `bitPos` 0→7.
- encode: `byte |= bitValue[bitPos]` when the code bit is 1; flush at `bitPos==8`.
- decode: bit = `(byte & bitValue[bitPos]) != 0`.

## 8.4 Encoder (tree build) and decoder

- **Encoder:** count byte frequencies; one leaf per byte with count>0; greedily
  merge the two lowest-weight unparented nodes (`MaxNodes=511`); `CreateBitSequences`
  = recursive DFS, **left child appends bit 0, right child bit 1**; single-symbol
  special case forces code length 1. Tie-break is a fixed nested comparison
  (deterministic, non-canonical) — only matters for byte-identical output.
- **Decoder:** does NOT need VARA's tree builder. Read Table 1 (symbol + bitlen),
  read Table 2 bits (LSB-first) to recover each symbol's codeword, insert each
  `(symbol,codeword)` into a trie, walk the body bit-by-bit emitting a symbol at
  each leaf, stop at Message-length, verify XOR parity.

## 8.5 Interop implication

For **decoding** VARA's output everything is exact (you read VARA's shipped
tables). For **encoding**, you do **not** need to reproduce VARA's exact
codewords: a real VARA decodes from *your* transmitted tables, so **any valid
Huffman coding** with correct framing + LSB packing + correct XOR parity
interoperates. Bit-exact encoder reproduction is needed only for identical test
vectors, not for interop.

## 8.6 Not specified here

All four are container-boundary questions; none touches the codec.

1. The exact HE3 field byte-offsets (parity and length position), per §8.2.
2. Version drift: this description follows the reference as published in 2021 and
   does not cover any divergence in v4.9.0.
3. VARA's HE0-vs-HE3 selection heuristic — the rule by which it decides to
   compress. This does not affect correctness. *Implementation note: this is not
   something VARA specifies to a peer, so an encoder here picks its own rule.*
4. How the HE0/HE3 blob is segmented and embedded into VARA's CRC+turbo+OFDM block
   framing ([`04`](04-frame-block-formats.md)).

## References
- EA5HVK, *VARA HUFFMAN COMPRESSION.pdf* + `VaraHuffOriginal.vb.txt` (Feb 2021).
- `github.com/pengowray/VaraHuffmanNet` (mirror + C# port; `VHuffman.cs`).
