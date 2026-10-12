# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 5066: the HF link layer NATO runs over 4285, 188-110 and 4539.

Written from STANAG 5066 edition 1.2, annexes A to C. The sublayers, bottom up, each as its annex has it:

- ``dpdu``        the data transfer sublayer's PDUs: formats, CRCs (annex C.3)
- ``dts``         the data transfer sublayer: selective-repeat ARQ, expedited
                  and non-ARQ delivery, segmentation, management (annex C)
- ``cas``         the channel access sublayer: making and breaking physical
                  links (annex B)
- ``sis``         the subnetwork interface sublayer: clients, soft and hard
                  links, broadcast, delivery confirmation (annex A.1, A.3)
- ``primitives``  the client interface: S_ primitives on the wire (annex A.2)
- ``node``        the three sublayers as one node, on a half-duplex modem
"""
