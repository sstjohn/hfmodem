# Mesh scope

Sabir implements two-party ARQ sessions and connectionless objects. It does not
implement a mesh MAC, routing, channel reservations or KISS host framing.

A future mesh mode would need an independently specified addressing and access
protocol, including hidden terminals, asymmetric reception, collisions,
reservation expiry and timing uncertainty. OFDM carrier groups alone do not
supply those functions or establish interference-free subchannels. No current
capability bit, frame type or host command is reserved for that hypothetical
mode.
