# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW500 / L4 inner FEC — parallel-concatenated turbo, (13,15)-octal RSC, log-MAP.

Built from spec/03 §3.6.4 (constituent RSC recurrence, rate-1/2 puncture, 12-bit
dual-tail termination) and §3.6.3 (turbo interleaver). Round-trip-validated in the
spec: the encoder is byte-exact and the log-MAP decoder reaches 0-error at ~11% raw
cell error on real captured audio.

The inner turbo interleaver is the 368-bit permutation IL[3::17][:368] of the BW500
stage-2 interleaver, which `kestrel/rx/tablegen.py` computes from the published PRNG
recurrence — see spec/03 §3.6.3.

Public API used by the RX:
    decode_from_coded_llr(Lc, N=368, iters=12, check=None) -> 368 hard bits
    encode(inp) -> 748+12 coded bits            (TX / round-trip)
"""

from functools import cache as _cache

import numpy as np


@_cache
def _il():
    # Built on first use rather than at import: the 150000-entry stage-2 walk costs
    # 0.8 s, and a station that never opens a BW500 link never pays it. The import
    # is deferred with it, which also keeps `kestrel.rx` out of this module's import
    # graph while `kestrel.rx` is still initialising.
    from ..rx.tablegen import interleaver as _generate       # noqa: PLC0415
    return _generate("bw500", 2)


@_cache
def interleaver(N):
    # spec/03 §3.6.3: the N-bit turbo permutation = IL[3::17][:N] (a full 0..N-1 perm).
    # Cached: the c0 scan re-encodes 72 candidate frames per burst, and this was
    # re-slicing the table and re-proving the bijection every time. The check is a
    # real assertion rather than an `assert`, which -O would remove after paying for it.
    p = _il()[3::17][:N].copy()
    if not (p.min() == 0 and p.max() == N - 1 and len(np.unique(p)) == N):
        raise ValueError(f"interleaver column is not a 0..{N-1} permutation")
    p.setflags(write=False)
    return p

# ---- (13,15) octal RSC, 8 states (spec/03 §3.6.4).  state s = reg0 | reg1<<1 | reg2<<2 ----
def _step(s, u):
    r0,r1,r2 = s&1,(s>>1)&1,(s>>2)&1
    a = (u ^ r1 ^ r2) & 1              # feedback
    par = (a ^ r0 ^ r2) & 1            # parity
    ns = (a) | (r0<<1) | (r1<<2)       # next state
    return ns, par
NS  = np.array([[_step(s,u)[0] for u in (0,1)] for s in range(8)])
PAR = np.array([[_step(s,u)[1] for u in (0,1)] for s in range(8)])
def _term_u(s):            # termination input that forces feedback a=0 (spec §3.6.4)
    return ((s>>1)&1) ^ ((s>>2)&1)

def _enc_constituent(bits):
    s=0; par=np.empty(len(bits),int)
    for n,u in enumerate(bits):
        par[n]=PAR[s,u]; s=NS[s,u]
    tsys=[]; tpar=[]
    for _ in range(3):
        u=_term_u(s); tsys.append(u); tpar.append(PAR[s,u]); s=NS[s,u]
    assert s==0
    return par, np.array(tsys), np.array(tpar)

def encode(inp, perm=None):
    # spec/03 §3.6.4 rate-1/2: systematic + punctured parity, +12-bit dual tail -> 2N+12
    # `perm` overrides the BW500 turbo interleaver (BW2300 uses its own il2 column).
    inp=np.asarray(inp,int); N=len(inp)
    if perm is None: perm=interleaver(N)
    par1,ts1,tp1=_enc_constituent(inp)
    par2,ts2,tp2=_enc_constituent(inp[perm])
    coded=np.empty(2*N+12,int)
    coded[0:2*N:2]=inp
    pd=par1.copy(); oi=np.arange(1,N,2); pd[oi]=par2[oi]   # parity1 at even i, parity2 at odd i
    coded[1:2*N:2]=pd
    t=[]
    for k in range(3): t+=[ts1[k],tp1[k],ts2[k],tp2[k]]    # tail interleaved [s1,p1,s2,p2]x3
    coded[2*N:]=t
    return coded

# ============================= log-MAP BCJR =============================
# max*(a,b) = max(a,b) + log1p(exp(-|a-b|)) is the identity numpy's logaddexp is built on,
# so this is the same arithmetic to the last bit, in one ufunc call rather than seven.
_maxstar=np.logaddexp
NEG=-1e9
_NEG8=np.full(8,NEG)
_SGN_U=1.0-2.0*np.arange(2)          # (1-2u)          -> (2,)
_SGN_P=1.0-2.0*PAR                   # (1-2*PAR[s,u])  -> (8,2)

def _predecessors():
    """PS[d,k], PU[d,k]: the two (state, input) pairs whose branch enters state d."""
    ps=np.zeros((8,2),int); pu=np.zeros((8,2),int); k=[0]*8
    for s in range(8):
        for u in (0,1):
            d=NS[s,u]; ps[d,k[d]]=s; pu[d,k[d]]=u; k[d]+=1
    assert (np.array(k)==2).all()
    return ps,pu
_PS,_PU=_predecessors()

def _bcjr(Ls, Lp, La, tail_s, tail_p):
    """LLR convention L=log P(0)/P(1). tail_s/tail_p: 3 tail (sys,par) LLRs -> forces term to state 0."""
    N=len(Ls); T=N+3
    sysL=np.concatenate([Ls, tail_s]); parL=np.concatenate([np.nan_to_num(Lp,nan=0.0), tail_p])
    aprL=np.concatenate([La, np.zeros(3)])
    # branch metrics for the whole trellis at once — they carry no recursion
    gam=(0.5*(sysL+aprL))[:,None,None]*_SGN_U + (0.5*parL)[:,None,None]*_SGN_P   # (T,8,2)
    gfwd=gam[:,_PS,_PU]                                                          # (T,8,2) by destination
    alpha=np.full((T+1,8),NEG); alpha[0,0]=0.0
    for t in range(T):
        a=alpha[t]
        c=a[_PS]+gfwd[t]
        # The sentinel mask runs at EVERY step, not just the t<3 trellis fill. All 8
        # states are reachable from t=3 on, so it looks like fill-only bookkeeping — but
        # the scalar original's `alpha[t,s] <= NEG/2` skip is also a metric-underflow
        # guard, and it fires at t>=3 once the normalised alpha spread reaches the
        # sentinel scale (empirically ~5.6x max|LLR|, so from max|LLR| ~ 9e7). Restricting
        # it to t<3 diverged from the original above ~2e8. Unreachable from this receiver
        # — real channel LLRs peak at 12 — but the guard costs one gather per step.
        live=(a>NEG/2)[_PS]
        c=np.where(live,c,NEG)
        nxt=np.where(live.any(1), _maxstar(c[:,0],c[:,1]), NEG)
        m=nxt.max()
        if m>NEG/2: nxt-=m
        alpha[t+1]=nxt
    beta=np.full((T+1,8),NEG); beta[T,0]=0.0    # terminated to state 0
    for t in range(T-1,-1,-1):
        c=beta[t+1][NS]+gam[t]
        b=_maxstar(_maxstar(_NEG8,c[:,0]),c[:,1])
        m=b.max()
        if m>NEG/2: b-=m
        beta[t]=b
    # extrinsic: fold the 8 states per input bit, the whole block at a time
    M=(alpha[:N,:,None]+(0.5*parL[:N])[:,None,None]*_SGN_P)+beta[1:N+1][:,NS]     # (N,8,2)
    l0=np.full(N,NEG); l1=np.full(N,NEG)
    for s in range(8):
        l0=_maxstar(l0,M[:,s,0]); l1=_maxstar(l1,M[:,s,1])
    return l0-l1

def _split_tail(Lt):
    """12 tail LLRs, 4-way interleaved [s1,p1,s2,p2] x3 -> (ts1,tp1,ts2,tp2)."""
    q=np.asarray(Lt,float).reshape(3,4)
    return q[:,0],q[:,1],q[:,2],q[:,3]

def _iterate(Lsys, Lp1, Lp2, tail, perm, iters, check):
    """Shared turbo iteration. `check(bits)->bool` stops early (CRC); None runs all `iters`."""
    N=len(Lsys)
    iperm=np.empty(N,int); iperm[perm]=np.arange(N)
    ts1,tp1,ts2,tp2=tail
    La=np.zeros(N)
    bits=(Lsys<0).astype(int)
    passed=None
    for _ in range(iters):
        Le1=_bcjr(Lsys, Lp1, La, ts1, tp1)
        La=_bcjr(Lsys[perm], Lp2, Le1[perm], ts2, tp2)[iperm]
        bits=(Lsys+Le1+La<0).astype(int)
        if check is not None and check(bits):
            # Stop on the SECOND consecutive pass over the same bits. Checking a 16-bit
            # CRC once per iteration instead of once per decode gives it up to `iters`
            # independent chances to accept noise — measured at ~12x the false-accept
            # rate on pure noise, where every iteration's candidate differs. Requiring
            # the frame to check twice with identical bits costs one extra iteration on
            # a frame that decodes, and puts the false-accept rate below where it was
            # before early termination existed.
            if passed is not None and np.array_equal(bits, passed): break
            passed=bits
        else:
            passed=None
    return bits

def decode_from_coded_llr(Lc, N, iters=12, perm=None, check=None):
    """Lc: length-(2N+12) channel LLRs (log P(0)/P(1)) for the coded bits. Returns N hard bits.

    `perm` overrides the BW500 turbo interleaver (BW2300 supplies its own il2 column).
    `check` is an optional predicate on the candidate hard bits, run after each iteration;
    the iteration stops once it passes twice running on identical bits."""
    if perm is None: perm=interleaver(N)
    Lpar=Lc[1:2*N:2]
    ei=np.arange(0,N,2); oi=np.arange(1,N,2)
    Lp1=np.full(N,np.nan); Lp1[ei]=Lpar[ei]     # parity1 known at even i
    Lp2=np.full(N,np.nan); Lp2[oi]=Lpar[oi]     # parity2 known at odd i
    return _iterate(Lc[0:2*N:2].copy(), Lp1, Lp2, _split_tail(Lc[2*N:]), perm, iters, check)


# ------------------------- RATE-1/3 variant (control levels 0/1) -------------------------
# spec/03 §3.6.4 rate-1/3 row: coded = 3N+12, systematic + BOTH parity streams (no
# puncture), same (13,15) constituents and same 12-bit tail. `perm` = the level's
# turbo interleaver (IL[col::17][:N] for the level-0/1 column).

def encode_r13(inp, perm):
    inp = np.asarray(inp, int); N = len(inp)
    par1, ts1, tp1 = _enc_constituent(inp)
    par2, ts2, tp2 = _enc_constituent(inp[perm])
    coded = np.empty(3 * N + 12, int)
    coded[0:3 * N:3] = inp
    coded[1:3 * N:3] = par1
    coded[2:3 * N:3] = par2
    t = []
    for k in range(3):
        t += [ts1[k], tp1[k], ts2[k], tp2[k]]
    coded[3 * N:] = t
    return coded


def decode_r13(Lc, N, perm, iters=12, check=None):
    """Rate-1/3 log-MAP turbo decode. Lc: length 3N+12 channel LLRs (log P0/P1)."""
    Lp1 = Lc[1:3 * N:3].copy()      # parity1 known at ALL i (no puncture)
    Lp2 = Lc[2:3 * N:3].copy()      # parity2 known at ALL i
    return _iterate(Lc[0:3 * N:3].copy(), Lp1, Lp2, _split_tail(Lc[3 * N:]),
                    perm, iters, check)


# ------------------- PUNCTURED variant (higher-rate high-throughput levels) -------------------
# spec/03 §3.5.4 / 06 §6.1d: the BW2300 high-throughput records reach rates 2/3, 4/5, 5/6
# above the 1/2 base by puncturing parity down to a target coded length. VARA's exact
# BW2300 puncture matrix is UNVALIDATED in the spec (§3.5.4), so this uses a deterministic,
# evenly-spaced parity puncture on the same (13,15) constituents + 12-bit dual tail. It is a
# self-consistent invertible code (TX/RX share the pattern); layout is
#   coded = [ systematic(N) | kept-parity(P) | tail(12) ],  P = coded_len − 12 − N,
# with parity multiplexed par1 at even i / par2 at odd i (matching the rate-1/2 base) and the
# kept subset chosen by even spacing over 0..N−1.

def _punct_keep(N, P):
    if P >= N:
        return np.arange(N)
    return np.unique(np.floor(np.arange(P) * N / P).astype(int))


def encode_punctured(inp, perm, coded_len):
    """(13,15) turbo encode punctured to exactly ``coded_len`` bits (coded_len ≥ N+12)."""
    inp = np.asarray(inp, int); N = len(inp)
    P = coded_len - 12 - N
    if not (0 <= P <= N):
        raise ValueError(f"coded_len {coded_len} out of range for N={N} (need N+12..2N+12)")
    par1, ts1, tp1 = _enc_constituent(inp)
    par2, ts2, tp2 = _enc_constituent(inp[perm])
    pmux = np.where(np.arange(N) % 2 == 0, par1, par2)
    keep = _punct_keep(N, P)
    coded = np.empty(coded_len, int)
    coded[:N] = inp
    coded[N:N + P] = pmux[keep]
    t = []
    for k in range(3):
        t += [ts1[k], tp1[k], ts2[k], tp2[k]]
    coded[N + P:] = t
    return coded


def decode_punctured(Lc, N, perm, coded_len, iters=8, check=None):
    """Log-MAP decode of :func:`encode_punctured`. Lc: length ``coded_len`` channel LLRs."""
    P = coded_len - 12 - N
    keep = _punct_keep(N, P)
    pmux = np.zeros(N); pmux[keep] = Lc[N:N + P]
    ev = np.arange(N) % 2 == 0
    Lp1 = np.where(ev, pmux, np.nan)          # parity1 known at kept even i
    Lp2 = np.where(~ev, pmux, np.nan)         # parity2 known at kept odd i
    return _iterate(Lc[:N].copy(), Lp1, Lp2, _split_tail(Lc[N + P:]), perm, iters, check)
