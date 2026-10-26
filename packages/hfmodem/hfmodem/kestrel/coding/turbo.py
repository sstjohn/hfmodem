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
from numpy.lib.stride_tricks import sliding_window_view


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
_DELTA0=np.where(np.arange(8)==0,0.0,NEG)

# Parallel-window BCJR: the trellis is cut into windows of _WIN steps, each entered from
# a uniform metric _WARM steps early, and all windows recurse at once. Eight constraint
# lengths of warm-up leave the frame error rate where the whole-trellis recursion has it;
# a trellis no longer than _WIN is one window with exact boundaries.
_WIN=128
_WARM=32

def _normalise(x):
    m=x.max(1,keepdims=True)
    return x-np.where(m>NEG/2,m,0.0)

def _bcjr(Ls, Lp, La, tail_s, tail_p):
    """LLR convention L=log P(0)/P(1). tail_s/tail_p: 3 tail (sys,par) LLRs -> forces term to state 0."""
    N=len(Ls); T=N+3
    sysL=np.concatenate([Ls, tail_s]); parL=np.concatenate([np.nan_to_num(Lp,nan=0.0), tail_p])
    aprL=np.concatenate([La, np.zeros(3)])
    L=min(_WIN,T); D=min(_WARM,L); nw=-(-T//L)
    gam=np.zeros((D+nw*L+D,8,2))         # branch metrics, zero-padded by the warm-up at each end
    gam[D:D+T]=(0.5*(sysL+aprL))[:,None,None]*_SGN_U + (0.5*parL)[:,None,None]*_SGN_P
    # (nw,8,2,L+D) views: window w's k-th forward step is gfwd[...,k], its k-th backward
    # step gbwd[...,L+D-1-k]
    gfwd=sliding_window_view(gam[:,_PS,_PU],L+D,axis=0)[:nw*L:L]
    gbwd=sliding_window_view(gam[D:],L+D,axis=0)[::L]
    alpha=np.empty((nw,L,8)); beta=np.empty((nw,L,8))
    a=np.zeros((nw,8)); b=np.zeros((nw,8))
    k_end=(np.arange(nw)+1)*L+D-T        # the step at which a window's beta reaches the tail's end
    for k in range(L+D):
        if k==D: a[0]=_DELTA0
        b[k_end==k]=_DELTA0
        if k>=D:
            alpha[:,k-D]=a; beta[:,L-1-(k-D)]=b
            if k==L+D-1: break
        c=a[:,_PS]+gfwd[...,k]
        # The sentinel mask is also a metric-underflow guard: it fires past the trellis
        # fill once the normalised alpha spread reaches the NEG scale (from max|LLR| ~ 9e7,
        # far above anything this receiver feeds it). Skipped only when nothing is dead.
        if (a<=NEG/2).any():
            live=(a>NEG/2)[:,_PS]
            c=np.where(live,c,NEG)
            a=_normalise(np.where(live.any(2),_maxstar(c[...,0],c[...,1]),NEG))
        else:
            a=_normalise(_maxstar(c[...,0],c[...,1]))
        c=b[:,NS]+gbwd[...,L+D-1-k]
        b=_normalise(_maxstar(np.maximum(c[...,0],NEG),c[...,1]))
    alpha=alpha.reshape(-1,8)[:N]; beta=beta.reshape(-1,8)[:N]
    M=(alpha[:,:,None]+(0.5*parL[:N])[:,None,None]*_SGN_P)+beta[:,NS]            # (N,8,2)
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
    work = _iterate_steps(Lsys, Lp1, Lp2, tail, perm, iters, check)
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def _iterate_steps(Lsys, Lp1, Lp2, tail, perm, iters, check):
    N=len(Lsys)
    iperm=np.empty(N,int); iperm[perm]=np.arange(N)
    ts1,tp1,ts2,tp2=tail
    La=np.zeros(N)
    bits=(Lsys<0).astype(int)
    passed=None
    for iteration in range(iters):
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
        if (iteration + 1) % 4 == 0 and iteration + 1 < iters:
            yield
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


def decode_from_coded_llr_steps(Lc, N, iters=12, check=None):
    perm = interleaver(N)
    Lpar = Lc[1:2*N:2]
    Lp1 = np.full(N, np.nan)
    Lp2 = np.full(N, np.nan)
    Lp1[::2], Lp2[1::2] = Lpar[::2], Lpar[1::2]
    return (yield from _iterate_steps(Lc[0:2*N:2].copy(), Lp1, Lp2,
                                      _split_tail(Lc[2*N:]), perm, iters, check))


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
# The group law read off high speed capable BW2300 and BW500 tapes (KPA-bw2300-const.md §5,
# KPA-bw500-const.md §7): the info bits go in groups of k, each group led by one parity bit
# after its first systematic bit, `s[kg], P, s[kg+1] .. s[kg+k-1]`, P = p1[kg] for even g and
# p2[kg] for odd g; then the 12-bit dual tail, then zero padding to `coded_len`. k = 1 is the
# rate-1/2 mux of `encode`.

@_cache
def _groups(N, coded_len):
    """(k, systematic positions, group parity positions, tail start) for the group law."""
    P = coded_len - N - 12
    k = round(N / P) if P > 0 else 0
    tail = N // k * (k + 1) if k else 0
    if k < 1 or N % k or tail + 12 > coded_len:
        raise ValueError(f"no group puncture carries N={N} in {coded_len} coded bits")
    i = np.arange(N)
    j = i % k
    spos = i // k * (k + 1) + j + (j > 0)
    ppos = np.arange(0, tail, k + 1) + 1
    for a in (spos, ppos):
        a.setflags(write=False)
    return k, spos, ppos, tail


def encode_punctured(inp, perm, coded_len):
    """(13,15) turbo encode, group-punctured to exactly ``coded_len`` bits."""
    inp = np.asarray(inp, int); N = len(inp)
    k, spos, ppos, tail = _groups(N, coded_len)
    par1, ts1, tp1 = _enc_constituent(inp)
    par2, ts2, tp2 = _enc_constituent(inp[perm])
    lead = np.arange(0, N, k)
    coded = np.zeros(coded_len, int)
    coded[spos] = inp
    coded[ppos] = np.where(np.arange(len(lead)) % 2 == 0, par1[lead], par2[lead])
    coded[tail:tail + 12] = np.column_stack([ts1, tp1, ts2, tp2]).ravel()
    return coded


def decode_punctured(Lc, N, perm, coded_len, iters=8, check=None):
    """Log-MAP decode of :func:`encode_punctured`. Lc: length ``coded_len`` channel LLRs."""
    k, spos, ppos, tail = _groups(N, coded_len)
    lead = np.arange(0, N, k)
    odd = np.arange(len(lead)) % 2 == 1
    Lp1 = np.full(N, np.nan); Lp1[lead[~odd]] = Lc[ppos[~odd]]
    Lp2 = np.full(N, np.nan); Lp2[lead[odd]] = Lc[ppos[odd]]
    return _iterate(Lc[spos], Lp1, Lp2, _split_tail(Lc[tail:tail + 12]), perm, iters, check)
