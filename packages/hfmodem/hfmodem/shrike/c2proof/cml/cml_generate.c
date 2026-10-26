/* SPDX-License-Identifier: LGPL-2.1-or-later
 * SPDX-FileCopyrightText: 2005-2006 Matthew C. Valenti and Yufei Wu
 * SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
 * Adapted from CML srandom.h. See upstream/srandom.h, ../NOTICE.md and ../COPYING.LESSER.
 * Deterministic seed, explicit accepted-swap budgets, portable 64-bit counters,
 * optional proposal cap, and an exact candidate cache. No tables are embedded.
 * stdout: final permutation, uint16 little endian. stderr: stage JSON records.
 * cc -O3 -std=c11 -Wall -Wextra -o cml_generate cml_generate.c -lm
 * Do not use -ffast-math: index mapping intentionally retains CML doubles.
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IM1 INT64_C(2147483563)
#define IM2 INT64_C(2147483399)
#define IMM1 (IM1-1)
#define NDIV (1+IMM1/32)
#define AM (1.0/IM1)
#define RNMX (1.0-1.2e-7)

static int64_t idum, idum2, iy, iv[32];
static uint64_t raw_draws;
static void fail(const char *s) { fprintf(stderr,"%s\n",s); exit(2); }
static uint64_t number(const char *s) {
    char *end; errno=0;
    if (!*s || *s=='-') fail("expected a nonnegative decimal integer");
    unsigned long long v=strtoull(s,&end,10);
    if (errno || *end) fail("invalid integer");
    return (uint64_t)v;
}
static void *alloc(size_t count,size_t size) {
    void *p=calloc(count,size); if (!p) fail("allocation failed"); return p;
}
static double ran2(void) {
    int64_t k=idum/53668;
    idum=40014*(idum-k*53668)-k*12211; if(idum<0) idum+=IM1;
    k=idum2/52774;
    idum2=40692*(idum2-k*52774)-k*3791; if(idum2<0) idum2+=IM2;
    int j=(int)(iy/NDIV);
    iy=iv[j]-idum2; iv[j]=idum; if(iy<1) iy+=IMM1;
    ++raw_draws;
    double x=AM*iy; return x>RNMX ? RNMX : x;
}
static void seed_rng(uint64_t seed) {
    idum=(int64_t)(seed ? seed : 1); idum2=idum;
    for(int j=39;j>=0;--j) {
        int64_t k=idum/53668;
        idum=40014*(idum-k*53668)-k*12211; if(idum<0) idum+=IM1;
        if(j<32) iv[j]=idum;
    }
    iy=iv[0]; raw_draws=0;
    (void)ran2(); /* CML genrand discards the seeding call's result. */
    raw_draws=0; /* Report draws after that one-time initialization. */
}
static void walk(int *a,int n) {
    int w=(int)sqrt((double)n),off=0,block=0;
    for(int i=0;i<n;++i) {
        if(w*off+block>=n) { off=0; ++block; }
        a[i]=w*off+block; ++off;
    }
}
/* Literal pre-swap, half-open guards, including old values at both positions. */
static int fits(const int *a,int n,int s,int v,int p) {
    int lo=p<s ? 0 : p-s, hi=p>=n-s ? n : p+s;
    for(int k=lo;k<hi;++k) if(abs(v-a[k])<s) return 0;
    return 1;
}
/* cover[v] counts forbidden window intervals containing v. A zero exists iff
 * at least one permutation value could fit. This is necessary, not sufficient.
 * Cache updates never consume RNG and never turn a rejection into acceptance. */
static uint32_t *cover;
static int free_values;
static void change_cover(int n,int s,int v,int delta) {
    int lo=v-s+1<0 ? 0 : v-s+1, hi=v+s>n ? n : v+s;
    for(int k=lo;k<hi;++k) {
        if(delta>0) { if(cover[k]++==0) --free_values; }
        else { if(!cover[k]) fail("cache underflow"); if(--cover[k]==0) ++free_values; }
    }
}
static void cache_range(const int *a,int n,int s,int lo,int hi,unsigned char *candidate) {
    int left=lo<s ? 0 : lo-s, right=left; free_values=n;
    for(int p=lo;p<hi;++p) {
        int start=p<s ? 0 : p-s, stop=p>=n-s ? n : p+s;
        for(;left<start;++left) change_cover(n,s,a[left],-1);
        for(;right<stop;++right) change_cover(n,s,a[right],1);
        candidate[p]=(unsigned char)(free_values>0);
    }
    for(;left<right;++left) change_cover(n,s,a[left],-1);
}
static void refresh(const int *a,int n,int s,int i,unsigned char *candidate) {
    /* i belongs to [p-s,p+s) iff p is in [i-s+1,i+s], clipped. */
    int lo=i-s+1<0 ? 0 : i-s+1, hi=i+s+1>n ? n : i+s+1;
    cache_range(a,n,s,lo,hi,candidate);
}
int main(int argc,char **argv) {
    if(argc<7 || (argc-4)%3) {
        fprintf(stderr,"usage: cml_generate scalar|cached|audit SEED MAX_PROPOSALS_PER_STAGE N S SWAPS [N S SWAPS ...]\n");
        return 2;
    }
    int cached=!strcmp(argv[1],"cached") || !strcmp(argv[1],"audit");
    int audit=!strcmp(argv[1],"audit");
    if(!cached && strcmp(argv[1],"scalar")) fail("unknown method");
    uint64_t seed=number(argv[2]),cap=number(argv[3]);
    if(seed>=(uint64_t)IM2) fail("seed must be less than 2147483399");
    seed_rng(seed); int *a=NULL,n=0;
    for(int q=4;q<argc;q+=3) {
        uint64_t nn=number(argv[q]),ss=number(argv[q+1]),want=number(argv[q+2]);
        if(nn<1 || nn>65535 || ss<1 || ss>nn) fail("require 1 <= S <= N <= 65535");
        n=(int)nn; int s=(int)ss;
        free(a); a=alloc((size_t)n,sizeof *a); walk(a,n);
        unsigned char *candidate=NULL;
        if(cached) {
            cover=alloc((size_t)n,sizeof *cover); candidate=alloc((size_t)n,1);
            cache_range(a,n,s,0,n,candidate);
        }
        uint64_t swaps=0,pairs=0;
        while(swaps<want) {
            if(cap && pairs==cap) {
                fprintf(stderr,"{\"stage\":%d,\"status\":\"incomplete_cap\",\"pairs\":%" PRIu64 ",\"swaps\":%" PRIu64 "}\n",(q-4)/3,pairs,swaps);
                free(candidate); free(cover); free(a); return 3;
            }
            if(pairs==UINT64_MAX || raw_draws>UINT64_MAX-2) fail("counter overflow");
            /* Separate statements guarantee original draw order in C. */
            int i=(int)floor(n*ran2()); int j=(int)floor(n*ran2()); ++pairs;
            int accept=0;
            if(i!=j && (!cached || (candidate[i] && candidate[j])))
                accept=fits(a,n,s,a[j],i) && fits(a,n,s,a[i],j);
            if(audit) {
                int literal=i!=j && fits(a,n,s,a[j],i) && fits(a,n,s,a[i],j);
                if(literal!=accept) fail("candidate cache differs from scalar decision");
            }
            if(!accept) continue;
            int t=a[i]; a[i]=a[j]; a[j]=t; ++swaps;
            if(cached) { refresh(a,n,s,i,candidate); refresh(a,n,s,j,candidate); }
        }
        fprintf(stderr,"{\"stage\":%d,\"status\":\"complete\",\"n\":%d,\"s\":%d,\"swaps\":%" PRIu64 ",\"pairs\":%" PRIu64 ",\"raw_draws\":%" PRIu64 "}\n",(q-4)/3,n,s,swaps,pairs,raw_draws);
        free(candidate); free(cover); cover=NULL;
    }
    for(int i=0;i<n;++i) {
        if(putchar(a[i]&255)==EOF || putchar((a[i]>>8)&255)==EOF) fail("output failed");
    }
    free(a); return fflush(stdout)==0 ? 0 : 2;
}
