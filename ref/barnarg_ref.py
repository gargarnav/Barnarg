"""Barnarg CPU reference: Mersenne PRP squaring with an FP32 FFT + GF(M31^2) NTT hybrid.

The GPU kernels must match this file word for word. Algorithm (one squaring of
x mod M_p, x held as N balanced digits, digit j worth 2^s_j, s_j = ceil(j*p/N)):

  1. FP32 side: weighted real FFT in float32, weights a_j = 2^(s_j - j*p/N),
     square pointwise, inverse, unweight -> approximate z_j (a float).
  2. GF(M31^2) side: the same weighted cyclic convolution done exactly mod
     M31 = 2^31 - 1, with a length-N NTT over GF(M31^2) = GF(M31)[i], i^2 = -1.
     The weight is a field element t^-jp * 2^s_j with t^N = 2; t = 2^u,
     u = N^-1 mod 31, so every weight is a power of 2 (a 31-bit rotation).
  3. CRT per word: z_j = r_j + M31 * round((zfp_j - r_j) / M31), exact while the
     FP32 error is below M31/2. The largest |zfp_j - z_j| is reported as the
     round-off error (ROE) that has to stay well below 2^30.
  4. Carry: turn z back into balanced digits of width b_j = s_{j+1} - s_j; the
     carry out of the top digit wraps to digit 0 (2^p = 1 mod M_p).

Usage:
  python3 ref/barnarg_ref.py selftest      # full PRPs + checkpoints vs GMP
  python3 ref/barnarg_ref.py prp P [N]      # one PRP, optional FFT length
  python3 ref/barnarg_ref.py limit          # round-off near the bits-per-word limit
"""
import json
import math
import os
import sys
import time

import numpy as np

M31 = (1 << 31) - 1
U64 = np.uint64
P64 = U64(M31)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- GF(M31) and GF(M31^2) arithmetic on uint64 arrays ----------


def mmul(a, b):
    return (a * b) % P64


def cmul(ar, ai, br, bi):
    """(ar + i ai)(br + i bi) in GF(M31^2), i^2 = -1."""
    rr = (mmul(ar, br) + (P64 - mmul(ai, bi))) % P64
    ri = (mmul(ar, bi) + mmul(ai, br)) % P64
    return rr, ri


def cpow(r, i, e):
    rr, ri = 1, 0
    while e:
        if e & 1:
            rr, ri = (rr * r - ri * i) % M31, (rr * i + ri * r) % M31
        r, i = (r * r - i * i) % M31, (2 * r * i) % M31
        e >>= 1
    return rr, ri


ORDER = M31 * M31 - 1          # = 2^32 * 3^2 * 7 * 11 * 31 * 151 * 331
ODD_PART = ORDER >> 32


def root_of_unity(N):
    """Primitive N-th root of unity in GF(M31^2), N a power of 2 up to 2^32."""
    for a in range(1, 100):
        for b in range(1, 100):
            g = cpow(a, b, ODD_PART)                     # element of 2-power order
            if cpow(*g, 1 << 31) != (1, 0):              # order exactly 2^32
                return cpow(*g, (1 << 32) // N)
    raise RuntimeError("no generator found")


class NTT:
    """Length-N radix-2 NTT over GF(M31^2), iterative, bit-reversed input order."""

    def __init__(self, N):
        self.N = N
        self.log = N.bit_length() - 1
        self.rev = np.array([int(f"{j:0{self.log}b}"[::-1], 2) for j in range(N)]) if N > 1 else np.zeros(1, int)
        w = root_of_unity(N)
        winv = cpow(*w, N - 1)
        self.tw = [self._twiddles(w), self._twiddles(winv)]
        self.ninv = pow(N, -1, M31)

    def _twiddles(self, w):
        out = []
        m = 2
        while m <= self.N:
            wm = cpow(*w, self.N // m)
            r = np.empty(m // 2, dtype=U64); i = np.empty(m // 2, dtype=U64)
            cr, ci = 1, 0
            for k in range(m // 2):
                r[k], i[k] = cr, ci
                cr, ci = (cr * wm[0] - ci * wm[1]) % M31, (cr * wm[1] + ci * wm[0]) % M31
            out.append((r, i))
            m *= 2
        return out

    def transform(self, re, im, inverse=False):
        re = re[self.rev].copy(); im = im[self.rev].copy()
        N = self.N
        for s, (wr, wi) in enumerate(self.tw[1 if inverse else 0]):
            m = 2 << s
            h = m // 2
            R = re.reshape(N // m, m); I = im.reshape(N // m, m)
            ur, ui = R[:, :h].copy(), I[:, :h].copy()
            vr, vi = cmul(R[:, h:], I[:, h:], wr, wi)
            R[:, :h] = (ur + vr) % P64; I[:, :h] = (ui + vi) % P64
            R[:, h:] = (ur + (P64 - vr)) % P64; I[:, h:] = (ui + (P64 - vi)) % P64
        if inverse:
            n = U64(self.ninv)
            re, im = mmul(re, n), mmul(im, n)
        return re, im


# ---------- the Mersenne IBDWT ----------


class Mersenne:
    def __init__(self, p, N):
        self.p, self.N = p, N
        s = np.array([-((-j * p) // N) for j in range(N + 1)], dtype=np.int64)  # ceil(j p / N)
        self.s = s[:-1]
        self.width = (s[1:] - s[:-1]).astype(np.int64)
        assert self.width.min() >= 1 and self.width.max() <= 40
        frac = self.s - np.arange(N, dtype=np.float64) * p / N               # in [0, 1)
        self.w32 = np.exp2(frac).astype(np.float32)                          # FP32 weights
        self.iw32 = (1.0 / np.exp2(frac)).astype(np.float32)
        # M31 weights: 2^(s_j - j*p*u) mod M31 with u = N^-1 mod 31 (2^31 = 1 mod M31)
        u = pow(N % 31, -1, 31)
        e = (self.s - np.arange(N, dtype=np.int64) * (p * u % 31)) % 31
        self.wexp = e                                                        # weight = 2^wexp
        self.ntt = NTT(N)
        self.roe = 0.0

    # x is an int64 array of balanced digits
    def from_int(self, v):
        v %= (1 << self.p) - 1
        d = np.zeros(self.N, dtype=np.int64)
        for j in range(self.N):
            d[j] = v & ((1 << int(self.width[j])) - 1)
            v >>= int(self.width[j])
        return self.carry(d.astype(object))

    def to_int(self, d):
        v = 0
        for j in reversed(range(self.N)):
            v = (v << int(self.width[j])) + int(d[j])
        return v % ((1 << self.p) - 1)

    def carry(self, z):
        """Exact integer carry propagation to balanced digits (Python ints, wrap-around)."""
        z = [int(t) for t in z]
        N = self.N
        c = 0
        for _ in range(3):  # the wrap carry is tiny after the first pass
            for j in range(N):
                t = z[j] + c
                b = int(self.width[j])
                lo = t & ((1 << b) - 1)
                if lo >= 1 << (b - 1):
                    lo -= 1 << b
                z[j] = lo
                c = (t - lo) >> b
            if c == 0:
                break
        assert c == 0
        return np.array(z, dtype=np.int64)

    def rot31(self, x, e):
        """x * 2^e mod M31 for x in [0, M31), e in [0, 31): a rotation of 31 bits."""
        e = e.astype(U64)
        return ((x << e) & P64) | (x >> (U64(31) - e))

    def square(self, d):
        N = self.N
        # FP32 path
        xf = d.astype(np.float32) * self.w32
        f = np.fft.rfft(xf)
        zf = np.fft.irfft(f * f, n=N).astype(np.float32) * self.iw32
        zf = zf.astype(np.float64)
        # GF(M31^2) path: digits mod M31, weighted by rotation, NTT, square, inverse
        xm = (d % M31).astype(U64)
        xm = self.rot31(xm, self.wexp) % P64
        re, im = self.ntt.transform(xm, np.zeros(N, dtype=U64))
        re, im = cmul(re, im, re, im)
        re, im = self.ntt.transform(re, im, inverse=True)
        assert not im.any(), "imaginary part should vanish for real input"
        r = self.rot31(re, (31 - self.wexp) % 31) % P64                   # unweight
        # CRT per word
        r = r.astype(np.int64)
        r = np.where(r > M31 // 2, r - M31, r)                             # balanced residue
        k = np.rint((zf - r) / M31)
        z = r.astype(object) + M31 * k.astype(np.int64).astype(object)
        err = np.max(np.abs(zf - np.array([float(t) for t in z])))
        self.roe = max(self.roe, err)
        return self.carry(z)


def res64(v):
    return format(v & ((1 << 64) - 1), "016x")


def pick_N(p, margin=0.5):
    """Smallest power-of-2 N within the model's bits-per-word limit minus a margin."""
    for logN in range(5, 33):
        bmax = (51 - math.log2(0.272 * logN) - logN / 2) / 2 - margin
        if p / (1 << logN) <= bmax:
            return 1 << logN


def prp(p, N=None, checkpoints=(), quiet=False):
    N = N or pick_N(p)
    m = Mersenne(p, N)
    d = m.from_int(3)
    got = {}
    t0 = time.time()
    for k in range(1, p + 1):
        d = m.square(d)
        if k in checkpoints:
            got[k] = res64(m.to_int(d))
    v = m.to_int(d)
    if not quiet:
        print(f"p={p} N={N} bpw={p / N:.2f} prp={v == 9} res64={res64(v)} "
              f"ROE={m.roe:.3g} (log2 {math.log2(max(m.roe, 1e-9)):.1f}, limit 30) {time.time() - t0:.1f}s")
    return v, got, m.roe


def selftest():
    vecs = {v["p"]: v for v in json.load(open(os.path.join(HERE, "ref_vectors.json")))}
    ok = True
    # full PRPs, small exponents (primes and composites)
    for p in [521, 523, 607, 1277, 1279, 2203, 2207, 2281, 3217, 4253, 4421, 4423]:
        v, got, roe = prp(p, checkpoints={1000}, quiet=True)
        want = vecs[p]
        good = (res64(v) == want["res64_at"][str(p)]) and (v == 9) == want["mersenne_prime"]
        if "1000" in want["res64_at"]:
            good &= got.get(1000) == want["res64_at"]["1000"]
        ok &= good
        print(f"{'ok ' if good else 'BAD'} p={p:6d} N={pick_N(p):5d} prp={v == 9} res64={res64(v)} ROE log2={math.log2(max(roe, 1e-9)):.1f}")
    # first 1000 squarings at larger exponents
    for p in [9689, 9941, 11213, 19937, 21701, 44497, 86243]:
        N = pick_N(p)
        m = Mersenne(p, N)
        d = m.from_int(3)
        for k in range(1000):
            d = m.square(d)
        good = res64(m.to_int(d)) == vecs[p]["res64_at"]["1000"]
        ok &= good
        print(f"{'ok ' if good else 'BAD'} p={p:6d} N={N:5d} bpw={p / N:5.2f} 1000 squarings res64 match, ROE log2={math.log2(max(m.roe, 1e-9)):.1f}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def limit():
    """Push bits per word up at fixed N to find where the CRT breaks."""
    print("N      bpw    ROE log2 (CRT fails near 30)   check vs GMP after 200 squarings")
    import gmpy2
    for logN in (8, 10, 12):
        N = 1 << logN
        for bpw in (16, 18, 20, 21, 22, 23):
            p = int(gmpy2.next_prime(bpw * N))
            m = Mersenne(p, N)
            mp = (gmpy2.mpz(1) << p) - 1
            d = m.from_int(3); x = gmpy2.mpz(3)
            for _ in range(200):
                d = m.square(d); x = x * x % mp
            good = m.to_int(d) == x
            print(f"{N:5d}  {p / N:5.2f}   {math.log2(max(m.roe, 1e-9)):5.1f}                      {'match' if good else 'WRONG'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "prp":
        prp(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == "limit":
        limit()
