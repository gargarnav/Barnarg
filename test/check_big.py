"""Check a Barnarg GPU dump at large p cheaply: compare the digits' value modulo a
61-bit prime q with GMP's 3^(2^k) mod M_p, allowing the balanced form to differ by
a small multiple of M_p. Avoids building a multi-million-word integer.

Usage: python3 test/check_big.py DUMP P
"""
import struct
import sys

import gmpy2
import numpy as np

path, p = sys.argv[1], int(sys.argv[2])
q = (1 << 61) - 1
with open(path, "rb") as f:
    k, N = struct.unpack("<II", f.read(8))
    d = np.frombuffer(f.read(4 * N), dtype="<i4")
j = np.arange(N + 1, dtype=np.int64)
s = -((-j * p) // N)                        # ceil(j p / N); fits int64 for these sizes
s = s[:-1]
# value mod q = sum d_j * 2^s_j mod q, with 2^s_j = 2^(s_j mod 61) since 2^61 = 1 mod q
e = (s % 61).astype(np.uint64)
v = 0
for ex in range(61):                        # group by exponent: 61 vector sums
    m = e == ex
    if m.any():
        v = (v + int(d[m].astype(np.int64).sum()) * pow(2, ex, q)) % q
mp = (gmpy2.mpz(1) << p) - 1
x = gmpy2.mpz(3)
for _ in range(k):
    x = x * x
    x = (x & mp) + (x >> p)                 # fast reduction mod 2^p - 1
    if x >= mp:
        x -= mp
mpq = int(mp % q)
ok = any((int(x % q) + t * mpq - v) % q == 0 for t in (-2, -1, 0, 1, 2))
print(f"k={k} N={N} value mod (2^61-1) {'matches GMP' if ok else 'WRONG'}")
sys.exit(0 if ok else 1)
