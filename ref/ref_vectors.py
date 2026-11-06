"""Reference residues for testing a Mersenne PRP tester, computed with GMP.

x_0 = 3, x_k = x_{k-1}^2 mod M_p, so x_k = 3^(2^k) mod M_p. A tester that
squares p times gets x_p = 3^(2^p) = 3^(M_p + 1) = 9 * 3^(M_p - 1) mod M_p.
M_p is a base-3 PRP iff x_p == 9. res64 is the low 64 bits, in hex.
Also records the GIMPS PRP type 1 res64, 3^(M_p - 1) mod M_p.

Usage: python3 ref/ref_vectors.py > ref/ref_vectors.json   (takes about 15 min, M216091 is slow)
"""
import json
import sys
import gmpy2

PRIMES = [521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213,
          19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091]
COMPOSITE = [523, 1277, 2207, 4421, 9973, 19927, 44483, 86249, 110501, 216103]


def vec(p):
    m = (gmpy2.mpz(1) << p) - 1
    x = gmpy2.mpz(3)
    checkpoints = {}
    marks = {1000, 10000} | {p}
    for k in range(1, p + 1):
        x = x * x % m
        if k in marks:
            checkpoints[str(k)] = format(int(x) & (2**64 - 1), "016x")
    type1 = gmpy2.powmod(3, m - 1, m)
    return {
        "p": p,
        "mersenne_prime": p in PRIMES,
        "prp": x == 9,
        "res64_at": checkpoints,
        "type1_res64": format(int(type1) & (2**64 - 1), "016x"),
    }


out = [vec(p) for p in sorted(PRIMES + COMPOSITE)]
for v in out:
    assert v["prp"] == v["mersenne_prime"], v["p"]
json.dump(out, sys.stdout, indent=1)
print()
