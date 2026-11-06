"""Check Barnarg GPU dumps against GMP and against the CPU reference, word for word.

Usage:
  python3 test/check.py DUMP P [--ref K]
    --ref K: also run barnarg_ref.py for the first K squarings and compare digits exactly.
Prints one line per checkpoint and exits 1 on any mismatch.
"""
import os
import struct
import sys

import gmpy2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
import barnarg_ref  # noqa: E402


def read(path):
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if not h:
                break
            k, N = struct.unpack("<II", h)
            d = np.frombuffer(f.read(4 * N), dtype="<i4").astype(np.int64)
            out.append((k, N, d))
    return out


def main():
    path, p = sys.argv[1], int(sys.argv[2])
    refk = int(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else 0
    dumps = read(path)
    mp = (gmpy2.mpz(1) << p) - 1
    ok = True
    m = barnarg_ref.Mersenne(p, dumps[0][1]) if dumps else None
    refd = {}
    if refk:
        d = m.from_int(3)
        for k in range(1, refk + 1):
            d = m.square(d)
            refd[k] = d.copy()
    x, kx = gmpy2.mpz(3), 0
    for k, N, d in dumps:
        while kx < k:
            x = x * x % mp
            kx += 1
        v = m.to_int(d)
        good = v == x
        line = f"k={k:7d} value {'matches GMP' if good else 'WRONG vs GMP'}  res64={barnarg_ref.res64(v)}"
        if k in refd:
            same = np.array_equal(refd[k], d)
            line += f"  digits {'identical to' if same else 'DIFFER from'} CPU reference"
            good &= same
        if k == p:
            line += f"  PRP={'yes' if v == 9 else 'no'}"
        ok &= good
        print(("ok  " if good else "BAD ") + line)
    print("CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
