# prp proofs (pietrzak), the slow and simple way, with gmpy2.
#
# same format as gpuowl / prmers / prime95 (proof file version 2):
#   header lines, then the residue at k = E, then the power middles,
#   each as ceil(E/8) little-endian bytes. x_k = 3^(2^k) mod 2^E - 1.
#
#   python3 ref/proof_ref.py make 86243 5 out.proof    # run the prp and write a proof
#   python3 ref/proof_ref.py verify out.proof           # check any proof file
import hashlib
import sys

import gmpy2


def points(E, power):
    # the iterations a run has to keep for a proof of this power, in order.
    # the last one is E itself (it takes the place of 0).
    pts = [0]
    span = (E + 1) // 2
    for _ in range(power):
        pts += [x + span for x in pts]
        span = (span + 1) // 2
    pts[0] = E
    return sorted(pts)


def to_bytes(x, E):
    return int(x).to_bytes((E - 1) // 8 + 1, "little")


def h_words(E, x, prefix=b""):
    return hashlib.sha3_256(prefix + to_bytes(x, E)).digest()


def h64(d):
    return int.from_bytes(d[:8], "little")


def residues(E, keep):
    mp = gmpy2.mpz(2) ** E - 1
    x = gmpy2.mpz(3)
    out = {}
    want = set(keep)
    for k in range(1, E + 1):
        x = x * x % mp
        if k in want:
            out[k] = x
    return out


def make(E, power):
    mp = gmpy2.mpz(2) ** E - 1
    pts = points(E, power)
    res = residues(E, pts)
    B = res[E]
    d = h_words(E, B)
    middles, hashes = [], []
    for p in range(power):
        s = 1 << (power - p - 1)
        stack = []
        for i in range(1 << p):
            stack.append(res[pts[s * (2 * i + 1) - 1]])
            k = 0
            while i & (1 << k):
                top = stack.pop()
                stack[-1] = gmpy2.powmod(stack[-1], hashes[p - 1 - k], mp) * top % mp
                k += 1
        assert len(stack) == 1
        M = stack[0]
        d = h_words(E, M, d)
        middles.append(M)
        hashes.append(h64(d))
    return B, middles


def verify(E, B, middles):
    mp = gmpy2.mpz(2) ** E - 1
    A = gmpy2.mpz(3)
    d = h_words(E, B)
    span = E
    for M in middles:
        d = h_words(E, M, d)
        h = h64(d)
        if span % 2:
            B = B * B % mp
        B = gmpy2.powmod(M, h, mp) * B % mp
        A = gmpy2.powmod(A, h, mp) * M % mp
        span = (span + 1) // 2
    return gmpy2.powmod(A, gmpy2.mpz(2) ** span, mp) == B


def write(path, E, B, middles):
    with open(path, "wb") as f:
        f.write(f"PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER={len(middles)}\nNUMBER=M{E}\n".encode())
        for x in [B] + middles:
            f.write(to_bytes(x, E))


def read(path):
    data = open(path, "rb").read()
    lines = data.split(b"\n", 5)
    assert lines[0] == b"PRP PROOF" and lines[1] == b"VERSION=2" and lines[2] == b"HASHSIZE=64"
    power = int(lines[3].split(b"=")[1])
    E = int(lines[4].split(b"=M")[1])
    body = lines[5]
    n = (E - 1) // 8 + 1
    assert len(body) == n * (power + 1), "wrong length"
    xs = [gmpy2.mpz(int.from_bytes(body[i * n:(i + 1) * n], "little")) for i in range(power + 1)]
    return E, xs[0], xs[1:]


if __name__ == "__main__":
    if sys.argv[1] == "make":
        E, power, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        B, middles = make(E, power)
        write(out, E, B, middles)
        print(f"M{E} power {power}: res64 {int(B) & (2**64 - 1):016x}, wrote {out}")
    else:
        E, B, middles = read(sys.argv[2])
        ok = verify(E, B, middles)
        print(f"M{E} power {len(middles)}: {'proof ok' if ok else 'proof BAD'}  res64 {int(B) & (2**64 - 1):016x}")
        sys.exit(0 if ok else 1)
