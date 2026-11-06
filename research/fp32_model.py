"""How many bits per word can an FP32 FFT carry in a Mersenne IBDWT square?

Builds random balanced IBDWT digits for M_p at length N, squares them with a
weighted FFT in float32 and in float64, and measures the float32 error against
the exact result. The exact result is the float64 result rounded, used only
where float64 is exact by a wide margin (2b + log2 N <= 40). The float32 error
scales as K * 2^(2b + log2 N - 24); K is fitted, then used to find the largest
b where:
  - an FP32+GF(M31^2) hybrid works: FP32 error below 2^27 (M31/2 = 2^30, with an 8x margin);
  - FP64 alone works: error below 0.25 (for comparison with PRPLL/Prime95 limits).
This is a CPU model with numpy's pocketfft twiddles; a GPU FP32 FFT will be somewhat worse.

Usage: python3 research/fp32_model.py
"""
import math
import numpy as np

rng = np.random.default_rng(1)


def digits(p, N):
    s = np.array([math.ceil(j * p / N) for j in range(N + 1)], dtype=np.int64)
    width = s[1:] - s[:-1]
    w = np.exp2((s[:-1] - np.arange(N) * p / N))  # IBDWT weights in [1, 2)
    x = np.array([rng.integers(-(1 << (int(b) - 1)), 1 << (int(b) - 1)) for b in width], dtype=np.float64)
    return x, w


def square(x, w, dtype):
    xw = (x * w).astype(dtype)
    f = np.fft.rfft(xw)
    f = f * f
    z = np.fft.irfft(f, n=len(x))
    return z.astype(np.float64) / w


rows = []
print(" log2N   bpw  fp64_err    fp32_err    K64=e64/2^(2b+logN/2-53)  K32=e32/2^(2b+logN/2-24)")
for logN in (12, 14, 16, 18, 20):
    N = 1 << logN
    for b in (6, 8, 10, 12):
        if 2 * b + logN > 40:
            continue
        p = b * N + 1  # near b bits per word (prime not needed for the error model)
        x, w = digits(p, N)
        z64 = square(x, w, np.float64)
        exact = np.round(z64)
        e64 = np.max(np.abs(z64 - exact))
        e32 = np.max(np.abs(square(x, w, np.float32) - exact))
        scale = 2 * b + logN / 2
        K64, K32 = e64 / 2.0 ** (scale - 53), e32 / 2.0 ** (scale - 24)
        rows.append((logN, K64, K32))
        print(f" {logN:5d} {b:5d}  {e64:.2e}    {e32:.2e}    {K64:10.2f}                {K32:10.2f}")

assert np.fft.rfft(np.zeros(8, np.float32)).dtype == np.complex64, "numpy fft upcast float32; model invalid"

# K grows about linearly in logN; take the worst K/logN seen and add 25%.
k64 = 1.25 * max(r[1] / r[0] for r in rows)
k32 = 1.25 * max(r[2] / r[0] for r in rows)
print(f"\nmodel: err64 <= {k64:.3f}*logN*2^(2b+logN/2-53), err32 <= {k32:.3f}*logN*2^(2b+logN/2-24)")
print("\nlog2N  FP64 alone (err<0.25)      FP32+GF(M31^2) (FP32 err<2^27)")
for logN in (20, 21, 22, 23, 24):
    b64 = (53 - 2 - math.log2(k64 * logN) - logN / 2) / 2
    b32 = (24 + 27 - math.log2(k32 * logN) - logN / 2) / 2
    print(f" {logN:5d}  {b64:5.2f} bpw, p ~ {b64 * 2**logN / 1e6:6.1f}M     {b32:5.2f} bpw, p ~ {b32 * 2**logN / 1e6:6.1f}M")
