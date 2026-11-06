// barnarg gpu kernels
//
// one squaring mod M_p: weight the digits, forward FFT in fp32 and NTT in GF(M31^2),
// square, inverse, then CRT each word and carry. ref/barnarg_ref.py does the same
// thing on the cpu, and the two should agree digit for digit.
#include <metal_stdlib>
using namespace metal;

constant uint P = 0x7fffffffu;   // M31 = 2^31 - 1

inline uint red(uint x) { x = (x & P) + (x >> 31); return x >= P ? x - P : x; }
inline uint addm(uint a, uint b) { return red(a + b); }
inline uint subm(uint a, uint b) { return red(a + P - b); }
inline uint mulm(uint a, uint b) {          // a, b < P; product < 2^62
  uint lo = a * b, hi = mulhi(a, b);
  return red((lo & P) + ((lo >> 31) | (hi << 1)));
}
inline uint2 cmulm(uint2 a, uint2 b) {      // GF(M31^2), i^2 = -1; 3 multiplies (Karatsuba)
  uint ac = mulm(a.x, b.x), bd = mulm(a.y, b.y);
  uint s = mulm(red(a.x + a.y), red(b.x + b.y));
  return uint2(subm(ac, bd), subm(s, addm(ac, bd)));
}
inline uint2 csqm(uint2 a) {                 // a^2 in GF(M31^2) with 2 multiplies: (x+y)(x-y) + 2xy i
  uint t = mulm(a.x, a.y);
  return uint2(mulm(red(a.x + a.y), subm(a.x, a.y)), addm(t, t));
}
inline uint rot(uint x, uint e) {           // x * 2^e mod M31, x < P, e < 31
  return e == 0 ? x : (((x << e) & P) | (x >> (31 - e)));
}
inline float2 cmulf(float2 a, float2 b) {
  return float2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}

inline uint2 addc(uint2 a, uint2 b) { return uint2(addm(a.x, b.x), addm(a.y, b.y)); }
inline uint2 subc(uint2 a, uint2 b) { return uint2(subm(a.x, b.x), subm(a.y, b.y)); }

#define DFT16_BODY(DFT4, TW1, TW2, TW3, TW4, TW6, TW9) \
  DFT4(v[0], v[4], v[8],  v[12]); DFT4(v[1], v[5], v[9],  v[13]); \
  DFT4(v[2], v[6], v[10], v[14]); DFT4(v[3], v[7], v[11], v[15]); \
  /* now v[4*m1 + k2] holds Y[m1][k2]; twiddle by W^(m1*k2) */ \
  v[5] = TW1(v[5]); v[6] = TW2(v[6]); v[7] = TW3(v[7]); \
  v[9] = TW2(v[9]); v[10] = TW4(v[10]); v[11] = TW6(v[11]); \
  v[13] = TW3(v[13]); v[14] = TW6(v[14]); v[15] = TW9(v[15]); \
  DFT4(v[0], v[1], v[2], v[3]); DFT4(v[4], v[5], v[6], v[7]); \
  DFT4(v[8], v[9], v[10], v[11]); DFT4(v[12], v[13], v[14], v[15]); \
  /* v[4*m1 + m2] holds X[m1 + 4*m2]; transpose to natural order */ \
  { auto t = v[1]; v[1] = v[4]; v[4] = t; t = v[2]; v[2] = v[8]; v[8] = t; t = v[3]; v[3] = v[12]; v[12] = t; \
    t = v[6]; v[6] = v[9]; v[9] = t; t = v[7]; v[7] = v[13]; v[13] = t; t = v[11]; v[11] = v[14]; v[14] = t; }


struct Prm { uint N; uint Ns; uint B; uint logN; };

// plain radix-2 path (--radix 2 --nopack). slow, but easy to check.

// digits -> weighted fp32 and M31 inputs, imaginary parts 0
kernel void weight(device const int* dig [[buffer(0)]], device float2* f [[buffer(1)]], device uint2* m [[buffer(2)]],
                   device const float* w [[buffer(3)]], device const uchar* wexp [[buffer(4)]],
                   uint j [[thread_position_in_grid]]) {
  int d = dig[j];
  f[j] = float2(float(d) * w[j], 0.0f);
  uint u = d < 0 ? uint(d + int(P)) : uint(d);
  m[j] = uint2(rot(u, wexp[j]), 0u);
}

// Stockham radix-2 stage, natural order in and out. Ns = 1, 2, 4, ..., N/2.
kernel void fft_f(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]],
                  device const float2* tw [[buffer(2)]], constant Prm& q [[buffer(3)]], constant uint& inv [[buffer(4)]],
                  uint j [[thread_position_in_grid]]) {
  uint N = q.N, Ns = q.Ns, k = j % Ns;
  float2 v0 = in[j], v1 = in[j + N / 2];
  float2 w = tw[k * (N / (2 * Ns))];
  if (inv) w.y = -w.y;
  v1 = cmulf(v1, w);
  uint d = (j / Ns) * Ns * 2 + k;
  out[d] = v0 + v1; out[d + Ns] = v0 - v1;
}

kernel void ntt_m(device const uint2* in [[buffer(0)]], device uint2* out [[buffer(1)]],
                  device const uint2* tw [[buffer(2)]], constant Prm& q [[buffer(3)]],
                  uint j [[thread_position_in_grid]]) {
  uint N = q.N, Ns = q.Ns, k = j % Ns;
  uint2 v0 = in[j], v1 = cmulm(in[j + N / 2], tw[k * (N / (2 * Ns))]);
  uint d = (j / Ns) * Ns * 2 + k;
  out[d] = uint2(addm(v0.x, v1.x), addm(v0.y, v1.y));
  out[d + Ns] = uint2(subm(v0.x, v1.x), subm(v0.y, v1.y));
}

kernel void square(device float2* f [[buffer(0)]], device uint2* m [[buffer(1)]], uint j [[thread_position_in_grid]]) {
  f[j] = cmulf(f[j], f[j]);
  m[j] = cmulm(m[j], m[j]);
}

// stat[0] = max |q - rint(q)| as float bits (CRT round-off, must stay < 0.5)
// stat[1] = nonzero imaginary parts on the M31 side (must be 0)
// stat[2] = carries that ran off the end of a block in carry2 (must be 0)
kernel void carry1(device const float2* fz [[buffer(0)]], device const uint2* mz [[buffer(1)]],
                   device int* dig [[buffer(2)]], device long* cout [[buffer(3)]],
                   device const float* iw [[buffer(4)]], device const uchar* wexp [[buffer(5)]],
                   device const uchar* width [[buffer(6)]], device atomic_uint* stat [[buffer(7)]],
                   constant Prm& q [[buffer(8)]], uint t [[thread_position_in_grid]]) {
  float invN = 1.0f / float(q.N);
  long c = 0;
  float maxe = 0.0f;
  uint bad = 0;
  for (uint i = 0; i < q.B; ++i) {
    uint j = t * q.B + i;
    float zf = fz[j].x * invN * iw[j];
    uint2 m = mz[j];
    bad += m.y != 0u;
    uint r = rot(m.x, (62u - uint(wexp[j]) - q.logN) % 31u);
    int rb = r > P / 2 ? int(r) - int(P) : int(r);
    float qq = (zf - float(rb)) * (1.0f / float(P));
    float k = rint(qq);
    maxe = max(maxe, fabs(qq - k));
    long z = long(rb) + long(k) * long(P) + c;
    uint b = width[j];
    long lo = z & ((1L << b) - 1);
    if (lo >= (1L << (b - 1))) lo -= (1L << b);
    dig[j] = int(lo);
    c = (z - lo) >> b;
  }
  cout[t] = c;
  atomic_fetch_max_explicit(&stat[0], as_type<uint>(maxe), memory_order_relaxed);
  if (bad) atomic_fetch_add_explicit(&stat[1], bad, memory_order_relaxed);
}

kernel void carry2(device int* dig [[buffer(0)]], device const long* cout [[buffer(1)]],
                   device const uchar* width [[buffer(2)]], device atomic_uint* stat [[buffer(3)]],
                   constant Prm& q [[buffer(4)]], uint t [[thread_position_in_grid]]) {
  uint nb = q.N / q.B;
  long c = cout[(t + nb - 1) % nb];      // block 0 takes the wrap carry (2^p = 1 mod M_p)
  for (uint i = 0; i < q.B && c != 0; ++i) {
    uint j = t * q.B + i;
    long z = long(dig[j]) + c;
    uint b = width[j];
    long lo = z & ((1L << b) - 1);
    if (lo >= (1L << (b - 1))) lo -= (1L << b);
    dig[j] = int(lo);
    c = (z - lo) >> b;
  }
  if (c != 0) atomic_fetch_add_explicit(&stat[2], 1u, memory_order_relaxed);
}

// ---- radix-R stockham passes (R = 2, 4, 8, 16) ----
// One pass does log2(R) radix-2 stages in registers. Gather v_k = in[j + k*N/R],
// twiddle v_k *= W_{Ns*R}^((j % Ns) * k), length-R DFT, scatter to
// out[(j/Ns)*Ns*R + j%Ns + k*Ns]. Ns = product of the radices of earlier passes.
// Twiddle powers come from one table read and repeated multiplication
// (exact on the M31 side; a few ulp on the FP32 side).
// wr: R/2 roots of unity of order R (forward); inverse uses conj (FP) or a separate table (M31).

inline void dft4f(thread float2& a, thread float2& b, thread float2& c, thread float2& d, uint inv) {
  float2 t0 = a + c, t1 = a - c, t2 = b + d, u = b - d;
  float2 t3 = inv ? float2(-u.y, u.x) : float2(u.y, -u.x);     // * W_4: -i forward, +i inverse
  a = t0 + t2; c = t0 - t2; b = t1 + t3; d = t1 - t3;
}
inline void dft16f(thread float2* v, constant float2* wr, uint inv) {
  float2 w1 = wr[1], w2 = wr[2], w3 = wr[3], w4 = wr[4], w6 = wr[6];
  if (inv) { w1.y = -w1.y; w2.y = -w2.y; w3.y = -w3.y; w4.y = -w4.y; w6.y = -w6.y; }
#define F_DFT4(a, b, c, d) dft4f(a, b, c, d, inv)
#define F_TW1(x) cmulf(x, w1)
#define F_TW2(x) cmulf(x, w2)
#define F_TW3(x) cmulf(x, w3)
#define F_TW4(x) cmulf(x, w4)
#define F_TW6(x) cmulf(x, w6)
#define F_TW9(x) (-cmulf(x, w1))
  DFT16_BODY(F_DFT4, F_TW1, F_TW2, F_TW3, F_TW4, F_TW6, F_TW9)
}

inline void dft8f(thread float2* v, constant float2* wr, uint inv) {
  float2 w1 = wr[1], w2 = wr[2], w3 = wr[3];
  if (inv) { w1.y = -w1.y; w2.y = -w2.y; w3.y = -w3.y; }
  dft4f(v[0], v[2], v[4], v[6], inv); dft4f(v[1], v[3], v[5], v[7], inv);
  v[3] = cmulf(v[3], w1); v[5] = cmulf(v[5], w2); v[7] = cmulf(v[7], w3);
  float2 t[8];
  for (uint a = 0; a < 4; ++a) { t[a] = v[2 * a] + v[2 * a + 1]; t[a + 4] = v[2 * a] - v[2 * a + 1]; }
  for (uint a = 0; a < 8; ++a) v[a] = t[a];
}

template <uint R>
inline void dft_f(thread float2* v, constant float2* wr, uint inv) {
  if (R == 16) { dft16f(v, wr, inv); return; }
  if (R == 8) { dft8f(v, wr, inv); return; }
  float2 y[R];
  #pragma unroll
  for (uint ns = 1; ns < R; ns *= 2) {
    #pragma unroll
    for (uint jj = 0; jj < R / 2; ++jj) {
      uint kk = jj % ns;
      float2 t = wr[kk * (R / (2 * ns))];
      if (inv) t.y = -t.y;
      float2 a = v[jj], b = cmulf(v[jj + R / 2], t);
      uint d = (jj / ns) * ns * 2 + kk;
      y[d] = a + b; y[d + ns] = a - b;
    }
    #pragma unroll
    for (uint k = 0; k < R; ++k) v[k] = y[k];
  }
}

// the host picks the roots so that W_8 = c = 2^15 (1 + i) (c^2 = 2^31 i = i). then any
// 8th root of unity is an add/sub plus a 15-bit rotation, with no multiply.
template <bool INV> inline uint2 mulc(uint2 a) {     // a * c, or a * conj(c) = a / c
  return INV ? uint2(rot(addm(a.x, a.y), 15), rot(subm(a.y, a.x), 15))
             : uint2(rot(subm(a.x, a.y), 15), rot(addm(a.x, a.y), 15));
}
template <bool INV> inline uint2 muli(uint2 a) { return INV ? uint2(a.y, subm(0u, a.x)) : uint2(subm(0u, a.y), a.x); }
template <bool INV> inline uint2 mul8(uint2 a, uint t) {   // a * W_8^t, t < 4
  return t == 0 ? a : t == 1 ? mulc<INV>(a) : t == 2 ? muli<INV>(a) : muli<INV>(mulc<INV>(a));
}

// hand-written length-16 dfts, 4 x 4: point k = 4*k1 + k2, output m = m1 + 4*m2.
// the 4-point dfts need no multiplies (W_4 = +-i). of the 9 twiddles W_16^(m1*k2) between
// the layers, 6 are 8th roots (rotations on the M31 side) and only W^1, W^3, W^9 multiply.
template <bool INV> inline void dft4m(thread uint2& a, thread uint2& b, thread uint2& c, thread uint2& d) {
  uint2 t0 = addc(a, c), t1 = subc(a, c), t2 = addc(b, d), t3 = muli<INV>(subc(b, d));
  a = addc(t0, t2); c = subc(t0, t2); b = addc(t1, t3); d = subc(t1, t3);
}
inline uint2 negm(uint2 a) { return uint2(subm(0u, a.x), subm(0u, a.y)); }

template <bool INV> inline void dft16m(thread uint2* v, constant uint2* wr) {
  uint2 w1 = wr[1], w3 = wr[3];
#define M_DFT4(a, b, c, d) dft4m<INV>(a, b, c, d)
#define M_TW1(x) cmulm(x, w1)
#define M_TW2(x) mul8<INV>(x, 1)
#define M_TW3(x) cmulm(x, w3)
#define M_TW4(x) mul8<INV>(x, 2)
#define M_TW6(x) mul8<INV>(x, 3)
#define M_TW9(x) negm(cmulm(x, w1))
  DFT16_BODY(M_DFT4, M_TW1, M_TW2, M_TW3, M_TW4, M_TW6, M_TW9)
}

// length-8 dft as 4 x 2: point k = 2*k1 + k2. every twiddle is an 8th root, so no multiplies.
template <bool INV> inline void dft8m(thread uint2* v) {
  dft4m<INV>(v[0], v[2], v[4], v[6]); dft4m<INV>(v[1], v[3], v[5], v[7]);
  v[3] = mul8<INV>(v[3], 1); v[5] = mul8<INV>(v[5], 2); v[7] = mul8<INV>(v[7], 3);
  uint2 t[8];
  for (uint a = 0; a < 4; ++a) { t[a] = addc(v[2 * a], v[2 * a + 1]); t[a + 4] = subc(v[2 * a], v[2 * a + 1]); }
  for (uint a = 0; a < 8; ++a) v[a] = t[a];
}

template <uint R, bool INV>
inline void dft_m(thread uint2* v, constant uint2* wr) {
  if (R == 16) { dft16m<INV>(v, wr); return; }
  if (R == 8) { dft8m<INV>(v); return; }
  uint2 y[R];
  #pragma unroll
  for (uint ns = 1; ns < R; ns *= 2) {
    #pragma unroll
    for (uint jj = 0; jj < R / 2; ++jj) {
      uint kk = jj % ns;
      uint2 a = v[jj], b = v[jj + R / 2];
      uint e = kk * (R / (2 * ns));                  // b *= W_R^e
      if ((8 * e) % R == 0) b = mul8<INV>(b, 8 * e / R);
      else b = cmulm(b, wr[e]);
      uint d = (jj / ns) * ns * 2 + kk;
      y[d] = uint2(addm(a.x, b.x), addm(a.y, b.y));
      y[d + ns] = uint2(subm(a.x, b.x), subm(a.y, b.y));
    }
    #pragma unroll
    for (uint k = 0; k < R; ++k) v[k] = y[k];
  }
}

template <uint R>
kernel void passf(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]],
                  device const float2* tw [[buffer(2)]], constant float2* wr [[buffer(3)]],
                  constant Prm& q [[buffer(4)]], constant uint& inv [[buffer(5)]],
                  uint j [[thread_position_in_grid]]) {
  uint N = q.N, Ns = q.Ns, stride = N / R, m = j % Ns;
  float2 v[R];
  #pragma unroll
  for (uint k = 0; k < R; ++k) v[k] = in[j + k * stride];
  if (Ns > 1) {
    float2 w = tw[m * (N / (Ns * R))];
    if (inv) w.y = -w.y;
    float2 wk = w;
    #pragma unroll
    for (uint k = 1; k < R; ++k) { v[k] = cmulf(v[k], wk); wk = cmulf(wk, w); }
  }
  dft_f<R>(v, wr, inv);
  uint base = (j / Ns) * Ns * R + m;
  #pragma unroll
  for (uint k = 0; k < R; ++k) out[base + k * Ns] = v[k];
}

template <uint R, bool INV>
kernel void passm(device const uint2* in [[buffer(0)]], device uint2* out [[buffer(1)]],
                  device const uint2* tw [[buffer(2)]], constant uint2* wr [[buffer(3)]],
                  constant Prm& q [[buffer(4)]],
                  uint j [[thread_position_in_grid]]) {
  uint N = q.N, Ns = q.Ns, stride = N / R, m = j % Ns;
  uint2 v[R];
  #pragma unroll
  for (uint k = 0; k < R; ++k) v[k] = in[j + k * stride];
  if (Ns > 1 && m != 0) {
    uint2 w = tw[m * (N / (Ns * R))];
    uint2 wk = w;
    #pragma unroll
    for (uint k = 1; k < R; ++k) { v[k] = cmulm(v[k], wk); wk = cmulm(wk, w); }
  }
  dft_m<R, INV>(v, wr);
  uint base = (j / Ns) * Ns * R + m;
  #pragma unroll
  for (uint k = 0; k < R; ++k) out[base + k * Ns] = v[k];
}

#define INST(R) \
  template [[host_name("passf" #R)]] kernel void passf<R>(device const float2*, device float2*, device const float2*, constant float2*, constant Prm&, constant uint&, uint); \
  template [[host_name("passm" #R "f")]] kernel void passm<R, false>(device const uint2*, device uint2*, device const uint2*, constant uint2*, constant Prm&, uint); \
  template [[host_name("passm" #R "i")]] kernel void passm<R, true>(device const uint2*, device uint2*, device const uint2*, constant uint2*, constant Prm&, uint);
INST(2) INST(4) INST(8) INST(16)

// ---- real input packed as N/2 complex values ----
// z_n = x_2n + i x_2n+1 (n < M = N/2). Transforms have length M. The "tail" kernel
// splits Z into the length-N spectrum X of the real data, squares it, and recombines:
//   E = Z_k + conj(Z_{M-k}),  O = -i (Z_k - conj(Z_{M-k})),  t = W_N^k O
//   X1 = E + t, X2 = E - t;  Y1 = X1^2, Y2 = X2^2
//   Z'_k = (Y1 + Y2) + i (Y1 - Y2) conj(W_N^k)
// The dropped factors of 1/2 give 8 * Z'; with the unnormalised inverse the result is
// scaled by 8M = 4N, removed in carry1p. On the M31 side conj(a + bi) = a - bi, and
// conj(w) = w^-1 because every root of order dividing 2^31 has norm 1 in GF(M31^2).

inline float2 conjf(float2 a) { return float2(a.x, -a.y); }
inline float2 mul_i(float2 a) { return float2(-a.y, a.x); }
inline float2 mul_mi(float2 a) { return float2(a.y, -a.x); }
inline uint2 conjm(uint2 a) { return uint2(a.x, subm(0u, a.y)); }
inline uint2 mul_im(uint2 a) { return uint2(subm(0u, a.y), a.x); }
inline uint2 mul_mim(uint2 a) { return uint2(a.y, subm(0u, a.x)); }

kernel void weight2(device const int* dig [[buffer(0)]], device float2* f [[buffer(1)]], device uint2* m [[buffer(2)]],
                    device const float* w [[buffer(3)]], device const uchar* wexp [[buffer(4)]],
                    uint n [[thread_position_in_grid]]) {
  int d0 = dig[2 * n], d1 = dig[2 * n + 1];
  f[n] = float2(float(d0) * w[2 * n], float(d1) * w[2 * n + 1]);
  uint u0 = d0 < 0 ? uint(d0 + int(P)) : uint(d0), u1 = d1 < 0 ? uint(d1 + int(P)) : uint(d1);
  m[n] = uint2(rot(u0, wexp[2 * n]), rot(u1, wexp[2 * n + 1]));
}

// tw: W_N^k for k < M (the length-N table). Out of place: reads Z_k and Z_{M-k}.
kernel void tail_f(device const float2* Z [[buffer(0)]], device float2* out [[buffer(1)]],
                   device const float2* tw [[buffer(2)]], constant Prm& q [[buffer(3)]],
                   uint k [[thread_position_in_grid]]) {
  uint M = q.N / 2;
  float2 a = Z[k], b = conjf(Z[(M - k) & (M - 1)]);
  float2 w = tw[k];
  float2 E = a + b, O = mul_mi(a - b), t = cmulf(w, O);
  float2 X1 = E + t, X2 = E - t;
  float2 Y1 = cmulf(X1, X1), Y2 = cmulf(X2, X2);
  out[k] = (Y1 + Y2) + mul_i(cmulf(Y1 - Y2, conjf(w)));
}

kernel void tail_m(device const uint2* Z [[buffer(0)]], device uint2* out [[buffer(1)]],
                   device const uint2* tw [[buffer(2)]], constant Prm& q [[buffer(3)]],
                   uint k [[thread_position_in_grid]]) {
  uint M = q.N / 2;
  uint2 a = Z[k], b = conjm(Z[(M - k) & (M - 1)]);
  uint2 w = tw[k];
  uint2 E = addc(a, b), O = mul_mim(subc(a, b)), t = cmulm(w, O);
  uint2 X1 = addc(E, t), X2 = subc(E, t);
  uint2 Y1 = csqm(X1), Y2 = csqm(X2);
  out[k] = addc(addc(Y1, Y2), mul_im(cmulm(subc(Y1, Y2), conjm(w))));
}

// carry1 for packed data: digit j is component (j & 1) of complex value j >> 1; scale 1/(4N).
kernel void carry1p(device const float2* fz [[buffer(0)]], device const uint2* mz [[buffer(1)]],
                    device int* dig [[buffer(2)]], device long* cout [[buffer(3)]],
                    device const float* iw [[buffer(4)]], device const uchar* wexp [[buffer(5)]],
                    device const uchar* width [[buffer(6)]], device atomic_uint* stat [[buffer(7)]],
                    constant Prm& q [[buffer(8)]], uint t [[thread_position_in_grid]]) {
  float sc = 1.0f / (4.0f * float(q.N));
  uint rsh = q.logN + 2;
  long c = 0;
  float maxe = 0.0f;
  for (uint i = 0; i < q.B; ++i) {
    uint j = t * q.B + i;
    float2 fv = fz[j >> 1];
    uint2 mv = mz[j >> 1];
    float zf = ((j & 1) ? fv.y : fv.x) * sc * iw[j];
    uint r = rot((j & 1) ? mv.y : mv.x, (93u - uint(wexp[j]) - rsh) % 31u);
    int rb = r > P / 2 ? int(r) - int(P) : int(r);
    float qq = (zf - float(rb)) * (1.0f / float(P));
    float k = rint(qq);
    maxe = max(maxe, fabs(qq - k));
    long z = long(rb) + long(k) * long(P) + c;
    uint b = width[j];
    long lo = z & ((1L << b) - 1);
    if (lo >= (1L << (b - 1))) lo -= (1L << b);
    dig[j] = int(lo);
    c = (z - lo) >> b;
  }
  cout[t] = c;
  atomic_fetch_max_explicit(&stat[0], as_type<uint>(maxe), memory_order_relaxed);
}

// carry1p with threadgroup staging: 128 threads x 16 digits = 2,048 digits (1,024 complex) per group.
// Loads and stores are contiguous; thread t still owns digits 16t..16t+15, so carry2 is unchanged.
kernel void carry1t(device const float2* fz [[buffer(0)]], device const uint2* mz [[buffer(1)]],
                    device int* dig [[buffer(2)]], device long* cout [[buffer(3)]],
                    device const float* iw [[buffer(4)]], device const uchar* wexp [[buffer(5)]],
                    device const uchar* width [[buffer(6)]], device atomic_uint* stat [[buffer(7)]],
                    constant Prm& q [[buffer(8)]], uint g [[threadgroup_position_in_grid]],
                    uint l [[thread_position_in_threadgroup]]) {
  threadgroup float2 sf[1024];
  threadgroup uint2 sm[1024];
  threadgroup float siw[2048];
  uint cb = g * 1024, db = g * 2048;
  for (uint i = l; i < 1024; i += 128) { sf[i] = fz[cb + i]; sm[i] = mz[cb + i]; }
  for (uint i = l; i < 2048; i += 128) siw[i] = iw[db + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sc = 1.0f / (4.0f * float(q.N));
  uint rsh = q.logN + 2;
  long c = 0;
  float maxe = 0.0f;
  int out[16];
  for (uint i = 0; i < 16; ++i) {
    uint lj = l * 16 + i, j = db + lj;
    float2 fv = sf[lj >> 1];
    uint2 mv = sm[lj >> 1];
    float zf = ((j & 1) ? fv.y : fv.x) * sc * siw[lj];
    uint r = rot((j & 1) ? mv.y : mv.x, (93u - uint(wexp[j]) - rsh) % 31u);
    int rb = r > P / 2 ? int(r) - int(P) : int(r);
    float qq = (zf - float(rb)) * (1.0f / float(P));
    float k = rint(qq);
    maxe = max(maxe, fabs(qq - k));
    long z = long(rb) + long(k) * long(P) + c;
    uint b = width[j];
    long lo = z & ((1L << b) - 1);
    if (lo >= (1L << (b - 1))) lo -= (1L << b);
    out[i] = int(lo);
    c = (z - lo) >> b;
  }
  cout[g * 128 + l] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup int* sd = (threadgroup int*)siw;
  for (uint i = 0; i < 16; ++i) sd[l * 16 + i] = out[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = l; i < 2048; i += 128) dig[db + i] = sd[i];
  maxe = simd_max(maxe);
  if (simd_is_first()) atomic_fetch_max_explicit(&stat[0], as_type<uint>(maxe), memory_order_relaxed);
}

// ---- radix R = A*A passes through threadgroup memory (R = 256 with A = 16, R = 64 with A = 8) ----
// same stockham step as passf/passm, but the length-R dft is two length-A dfts with an
// exchange in threadgroup memory between them. point k = A*k1 + k2.
//   step 1: thread (i, k2) loads k1 = 0..A-1, twiddles, dft over k1, multiplies by W_R^(m1*k2)
//   step 2: thread (i, m1) loads k2 = 0..A-1, dft over k2, writes output m = m1 + A*m2
// J = 8 instances (consecutive j) per threadgroup, so loads for a fixed k are contiguous.
// inter-pass twiddle W^((j%Ns)*k) = a * b^k1 with a = W^((j%Ns)*k2), b = W^((j%Ns)*A),
// both read from the table, b^k1 by repeated multiplication.

constant uint TJ = 8;

// W = true: first forward pass reading the digits directly and weighting them (replaces weight2).
template <uint A, bool W>
kernel void passTf(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]],
                   device const float2* tw [[buffer(2)]], constant float2* wa [[buffer(3)]],
                   device const float2* wR [[buffer(4)]], constant Prm& q [[buffer(5)]], constant uint& inv [[buffer(6)]],
                   device const int2* dig [[buffer(7)]], device const float2* wt [[buffer(8)]],
                   uint g [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  constexpr uint R = A * A, S = R + 1;
  threadgroup float2 sh[TJ * S];
  uint N = q.N, Ns = q.Ns, stride = N / R;
  uint i = l % TJ, c = l / TJ;          // instance in the group, column (k2, then m1)
  uint j = g * TJ + i, m = j % Ns;
  float2 v[A];
  #pragma unroll
  for (uint k1 = 0; k1 < A; ++k1) {
    uint n = j + (A * k1 + c) * stride;
    if (W) { int2 d = dig[n]; v[k1] = float2(d) * wt[n]; }
    else v[k1] = in[n];
  }
  if (!W && Ns > 1) {
    uint step = N / (Ns * R);
    float2 a = tw[m * c * step], b = tw[m * A * step];
    if (inv) { a.y = -a.y; b.y = -b.y; }
    float2 w = a;
    #pragma unroll
    for (uint k1 = 0; k1 < A; ++k1) { v[k1] = cmulf(v[k1], w); w = cmulf(w, b); }
  }
  dft_f<A>(v, wa, inv);
  #pragma unroll
  for (uint m1 = 0; m1 < A; ++m1) {
    float2 t = wR[m1 * c];
    if (inv) t.y = -t.y;
    sh[i * S + m1 * A + c] = cmulf(v[m1], t);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < A; ++k2) v[k2] = sh[i * S + c * A + k2];
  dft_f<A>(v, wa, inv);
  uint base = (j / Ns) * Ns * R + m;
  #pragma unroll
  for (uint m2 = 0; m2 < A; ++m2) out[base + (c + A * m2) * Ns] = v[m2];
}

template <uint A, bool INV, bool W>
kernel void passTm(device const uint2* in [[buffer(0)]], device uint2* out [[buffer(1)]],
                   device const uint2* tw [[buffer(2)]], constant uint2* wa [[buffer(3)]],
                   device const uint2* wR [[buffer(4)]], constant Prm& q [[buffer(5)]],
                   device const int2* dig [[buffer(6)]], device const uchar2* wexp [[buffer(7)]],
                   uint g [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  constexpr uint R = A * A, S = R + 1;
  threadgroup uint2 sh[TJ * S];
  uint N = q.N, Ns = q.Ns, stride = N / R;
  uint i = l % TJ, c = l / TJ;
  uint j = g * TJ + i, m = j % Ns;
  uint2 v[A];
  #pragma unroll
  for (uint k1 = 0; k1 < A; ++k1) {
    uint n = j + (A * k1 + c) * stride;
    if (W) {
      int2 d = dig[n]; uchar2 e = wexp[n];
      uint u0 = d.x < 0 ? uint(d.x + int(P)) : uint(d.x), u1 = d.y < 0 ? uint(d.y + int(P)) : uint(d.y);
      v[k1] = uint2(rot(u0, e.x), rot(u1, e.y));
    } else v[k1] = in[n];
  }
  if (!W && Ns > 1 && m != 0) {
    uint step = N / (Ns * R);
    uint2 w = tw[m * c * step], b = tw[m * A * step];
    #pragma unroll
    for (uint k1 = 0; k1 < A; ++k1) { v[k1] = cmulm(v[k1], w); w = cmulm(w, b); }
  }
#ifndef SKIP_DFT
  dft_m<A, INV>(v, wa);
#endif
  #pragma unroll
#ifdef SKIP_INNER
  for (uint m1 = 0; m1 < A; ++m1) sh[i * S + m1 * A + c] = v[m1];
#else
  for (uint m1 = 0; m1 < A; ++m1) sh[i * S + m1 * A + c] = (m1 * c) ? cmulm(v[m1], wR[m1 * c]) : v[m1];
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < A; ++k2) v[k2] = sh[i * S + c * A + k2];
#ifndef SKIP_DFT
  dft_m<A, INV>(v, wa);
#endif
  uint base = (j / Ns) * Ns * R + m;
  #pragma unroll
  for (uint m2 = 0; m2 < A; ++m2) out[base + (c + A * m2) * Ns] = v[m2];
}

#define TF_ARGS device const float2*, device float2*, device const float2*, constant float2*, device const float2*, constant Prm&, constant uint&, device const int2*, device const float2*, uint, uint
#define TM_ARGS device const uint2*, device uint2*, device const uint2*, constant uint2*, device const uint2*, constant Prm&, device const int2*, device const uchar2*, uint, uint
#define INSTT(A) \
  template [[host_name("passTf" #A)]] kernel void passTf<A, false>(TF_ARGS); \
  template [[host_name("passTf" #A "w")]] kernel void passTf<A, true>(TF_ARGS); \
  template [[host_name("passTm" #A "f")]] kernel void passTm<A, false, false>(TM_ARGS); \
  template [[host_name("passTm" #A "i")]] kernel void passTm<A, true, false>(TM_ARGS); \
  template [[host_name("passTm" #A "w")]] kernel void passTm<A, false, true>(TM_ARGS);
INSTT(8) INSTT(16)

// ---- fused middle: last forward pass + tail + first inverse pass ----
// needs the forward plan to end with R = 64 (so Ns = M/64 there, and instance j holds
// Z[j + Ns*m], m < 64) and the inverse plan to start with R = 64 at Ns = 1 (which reads
// exactly the same 64 values). the tail pairs k with M - k, which lives in instance Ns - j,
// so each group takes 4 instances j and their 4 partners. j = 0 and j = Ns/2 pair with
// themselves; group 0 puts Ns/2 in the slot that would hold the partner of 0.
// 8 slots x 8 threads, two 64-point dfts (8 x 8) each way.

inline uint mid_slot_j(uint g, uint i, uint Ns) {
  uint jA = 4 * g + (i & 3);
  return i < 4 ? jA : (jA == 0 ? Ns / 2 : Ns - jA);
}
inline uint mid_partner(uint i, uint j, uint Ns) {
  return i < 4 ? (j == 0 ? i : i + 4) : (j == Ns / 2 ? i : i - 4);
}

kernel void midf(device const float2* in [[buffer(0)]], device float2* out [[buffer(1)]],
                 device const float2* twT [[buffer(2)]], constant float2* wa [[buffer(3)]],
                 device const float2* wR [[buffer(4)]], device const float2* twN [[buffer(5)]],
                 constant Prm& q [[buffer(6)]],
                 uint g [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  constexpr uint S = 65;
  threadgroup float2 sa[8 * S], sb[8 * S];
  uint M = q.N, Ns = M / 64;
  uint i = l % 8, c = l / 8, j = mid_slot_j(g, i, Ns);
  float2 v[8];
  // forward: twiddle W_M^(j*k), then 8 x 8
  #pragma unroll
  for (uint k1 = 0; k1 < 8; ++k1) v[k1] = in[j + (8 * k1 + c) * Ns];
  if (j) {
    float2 w = twT[j * c], b = twT[j * 8];
    #pragma unroll
    for (uint k1 = 0; k1 < 8; ++k1) { v[k1] = cmulf(v[k1], w); w = cmulf(w, b); }
  }
  dft8f(v, wa, 0);
  #pragma unroll
  for (uint m1 = 0; m1 < 8; ++m1) sa[i * S + m1 * 8 + c] = cmulf(v[m1], wR[m1 * c]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < 8; ++k2) v[k2] = sa[i * S + c * 8 + k2];
  dft8f(v, wa, 0);
  #pragma unroll
  for (uint m2 = 0; m2 < 8; ++m2) sb[i * S + c + 8 * m2] = v[m2];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // tail: same algebra as tail_f
  uint ip = mid_partner(i, j, Ns);
  #pragma unroll
  for (uint t = 0; t < 8; ++t) {
    uint m = c + 8 * t, k = j + Ns * m, kp = (M - k) & (M - 1), mp = kp / Ns;
    float2 a = sb[i * S + m], b = conjf(sb[ip * S + mp]), w = twN[k];
    float2 E = a + b, O = mul_mi(a - b), tt = cmulf(w, O);
    float2 X1 = E + tt, X2 = E - tt;
    float2 Y1 = cmulf(X1, X1), Y2 = cmulf(X2, X2);
    sa[i * S + m] = (Y1 + Y2) + mul_i(cmulf(Y1 - Y2, conjf(w)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // inverse, Ns = 1 (no inter-pass twiddle): points 0..63 of instance j are sa[i*S + 0..63]
  #pragma unroll
  for (uint k1 = 0; k1 < 8; ++k1) v[k1] = sa[i * S + 8 * k1 + c];
  dft8f(v, wa, 1);
  #pragma unroll
  for (uint m1 = 0; m1 < 8; ++m1) sb[i * S + m1 * 8 + c] = cmulf(v[m1], conjf(wR[m1 * c]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < 8; ++k2) v[k2] = sb[i * S + c * 8 + k2];
  dft8f(v, wa, 1);
  #pragma unroll
  for (uint m2 = 0; m2 < 8; ++m2) out[j * 64 + c + 8 * m2] = v[m2];
}

kernel void midm(device const uint2* in [[buffer(0)]], device uint2* out [[buffer(1)]],
                 device const uint2* twT [[buffer(2)]], device const uint2* wRf [[buffer(3)]],
                 device const uint2* wRi [[buffer(4)]], device const uint2* twN [[buffer(5)]],
                 constant Prm& q [[buffer(6)]],
                 uint g [[threadgroup_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
  constexpr uint S = 65;
  threadgroup uint2 sa[8 * S], sb[8 * S];
  uint M = q.N, Ns = M / 64;
  uint i = l % 8, c = l / 8, j = mid_slot_j(g, i, Ns);
  uint2 v[8];
  #pragma unroll
  for (uint k1 = 0; k1 < 8; ++k1) v[k1] = in[j + (8 * k1 + c) * Ns];
  if (j) {
    uint2 w = twT[j * c], b = twT[j * 8];
    #pragma unroll
    for (uint k1 = 0; k1 < 8; ++k1) { v[k1] = cmulm(v[k1], w); w = cmulm(w, b); }
  }
  dft8m<false>(v);
  #pragma unroll
  for (uint m1 = 0; m1 < 8; ++m1) sa[i * S + m1 * 8 + c] = (m1 * c) ? cmulm(v[m1], wRf[m1 * c]) : v[m1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < 8; ++k2) v[k2] = sa[i * S + c * 8 + k2];
  dft8m<false>(v);
  #pragma unroll
  for (uint m2 = 0; m2 < 8; ++m2) sb[i * S + c + 8 * m2] = v[m2];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint ip = mid_partner(i, j, Ns);
  #pragma unroll
  for (uint t = 0; t < 8; ++t) {
    uint m = c + 8 * t, k = j + Ns * m, kp = (M - k) & (M - 1), mp = kp / Ns;
#ifdef SKIP_TAIL
    sa[i * S + m] = sb[i * S + m] ^ sb[ip * S + mp];
#else
    uint2 a = sb[i * S + m], b = conjm(sb[ip * S + mp]), w = twN[k];
    uint2 E = addc(a, b), O = mul_mim(subc(a, b)), tt = cmulm(w, O);
    uint2 X1 = addc(E, tt), X2 = subc(E, tt);
    uint2 Y1 = csqm(X1), Y2 = csqm(X2);
    sa[i * S + m] = addc(addc(Y1, Y2), mul_im(cmulm(subc(Y1, Y2), conjm(w))));
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k1 = 0; k1 < 8; ++k1) v[k1] = sa[i * S + 8 * k1 + c];
  dft8m<true>(v);
  #pragma unroll
  for (uint m1 = 0; m1 < 8; ++m1) sb[i * S + m1 * 8 + c] = (m1 * c) ? cmulm(v[m1], wRi[m1 * c]) : v[m1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for (uint k2 = 0; k2 < 8; ++k2) v[k2] = sb[i * S + c * 8 + k2];
  dft8m<true>(v);
  #pragma unroll
  for (uint m2 = 0; m2 < 8; ++m2) out[j * 64 + c + 8 * m2] = v[m2];
}
