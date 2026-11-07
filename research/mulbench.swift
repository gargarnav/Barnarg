// how fast are GF(M31^2) multiplies on this gpu, written different ways.
// each thread runs a chain of complex multiplies by a fixed "twiddle", like the passes do.
//   swiftc -O research/mulbench.swift -o build/mulbench && ./build/mulbench
import Foundation
import Metal

let src = """
#include <metal_stdlib>
using namespace metal;
constant uint P = 0x7fffffffu;
inline uint red(uint x) { x = (x & P) + (x >> 31); return x >= P ? x - P : x; }
inline uint redmin(uint x) { x = (x & P) + (x >> 31); return min(x, x - P); }
inline uint m0(uint a, uint b) { uint lo = a * b, hi = mulhi(a, b); return red((lo & P) + ((lo >> 31) | (hi << 1))); }
inline uint m1(uint a, uint b) { uint b2 = b << 1; return red(mulhi(a, b2) + ((a * b2) >> 1)); }
inline uint m2(uint a, uint b) { uint b2 = b << 1; return redmin(mulhi(a, b2) + ((a * b2) >> 1)); }
inline uint m3(uint a, uint b2) { return redmin(mulhi(a, b2) + ((a * b2) >> 1)); }   // b already doubled
#ifdef CHEAPADD
inline uint addm(uint a, uint b) { uint x = a + b; return min(x, x - P); }   // a, b < P
inline uint subm(uint a, uint b) { uint x = a - b; return min(x, x + P); }
#else
inline uint addm(uint a, uint b) { return red(a + b); }
inline uint subm(uint a, uint b) { return red(a + P - b); }
#endif
inline uint m4(uint a, uint b) { uint b2 = b << 1, x = mulhi(a, b2) + ((a * b2) >> 1); return min(x, x - P); }
inline uint m5(uint a, uint b2) { uint x = mulhi(a, b2) + ((a * b2) >> 1); return min(x, x - P); }

// karatsuba, b = (x, y, x + y) with the sum precomputed, MUL decides the multiply
#define KARA(MUL) \\
  inline uint2 kara_##MUL(uint2 a, uint3 b) { uint ac = MUL(a.x, b.x), bd = MUL(a.y, b.y); \\
    uint s = MUL(addm(a.x, a.y), b.z); return uint2(subm(ac, bd), subm(s, addm(ac, bd))); }
KARA(m0) KARA(m1) KARA(m2) KARA(m3) KARA(m4) KARA(m5)
#define FOUR(MUL) \\
  inline uint2 four_##MUL(uint2 a, uint3 b) { \\
    return uint2(subm(MUL(a.x, b.x), MUL(a.y, b.y)), addm(MUL(a.x, b.y), MUL(a.y, b.x))); }
FOUR(m0) FOUR(m1) FOUR(m2) FOUR(m3) FOUR(m4) FOUR(m5)

#define BENCH(NAME, F) \\
kernel void NAME(device uint2* io [[buffer(0)]], constant uint3& w [[buffer(1)]], uint j [[thread_position_in_grid]]) { \\
  uint2 v0 = io[j], v1 = io[j] ^ 0x1234u, v2 = io[j] ^ 0x5678u, v3 = io[j] ^ 0x9abcu; \\
  for (int i = 0; i < 64; ++i) { v0 = F(v0, w); v1 = F(v1, w); v2 = F(v2, w); v3 = F(v3, w); } \\
  io[j] = v0 ^ v1 ^ v2 ^ v3; }
BENCH(k_kara0, kara_m0) BENCH(k_kara1, kara_m1) BENCH(k_kara2, kara_m2) BENCH(k_kara3, kara_m3)
BENCH(k_kara4, kara_m4) BENCH(k_kara5, kara_m5) BENCH(k_four4, four_m4) BENCH(k_four5, four_m5)
BENCH(k_four0, four_m0) BENCH(k_four1, four_m1) BENCH(k_four2, four_m2) BENCH(k_four3, four_m3)
"""

let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
if CommandLine.arguments.contains("--cheapadd") { opts.preprocessorMacros = ["CHEAPADD": NSNumber(value: 1)] }
let lib = try! dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!
let n = 1 << 22
let io = dev.makeBuffer(length: n * 8, options: .storageModeShared)!
let p = io.contents().bindMemory(to: UInt32.self, capacity: 2 * n)
for i in 0..<(2 * n) { p[i] = UInt32(truncatingIfNeeded: i &* 2654435761) & 0x3fffffff }
let wx: UInt32 = 123456789, wy: UInt32 = 987654321
for name in ["k_kara0", "k_kara1", "k_kara3", "k_kara4", "k_kara5", "k_four0", "k_four4", "k_four5"] {
  let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
  let doubled = name.hasSuffix("3") || name.hasSuffix("5")
  var w: SIMD3<UInt32> = doubled ? [wx << 1, wy << 1, ((wx + wy) % 0x7fffffff) << 1] : [wx, wy, (wx + wy) % 0x7fffffff]
  var best = Double.infinity
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso); e.setBuffer(io, offset: 0, index: 0); e.setBytes(&w, length: 16, index: 1)
    e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let cmuls = Double(n) * 64 * 4
  print(String(format: "%@  %.1f G complex mul/s", name, cmuls / best / 1e9))
}
