// Metal microbenchmark for an Apple Silicon PRPLL port: measures the two
// numbers that decide a Mersenne PRP iteration's speed on the GPU:
//   1. device-memory bandwidth (stream copy), since PRPLL is bandwidth bound;
//   2. M31 (2^31-1) modular butterfly throughput in registers (integer NTT core).
// Runs a few seconds. Build: swiftc -O research/bench.swift -o build/bench
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
constant uint P = 0x7fffffffu;
inline uint red(uint x) { x = (x & P) + (x >> 31); return x == P ? 0u : x; }
inline uint mulm(uint a, uint b) {
  uint lo = a * b, hi = mulhi(a, b);
  return red((lo & P) + ((lo >> 31) | (hi << 1)));
}
inline uint addm(uint a, uint b) { return red(a + b); }
inline uint subm(uint a, uint b) { return red(a + P - b); }

kernel void copy4(device const uint4* in [[buffer(0)]], device uint4* out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { out[i] = in[i]; }

// Each thread: 8 values, ROUNDS rounds of 4 butterflies (8 values) -> 4 mulmod + 8 add/sub each.
kernel void bfly(device uint* io [[buffer(0)]], constant uint& rounds [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
  uint v[8];
  for (int k = 0; k < 8; ++k) v[k] = io[i * 8 + k] & P;
  uint w = (i * 2654435761u) & P;
  for (uint r = 0; r < rounds; ++r) {
    for (int k = 0; k < 4; ++k) {
      uint t = mulm(v[k + 4], w);
      uint a = v[k];
      v[k] = addm(a, t); v[k + 4] = subm(a, t);
    }
    w = mulm(w, w | 1u);
    uint t0 = v[0]; v[0] = v[1]; v[1] = v[2]; v[2] = v[3]; v[3] = t0;
  }
  for (int k = 0; k < 8; ++k) io[i * 8 + k] = v[k];
}
"""

let dev = MTLCreateSystemDefaultDevice()!
print("device:", dev.name)
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

func time(_ reps: Int, _ enc: (MTLComputeCommandEncoder) -> Void) -> Double {
  var best = Double.infinity
  for _ in 0..<reps {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    enc(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  return best
}

// 1. Bandwidth: 256 MiB in, 256 MiB out (far above the SLC).
let bytes = 256 << 20
let a = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
let b = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
let cp = pso("copy4")
let n4 = bytes / 16
let tc = time(10) { e in
  e.setComputePipelineState(cp); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
  e.dispatchThreads(MTLSize(width: n4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
print(String(format: "copy bandwidth: %.1f GB/s (read+write)", Double(2 * bytes) / tc / 1e9))

// 2. M31 butterflies.
let threads = 1 << 20
let io = dev.makeBuffer(length: threads * 8 * 4, options: .storageModeShared)!
let p = io.contents().bindMemory(to: UInt32.self, capacity: threads * 8)
for k in 0..<(threads * 8) { p[k] = UInt32(truncatingIfNeeded: k &* 2654435761) }
var rounds: UInt32 = 256
let bf = pso("bfly")
let tb = time(5) { e in
  e.setComputePipelineState(bf); e.setBuffer(io, offset: 0, index: 0)
  e.setBytes(&rounds, length: 4, index: 1)
  e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}
let bflies = Double(threads) * Double(rounds) * 4
print(String(format: "M31 butterflies: %.1f G/s", bflies / tb / 1e9))
