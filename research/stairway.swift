// Carry-stairway test on Metal: can threadgroup g safely spin-wait for a value
// published by threadgroup g-1? Metal promises no forward progress between
// threadgroups, so this checks it on real hardware. Every spin is bounded, so a
// stuck wait shows up as a counted timeout instead of a GPU hang.
//
// Modes:
//   0 = static order: group g waits on hardware group id g-1
//   1 = ticket order: group takes its logical id from an atomic counter at start,
//       so any group it waits on has already started (the usual fix)
// Payload: 8 words in a plain buffer, published with a device fence + flag,
// read after seeing the flag. Checks that every group saw the right payload.
// Build: swiftc -O research/stairway.swift -o build/stairway
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;

kernel void stair(device atomic_uint* flags   [[buffer(0)]],
                  device uint* payload        [[buffer(1)]],
                  device atomic_uint* ticket  [[buffer(2)]],
                  device atomic_uint* stats   [[buffer(3)]],   // 0 timeouts, 1 bad payload
                  constant uint& mode         [[buffer(4)]],
                  constant uint& work         [[buffer(5)]],
                  uint gid  [[threadgroup_position_in_grid]],
                  uint lid  [[thread_position_in_threadgroup]]) {
  threadgroup uint g_s;
  if (lid == 0) g_s = mode == 1 ? atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed) : gid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint g = g_s;

  // Fake work before the carry, so groups overlap.
  uint x = g * 2654435761u + lid;
  for (uint i = 0; i < work; ++i) x = x * 1664525u + 1013904223u;

  threadgroup uint in_s[8];
  if (lid == 0) {
    bool ok = true;
    if (g > 0) {
      uint spins = 0;
      while (atomic_load_explicit(&flags[g - 1], memory_order_relaxed) == 0u) {
        if (++spins > (1u << 24)) { atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed); ok = false; break; }
      }
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    for (int k = 0; k < 8; ++k) {
      uint expect = (g == 0) ? 0u : (g - 1) * 8u + k;   // what group g-1 wrote
      uint v = (g == 0) ? 0u : payload[(g - 1) * 8 + k];
      if (g > 0 && ok && v != expect) atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
      in_s[k] = v;
    }
    for (int k = 0; k < 8; ++k) payload[g * 8 + k] = g * 8u + k;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_store_explicit(&flags[g], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (x == 0xdeadbeefu && in_s[0] == 7u) payload[0] = x;   // keep the work alive
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
opts.languageVersion = .version3_2
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("compile failed:", error); exit(1) }
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "stair")!)
let q = dev.makeCommandQueue()!
print("device:", dev.name, " max threads/group:", pso.maxTotalThreadsPerThreadgroup)

func run(mode: UInt32, groups: Int, tpg: Int, work: UInt32) -> (UInt32, UInt32, Double, String) {
  let flags = dev.makeBuffer(length: groups * 4, options: .storageModeShared)!
  memset(flags.contents(), 0, groups * 4)
  let payload = dev.makeBuffer(length: groups * 32, options: .storageModeShared)!
  let ticket = dev.makeBuffer(length: 4, options: .storageModeShared)!
  memset(ticket.contents(), 0, 4)
  let stats = dev.makeBuffer(length: 8, options: .storageModeShared)!
  memset(stats.contents(), 0, 8)
  var m = mode, w = work
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pso)
  e.setBuffer(flags, offset: 0, index: 0); e.setBuffer(payload, offset: 0, index: 1)
  e.setBuffer(ticket, offset: 0, index: 2); e.setBuffer(stats, offset: 0, index: 3)
  e.setBytes(&m, length: 4, index: 4); e.setBytes(&w, length: 4, index: 5)
  e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let s = stats.contents().bindMemory(to: UInt32.self, capacity: 2)
  let err = cb.error.map { "\($0)" } ?? "ok"
  return (s[0], s[1], (cb.gpuEndTime - cb.gpuStartTime) * 1e3, err)
}

print("mode  groups  tpg  work   timeouts  bad  ms     status")
for mode in [UInt32(1), 0] {
  for (groups, tpg, work) in [(1024, 256, 0), (16384, 256, 0), (16384, 256, 2000), (65536, 64, 500), (4096, 1024, 200)] {
    for rep in 0..<3 {
      let (t, b, ms, st) = run(mode: mode, groups: groups, tpg: tpg, work: UInt32(work))
      print(String(format: "%-5d %-7d %-4d %-6d %-9d %-4d %-6.2f %@ %@", mode, groups, tpg, work, t, b, ms, st, rep == 0 ? "" : "(rep)"))
    }
  }
}
