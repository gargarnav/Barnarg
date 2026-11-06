// barnarg: PRP squarings of 3 mod M_p on the mac gpu (metal).
// Usage: barnarg_gpu P N ITERS [dumpfile k1,k2,...] [--kernels path]
// Dump format: per checkpoint, u32 k, u32 N, then N x i32 balanced digits.
// Prints the round-off (max |q - round(q)|, must stay < 0.5), error counters and us/iter.
import Foundation
import Metal

let M31: UInt64 = (1 << 31) - 1

func cpow(_ r0: UInt64, _ i0: UInt64, _ e0: UInt64) -> (UInt64, UInt64) {
  var (rr, ri): (UInt64, UInt64) = (1, 0)
  var (r, i, e) = (r0, i0, e0)
  while e > 0 {
    if e & 1 == 1 { (rr, ri) = ((rr * r % M31 + M31 - ri * i % M31) % M31, (rr * i % M31 + ri * r % M31) % M31) }
    (r, i) = ((r * r % M31 + M31 - i * i % M31) % M31, (2 * (r * i % M31)) % M31)
    e >>= 1
  }
  return (rr, ri)
}

func rootOfUnity(_ N: Int) -> (UInt64, UInt64) {
  let order = M31 * M31 - 1, odd = order >> 32
  for a in UInt64(1)..<100 { for b in UInt64(1)..<100 {
    let g = cpow(a, b, odd)
    if cpow(g.0, g.1, 1 << 31) != (1, 0) { return cpow(g.0, g.1, (1 << 32) / UInt64(N)) }
  } }
  fatalError("no generator")
}

var args = CommandLine.arguments.dropFirst().map { $0 }
var kernelPath = "src/barnarg.metal"
if let k = args.firstIndex(of: "--kernels") { kernelPath = args[k + 1]; args.removeSubrange(k...(k + 1)) }
var maxRadix = 16   // 2 = plain radix-2 stages, 16 = radix-16 passes
if let k = args.firstIndex(of: "--radix") { maxRadix = Int(args[k + 1])!; args.removeSubrange(k...(k + 1)) }
let profileMode = CommandLine.arguments.contains("--profile")
var pack = true      // real input packed as N/2 complex (needs maxRadix > 2)
if let k = args.firstIndex(of: "--nopack") { pack = false; args.remove(at: k) }
args.removeAll { $0 == "--profile" }
guard args.count >= 3, let p = Int(args[0]), let N = Int(args[1]), let iters = Int(args[2]) else {
  print("usage: barnarg_gpu P N ITERS [dumpfile k1,k2,...]"); exit(2)
}
let dumpPath: String? = args.count >= 4 ? args[3] : nil
let checkpoints = Set((args.count >= 5 ? args[4] : "").split(separator: ",").compactMap { Int($0) })
let logN = N.trailingZeroBitCount
precondition(N == 1 << logN && N >= 64)
let B = 16
precondition(N % B == 0)

// ---- tables (same definitions as barnarg_ref.py) ----
var width = [UInt8](repeating: 0, count: N), wexp = [UInt8](repeating: 0, count: N)
var w32 = [Float](repeating: 0, count: N), iw32 = [Float](repeating: 0, count: N)
func ceilDiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }
var uinv = 0
for u in 1..<31 where (N % 31) * u % 31 == 1 { uinv = u }
for j in 0..<N {
  let s = ceilDiv(j * p, N), s1 = ceilDiv((j + 1) * p, N)
  width[j] = UInt8(s1 - s)
  let frac = Double(s * N - j * p) / Double(N)
  w32[j] = Float(exp2(frac)); iw32[j] = Float(1.0 / exp2(frac))
  wexp[j] = UInt8(((s - j * (p * uinv % 31)) % 31 + 31) % 31)
}
precondition(width.max()! <= 30 && width.min()! >= 2)
var twf = [SIMD2<Float>](repeating: .zero, count: N / 2)
for k in 0..<(N / 2) { let a = -2.0 * Double.pi * Double(k) / Double(N); twf[k] = SIMD2(Float(cos(a)), Float(sin(a))) }
let w = rootOfUnity(N), winv = cpow(w.0, w.1, UInt64(N - 1))
var twm = [SIMD2<UInt32>](repeating: .zero, count: N / 2), twmi = twm
var c = (UInt64(1), UInt64(0)), ci = c
for k in 0..<(N / 2) {
  twm[k] = SIMD2(UInt32(c.0), UInt32(c.1)); twmi[k] = SIMD2(UInt32(ci.0), UInt32(ci.1))
  c = ((c.0 * w.0 % M31 + M31 - c.1 * w.1 % M31) % M31, (c.0 * w.1 % M31 + c.1 * w.0 % M31) % M31)
  ci = ((ci.0 * winv.0 % M31 + M31 - ci.1 * winv.1 % M31) % M31, (ci.0 * winv.1 % M31 + ci.1 * winv.0 % M31) % M31)
}

// ---- Metal setup ----
let dev = MTLCreateSystemDefaultDevice()!
let src = try! String(contentsOfFile: kernelPath, encoding: .utf8)
let lib = try! dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let kWeight = pso("weight"), kFFT = pso("fft_f"), kNTT = pso("ntt_m"), kSq = pso("square"), kC1 = pso("carry1"), kC2 = pso("carry2")
func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let bWidth = buf(width), bWexp = buf(wexp), bW = buf(w32), bIW = buf(iw32), bTwf = buf(twf), bTwm = buf(twm), bTwmi = buf(twmi)
let bDig = dev.makeBuffer(length: N * 4, options: .storageModeShared)!
let fA = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!, fB = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!
let mA = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!, mB = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!
let bCout = dev.makeBuffer(length: (N / B) * 8, options: .storageModePrivate)!
let bStat = dev.makeBuffer(length: 16, options: .storageModeShared)!
memset(bStat.contents(), 0, 16)
let dig = bDig.contents().bindMemory(to: Int32.self, capacity: N)
for j in 0..<N { dig[j] = 0 }
dig[0] = 3
let q = dev.makeCommandQueue()!

// radix plan: 16, 16, ..., then whatever power of 2 is left
if maxRadix <= 2 { pack = false }
let TN = pack ? N / 2 : N          // transform length
var radices: [Int] = []
if maxRadix > 2 {
  var rem = TN.trailingZeroBitCount
  while rem >= 4 { radices.append(16); rem -= 4 }
  if rem > 0 { radices.append(1 << rem) }
}
var kPF: [Int: MTLComputePipelineState] = [:], kPM: [Int: MTLComputePipelineState] = [:]
var wrF: [Int: MTLBuffer] = [:], wrM: [Int: MTLBuffer] = [:], wrMi: [Int: MTLBuffer] = [:]
for R in Set(radices) {
  kPF[R] = pso("passf\(R)"); kPM[R] = pso("passm\(R)")
  wrF[R] = buf((0..<(R / 2)).map { twf[$0 * N / R] })
  wrM[R] = buf((0..<(R / 2)).map { twm[$0 * N / R] })
  wrMi[R] = buf((0..<(R / 2)).map { twmi[$0 * N / R] })
}
// half-length tables (W_M^k = W_N^2k, k < M/2) for the packed transforms
let bTwfT = pack ? buf((0..<(TN / 2)).map { twf[2 * $0] }) : bTwf
let bTwmT = pack ? buf((0..<(TN / 2)).map { twm[2 * $0] }) : bTwm
let bTwmiT = pack ? buf((0..<(TN / 2)).map { twmi[2 * $0] }) : bTwmi
let kW2 = pso("weight2"), kTF = pso("tail_f"), kTM = pso("tail_m"), kC1P = pso("carry1p"), kC1T = pso("carry1t")
let useC1T = pack && N % 2048 == 0 && B == 16

struct Prm { var N: UInt32; var Ns: UInt32; var B: UInt32; var logN: UInt32 }
func grid(_ e: MTLComputeCommandEncoder, _ k: MTLComputePipelineState, _ n: Int) {
  e.setComputePipelineState(k)
  e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 256), height: 1, depth: 1))
}

func encodeSquaring(_ e: MTLComputeCommandEncoder) {
  e.setBuffer(bDig, offset: 0, index: 0); e.setBuffer(fA, offset: 0, index: 1); e.setBuffer(mA, offset: 0, index: 2)
  e.setBuffer(bW, offset: 0, index: 3); e.setBuffer(bWexp, offset: 0, index: 4)
  grid(e, pack ? kW2 : kWeight, TN)
  var fin = fA, fout = fB, min_ = mA, mout = mB
  for pass in 0..<2 {   // 0 forward, 1 inverse
    var inv: UInt32 = UInt32(pass)
    var Ns = 1
    if maxRadix > 2 {
      for R in radices {
        var prm = Prm(N: UInt32(TN), Ns: UInt32(Ns), B: UInt32(B), logN: UInt32(logN))
        e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwfT, offset: 0, index: 2)
        e.setBuffer(wrF[R]!, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4); e.setBytes(&inv, length: 4, index: 5)
        grid(e, kPF[R]!, TN / R)
        e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1)
        e.setBuffer(pass == 0 ? bTwmT : bTwmiT, offset: 0, index: 2); e.setBuffer(pass == 0 ? wrM[R]! : wrMi[R]!, offset: 0, index: 3)
        e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
        grid(e, kPM[R]!, TN / R)
        swap(&fin, &fout); swap(&min_, &mout)
        Ns *= R
      }
      Ns = N
    }
    while Ns < N {
      var prm = Prm(N: UInt32(N), Ns: UInt32(Ns), B: UInt32(B), logN: UInt32(logN))
      e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwf, offset: 0, index: 2)
      e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 3); e.setBytes(&inv, length: 4, index: 4)
      grid(e, kFFT, N / 2)
      e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1)
      e.setBuffer(pass == 0 ? bTwm : bTwmi, offset: 0, index: 2); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 3)
      grid(e, kNTT, N / 2)
      swap(&fin, &fout); swap(&min_, &mout)
      Ns *= 2
    }
    if pass == 0 && pack {
      var prm = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
      e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwf, offset: 0, index: 2)
      e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 3)
      grid(e, kTF, TN)
      e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1); e.setBuffer(bTwm, offset: 0, index: 2)
      e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 3)
      grid(e, kTM, TN)
      swap(&fin, &fout); swap(&min_, &mout)
    } else if pass == 0 {
      e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(min_, offset: 0, index: 1)
      grid(e, kSq, N)
    }
  }
  var prm = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
  e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(min_, offset: 0, index: 1); e.setBuffer(bDig, offset: 0, index: 2)
  e.setBuffer(bCout, offset: 0, index: 3); e.setBuffer(bIW, offset: 0, index: 4); e.setBuffer(bWexp, offset: 0, index: 5)
  e.setBuffer(bWidth, offset: 0, index: 6); e.setBuffer(bStat, offset: 0, index: 7); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 8)
  if useC1T {
    e.setComputePipelineState(kC1T)
    e.dispatchThreadgroups(MTLSize(width: N / 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  } else { grid(e, pack ? kC1P : kC1, N / B) }
  e.setBuffer(bDig, offset: 0, index: 0); e.setBuffer(bCout, offset: 0, index: 1); e.setBuffer(bWidth, offset: 0, index: 2)
  e.setBuffer(bStat, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
  grid(e, kC2, N / B)
}

// --profile: time each kernel type alone (10 repeats in one command buffer) and report GB/s
if profileMode {
  func timeIt(_ name: String, _ bytes: Double, _ body: (MTLComputeCommandEncoder) -> Void) {
    var best = Double.infinity
    for _ in 0..<3 {
      let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
      for _ in 0..<10 { body(e) }
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / 10)
    }
    print(String(format: "%-14@ %8.3f ms  %6.1f GB/s", name, best * 1e3, bytes / best / 1e9))
  }
  let nb = Double(TN) * 8 * 2   // read + write of 8-byte complex values
  var inv: UInt32 = 0
  var Ns = 1
  var total = 0.0
  for R in radices {
    var prm = Prm(N: UInt32(TN), Ns: UInt32(Ns), B: UInt32(B), logN: UInt32(logN))
    timeIt("passf\(R) Ns=\(Ns)", nb) { e in
      e.setBuffer(fA, offset: 0, index: 0); e.setBuffer(fB, offset: 0, index: 1); e.setBuffer(bTwfT, offset: 0, index: 2)
      e.setBuffer(wrF[R]!, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4); e.setBytes(&inv, length: 4, index: 5)
      grid(e, kPF[R]!, TN / R) }
    timeIt("passm\(R) Ns=\(Ns)", nb) { e in
      e.setBuffer(mA, offset: 0, index: 0); e.setBuffer(mB, offset: 0, index: 1); e.setBuffer(bTwmT, offset: 0, index: 2)
      e.setBuffer(wrM[R]!, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
      grid(e, kPM[R]!, TN / R) }
    Ns *= R
  }
  if pack {
    var prmN = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
    timeIt("weight2", Double(N) * (4 + 4 + 1) + Double(TN) * 16) { e in
      e.setBuffer(bDig, offset: 0, index: 0); e.setBuffer(fA, offset: 0, index: 1); e.setBuffer(mA, offset: 0, index: 2)
      e.setBuffer(bW, offset: 0, index: 3); e.setBuffer(bWexp, offset: 0, index: 4); grid(e, kW2, TN) }
    timeIt("tail_f", nb) { e in
      e.setBuffer(fA, offset: 0, index: 0); e.setBuffer(fB, offset: 0, index: 1); e.setBuffer(bTwf, offset: 0, index: 2)
      e.setBytes(&prmN, length: MemoryLayout<Prm>.size, index: 3); grid(e, kTF, TN) }
    timeIt("tail_m", nb) { e in
      e.setBuffer(mA, offset: 0, index: 0); e.setBuffer(mB, offset: 0, index: 1); e.setBuffer(bTwm, offset: 0, index: 2)
      e.setBytes(&prmN, length: MemoryLayout<Prm>.size, index: 3); grid(e, kTM, TN) }
    timeIt("carry1p", Double(TN) * 16 + Double(N) * (4 + 4 + 2)) { e in
      e.setBuffer(fA, offset: 0, index: 0); e.setBuffer(mA, offset: 0, index: 1); e.setBuffer(bDig, offset: 0, index: 2)
      e.setBuffer(bCout, offset: 0, index: 3); e.setBuffer(bIW, offset: 0, index: 4); e.setBuffer(bWexp, offset: 0, index: 5)
      e.setBuffer(bWidth, offset: 0, index: 6); e.setBuffer(bStat, offset: 0, index: 7); e.setBytes(&prmN, length: MemoryLayout<Prm>.size, index: 8)
      if useC1T { e.setComputePipelineState(kC1T)
        e.dispatchThreadgroups(MTLSize(width: N / 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      } else { grid(e, kC1P, N / B) } }
    timeIt("carry2", Double(N) * 4) { e in
      e.setBuffer(bDig, offset: 0, index: 0); e.setBuffer(bCout, offset: 0, index: 1); e.setBuffer(bWidth, offset: 0, index: 2)
      e.setBuffer(bStat, offset: 0, index: 3); e.setBytes(&prmN, length: MemoryLayout<Prm>.size, index: 4); grid(e, kC2, N / B) }
  }
  timeIt("full squaring", 0) { e in encodeSquaring(e) }
  _ = total
  exit(0)
}

var dump: FileHandle? = nil
if let dp = dumpPath { FileManager.default.createFile(atPath: dp, contents: nil); dump = FileHandle(forWritingAtPath: dp) }
let sortedCk = checkpoints.sorted()
var done = 0
var gpuTime = 0.0
let t0 = Date()
while done < iters {
  let next = sortedCk.first(where: { $0 > done }) ?? iters
  let batch = min(min(next, iters) - done, 256)
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  for _ in 0..<batch { encodeSquaring(e) }
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("GPU error:", err); exit(1) }
  gpuTime += cb.gpuEndTime - cb.gpuStartTime
  done += batch
  if checkpoints.contains(done), let d = dump {
    var hdr: [UInt32] = [UInt32(done), UInt32(N)]
    d.write(Data(bytes: &hdr, count: 8)); d.write(Data(bytes: dig, count: N * 4))
  }
}
let st = bStat.contents().bindMemory(to: UInt32.self, capacity: 4)
let roe = Float(bitPattern: st[0])
print(String(format: "p=%d N=%d iters=%d bpw=%.2f  roundoff=%.4f (limit 0.5)  imag_errors=%u  carry_overflows=%u  %.1f us/iter (GPU), wall %.1fs",
             p, N, iters, Double(p) / Double(N), roe, st[1], st[2], gpuTime / Double(iters) * 1e6, Date().timeIntervalSince(t0)))
dump?.closeFile()
