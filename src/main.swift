// barnarg: PRP squarings of 3 mod M_p on the mac gpu (metal).
// Usage: barnarg P N ITERS [dumpfile k1,k2,...] [--kernels path]
//        barnarg P N P --gerbicz [--proof POWER]     a full checked prp, with a proof
//        barnarg verify FILE N                       check a proof file
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
var kernelPath = "src/barnarg.metal"   // from the repo, or next to build/ when run from elsewhere
if !FileManager.default.fileExists(atPath: kernelPath) {
  kernelPath = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    .deletingLastPathComponent().appendingPathComponent("../src/barnarg.metal").standardized.path
}
if let k = args.firstIndex(of: "--kernels") { kernelPath = args[k + 1]; args.removeSubrange(k...(k + 1)) }
var maxRadix = 256  // 2 = plain radix-2 stages, 16 = radix-16 passes, 256 = threadgroup passes
if let k = args.firstIndex(of: "--radix") { maxRadix = Int(args[k + 1])!; args.removeSubrange(k...(k + 1)) }
let profileMode = CommandLine.arguments.contains("--profile")
var pack = true      // real input packed as N/2 complex (needs maxRadix > 2)
if let k = args.firstIndex(of: "--nopack") { pack = false; args.remove(at: k) }
args.removeAll { $0 == "--profile" }
let noMidFlag = args.contains("--nomid"); args.removeAll { $0 == "--nomid" }
// --gerbicz: run ITERS squarings with gerbicz-li checks (block L, check every C blocks)
let gerbicz = args.contains("--gerbicz"); args.removeAll { $0 == "--gerbicz" }
func intFlag(_ name: String, _ def: Int?) -> Int? {
  guard let k = args.firstIndex(of: name) else { return def }
  let v = Int(args[k + 1]); args.removeSubrange(k...(k + 1)); return v
}
let gL = intFlag("--L", 1000)!, gC = intFlag("--C", 400)!
let injectAt = intFlag("--inject", nil)   // add 1 to a digit of x once at this iteration (tests the check)
let saveEvery = intFlag("--save-every", 600)!   // seconds between save files
var savePathFlag: String? = nil
if let k = args.firstIndex(of: "--save") { savePathFlag = args[k + 1]; args.removeSubrange(k...(k + 1)) }
let noSave = args.contains("--nosave"); args.removeAll { $0 == "--nosave" }
// ctrl-c or kill: finish the current block, save, exit
var stopRequested: sig_atomic_t = 0
signal(SIGINT) { _ in stopRequested = 1 }
signal(SIGTERM) { _ in stopRequested = 1 }
// --proof P: keep residues for a prp proof of power P (full checked runs only, 0 = no proof)
let proofPower = intFlag("--proof", 8)!
// barnarg verify FILE N: check a proof file
var verifyFile: (path: String, E: Int, B: [UInt32], middles: [[UInt32]])? = nil
if args.first == "verify" {
  guard args.count >= 3, let pf = readProof(args[1]) else { print("usage: barnarg verify FILE N (and FILE must be a v2 proof file)"); exit(2) }
  verifyFile = (args[1], pf.E, pf.B, pf.middles)
  args = [String(pf.E), args[2], "0"]
}
guard args.count >= 3, let p = Int(args[0]), let N = Int(args[1]), let iters = Int(args[2]) else {
  print("usage: barnarg P N ITERS [dumpfile k1,k2,...]  or  barnarg verify FILE N"); exit(2)
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
// pick the primitive N-th root whose N/8-th power is c = 2^15 (1 + i), so the kernels can
// multiply by 8th roots of unity with rotations (see mulc in barnarg.metal)
let c8: (UInt64, UInt64) = (1 << 15, 1 << 15)
var w = rootOfUnity(N)
if N >= 8 {
  let w0 = w
  for k in stride(from: UInt64(1), to: 8, by: 2) {
    let cand = cpow(w0.0, w0.1, k)
    if cpow(cand.0, cand.1, UInt64(N / 8)) == c8 { w = cand; break }
  }
  precondition(cpow(w.0, w.1, UInt64(N / 8)) == c8)
}
let winv = cpow(w.0, w.1, UInt64(N - 1))
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
let copts = MTLCompileOptions()
if let defs = ProcessInfo.processInfo.environment["BARNARG_DEFS"] {   // timing experiments only
  var m: [String: NSObject] = [:]
  for d in defs.split(separator: ",") { m[String(d)] = NSNumber(value: 1) }
  copts.preprocessorMacros = m
}
let lib = try! dev.makeLibrary(source: src, options: copts)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let kWeight = pso("weight"), kFFT = pso("fft_f"), kNTT = pso("ntt_m"), kSq = pso("square"), kC1 = pso("carry1"), kC2 = pso("carry2")
func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let bWidth = buf(width), bWexp = buf(wexp), bW = buf(w32), bIW = buf(iw32), bTwf = buf(twf), bTwm = buf(twm), bTwmi = buf(twmi)
let bDig = dev.makeBuffer(length: N * 4, options: .storageModeShared)!
var curDig = bDig   // the number the next encoded squaring works on
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
  if maxRadix >= 256 {
    while rem >= 8 && TN >= 256 * 8 { radices.append(256); rem -= 8 }
    if rem >= 6 && TN >= 64 * 8 { radices.append(64); rem -= 6 }
  }
  while rem >= 4 { radices.append(16); rem -= 4 }
  if rem > 0 { radices.append(1 << rem) }
}
let tgA: [Int: Int] = [256: 16, 64: 8]   // threadgroup passes: R = A * A
var kPF: [Int: MTLComputePipelineState] = [:], kPM: [Int: MTLComputePipelineState] = [:], kPMi: [Int: MTLComputePipelineState] = [:]
var kPFw: [Int: MTLComputePipelineState] = [:], kPMw: [Int: MTLComputePipelineState] = [:]
var wrF: [Int: MTLBuffer] = [:], wrM: [Int: MTLBuffer] = [:], wrMi: [Int: MTLBuffer] = [:]
var wRF: [Int: MTLBuffer] = [:], wRM: [Int: MTLBuffer] = [:], wRMi: [Int: MTLBuffer] = [:]
func cmulM(_ a: (UInt64, UInt64), _ b: (UInt64, UInt64)) -> (UInt64, UInt64) {
  ((a.0 * b.0 % M31 + M31 - a.1 * b.1 % M31) % M31, (a.0 * b.1 % M31 + a.1 * b.0 % M31) % M31)
}
for R in Set(radices) where tgA[R] != nil {
  let A = tgA[R]!
  kPF[R] = pso("passTf\(A)"); kPM[R] = pso("passTm\(A)f"); kPMi[R] = pso("passTm\(A)i")
  kPFw[R] = pso("passTf\(A)w"); kPMw[R] = pso("passTm\(A)w")
  wrF[A] = buf((0..<(A / 2)).map { twf[$0 * N / A] })
  wrM[A] = buf((0..<(A / 2)).map { twm[$0 * N / A] })
  wrMi[A] = buf((0..<(A / 2)).map { twmi[$0 * N / A] })
  wRF[R] = buf((0..<R).map { k -> SIMD2<Float> in let a = -2.0 * Double.pi * Double(k) / Double(R); return SIMD2(Float(cos(a)), Float(sin(a))) })
  let r = cpow(w.0, w.1, UInt64(N / R)), ri = cpow(winv.0, winv.1, UInt64(N / R))
  var x = (UInt64(1), UInt64(0)), xi = x
  var tf = [SIMD2<UInt32>](), ti = [SIMD2<UInt32>]()
  for _ in 0..<R { tf.append(SIMD2(UInt32(x.0), UInt32(x.1))); ti.append(SIMD2(UInt32(xi.0), UInt32(xi.1))); x = cmulM(x, r); xi = cmulM(xi, ri) }
  wRM[R] = buf(tf); wRMi[R] = buf(ti)
}
for R in Set(radices) where tgA[R] == nil {
  kPF[R] = pso("passf\(R)"); kPM[R] = pso("passm\(R)f"); kPMi[R] = pso("passm\(R)i")
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

// one stockham pass of radix R on both the fp32 and M31 buffers
// weighted: first forward pass reads the digits itself (fused weight); fin/min_ are ignored then
let fuseWeight = pack && radices.first.map { tgA[$0] != nil } == true
func encodePass(_ e: MTLComputeCommandEncoder, _ R: Int, _ Ns: Int, _ pass: Int,
                _ fin: MTLBuffer, _ fout: MTLBuffer, _ min_: MTLBuffer, _ mout: MTLBuffer, fp: Bool = true, m31: Bool = true,
                weighted: Bool = false) {
  var prm = Prm(N: UInt32(TN), Ns: UInt32(Ns), B: UInt32(B), logN: UInt32(logN))
  var inv = UInt32(pass)
  if let A = tgA[R] {
    let groups = MTLSize(width: TN / R / 8, height: 1, depth: 1), threads = MTLSize(width: 8 * A, height: 1, depth: 1)
    if fp {
      e.setComputePipelineState(weighted ? kPFw[R]! : kPF[R]!)
      e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwfT, offset: 0, index: 2)
      e.setBuffer(wrF[A]!, offset: 0, index: 3); e.setBuffer(wRF[R]!, offset: 0, index: 4)
      e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 5); e.setBytes(&inv, length: 4, index: 6)
      e.setBuffer(curDig, offset: 0, index: 7); e.setBuffer(bW, offset: 0, index: 8)
      e.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }
    if m31 {
      e.setComputePipelineState(weighted ? kPMw[R]! : pass == 0 ? kPM[R]! : kPMi[R]!)
      e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1); e.setBuffer(pass == 0 ? bTwmT : bTwmiT, offset: 0, index: 2)
      e.setBuffer(pass == 0 ? wrM[A]! : wrMi[A]!, offset: 0, index: 3); e.setBuffer(pass == 0 ? wRM[R]! : wRMi[R]!, offset: 0, index: 4)
      e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 5)
      e.setBuffer(curDig, offset: 0, index: 6); e.setBuffer(bWexp, offset: 0, index: 7)
      e.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    }
    return
  }
  if fp {
    e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwfT, offset: 0, index: 2)
    e.setBuffer(wrF[R]!, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4); e.setBytes(&inv, length: 4, index: 5)
    grid(e, kPF[R]!, TN / R)
  }
  if m31 {
    e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1)
    e.setBuffer(pass == 0 ? bTwmT : bTwmiT, offset: 0, index: 2); e.setBuffer(pass == 0 ? wrM[R]! : wrMi[R]!, offset: 0, index: 3)
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
    grid(e, pass == 0 ? kPM[R]! : kPMi[R]!, TN / R)
  }
}

// fused middle (last forward pass + tail + first inverse pass), when the plan ends in 64
let fuseMid = pack && maxRadix >= 256 && radices.count >= 2 && radices.last == 64 && TN / 64 >= 8
let kMidF = pso("midf"), kMidM = pso("midm")
var noMid = noMidFlag   // --nomid, for comparisons
func encodeMid(_ e: MTLComputeCommandEncoder, _ fin: MTLBuffer, _ fout: MTLBuffer, _ min_: MTLBuffer, _ mout: MTLBuffer,
               fp: Bool = true, m31: Bool = true) {
  var prm = Prm(N: UInt32(TN), Ns: 0, B: UInt32(B), logN: UInt32(logN))
  let groups = MTLSize(width: TN / 512, height: 1, depth: 1), threads = MTLSize(width: 64, height: 1, depth: 1)
  if fp {
    e.setComputePipelineState(kMidF)
    e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(fout, offset: 0, index: 1); e.setBuffer(bTwfT, offset: 0, index: 2)
    e.setBuffer(wrF[8]!, offset: 0, index: 3); e.setBuffer(wRF[64]!, offset: 0, index: 4); e.setBuffer(bTwf, offset: 0, index: 5)
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 6)
    e.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
  }
  if m31 {
    e.setComputePipelineState(kMidM)
    e.setBuffer(min_, offset: 0, index: 0); e.setBuffer(mout, offset: 0, index: 1); e.setBuffer(bTwmT, offset: 0, index: 2)
    e.setBuffer(wRM[64]!, offset: 0, index: 3); e.setBuffer(wRMi[64]!, offset: 0, index: 4); e.setBuffer(bTwm, offset: 0, index: 5)
    e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 6)
    e.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
  }
}

func encodeSquaring(_ e: MTLComputeCommandEncoder) {
  e.setBuffer(curDig, offset: 0, index: 0); e.setBuffer(fA, offset: 0, index: 1); e.setBuffer(mA, offset: 0, index: 2)
  e.setBuffer(bW, offset: 0, index: 3); e.setBuffer(bWexp, offset: 0, index: 4)
  if !fuseWeight { grid(e, pack ? kW2 : kWeight, TN) }
  var fin = fA, fout = fB, min_ = mA, mout = mB
  if fuseMid && !noMid {
    var Ns = 1
    for (ri, R) in radices.dropLast().enumerated() {
      encodePass(e, R, Ns, 0, fin, fout, min_, mout, weighted: fuseWeight && ri == 0)
      swap(&fin, &fout); swap(&min_, &mout); Ns *= R
    }
    encodeMid(e, fin, fout, min_, mout)
    swap(&fin, &fout); swap(&min_, &mout)
    Ns = 64
    for R in radices.dropLast() {
      encodePass(e, R, Ns, 1, fin, fout, min_, mout)
      swap(&fin, &fout); swap(&min_, &mout); Ns *= R
    }
  } else {
  for pass in 0..<2 {   // 0 forward, 1 inverse
    var inv: UInt32 = UInt32(pass)
    var Ns = 1
    if maxRadix > 2 {
      for (ri, R) in radices.enumerated() {
        encodePass(e, R, Ns, pass, fin, fout, min_, mout, weighted: fuseWeight && pass == 0 && ri == 0)
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
  }
  var prm = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
  e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(min_, offset: 0, index: 1); e.setBuffer(curDig, offset: 0, index: 2)
  e.setBuffer(bCout, offset: 0, index: 3); e.setBuffer(bIW, offset: 0, index: 4); e.setBuffer(bWexp, offset: 0, index: 5)
  e.setBuffer(bWidth, offset: 0, index: 6); e.setBuffer(bStat, offset: 0, index: 7); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 8)
  if useC1T {
    e.setComputePipelineState(kC1T)
    e.dispatchThreadgroups(MTLSize(width: N / 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  } else { grid(e, pack ? kC1P : kC1, N / B) }
  e.setBuffer(curDig, offset: 0, index: 0); e.setBuffer(bCout, offset: 0, index: 1); e.setBuffer(bWidth, offset: 0, index: 2)
  e.setBuffer(bStat, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
  grid(e, kC2, N / B)
}

// carry of the inverse transform in (fin, min_) into curDig
func encodeCarry(_ e: MTLComputeCommandEncoder, _ fin: MTLBuffer, _ min_: MTLBuffer) {
  var prm = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
  e.setBuffer(fin, offset: 0, index: 0); e.setBuffer(min_, offset: 0, index: 1); e.setBuffer(curDig, offset: 0, index: 2)
  e.setBuffer(bCout, offset: 0, index: 3); e.setBuffer(bIW, offset: 0, index: 4); e.setBuffer(bWexp, offset: 0, index: 5)
  e.setBuffer(bWidth, offset: 0, index: 6); e.setBuffer(bStat, offset: 0, index: 7); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 8)
  if useC1T {
    e.setComputePipelineState(kC1T)
    e.dispatchThreadgroups(MTLSize(width: N / 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  } else { grid(e, kC1P, N / B) }
  e.setBuffer(curDig, offset: 0, index: 0); e.setBuffer(bCout, offset: 0, index: 1); e.setBuffer(bWidth, offset: 0, index: 2)
  e.setBuffer(bStat, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
  grid(e, kC2, N / B)
}

// a = a * b mod M_p. slow path (no fused passes), only used once every L squarings.
let kTMF = pso("tailmul_f"), kTMM = pso("tailmul_m")
let fC = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!, fD = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!
let mC = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!, mD = dev.makeBuffer(length: N * 8, options: .storageModePrivate)!
func encodeForward(_ e: MTLComputeCommandEncoder, _ digits: MTLBuffer,
                   _ f0: MTLBuffer, _ f1: MTLBuffer, _ m0: MTLBuffer, _ m1: MTLBuffer) -> (MTLBuffer, MTLBuffer) {
  e.setBuffer(digits, offset: 0, index: 0); e.setBuffer(f0, offset: 0, index: 1); e.setBuffer(m0, offset: 0, index: 2)
  e.setBuffer(bW, offset: 0, index: 3); e.setBuffer(bWexp, offset: 0, index: 4)
  grid(e, kW2, TN)
  var fin = f0, fout = f1, min_ = m0, mout = m1, Ns = 1
  for R in radices { encodePass(e, R, Ns, 0, fin, fout, min_, mout); swap(&fin, &fout); swap(&min_, &mout); Ns *= R }
  return (fin, min_)
}
func encodeMul(_ e: MTLComputeCommandEncoder, _ a: MTLBuffer, _ b: MTLBuffer) {
  precondition(pack && maxRadix > 2)
  let (fb, mb) = encodeForward(e, b, fC, fD, mC, mD)
  let (fa, ma) = encodeForward(e, a, fA, fB, mA, mB)
  let fo = fa === fA ? fB : fA, mo = ma === mA ? mB : mA
  var prm = Prm(N: UInt32(N), Ns: 0, B: UInt32(B), logN: UInt32(logN))
  e.setBuffer(fa, offset: 0, index: 0); e.setBuffer(fb, offset: 0, index: 1); e.setBuffer(fo, offset: 0, index: 2)
  e.setBuffer(bTwf, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
  grid(e, kTMF, TN)
  e.setBuffer(ma, offset: 0, index: 0); e.setBuffer(mb, offset: 0, index: 1); e.setBuffer(mo, offset: 0, index: 2)
  e.setBuffer(bTwm, offset: 0, index: 3); e.setBytes(&prm, length: MemoryLayout<Prm>.size, index: 4)
  grid(e, kTMM, TN)
  var fin = fo, fout = fa, min_ = mo, mout = ma, Ns = 1
  for R in radices { encodePass(e, R, Ns, 1, fin, fout, min_, mout); swap(&fin, &fout); swap(&min_, &mout); Ns *= R }
  curDig = a
  encodeCarry(e, fin, min_)
}

// ---- reading numbers back on the cpu ----
// value mod q = 2^61 - 1 (2^61 = 1 mod q, so digit j weighs 2^(s_j mod 61)), used to compare
// two numbers that may be written with different balanced digits, and res64 (low 64 bits).
let q61: UInt64 = (1 << 61) - 1
let sExp: [Int] = (0..<N).map { ceilDiv($0 * p, N) }
func mod61(_ x: UInt64, _ y: UInt64) -> UInt64 {
  let (hi, lo) = x.multipliedFullWidth(by: y)
  var r = (lo & q61) + (lo >> 61) + (hi << 3)
  r = (r & q61) + (r >> 61)
  return r >= q61 ? r - q61 : r
}
let mpMod61: UInt64 = ((1 << UInt64(p % 61)) - 1) % q61
func hash61(_ b: MTLBuffer) -> UInt64 {
  let d = b.contents().bindMemory(to: Int32.self, capacity: N)
  var sum = [Int64](repeating: 0, count: 61)
  for j in 0..<N { sum[sExp[j] % 61] += Int64(d[j]) }
  var h: UInt64 = 0
  for e in 0..<61 {
    let v = UInt64(((sum[e] % Int64(q61)) + Int64(q61)) % Int64(q61))
    h = (h + mod61(v, 1 << UInt64(e))) % q61
  }
  return h
}
func sameValue(_ a: MTLBuffer, _ b: MTLBuffer) -> Bool {
  if memcmp(a.contents(), b.contents(), N * 4) == 0 { return true }
  let ha = hash61(a), hb = hash61(b)
  return ha == hb || (ha + mpMod61) % q61 == hb || (hb + mpMod61) % q61 == ha
}
func res64(_ b: MTLBuffer) -> UInt64 {   // low 64 bits of the value in [0, M_p)
  let d = b.contents().bindMemory(to: Int32.self, capacity: N)
  var v: UInt64 = 0
  for j in 0..<N where sExp[j] < 64 { v = v &+ (UInt64(bitPattern: Int64(d[j])) << UInt64(sExp[j])) }
  var top = N - 1
  while top > 0 && d[top] == 0 { top -= 1 }
  if d[top] < 0 { v = v &- 1 }           // negative: add M_p = 2^p - 1, which is -1 mod 2^64
  return v
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
    let ns = Ns
    timeIt("fp32 R=\(R) Ns=\(ns)", nb) { e in encodePass(e, R, ns, 0, fA, fB, mA, mB, m31: false) }
    timeIt("m31  R=\(R) Ns=\(ns)", nb) { e in encodePass(e, R, ns, 0, fA, fB, mA, mB, fp: false) }
    Ns *= R
  }
  _ = inv
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
  if fuseMid {
    timeIt("mid fp32", nb) { e in encodeMid(e, fA, fB, mA, mB, m31: false) }
    timeIt("mid m31", nb) { e in encodeMid(e, fA, fB, mA, mB, fp: false) }
  }
  timeIt("full squaring", 0) { e in encodeSquaring(e) }
  _ = total
  exit(0)
}

// ---- small helpers for the checked prp and the proofs ----
func runGPU(_ body: (MTLComputeCommandEncoder) -> Void) {
  let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
  body(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("gpu error:", err); exit(1) }
}
func square(_ b: MTLBuffer, _ n: Int) {
  var left = n
  while left > 0 {
    let m = min(left, 256)
    runGPU { e in for _ in 0..<m { curDig = b; encodeSquaring(e) } }
    left -= m
  }
}
func newDigits(_ v: Int32 = 0) -> MTLBuffer {
  let b = dev.makeBuffer(length: N * 4, options: .storageModeShared)!
  memset(b.contents(), 0, N * 4); b.contents().bindMemory(to: Int32.self, capacity: N)[0] = v
  return b
}
func copyDigits(_ to: MTLBuffer, _ from: MTLBuffer) { memcpy(to.contents(), from.contents(), N * 4) }
func toWords(_ b: MTLBuffer) -> [UInt32] { digitsToWords(b.contents().bindMemory(to: Int32.self, capacity: N), width, p) }
func fromWords(_ w: [UInt32], _ b: MTLBuffer) { wordsToDigits(w, width, b.contents().bindMemory(to: Int32.self, capacity: N)) }

// x = x^h * y, left to right. y must not be x.
let bBase = newDigits()
func powMul(_ x: MTLBuffer, _ h: UInt64, _ y: MTLBuffer) {
  if h == 0 { copyDigits(x, y); return }
  copyDigits(bBase, x)
  let top = 63 - h.leadingZeroBitCount
  runGPU { e in
    for i in stride(from: top - 1, through: 0, by: -1) {
      curDig = x; encodeSquaring(e)
      if (h >> UInt64(i)) & 1 == 1 { encodeMul(e, x, bBase) }
    }
    encodeMul(e, x, y)
  }
}

// ---- prp proofs ----
// the run keeps x_k at the proof points in barnarg-<p>-proof/. after the test they are folded
// into the middles, level by level, the same way as gpuowl (a stack of power + 1 numbers).
let proofDir = "barnarg-\(p)-proof"
func residuePath(_ k: Int) -> String { "\(proofDir)/\(k)" }
func buildProof(_ power: Int) -> (B: [UInt32], middles: [[UInt32]])? {
  let pts = proofPoints(p, power)
  func load(_ k: Int) -> [UInt32]? {
    let r = readResidue(residuePath(k), p)
    if r == nil { print("\(residuePath(k)) is missing or damaged, can't make the proof") }
    return r
  }
  guard let B = load(p) else { return nil }
  var hash = proofHash(p, [], B)
  var middles: [[UInt32]] = [], hashes: [UInt64] = []
  let stack = (0...power).map { _ in newDigits() }
  for level in 0..<power {
    let s = 1 << (power - level - 1)
    var top = 0
    for i in 0..<(1 << level) {
      guard let w = load(pts[s * (2 * i + 1) - 1]) else { return nil }
      fromWords(w, stack[top]); top += 1
      var b = 0
      while i & (1 << b) != 0 {   // fold the top two: a = a^h * b
        top -= 1
        powMul(stack[top - 1], hashes[level - 1 - b], stack[top])
        b += 1
      }
    }
    let M = toWords(stack[0])
    hash = proofHash(p, hash, M)
    middles.append(M); hashes.append(h64(hash))
    print(String(format: "proof level %d: middle res64 %016llx, h %016llx", level, UInt64(M[0]) | UInt64(M[1]) << 32, hashes.last!))
  }
  return (B, middles)
}
// the verifier's side: fold A = 3 and B down with the same hashes, then A^(2^span) must be B
func verifyProof(_ B0: [UInt32], _ middles: [[UInt32]]) -> Bool {
  if ([B0] + middles).contains(where: { $0.allSatisfy { $0 == 0 } }) { return false }
  let A = newDigits(3)
  var Bc = newDigits(), Mb = newDigits()
  fromWords(B0, Bc)
  var hash = proofHash(p, [], B0)
  var span = p
  for mw in middles {
    hash = proofHash(p, hash, mw)
    let h = h64(hash)
    if span % 2 == 1 { square(Bc, 1) }
    fromWords(mw, Mb); powMul(Mb, h, Bc); swap(&Bc, &Mb)   // B = M^h * B
    fromWords(mw, Mb); powMul(A, h, Mb)                   // A = A^h * M
    span = (span + 1) / 2
  }
  print("checking the proof: \(span) squarings")
  square(A, span)
  return sameValue(A, Bc)
}

if let vf = verifyFile {
  let t = Date()
  let ok = verifyProof(vf.B, vf.middles)
  let r = UInt64(vf.B[0]) | UInt64(vf.B[1]) << 32
  print(String(format: "%@: M%d power %d, res64 %016llx, %@ (%.1f s)", vf.path, p, vf.middles.count, r,
               ok ? "proof ok" : "proof BAD", Date().timeIntervalSince(t)))
  exit(ok ? 0 : 1)
}

// ---- PRP with gerbicz-li checks ----
// x_k = 3^(2^k). every L squarings d <- d * x, so d_t = x_0 x_L ... x_tL. that gives
// d_t = 3 * d_(t-1)^(2^L), and an error anywhere since the last check breaks it. every C blocks
// we check it (L extra squarings of the old d), and on a failure go back to the last good state.
// the last ITERS mod L squarings are done twice and compared instead.
if gerbicz {
  precondition(pack && maxRadix > 2, "gerbicz mode needs the packed transform")
  let L = gL, C = gC
  let bD = newDigits(3), bT = newDigits(), bThree = newDigits(3)
  // proof residues are written as soon as x reaches a point. if a later check fails, the
  // rollback goes back before that point and the file is written again.
  let wantProof = proofPower > 0 && iters == p
  let pts = wantProof ? proofPoints(p, proofPower) : []
  let ptSet = Set(pts)
  if wantProof {
    try? FileManager.default.createDirectory(atPath: proofDir, withIntermediateDirectories: true)
    print(String(format: "keeping %d residues for a power %d proof in %@ (%.1f gb)", pts.count, proofPower, proofDir,
                 Double(pts.count * (residueBytes(p) + 8)) / 1e9))
  }
  func keep(_ k: Int, _ b: MTLBuffer) {
    do { try writeResidue(residuePath(k), p, toWords(b)) } catch { print("could not write \(residuePath(k)): \(error)"); exit(1) }
  }
  func squareX(_ k0: Int, _ n: Int) {   // x from k0 to k0 + n, stopping at proof points
    var k = k0
    while k < k0 + n {
      let next = pts.first { $0 > k && $0 <= k0 + n } ?? (k0 + n)
      square(bDig, next - k); k = next
      if ptSet.contains(k) && k < p { keep(k, bDig) }
    }
  }
  var savedX = [Int32](repeating: 0, count: N), savedD = savedX, savedK = 0
  func save(_ k: Int) {
    savedX.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, bDig.contents(), N * 4) }
    savedD.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, bD.contents(), N * 4) }
    savedK = k
  }
  func restore() -> Int {
    savedX.withUnsafeBytes { _ = memcpy(bDig.contents(), $0.baseAddress!, N * 4) }
    savedD.withUnsafeBytes { _ = memcpy(bD.contents(), $0.baseAddress!, N * 4) }
    return savedK
  }
  // save file: "BARNARG1", then p, N, k, verified k as u64, then x, d, verified x, verified d
  // (N x i32 each), then a 64-bit hash of everything before it. written to .tmp and renamed.
  let savePath = savePathFlag ?? "barnarg-\(p).save"
  func writeSave(_ k: Int) {
    if noSave { return }
    var data = Data("BARNARG1".utf8)
    var hdr: [UInt64] = [UInt64(p), UInt64(N), UInt64(k), UInt64(savedK)]
    data.append(Data(bytes: &hdr, count: 32))
    data.append(Data(bytes: bDig.contents(), count: N * 4))
    data.append(Data(bytes: bD.contents(), count: N * 4))
    savedX.withUnsafeBytes { data.append(Data($0)) }
    savedD.withUnsafeBytes { data.append(Data($0)) }
    var h = hash64(data)
    data.append(Data(bytes: &h, count: 8))
    let tmp = savePath + ".tmp"
    do {
      try data.write(to: URL(fileURLWithPath: tmp))
      _ = try? FileManager.default.removeItem(atPath: savePath)
      try FileManager.default.moveItem(atPath: tmp, toPath: savePath)
    } catch { print("could not write \(savePath): \(error)") }
  }
  func loadSave() -> Int? {
    guard let data = FileManager.default.contents(atPath: savePath) else { return nil }
    let want = 8 + 32 + 4 * N * 4 + 8
    guard data.count == want, data.prefix(8) == Data("BARNARG1".utf8) else {
      print("\(savePath) is not a barnarg save file for this size, ignoring it"); return nil
    }
    let body = data.prefix(want - 8)
    let stored = data.suffix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    guard hash64(Data(body)) == stored else { print("\(savePath) is damaged (hash mismatch), ignoring it"); return nil }
    let hdr = data.subdata(in: 8..<40).withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    guard hdr[0] == UInt64(p), hdr[1] == UInt64(N) else { print("\(savePath) is for p=\(hdr[0]) N=\(hdr[1]), ignoring it"); return nil }
    let off = 40, sz = N * 4
    data.withUnsafeBytes { raw in
      memcpy(bDig.contents(), raw.baseAddress! + off, sz)
      memcpy(bD.contents(), raw.baseAddress! + off + sz, sz)
      savedX.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, raw.baseAddress! + off + 2 * sz, sz) }
      savedD.withUnsafeMutableBytes { _ = memcpy($0.baseAddress!, raw.baseAddress! + off + 3 * sz, sz) }
    }
    if Int(hdr[2]) > iters { print("\(savePath) is already past \(iters) squarings (k=\(hdr[2])), ignoring it"); return nil }
    savedK = Int(hdr[3])
    return Int(hdr[2])
  }

  let run = runGPU
  save(0)
  var k = 0, checks = 0, failures = 0, injected = false, resumedAt = 0
  if let rk = loadSave() {
    k = rk
    resumedAt = rk
    print("resuming from \(savePath) at k=\(k) (last verified k=\(savedK))")
  }
  var lastSave = Date()
  let kFull = iters / L * L
  let start = Date()
  while k < kFull {
    squareX(k, L); k += L
    if let at = injectAt, !injected, k >= at {
      bDig.contents().bindMemory(to: Int32.self, capacity: N)[1] += 1
      injected = true
      print("injected an error into x at iteration \(k)")
    }
    let checkNow = k % (C * L) == 0 || k == kFull
    if checkNow { copyDigits(bT, bD) }
    run { e in encodeMul(e, bD, bDig) }
    if checkNow {
      square(bT, L)
      run { e in encodeMul(e, bT, bThree) }
      checks += 1
      if sameValue(bT, bD) {
        save(k)
        writeSave(k); lastSave = Date()
        let el = Date().timeIntervalSince(start)
        print(String(format: "k=%d  check ok  res64=%016llx  %.1f iter/s", k, res64(bDig), Double(k - resumedAt) / el))
      } else {
        failures += 1
        let bad = k
        k = restore()
        print("k=\(bad)  gerbicz check FAILED, going back to k=\(k)")
      }
    }
    if stopRequested != 0 { writeSave(k); print("stopped at k=\(k), saved to \(savePath)"); exit(0) }
    if Date().timeIntervalSince(lastSave) >= Double(saveEvery) { writeSave(k); lastSave = Date() }
  }
  let r = iters - kFull
  if stopRequested != 0 { writeSave(k); print("stopped at k=\(k), saved to \(savePath)"); exit(0) }
  if r > 0 {
    while true {
      copyDigits(bT, bDig)
      squareX(kFull, r); square(bT, r)
      if let at = injectAt, !injected, at > kFull {
        bDig.contents().bindMemory(to: Int32.self, capacity: N)[1] += 1
        injected = true
        print("injected an error into x in the last \(r) squarings")
      }
      if sameValue(bT, bDig) { break }
      failures += 1
      print("last \(r) squarings disagreed, redoing them")
      _ = restore()
    }
  }
  let st = bStat.contents().bindMemory(to: UInt32.self, capacity: 4)
  let rv = res64(bDig)
  let isPRP = rv == 9 && (hash61(bDig) == 9 || (hash61(bDig) + mpMod61) % q61 == 9)
  let el = Date().timeIntervalSince(start)
  print(String(format: "p=%d N=%d iters=%d  res64=%016llx  %@  checks=%d failures=%d  roundoff=%.4f  %.1f iter/s (wall, checks included)",
               p, N, iters, rv, iters == p ? (isPRP ? "PRP" : "not PRP") : "", checks, failures, Float(bitPattern: st[0]), Double(iters - resumedAt) / el))
  var proofNote = ""
  if wantProof {
    keep(p, bDig)
    let tp = Date()
    let proofPath = "\(p)-\(proofPower).proof"
    if let pr = buildProof(proofPower) {
      if verifyProof(pr.B, pr.middles) {
        do {
          try writeProof(proofPath, p, pr.B, pr.middles)
          proofNote = "  proof=\(proofPath)"
          _ = try? FileManager.default.removeItem(atPath: proofDir)
          print(String(format: "wrote %@ and checked it (%.1f s)", proofPath, Date().timeIntervalSince(tp)))
        } catch { print("could not write \(proofPath): \(error)") }
      } else {
        print("the proof did not check out, keeping \(proofDir) so it can be made again")
      }
    }
    if proofNote.isEmpty { failures += 1 }
  }
  if iters == p && failures == 0 || iters == p && injectAt != nil {
    let line = String(format: "%@  M%d  %@  res64=%016llx  N=%d  checks=%d  failures=%d%@\n",
                      ISO8601DateFormatter().string(from: Date()), p, isPRP ? "PRP" : "not PRP", rv, N, checks, failures, proofNote)
    if let fh = FileHandle(forWritingAtPath: "results.txt") { fh.seekToEndOfFile(); fh.write(Data(line.utf8)); fh.closeFile() }
    else { FileManager.default.createFile(atPath: "results.txt", contents: Data(line.utf8)) }
    _ = try? FileManager.default.removeItem(atPath: savePath)
  }
  if let dp = dumpPath {
    FileManager.default.createFile(atPath: dp, contents: nil)
    let d = FileHandle(forWritingAtPath: dp)!
    var hdr: [UInt32] = [UInt32(iters), UInt32(N)]
    d.write(Data(bytes: &hdr, count: 8)); d.write(Data(bytes: bDig.contents(), count: N * 4)); d.closeFile()
  }
  exit(failures > 0 && injectAt == nil ? 3 : 0)
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
