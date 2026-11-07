// prp proofs: the parts that don't touch the gpu.
//
// a proof (pietrzak, as in gpuowl and prime95) lets the server check a whole prp test with
// about E / 2^power squarings instead of E. the run keeps x_k = 3^(2^k) at 2^power points,
// and afterwards these are folded into `power` middle values with random exponents taken from
// a sha3 hash chain. the file format is gpuowl's version 2, which primenet takes.
import Foundation

// ---- sha3-256 ----
struct SHA3 {
  private var s = [UInt64](repeating: 0, count: 25)
  private var buf = [UInt8]()
  private static let rate = 136
  private static let rc: [UInt64] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
  private static let rot: [UInt64] = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14]

  private mutating func permute() {
    var c = [UInt64](repeating: 0, count: 5), b = [UInt64](repeating: 0, count: 25)
    for round in 0..<24 {
      for x in 0..<5 { c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20] }
      for x in 0..<5 {
        let d = c[(x + 4) % 5] ^ ((c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63))
        for y in 0..<5 { s[x + 5 * y] ^= d }
      }
      for x in 0..<5 { for y in 0..<5 {   // rho and pi
        let r = SHA3.rot[x + 5 * y], v = s[x + 5 * y]
        b[y + 5 * ((2 * x + 3 * y) % 5)] = r == 0 ? v : (v << r) | (v >> (64 - r))
      } }
      for y in 0..<5 { for x in 0..<5 {   // chi
        s[x + 5 * y] = b[x + 5 * y] ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y])
      } }
      s[0] ^= SHA3.rc[round]
    }
  }
  private mutating func absorb(_ block: UnsafeRawBufferPointer) {
    for i in 0..<(SHA3.rate / 8) { s[i] ^= block.loadUnaligned(fromByteOffset: 8 * i, as: UInt64.self) }
    permute()
  }
  mutating func update(_ data: UnsafeRawBufferPointer) {
    var off = 0
    if !buf.isEmpty {
      let take = min(SHA3.rate - buf.count, data.count)
      buf += data[0..<take]; off = take
      if buf.count == SHA3.rate { buf.withUnsafeBytes { absorb($0) }; buf.removeAll() }
    }
    while data.count - off >= SHA3.rate { absorb(UnsafeRawBufferPointer(rebasing: data[off..<(off + SHA3.rate)])); off += SHA3.rate }
    buf += data[off...]
  }
  mutating func update(_ bytes: [UInt8]) { bytes.withUnsafeBytes { update($0) } }
  mutating func finish() -> [UInt8] {
    var last = buf + [UInt8](repeating: 0, count: SHA3.rate - buf.count)
    last[buf.count] ^= 0x06; last[SHA3.rate - 1] ^= 0x80
    last.withUnsafeBytes { absorb($0) }
    return s[0..<4].flatMap { v in (0..<8).map { UInt8(truncatingIfNeeded: v >> (8 * $0)) } }
  }
}

// ---- residues as E-bit numbers ----
func residueBytes(_ E: Int) -> Int { (E - 1) / 8 + 1 }

// hash for a residue: sha3(prefix || bytes). h is its first 8 bytes, little-endian.
func proofHash(_ E: Int, _ prefix: [UInt8], _ words: [UInt32]) -> [UInt8] {
  var h = SHA3()
  h.update(prefix)
  words.withUnsafeBytes { h.update(UnsafeRawBufferPointer(rebasing: $0[0..<residueBytes(E)])) }
  return h.finish()
}
func h64(_ d: [UInt8]) -> UInt64 { d[0..<8].reversed().reduce(0) { $0 << 8 | UInt64($1) } }

// balanced digits (digit j has width[j] bits) -> the value in [0, 2^E - 2] as ceil(E/32) words.
func digitsToWords(_ d: UnsafePointer<Int32>, _ width: [UInt8], _ E: Int) -> [UInt32] {
  let N = width.count
  var v = (0..<N).map { Int64(d[$0]) }
  var c: Int64 = 0
  for j in 0..<N { let t = v[j] + c; v[j] = t & ((1 << Int64(width[j])) - 1); c = t >> Int64(width[j]) }
  var j = 0
  while c != 0 {   // 2^E = 1, so the carry out of the top goes back in at the bottom
    let t = v[j] + c; v[j] = t & ((1 << Int64(width[j])) - 1); c = t >> Int64(width[j])
    j = (j + 1) % N
  }
  var out = [UInt32](repeating: 0, count: (E + 31) / 32)
  var acc: UInt64 = 0, bits = 0, wi = 0, allOnes = true
  for j in 0..<N {
    if v[j] != (1 << Int64(width[j])) - 1 { allOnes = false }
    acc |= UInt64(v[j]) << UInt64(bits); bits += Int(width[j])
    while bits >= 32 { out[wi] = UInt32(truncatingIfNeeded: acc); wi += 1; acc >>= 32; bits -= 32 }
  }
  if bits > 0 { out[wi] = UInt32(truncatingIfNeeded: acc) }
  if allOnes { return [UInt32](repeating: 0, count: out.count) }   // 2^E - 1 is 0
  return out
}

// the other way, into balanced digits
func wordsToDigits(_ w: [UInt32], _ width: [UInt8], _ d: UnsafeMutablePointer<Int32>) {
  var acc: UInt64 = 0, bits = 0, wi = 0
  var c: Int64 = 0
  for j in 0..<width.count {
    let wd = Int(width[j])
    while bits < wd { acc |= UInt64(wi < w.count ? w[wi] : 0) << UInt64(bits); wi += 1; bits += 32 }
    var t = Int64(acc & ((1 << UInt64(wd)) - 1)) + c
    acc >>= UInt64(wd); bits -= wd
    c = 0
    if t >= 1 << Int64(wd - 1) { t -= 1 << Int64(wd); c = 1 }
    d[j] = Int32(t)
  }
  d[0] += Int32(c)
}

// ---- which iterations to keep ----
// the same points as gpuowl: spans E/2, E/4, ... rounded up, and E itself in place of 0.
func proofPoints(_ E: Int, _ power: Int) -> [Int] {
  var pts = [0], span = (E + 1) / 2
  for _ in 0..<power { pts += pts.map { $0 + span }; span = (span + 1) / 2 }
  pts[0] = E
  return pts.sorted()
}

// ---- files ----
func hash64(_ data: Data) -> UInt64 {
  var h: UInt64 = 0xcbf29ce484222325
  data.withUnsafeBytes { raw in
    let n = raw.count / 8
    for i in 0..<n { h = (h ^ raw.loadUnaligned(fromByteOffset: 8 * i, as: UInt64.self)) &* 0x100000001b3; h ^= h >> 29 }
    for b in raw.suffix(raw.count % 8) { h = (h ^ UInt64(b)) &* 0x100000001b3 }
  }
  return h
}

// a kept residue: its bytes, then a 64-bit hash of them
func writeResidue(_ path: String, _ E: Int, _ w: [UInt32]) throws {
  var data = w.withUnsafeBytes { Data($0.prefix(residueBytes(E))) }
  var h = hash64(data)
  data.append(Data(bytes: &h, count: 8))
  try data.write(to: URL(fileURLWithPath: path + ".tmp"))
  _ = try? FileManager.default.removeItem(atPath: path)
  try FileManager.default.moveItem(atPath: path + ".tmp", toPath: path)
}
func readResidue(_ path: String, _ E: Int) -> [UInt32]? {
  guard let data = FileManager.default.contents(atPath: path), data.count == residueBytes(E) + 8 else { return nil }
  let body = data.prefix(residueBytes(E))
  guard hash64(Data(body)) == data.suffix(8).withUnsafeBytes({ $0.loadUnaligned(as: UInt64.self) }) else { return nil }
  var w = [UInt32](repeating: 0, count: (E + 31) / 32)
  w.withUnsafeMutableBytes { $0.copyBytes(from: body) }
  return w
}

func writeProof(_ path: String, _ E: Int, _ B: [UInt32], _ middles: [[UInt32]]) throws {
  var data = Data("PRP PROOF\nVERSION=2\nHASHSIZE=64\nPOWER=\(middles.count)\nNUMBER=M\(E)\n".utf8)
  for w in [B] + middles { w.withUnsafeBytes { data.append(Data($0.prefix(residueBytes(E)))) } }
  try data.write(to: URL(fileURLWithPath: path))
}
func readProof(_ path: String) -> (E: Int, B: [UInt32], middles: [[UInt32]])? {
  guard let data = FileManager.default.contents(atPath: path) else { return nil }
  var lines: [String] = [], off = 0
  while lines.count < 5, let nl = data[off...].firstIndex(of: 10) {
    lines.append(String(decoding: data[off..<nl], as: UTF8.self)); off = nl + 1
  }
  guard lines.count == 5, lines[0] == "PRP PROOF", lines[1] == "VERSION=2", lines[2] == "HASHSIZE=64",
        lines[3].hasPrefix("POWER="), let power = Int(lines[3].dropFirst(6)),
        lines[4].hasPrefix("NUMBER=M"), let E = Int(lines[4].dropFirst(8)) else { return nil }
  let n = residueBytes(E)
  guard data.count - off == n * (power + 1) else { return nil }
  let xs: [[UInt32]] = (0...power).map { i in
    var w = [UInt32](repeating: 0, count: (E + 31) / 32)
    w.withUnsafeMutableBytes { $0.copyBytes(from: data[(off + i * n)..<(off + (i + 1) * n)]) }
    return w
  }
  return (E, xs[0], Array(xs[1...]))
}
