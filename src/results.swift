// results for primenet: the json line, the worktodo file.
//
// the format is the one prime95, gpuowl and prmers write to results.txt, so autoprimenet
// can pick it up. the residue is type 1, 3^(M_p - 1) = 3^(2^p - 2), which is 1 for a prime.
// barnarg's x_p is 3^(2^p), so type 1 is x_p / 9.
import CryptoKit
import Foundation

let barnargVersion = "3.7"

// v / 9 mod 2^p - 1, v given as ceil(p/32) words in [0, 2^p - 2]
func divideBy9(_ v: [UInt32], _ p: Int) -> [UInt32] {
  let n = v.count
  func mod9(_ w: [UInt32]) -> UInt64 { w.reversed().reduce(UInt64(0)) { ($0 << 32 | UInt64($1)) % 9 } }
  var mp = [UInt32](repeating: 0xffffffff, count: n)
  if p % 32 != 0 { mp[n - 1] = (1 << UInt32(p % 32)) - 1 }
  let vm = mod9(v), mm = mod9(mp)
  let k = (0..<9).first { (vm + UInt64($0) * mm) % 9 == 0 }!   // 9 doesn't divide 2^p - 1 for odd p
  var t = [UInt32](repeating: 0, count: n + 1), c: UInt64 = 0   // t = v + k * mp
  for i in 0..<n { let s = UInt64(v[i]) + UInt64(k) * UInt64(mp[i]) + c; t[i] = UInt32(truncatingIfNeeded: s); c = s >> 32 }
  t[n] = UInt32(c)
  var r: UInt64 = 0
  for i in stride(from: n, through: 0, by: -1) { let x = r << 32 | UInt64(t[i]); t[i] = UInt32(x / 9); r = x % 9 }
  precondition(r == 0 && t[n] == 0)
  return Array(t[0..<n])
}

func hexWords(_ w: [UInt32], bits: Int) -> String {   // low `bits` bits as hex, top first
  (0..<(bits / 32)).reversed().map { String(format: "%08x", $0 < w.count ? w[$0] : 0) }.joined()
}

func crc32(_ s: String) -> UInt32 {
  var c: UInt32 = 0xffffffff
  for b in s.utf8 {
    c ^= UInt32(b)
    for _ in 0..<8 { c = (c >> 1) ^ (0xedb88320 & (0 &- (c & 1))) }
  }
  return ~c
}

func md5Hex(_ path: String) -> String? {
  guard let d = FileManager.default.contents(atPath: path) else { return nil }
  return Insecure.MD5.hash(data: d).map { String(format: "%02x", $0) }.joined()
}

// one line for results.txt. proof is (power, md5) or nil.
func resultJSON(p: Int, residue: [UInt32], N: Int, gerbiczErrors: Int, proof: (Int, String)?, aid: String?) -> (String, Bool) {
  let r = divideBy9(residue, p)
  let isPrime = r[0] == 1 && r.dropFirst().allSatisfy { $0 == 0 }
  let res64 = hexWords(r, bits: 64).uppercased(), res2048 = hexWords(r, bits: 2048)
  let f = DateFormatter()
  f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd HH:mm:ss"
  let ts = f.string(from: Date())
  var j = "{\"status\":\"\(isPrime ? "P" : "C")\",\"exponent\":\(p),\"worktype\":\"PRP-3\""
  j += ",\"res64\":\"\(res64)\",\"res2048\":\"\(res2048)\",\"residue-type\":1"
  j += ",\"errors\":{\"gerbicz\":\(gerbiczErrors)},\"shift-count\":0,\"fft-length\":\(N)"
  if let (power, md5) = proof { j += ",\"proof\":{\"version\":2,\"power\":\(power),\"hashsize\":64,\"md5\":\"\(md5)\"}" }
  j += ",\"program\":{\"name\":\"barnarg\",\"version\":\"\(barnargVersion)\"},\"os\":{\"os\":\"macOS\",\"architecture\":\"arm64\"}"
  if let aid = aid { j += ",\"aid\":\"\(aid)\"" }
  j += ",\"timestamp\":\"\(ts)\""
  // checksum: crc32 of these fields joined by ';', the same as prime95 and prmers
  let canon = ["\(p)", "PRP", "", "", res64.lowercased(), res2048, "0_3_1", "\(N)", "gerbicz:\(gerbiczErrors)",
               "barnarg", barnargVersion, "", "", "macOS", "arm64", ts].joined(separator: ";")
  j += String(format: ",\"checksum\":{\"version\":1,\"checksum\":\"%08X\"}}", crc32(canon))
  return (j, isPrime)
}

func appendLine(_ path: String, _ line: String) {
  let data = Data((line + "\n").utf8)
  if let fh = FileHandle(forWritingAtPath: path) { fh.seekToEndOfFile(); fh.write(data); fh.closeFile() }
  else { FileManager.default.createFile(atPath: path, contents: data) }
}

// ---- worktodo.txt ----
// PRP=AID,k,b,n,c[,how far factored,tests saved[,base,residue type]][,"known factors"]
// (and PRPDC= for a double check). barnarg does k = 1, b = 2, c = -1, base 3, no known factors.
// Test=AID,p,how far factored,p-1 done (and DoubleCheck=) is a lucas-lehmer test.
struct Assignment { var line: String; var aid: String?; var p: Int; var ll = false }

func nextAssignment(_ path: String) -> Assignment? {
  guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
  for raw in text.split(whereSeparator: \.isNewline) {
    let line = String(raw).trimmingCharacters(in: .whitespaces)
    let ll = line.hasPrefix("Test=") || line.hasPrefix("DoubleCheck=")
    guard ll || line.hasPrefix("PRP=") || line.hasPrefix("PRPDC=") else { continue }
    var f = line.split(separator: "=", maxSplits: 1)[1].split(separator: ",", omittingEmptySubsequences: false).map(String.init)
    var aid: String? = nil
    if let first = f.first, first == "N/A" || first == "0" && f.count > 1 && ll || (first.count == 32 && first.allSatisfy(\.isHexDigit)) {
      aid = first.count == 32 ? first : nil; f.removeFirst()
    }
    if ll {
      guard let p = f.first.flatMap({ Int($0) }), p > 2 else { print("skipping \(line): no exponent"); continue }
      return Assignment(line: line, aid: aid, p: p, ll: true)
    }
    guard f.count >= 4, f[0] == "1", f[1] == "2", let p = Int(f[2]), f[3] == "-1" else {
      print("skipping \(line): barnarg only does 2^p - 1"); continue
    }
    if f.contains(where: { $0.hasPrefix("\"") }) { print("skipping \(line): known factors (cofactor tests) aren't supported"); continue }
    if f.count >= 8, f[6] != "3" || f[7] != "1" { print("skipping \(line): only base 3, residue type 1"); continue }
    return Assignment(line: line, aid: aid, p: p)
  }
  return nil
}

func removeLine(_ path: String, _ line: String) {
  guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return }
  var lines = text.components(separatedBy: "\n")
  if let i = lines.firstIndex(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines) == line }) { lines.remove(at: i) }
  let tmp = path + ".tmp"
  do {
    try lines.joined(separator: "\n").write(toFile: tmp, atomically: false, encoding: .utf8)
    _ = try FileManager.default.replaceItemAt(URL(fileURLWithPath: path), withItemAt: URL(fileURLWithPath: tmp))
  } catch { print("could not update \(path): \(error)") }
}

// smallest fft that keeps the fp32 side well inside its limit. the model in
// research/fp32_model.py gives about 18.35 bits per word at 4M and 0.28 less per doubling;
// this stays about 0.35 under that.
func chooseN(_ p: Int) -> Int {
  var logN = 6
  while Double(p) / Double(1 << logN) > 18.0 - 0.25 * Double(logN - 22) { logN += 1 }
  return 1 << logN
}
