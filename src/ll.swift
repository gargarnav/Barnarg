// lucas-lehmer: s_0 = 4, s_(k+1) = s_k^2 - 2, and M_p is prime iff s_(p-2) = 0.
//
// there is no gerbicz check for ll. the error check is the jacobi symbol: (s_k - 2 / M_p) = -1
// for every k >= 1, prime or not, so a wrong s_k fails it about half the time. that is what
// prime95 and gpuowl check too. the symbol is computed with gmp, loaded at run time so the prp
// side doesn't need it (brew install gmp).
import Foundation

// an mpz_t is 16 bytes (int alloc, int size, limb pointer); it is handled here as raw memory
final class GMP {
  typealias Z = UnsafeMutableRawPointer
  typealias Init = @convention(c) (Z) -> Void
  typealias Import = @convention(c) (Z, Int, Int32, Int, Int32, Int, UnsafeRawPointer) -> Void
  typealias Jacobi = @convention(c) (Z, Z) -> Int32
  typealias SetBit = @convention(c) (Z, UInt) -> Void
  typealias SubUI = @convention(c) (Z, Z, UInt) -> Void
  typealias Add = @convention(c) (Z, Z, Z) -> Void
  let initz: Init, clear: Init, importz: Import, jacobi: Jacobi, setbit: SetBit, subui: SubUI, add: Add

  init?() {
    let names = ["/opt/homebrew/lib/libgmp.10.dylib", "/usr/local/lib/libgmp.10.dylib", "/opt/local/lib/libgmp.10.dylib", "libgmp.10.dylib"]
    guard let h = names.lazy.compactMap({ dlopen($0, RTLD_NOW) }).first else { return nil }
    func f<T>(_ n: String, _ t: T.Type) -> T? { dlsym(h, n).map { unsafeBitCast($0, to: t) } }
    guard let a = f("__gmpz_init", Init.self), let b = f("__gmpz_clear", Init.self), let c = f("__gmpz_import", Import.self),
          let d = f("__gmpz_jacobi", Jacobi.self), let e = f("__gmpz_setbit", SetBit.self),
          let g = f("__gmpz_sub_ui", SubUI.self), let k = f("__gmpz_add", Add.self) else { return nil }
    initz = a; clear = b; importz = c; jacobi = d; setbit = e; subui = g; add = k
  }

  // (x - 2 / 2^p - 1), x given as words in [0, 2^p - 2]
  func llJacobi(_ x: [UInt32], _ p: Int) -> Int {
    let m = Z.allocate(byteCount: 16, alignment: 8), a = Z.allocate(byteCount: 16, alignment: 8)
    initz(m); initz(a)
    defer { clear(m); clear(a); m.deallocate(); a.deallocate() }
    setbit(m, UInt(p)); subui(m, m, 1)
    x.withUnsafeBytes { importz(a, x.count, -1, 4, 0, 0, $0.baseAddress!) }
    subui(a, a, 2)
    if a.load(fromByteOffset: 4, as: Int32.self) < 0 { add(a, a, m) }   // the size field is negative for a < 0
    return Int(jacobi(a, m))
  }
}

// one line for results.txt, the fields prime95 and gpuowl write for ll. s is s_(p-2).
func llResultJSON(p: Int, s: [UInt32], N: Int, jacobiErrors: Int, aid: String?) -> (String, Bool) {
  let isPrime = s.allSatisfy { $0 == 0 }
  let res64 = hexWords(s, bits: 64).uppercased()
  let f = DateFormatter()
  f.locale = Locale(identifier: "en_US_POSIX"); f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd HH:mm:ss"
  let ts = f.string(from: Date())
  var j = "{\"status\":\"\(isPrime ? "P" : "C")\",\"exponent\":\(p),\"worktype\":\"LL\",\"res64\":\"\(res64)\""
  j += ",\"errors\":{\"jacobi\":\(jacobiErrors)},\"shift-count\":0,\"fft-length\":\(N)"
  j += ",\"program\":{\"name\":\"barnarg\",\"version\":\"\(barnargVersion)\"},\"os\":{\"os\":\"macOS\",\"architecture\":\"arm64\"}"
  if let aid = aid { j += ",\"aid\":\"\(aid)\"" }
  j += ",\"timestamp\":\"\(ts)\""
  // same checksum layout as the prp line, with the ll fields (res64, no res2048, shift count)
  let canon = ["\(p)", "LL", "", "", res64.lowercased(), "", "0", "\(N)", "jacobi:\(jacobiErrors)",
               "barnarg", barnargVersion, "", "", "macOS", "arm64", ts].joined(separator: ";")
  j += String(format: ",\"checksum\":{\"version\":1,\"checksum\":\"%08X\"}}", crc32(canon))
  return (j, isPrime)
}
