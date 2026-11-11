# barnarg

a mersenne prime tester (prp and lucas-lehmer) for the gpu in apple silicon macs, written in metal.

the name is a cross of two old english apples, barnack beauty and margil (b**arn**ack, m**arg**il).

## why

apple gpus have no fp64 at all, so the usual double precision fft that prime95 and gpuowl use can't run on them. the only gpu option on a mac right now is [prmers](https://github.com/cherubrock-seb/PrMers), which goes through apple's old opencl 1.2 and does 64-bit integer maths that the gpu has to emulate. on my m4 air it does M136279841 at about 70 iterations per second, so one wavefront test takes about 22 days.

barnarg only uses 32-bit arithmetic, which is what the hardware is good at.

## how it works

each squaring of x mod 2^p - 1 is done twice, in two different number systems, and the results are glued together:

- an fp32 fft gives every output word approximately.
- an ntt over GF(M31^2), where M31 = 2^31 - 1, gives every output word exactly, mod M31.
- if the fp32 error is under M31/2, the two together pin down the exact word (crt).

both use the irrational base weighting from crandall and fagin, so there's no zero padding. a few things make M31 a nice fit:

- it's the only 31-bit prime where 2 has N-th roots for every power-of-2 N (2 has odd order, 31), so the mersenne weights exist. babybear and koalabear don't work.
- the N-th root of 2 is itself a power of 2 (2^u with u = N^-1 mod 31), so applying a weight is a 31-bit rotation, not a multiply.
- every root of unity of order up to 2^31 has norm 1, so conj(w) = 1/w, same as complex numbers. that lets the real input be packed into N/2 complex values and split back after the transform, exactly.

at an 8M fft the fp32 side carries about 18.4 bits per word, the same as fp64 on an nvidia card, with 8 bytes per word.

## status

works, and about 1.2 times faster than prmers on an m4 air. every result so far matches gmp.

| test | fft | result |
|---|---|---|
| M86243, full prp | 4096 | prp (a known mersenne prime) |
| M86249, full prp | 4096 | not prp, res64 `062d6633d5052b5f` (of 3^(2^p); the type 1 residue in results.txt is `56050b5b17ab3db5`) |
| M216091, full prp | 16384 | prp (a known mersenne prime) |
| M136279841, 60 squarings | 8M | matches gmp, round-off 0.012 |
| M86243, M110503, M132049, M216091, full ll | 4096 to 16384 | prime (known mersenne primes) |
| M86249, M132059, full ll | 8192 | composite, res64 matches gmp |
| M136279841, 40 ll iterations | 8M | matches gmp |

speed at M136279841 on an m4 macbook air (10 core gpu). whole numbers are big jumps (about 1.4x or more), decimals are smaller steps:

| version | change | ms/iter |
|---|---|---|
| 0 | radix-2 stages, 92 passes over memory per squaring | 232 |
| 1 | radix-16 passes in registers | 54 |
| 2 | real input packed as N/2 complex | 35 |
| 2.1 | carry through threadgroup memory, 3-multiply GF(M31^2) product | 29 |
| 2.2 | radix-256 and radix-64 passes through threadgroup memory (3 passes per transform instead of 6), 8th roots of unity as rotations | 24.5 |
| 3 | hand-written 16 and 8 point dfts (4x4 and 4x2), weighting folded into the first pass | 16.0 |
| 3.1 | last forward pass, tail and first inverse pass in one kernel; 2-multiply squaring in GF(M31^2) | 13.3 |
| 3.2 | gerbicz-li checks (about 0.4% slower with the default settings) | 13.3 |
| 3.3 | save files, so a long test can stop and resume | 13.3 |
| 3.4 | prp proofs, the same files gpuowl and prmers make | 13.3 |
| 3.5 | cheaper M31 add, subtract and multiply (one `min` instead of a full reduction) | 11.2 |
| 3.6 | worktodo.txt and json results, for autoprimenet | 11.2 |
| 3.7 | lucas-lehmer tests with jacobi checks | 11.2 |

the air has no fan, so a long run is slower than these numbers. over 5000 squarings (about a minute, starting cool) 3.4 held 13.8 to 13.9 ms and 3.5 held 12.1 to 12.4 ms, about 81 iterations per second (two runs each, alternating, 2 minutes apart). prmers averaged about 68 over a run that long, including its gerbicz checks, so barnarg is now about 1.2 times faster than it on this mac. the checks cost barnarg about 0.4%.

the M31 change: every value stays in [0, P), so a + b and a - b + P are both under 2P, and one conditional subtract is enough. `min(x, x - P)` does that in 2 instructions, since x - P wraps to something huge when x < P. the multiply works the same way if you multiply by 2b instead of b: a * 2b = hi * 2^32 + lo, so a * b = hi + lo / 2 mod P, and both halves are already under P. that's 7 instructions instead of 12. on its own the cheaper multiply was only 12% faster in `research/mulbench.swift`, because the adds and subtracts around it cost as much. with both, a GF(M31^2) product went from 30 to 47 billion a second.

the 8th roots trick: the host picks the roots of unity so that W_8 = 2^15 (1 + i). that works because (2^15 (1 + i))^2 = 2^31 i = i. multiplying by any 8th root is then an add, a subtract and a 15-bit rotation, so radix-8 butterflies on the M31 side need no multiplies at all.

the biggest single jump (24.5 to 16 ms) came from writing the 16 point dft out by hand. the generic loop version wasn't being fully unrolled, so its arrays ended up in slow memory. as 4x4, the 4 point dfts need no multiplies (W_4 = ±i) and only 3 of the 9 twiddles between them aren't 8th roots.

the middle kernel works because of where things land. with the plan 256 x 256 x 64, the last forward pass leaves instance j holding Z[j + 65536 m]. the tail pairs k with M - k, which is in instance 65536 - j. so one threadgroup takes 4 instances and their 4 partners, finishes the forward dft, squares, and then runs the first inverse pass too, since an inverse plan that starts with 64 reads exactly those same 64 values. j = 0 and j = 32768 are their own partners.

where the time goes now (`--profile`, 11.2 ms total):

- 4 ordinary passes each way, 0.75 to 1.07 ms each. the M31 ones now move 84 to 89 gb/s, against 98 for a plain copy, so they're close to memory bound.
- the middle kernel, 1.20 ms fp32 and 1.32 ms M31 (2.05 before 3.5).
- the carry, 2.0 ms.

i tried folding the carry into the last inverse pass: the M31 kernel did its last 256 point pass, read the fp32 result, did the crt and carried each 16 digit block in threadgroup memory. it was correct but not faster. 13.7 ms against 13.2 without it, and 14.3 when i also staged the digit stores. my guess is the extra threadgroup memory leaves too few groups per core to hide the memory latency. the separate carry kernel stays for now.

a second try (branch `fused-carry`) did the last two inverse passes, the crt and the carry in one kernel, and added each block's carry-out in the next squaring's first pass. that one was correct and took 11.03 ms against 11.2. the crt is mostly 64-bit integer work, which apple gpus don't have in hardware, and in the separate kernel it was hidden behind memory waits anyway. 1.5% wasn't worth the extra code, so it stays on the branch.

## gerbicz checks

a hardware error in a week-long test wastes the whole week, so `--gerbicz` checks the work as it goes. x_k = 3^(2^k), and every L squarings barnarg multiplies a running product d by x:

```
d_t = x_0 * x_L * x_2L * ... * x_tL      so      d_t = 3 * d_(t-1)^(2^L)
```

every C blocks it squares the old d L more times, multiplies by 3 and compares. an error anywhere since the last check breaks the equation, and barnarg goes back to the last state that passed. the last ITERS mod L squarings are just done twice and compared.

defaults are L = 1000 and C = 400, so a check every 400,000 squarings (about 1.6 hours here), plus one multiply per 1000 squarings. `--inject K` adds 1 to a digit at iteration K, to see it work:

```sh
./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --inject 30000
# injected an error into x at iteration 30000
# k=30000  gerbicz check FAILED, going back to k=25000
# ...
# p=86243 N=4096 iters=86243  res64=0000000000000009  PRP  checks=19 failures=1
```

## save files

every 5 minutes a full run prints a progress line (`--progress-every`, in seconds, 0 for none) with the speed since the last line and an eta from it:

```
k=1000  25.00%  13.75 ms/iter  eta 0h 00m 41s
```

a checked run saves to `barnarg-<p>.save` every 10 minutes (`--save-every`, in seconds), after every check that passes, and when you press ctrl-c (it finishes the block it's on first). running the same command again picks up where it left off:

```sh
./build/barnarg 136279841 8388608 136279841 --gerbicz
# ... ctrl-c
# stopped at k=2000000, saved to barnarg-136279841.save
./build/barnarg 136279841 8388608 136279841 --gerbicz
# resuming from barnarg-136279841.save at k=2000000 (last verified k=1600000)
```

the file holds the current x and d and also the last verified pair, so an error that got in before a save still gets caught by the next check and rolled back. it ends with a 64-bit hash and is written to a `.tmp` file first and then renamed, so a crash mid-save can't leave a half-written file. a damaged file, or one for a different p or N, is ignored. at 8M words it's 134 mb. when a full test finishes, barnarg writes the result to `results.txt` (below) and deletes the save file.

the multiply (x times d, not a square) goes through the plain passes and a two-input version of the tail kernel, since it only runs once per 1000 squarings.

## prp proofs

gimps wants a proof with every first-time prp test, so the server can check the result without anyone running the whole test again. it's pietrzak's verifiable delay function: the run keeps x_k at 2^power points, and afterwards they get folded into `power` middle values, with exponents taken from a sha3-256 hash chain. checking the proof then takes about p / 2^power squarings instead of p.

a full checked run (`--gerbicz` with ITERS = p) does this by default with power 8. the residues go into `barnarg-<p>-proof/` as they're reached. if a gerbicz check fails later, the rollback goes back before them and they get written again. at the end barnarg builds the proof, checks it itself and writes `proof/<p>-<power>.proof`:

```sh
./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 5
# proof level 0: middle res64 adabb277f01dcae0, h 56d222f32855a5d7
# ...
# checking the proof: 2696 squarings
# wrote proof/86243-5.proof and checked it (0.6 s)
./build/barnarg verify proof/86243-5.proof 4096
# proof/86243-5.proof: M86243 power 5, res64 0000000000000009, proof ok (0.3 s)
```

the file is gpuowl's version 2 format, the one primenet takes. i used the same points and the same folding order as gpuowl, so the file depends only on the number, not on the program. for M86243 at power 5, barnarg's proof is byte for byte the same as the one prmers writes (md5 `60b69c35a3d3403c2173ba598ba0f243`), and `make check` tests for that. `ref/proof_ref.py` does the same thing in gmpy2. it gave identical files for M86243, M86249 and M216091 (that one stopped with ctrl-c halfway and resumed), and it verified the M1257787 power 8 proof.

at M136279841 power 8 keeps 256 residues of 17 mb, about 4.4 gb of disk until the proof is written. building the proof takes roughly 40,000 squarings' worth of work, and checking it 532,000 squarings (about 2 hours here), about 1% on top of the 19 days a test now takes. `--proof N` changes the power, `--proof 0` turns it off.

## lucas-lehmer

s_0 = 4, s_(k+1) = s_k^2 - 2, and 2^p - 1 is prime exactly when s_(p-2) = 0. the squaring is the same as for prp; the - 2 goes into the carry kernel, as an extra carry into digit 0. ll is mostly used for double checks of old ll results now, since new tests are prp with proofs.

there's no gerbicz check for ll. the check is the jacobi symbol: (s_k - 2 / 2^p - 1) = -1 for every k >= 1, prime or not. a wrong s_k fails it about half the time, so it catches half the errors, same as in prime95 and gpuowl. barnarg checks every 30 minutes (`--jacobi-every`, in seconds) and at the end, and goes back to the last s_k that passed. at M136279841 one check takes about 20 s with gmp, under 1.2% of the run.

```sh
./build/barnarg 86249 8192 86247 --ll --inject 20000 --jacobi-every 0
# injected an error into s at iteration 20000
# k=20000  jacobi check FAILED, going back to k=19000
# ...
# p=86249 N=8192 LL  res64=422c56c4f9e3f2e3  composite  checks=88 failures=1
```

the save file is `barnarg-<p>.ll.save`, with the current s and the last one that passed. the result line has `"worktype":"LL"`, the res64 of s_(p-2), `"errors":{"jacobi":n}` and `"shift-count":0`.

## gimps work

run `barnarg` with no exponent and it works through `worktodo.txt` in the current folder, the same file prime95 and the gpu programs use. it takes the first `PRP=` (or `PRPDC=`) line for 2^p - 1, or `Test=` / `DoubleCheck=` for a lucas-lehmer test, picks the fft size, runs a checked prp with a proof (or a checked ll), writes the result and takes the line out. then it starts again on the next line. ctrl-c stops it, and running `barnarg` again resumes from the save file.

```sh
cat worktodo.txt
# PRP=0123456789ABCDEF0123456789ABCDEF,1,2,86249,-1,75,0,3,1
barnarg --proof 4
# M86249 from worktodo.txt, assignment 0123456789ABCDEF0123456789ABCDEF, fft 8192
# ...
# wrote proof/86249-4.proof and checked it (0.7 s)
# M86249 is composite, result written to results.txt
```

the result is one json line in `results.txt`, in the format prime95, gpuowl and prmers use:

```
{"status":"C","exponent":86249,"worktype":"PRP-3","res64":"56050B5B17AB3DB5","res2048":"6309...3db5","residue-type":1,
 "errors":{"gerbicz":0},"shift-count":0,"fft-length":8192,"proof":{"version":2,"power":4,"hashsize":64,"md5":"277e80b2..."},
 "program":{"name":"barnarg","version":"3.7"},...,"checksum":{"version":1,"checksum":"6D2C9BBF"}}
```

the residue is type 1, 3^(2^p - 2), so a prime gives 1. barnarg computes 3^(2^p), so it divides by 9 first (exactly, on the cpu). for M86249 the res64, res2048 and proof md5 are the same as prmers gives. the checksum is a crc32 over the fields joined by `;`, and the same code gives prmers's checksum for its own line.

the fft size is the smallest power of 2 under about 18 bits per word at 4M, less 0.25 per doubling, which leaves some room under the limit from `research/fp32_model.py`. `--fft N` overrides it. M136279841 gets 8M.

autoprimenet (by teal dulcet and daniel connelly) fetches the assignments, reports the results and uploads the proofs. it doesn't know barnarg, but barnarg uses the same files as prmers (`worktodo.txt`, `results.txt`, `proof/`), so its `--prmers` mode should work, except for the progress reports, which read prmers's own save files. i haven't run it against primenet yet. george woltman said on 8 nov 2026 that proof files that verify are enough for him, and james heinrich checked a sample prp line and ll line the same day: both looked right and both checksums validated.

skipped for now: known-factor (cofactor) tests, other bases and residue types, and certification work.

next:

- run a real assignment through autoprimenet.

## download

the [releases page](https://github.com/gargarnav/Barnarg/releases) has a zip with the binary (apple silicon, macos 13 or later). the gpu kernels are inside the binary, so it's the only file you need. it isn't signed by apple, so macos blocks it the first time if it came through a browser. this takes the block off:

```sh
unzip barnarg-3.7-macos-arm64.zip && cd barnarg-3.7
xattr -d com.apple.quarantine barnarg
./barnarg 86243 4096 86243 --gerbicz --proof 0     # about 10 s: "M86243 is a probable prime"
```

lucas-lehmer tests also need gmp (`brew install gmp`).

## build and run

needs an apple silicon mac (m1 or later) and swiftc, which comes with xcode or just the command line tools (`xcode-select --install`). the kernels are compiled by metal at run time, so there's no separate shader build. lucas-lehmer tests also need gmp (`brew install gmp`) for the jacobi check; prp doesn't. the tests need python with numpy and gmpy2 (`pip install -r requirements.txt`).

i've only run it on an m4 macbook air so far, so results from other chips would be useful.

```sh
git clone https://github.com/gargarnav/Barnarg.git && cd Barnarg
make
make install            # copies build/barnarg to /usr/local/bin, or PREFIX=~/.local
make dist               # the release zip, in build/dist/
./build/barnarg 86243 4096 86243 build/m86243.bin 1000,86243
python3 test/check.py build/m86243.bin 86243
# ok  k=   1000 value matches GMP  res64=e52db002b73401a8
# ok  k=  86243 value matches GMP  res64=0000000000000009  PRP=yes
```

arguments are `p N iterations [dumpfile k1,k2,...]`, plus `--gerbicz` for a checked run. the dump has the digits at each checkpoint, which `test/check.py` compares against gmp (and against the cpu reference with `--ref K`). `--profile` times each kernel on its own.

`make check` runs a small set of prps, a proof, three lls and two injected errors, and compares them with gmp and with prmers's proof file. the gpu kernels are built into the binary, so after `make install` it runs from any folder (`--kernels FILE` compiles a different .metal file instead).

## files

- `src/barnarg.metal`: the gpu kernels.
- `src/main.swift`: host side, tables and the iteration loop.
- `src/proof.swift`: sha3, proof points, residue and proof files.
- `src/results.swift`: the json results line, worktodo.txt, fft size.
- `src/ll.swift`: the jacobi check (through gmp) and the ll results line.
- `ref/proof_ref.py`: proofs in gmpy2 (`make` and `verify`).
- `ref/barnarg_ref.py`: the same algorithm in numpy, slow but simple.
- `ref/ref_vectors.json`: gmp residues for 19 mersenne primes and 10 composites.
- `test/`: checkers for gpu dumps.
- `research/`: the bandwidth benchmark, the threadgroup handoff test and the fp32 error model.

## license

gpl-3.0, see `LICENSE`. copyright 2026 arnav garg.
