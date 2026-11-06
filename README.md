# barnarg

a mersenne prp tester for the gpu in apple silicon macs, written in metal.

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

works, not fast yet. every result so far matches gmp.

| test | fft | result |
|---|---|---|
| M86243, full prp | 4096 | prp (a known mersenne prime) |
| M86249, full prp | 4096 | not prp, res64 `062d6633d5052b5f` |
| M216091, full prp | 16384 | prp (a known mersenne prime) |
| M136279841, 60 squarings | 8M | matches gmp, round-off 0.012 |

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

over an 80 second run (5500 squarings) barnarg holds 14.5 ms, about 69 iterations per second. prmers averages about 68 over the same length of run, and that includes its gerbicz checks. so they're level on this air, and barnarg doesn't have the checks yet.

the 8th roots trick: the host picks the roots of unity so that W_8 = 2^15 (1 + i). that works because (2^15 (1 + i))^2 = 2^31 i = i. multiplying by any 8th root is then an add, a subtract and a 15-bit rotation, so radix-8 butterflies on the M31 side need no multiplies at all.

the biggest single jump (24.5 to 16 ms) came from writing the 16 point dft out by hand. the generic loop version wasn't being fully unrolled, so its arrays ended up in slow memory. as 4x4, the 4 point dfts need no multiplies (W_4 = ±i) and only 3 of the 9 twiddles between them aren't 8th roots.

the middle kernel works because of where things land. with the plan 256 x 256 x 64, the last forward pass leaves instance j holding Z[j + 65536 m]. the tail pairs k with M - k, which is in instance 65536 - j. so one threadgroup takes 4 instances and their 4 partners, finishes the forward dft, squares, and then runs the first inverse pass too, since an inverse plan that starts with 64 reads exactly those same 64 values. j = 0 and j = 32768 are their own partners.

where the time goes now (`--profile`, 13.3 ms total):

- 4 ordinary passes each way, 0.74 to 1.07 ms each.
- the middle kernel, 1.25 ms fp32 and 2.05 ms M31. the M31 one is limited by multiplies: the tail does 4M x 5 complex products, which matches the measured 72.7 billion modular multiplies a second.
- the carry, 2.0 ms.

next:

- fold the carry into the last inverse pass.
- a cheaper M31 multiply (it's about 10 integer ops now, not just the 2 multiplies).
- gerbicz error checks, prp proofs, save files and autoprimenet.

## build and run

needs xcode (for swiftc and metal) and, for the checks, python with numpy and gmpy2.

```sh
make
./build/barnarg 86243 4096 86243 build/m86243.bin 1000,86243
python3 test/check.py build/m86243.bin 86243
# ok  k=   1000 value matches GMP  res64=e52db002b73401a8
# ok  k=  86243 value matches GMP  res64=0000000000000009  PRP=yes
```

arguments are `p N iterations [dumpfile k1,k2,...]`. the dump has the digits at each checkpoint, which `test/check.py` compares against gmp (and against the cpu reference with `--ref K`). `--profile` times each kernel on its own.

`make check` runs a small set of prps and compares them.

## files

- `src/barnarg.metal`: the gpu kernels.
- `src/main.swift`: host side, tables and the iteration loop.
- `ref/barnarg_ref.py`: the same algorithm in numpy, slow but simple.
- `ref/ref_vectors.json`: gmp residues for 19 mersenne primes and 10 composites.
- `test/`: checkers for gpu dumps.
- `research/`: the bandwidth benchmark, the threadgroup handoff test and the fp32 error model.
