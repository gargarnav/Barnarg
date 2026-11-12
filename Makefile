PREFIX ?= /usr/local
SRC = src/main.swift src/proof.swift src/results.swift src/ll.swift src/logo.swift

build/barnarg: $(SRC) build/kernels.swift
	swiftc -O -target arm64-apple-macos13.0 $(SRC) build/kernels.swift -o build/barnarg

# the metal source goes into the binary as a string, so build/barnarg works from any folder
build/kernels.swift: src/barnarg.metal
	mkdir -p build
	{ printf 'let embeddedKernels = #"""\n'; cat src/barnarg.metal; printf '\n"""#\n'; } > $@

# the zip for a github release: the binary, readme and license
VERSION = $(shell sed -n 's/.*barnargVersion = "\(.*\)".*/\1/p' src/results.swift)
dist: build/barnarg
	rm -rf build/dist && mkdir -p build/dist/barnarg-$(VERSION)
	cp build/barnarg README.md LICENSE build/dist/barnarg-$(VERSION)/
	cd build/dist && zip -qr barnarg-$(VERSION)-macos-arm64.zip barnarg-$(VERSION)
	shasum -a 256 build/dist/barnarg-$(VERSION)-macos-arm64.zip

install: build/barnarg
	mkdir -p $(PREFIX)/bin
	install -m 755 build/barnarg $(PREFIX)/bin/barnarg

# make check starts barnarg a dozen times, so the logo goes once at the top and not after every line
check: export BARNARG_NOLOGO = 1
check: build/barnarg
	@./build/barnarg logo
	./build/barnarg 521 64 521 build/m521.bin 1,2,10,521
	python3 test/check.py build/m521.bin 521 --ref 10
	./build/barnarg 86243 4096 1000 build/m86243.bin 1,2,20,1000
	python3 test/check.py build/m86243.bin 86243 --ref 20
	./build/barnarg 216091 16384 1000 build/m216091.bin 1,20,1000
	python3 test/check.py build/m216091.bin 216091 --ref 20
	./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 0 | tail -1
	./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 0 --inject 30000 | grep -E "FAILED|PRP"
	cd build && ./barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 5 --nosave | grep -E "PRP|wrote"
	test "$$(md5 -q build/proof/86243-5.proof)" = 60b69c35a3d3403c2173ba598ba0f243
	./build/barnarg verify build/proof/86243-5.proof 4096
	./build/barnarg 86243 4096 86241 --ll --jacobi-every 0 --nosave | grep "LL  res64=0000000000000000  PRIME"
	./build/barnarg 86249 8192 86247 --ll --jacobi-every 0 --nosave | grep "res64=422c56c4f9e3f2e3  composite"
	./build/barnarg 86249 8192 86247 --ll --jacobi-every 0 --nosave --inject 20000 | grep -E "FAILED|res64=422c56c4f9e3f2e3  composite"

clean:
	rm -rf build

.PHONY: check clean install dist
