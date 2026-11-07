build/barnarg: src/main.swift src/proof.swift
	mkdir -p build
	swiftc -O src/main.swift src/proof.swift -o build/barnarg

check: build/barnarg
	./build/barnarg 521 64 521 build/m521.bin 1,2,10,521
	python3 test/check.py build/m521.bin 521 --ref 10
	./build/barnarg 86243 4096 1000 build/m86243.bin 1,2,20,1000
	python3 test/check.py build/m86243.bin 86243 --ref 20
	./build/barnarg 216091 16384 1000 build/m216091.bin 1,20,1000
	python3 test/check.py build/m216091.bin 216091 --ref 20
	./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 0 | tail -1
	./build/barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 0 --inject 30000 | grep -E "FAILED|PRP"
	cd build && ./barnarg 86243 4096 86243 --gerbicz --L 100 --C 50 --proof 5 --nosave | grep -E "PRP|wrote"
	test "$$(md5 -q build/86243-5.proof)" = 60b69c35a3d3403c2173ba598ba0f243
	./build/barnarg verify build/86243-5.proof 4096

clean:
	rm -rf build

.PHONY: check clean
