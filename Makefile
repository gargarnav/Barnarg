build/barnarg: src/main.swift
	mkdir -p build
	swiftc -O src/main.swift -o build/barnarg

check: build/barnarg
	./build/barnarg 521 64 521 build/m521.bin 1,2,10,521
	python3 test/check.py build/m521.bin 521 --ref 10
	./build/barnarg 86243 4096 1000 build/m86243.bin 1,2,20,1000
	python3 test/check.py build/m86243.bin 86243 --ref 20
	./build/barnarg 216091 16384 1000 build/m216091.bin 1,20,1000
	python3 test/check.py build/m216091.bin 216091 --ref 20

clean:
	rm -rf build

.PHONY: check clean
