## rocmbandwidth

To configure, you need `cmake`, `make`, `clang` and `hipcc`. Then, run

```bash
cmake . -B build
make -C build
```

To run all tests, execute the output binary under `build/rocmbandwidth`.

To run on the `n015` or `n016` node on eX³:

```
module load cmake
cmake . -B build
make -C build
build/rocmbandwidth
```