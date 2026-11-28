# Glacier.Quant Empirical Benchmarks

All measurements on this page were executed directly in `Release` mode (`-c Release`) using `benchmarks/Glacier.Quant.Benchmarks`.

---

## Reference Workstation

- **CPU:** AMD Ryzen AI 9 HX 370 (12C/24T Zen 5, AVX-512)
- **Discrete GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, `sm_89`)
- **Integrated GPU:** AMD Radeon 890M Graphics (RDNA 3.5)
- **RAM:** 31 GB
- **OS:** Windows 11 (`10.0.26300.0`)
- **Runtime:** `.NET 10.0.15` (`.NET SDK 11.0.100`, Release x64, `AVX-512F=True`, `AVX2=True`, `FMA=True`, `24` threads)

---

## How to Reproduce

```bash
dotnet run --project benchmarks/Glacier.Quant.Benchmarks/Glacier.Quant.Benchmarks.csproj -c Release
```

---

## 1. Sylvester Block FWHT (`16,384` Floats, Normalized $H \cdot H = I$)

| Block Length | Iterations | Measured Latency | Throughput |
| ---: | ---: | ---: | ---: |
| `128` | 5,000 | 5.69 us | 11.52 GB/s |
| `512` | 5,000 | 4.42 us | 14.84 GB/s |
| `4096` | 5,000 | 6.44 us | 10.18 GB/s |

---

## 2. Paley Type-I & Kronecker Paley-Sylvester Transforms

| Transform | Shape / Order | Measured Latency | Throughput |
| :--- | :--- | ---: | ---: |
| `ApplyPaleyInPlace` | `m = 12` (500k iters) | 67.9 ns/call | — |
| `ApplyPaleyInPlace` | `m = 20` (500k iters) | 101.5 ns/call | — |
| `ApplyPaleyInPlace` | `m = 68` (500k iters) | 385.2 ns/call | — |
| `ApplyKroneckerPaleyInPlace` | `5120` (`20 * 256`, 5k iters) | 7.34 us | 2.79 GB/s |
| `ApplyKroneckerPaleyInPlace` | `17408` (`68 * 256`, 5k iters) | 71.71 us | 0.97 GB/s |

---

## 3. 65,536-State NariQ Codebook Decode (`0x83DCD12Du`)

| Implementation | Scope | Latency | Speedup |
| :--- | :--- | ---: | ---: |
| `DecodeStateScalar` | 1,000 full 65,536-state sweeps | 1.058 ns/state | 1.00x |
| `DecodeStateSwar` | 1,000 full 65,536-state sweeps | 0.798 ns/state | 1.33x |

---

## 4. NariQ `16x16` Tile Matrix (`5120 x 256`, 5,120 Tiles = 1.31M Weights)

| Bitrate | `DecodeTileInt8` | Decode Rate | `TileMatVecAdd` | MatVec Rate |
| :--- | ---: | ---: | ---: | ---: |
| `TRL2` (2-bit) | 3.63 ms | 0.36 Gw/s | 3.61 ms | 0.36 Gw/s |
| `TRL3` (3-bit) | 3.47 ms | 0.38 Gw/s | 3.66 ms | 0.36 Gw/s |
| `TRL4` (4-bit) | 3.34 ms | 0.39 Gw/s | 3.57 ms | 0.37 Gw/s |
