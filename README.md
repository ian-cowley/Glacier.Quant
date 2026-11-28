# Glacier.Quant

[![NuGet CI](https://github.com/ian-cowley/Glacier.Quant/actions/workflows/publish-nuget.yml/badge.svg)](https://github.com/ian-cowley/Glacier.Quant/actions/workflows/publish-nuget.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)
![NativeAOT](https://img.shields.io/badge/NativeAOT-Compatible-brightgreen.svg)

`Glacier.Quant` is a cross-platform .NET 10 tensor quantization library for compressing, calibrating, evaluating, and exporting large language model weights in pure C#. It includes SIMD quantization kernels (`Q4_K_M`, `Q8_0`, `FP8`, 1-bit binary), sub-1-bit LittleBit-2 SVD/ADMM codecs, Sylvester and Paley Hadamard rotations, NariQ trellis decoding (`TRL1..TRL6`), PTQ calibrators (`AWQ`, `GPTQ`, `imatrix`), and streaming `GGUF` / `Safetensors` writers.

---

## Install

```bash
dotnet add package Glacier.Quant
```

Or build and test from source:

```bash
git clone https://github.com/ian-cowley/Glacier.Quant.git
cd Glacier.Quant
dotnet test Glacier.Quant.slnx -c Release
```

---

## Quick Start

Apply an in-place orthogonal Sylvester Hadamard rotation (QuaRot / SpinQuant) and quantize rows to `Q8_0` blocks:

```csharp
using Glacier.Quant.Kernels;

float[] weights = new float[16_384];
for (int i = 0; i < weights.Length; i++)
{
    weights[i] = ((i % 29) - 14) * 0.1f;
}

// Normalized block FWHT (H * H = I)
HadamardTransform.ApplySylvesterInPlace(weights, blockLen: 512);

// Quantize 32-element blocks to Q8_0 using AVX-512 / AVX2
var q8Blocks = new BlockQ8_0[weights.Length / BlockQ8_0.BlockSize];
SimdQuantizer.QuantizeRowQ8_0(weights, q8Blocks);
```

Decode a NariQ `16x16` trellis tile (`TRL3`, 3 bits/weight):

```csharp
using Glacier.Quant.Kernels;

int wordsPerTile = NariQTrellisCodec.GetTileWordCount(bits: 3);
uint[] packedTile = new uint[wordsPerTile];
sbyte[] decoded = new sbyte[NariQTrellisCodec.TileElements];

NariQTrellisCodec.DecodeTileInt8(packedTile, bits: 3, decoded);
```

---

## What You Can Do

- **Block & Scalar Quantization (`SimdQuantizer`, `Fp8DynamicScaler`):** Quantize and dequantize `Q4_K_M`, `Q8_0`, `FP8_E4M3` (canonical GGUF type ID `43u` with legacy `104u` read compatibility), and 1-bit binary blocks (`BlockBinary1Bit`).
- **Hadamard Outlier Suppression (`HadamardTransform`):** Run normalized block Sylvester FWHT, Paley Type-I (`m in {12, 20, 68}`), Kronecker Paley-Sylvester transforms (`m * 2^p`), and randomized sign rotations.
- **NariQ Trellis Codecs (`NariQTrellisCodec`):** Decode `TRL1..TRL6` circular bitstreams via portable 32-bit SWAR (`0x83DCD12Du`), compute `TileMatVecAdd`, and measure `TileErrorMetrics` (RMSE, MAE, SNR).
- **LittleBit-2 Sub-1-Bit Quantization (`LittleBit2Quantizer`):** Factor matrices via Dual-SVID, Joint-ITQ, ADMM refinement, activation column scale folding, and Samsung checkpoint interop (`LittleBit2SamsungCodec`).
- **PTQ Calibration, Evaluation & Export:** Run `AwqCalibrator`, `GptqCalibrator`, `ImatrixCalibrator`, and `ActivationSalientSelector`, evaluate quality with `KlDivergenceEvaluator` (`Log1p`-stabilized) and `QuantQualityChecker`, and export checkpoints with `GgufWriter` / `SafetensorsWriter`.

---

## Platform & Compatibility

- **Runtime:** .NET 10 (`net10.0`), Native AOT compatible (`IsAotCompatible=true`).
- **SIMD & GPU Dispatch:** AVX-512F, AVX2/FMA, and `TensorPrimitives` across CPU cores; optional CUDA dispatch hooks in `GpuQuantizationDispatcher`.
- **Cross-Platform:** Tested on Windows and Linux x64 in GitHub Actions CI.

---

## Benchmarks

Measured in `Release` on Windows 11 with an AMD Ryzen AI 9 HX 370 (AVX-512F, 24 threads). Reproduce with `dotnet run --project benchmarks/Glacier.Quant.Benchmarks/Glacier.Quant.Benchmarks.csproj -c Release`:

| Kernel / Workload | Shape / Scope | Latency | Throughput / Metric |
| :--- | :--- | ---: | :--- |
| `HadamardTransform.ApplySylvesterInPlace` | 16,384 floats, `blockLen = 128` | 5.69 us | 11.52 GB/s |
| `HadamardTransform.ApplySylvesterInPlace` | 16,384 floats, `blockLen = 512` | 4.42 us | 14.84 GB/s |
| `HadamardTransform.ApplyKroneckerPaleyInPlace` | `5120` (`20 * 256`) | 7.34 us | 2.79 GB/s |
| `NariQTrellisCodec.DecodeStateSwar` vs `Scalar` | 65,536-state codebook | 0.798 vs 1.058 ns/state | 1.33x SWAR speedup |
| `NariQTrellisCodec.DecodeTileInt8` (`TRL3` / `TRL4`) | `5120 x 256` (`1.31M` weights) | 3.47 ms / 3.34 ms | 0.38 / 0.39 Gw/s |
| `NariQTrellisCodec.TileMatVecAdd` (`TRL2` / `TRL4`) | `5120 x 256` (`1.31M` weights) | 3.61 ms / 3.57 ms | 0.36 / 0.37 Gw/s |

See [the full benchmark suite](docs/benchmarks.md) for all measured block lengths and bitrates.

---

## Documentation

- [User Manual](docs/manual.md) — API reference for kernels, Hadamard rotations, NariQ trellis, LittleBit-2, calibrators, evaluators, and GGUF/Safetensors I/O
- [Architecture](docs/architecture.md) — Block formats, SWAR/SIMD kernel designs, ADMM/Cholesky calibration, and streaming writers
- [Benchmarks](docs/benchmarks.md) — Workstation specs, reproduction command, and complete kernel benchmark tables
- [Changelog](CHANGELOG.md) — Version history (`v1.3.0` to `v1.4.2`)

---

## License & Credits

Licensed under the MIT License.

Developed by Ian Cowley, Antigravity (Google DeepMind) and Claude.
