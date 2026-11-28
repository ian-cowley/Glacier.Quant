# Glacier.Quant

Cross-platform .NET 10 (Native AOT compatible) tensor quantization library covering `Q4_K_M`, `Q8_0`, `FP8`, 1-bit binary, LittleBit-2 sub-1-bit kernels, Hadamard rotations, NariQ trellis decoding (`TRL1..TRL6`), PTQ calibration (`AWQ`, `GPTQ`, `imatrix`, activation-weighted salient column selection), and streaming GGUF/Safetensors I/O.

## What's in v1.4.0 — Hadamard & NariQ Trellis Kernels (`Glacier.Quant.Kernels`)

- **`HadamardTransform`**:
  - `ApplySylvesterInPlace(Span<float> data, int blockLen)` — AVX-512F / AVX2 / scalar normalized block Sylvester FWHT (`H * H = I`, fused in-register `h = 1, 2, 4, 8` butterflies and final `1 / sqrt(blockLen)` scaling).
  - `ApplyPaleyInPlace` / `ApplyPaleyInverseInPlace` & `ApplyKroneckerPaleyInPlace` / `ApplyKroneckerPaleyInverseInPlace` — SIMD Paley Type-I orthogonal Hadamard (`m in {12, 20, 68}`) and full-slice Kronecker transforms (`m * 2^p`).
  - `GenerateRandomSigns`, `ApplyRandomizedRotation`, `ApplyRandomizedRotationInverse` — QuaRot / SpinQuant / QuIP# / NariQ randomized Hadamard rotation.
- **`NariQTrellisCodec`**:
  - Bit-exact `TRL1..TRL6` (`1..6` bits/weight) `16x16` circular bitstream decoder using the `0x83DCD12Du` procedural codebook and portable 32-bit SWAR horizontal byte sum (`DecodeStateSwar` vs `DecodeStateScalar`).
  - `DecodeTileInt8`, `DecodeTileFloat`, `DecodeRows`, `DotTileRowSimd`, AVX-512F / AVX2+FMA `TileMatVecAdd`, and `InspectTile` (`TileErrorMetrics` RMSE/MAE/SNR).

### Empirical Benchmarks (Windows 11, .NET 10.0 Release, Zen 5 AVX-512F, 24 Threads)

Measured via `dotnet run --project benchmarks/Glacier.Quant.Benchmarks/Glacier.Quant.Benchmarks.csproj -c Release`:

| Kernel / Workload | Shape / Scope | Measured Latency | Throughput |
| :--- | :--- | :---: | :---: |
| **`HadamardTransform.ApplySylvesterInPlace`** | 16,384 floats, `blockLen = 128` | **4.25 &mu;s** | **15.40 GB/s** |
| **`HadamardTransform.ApplySylvesterInPlace`** | 16,384 floats, `blockLen = 512` | **4.77 &mu;s** | **13.75 GB/s** |
| **`HadamardTransform.ApplySylvesterInPlace`** | 16,384 floats, `blockLen = 4096` | **5.27 &mu;s** | **12.43 GB/s** |
| **`HadamardTransform.ApplyPaleyInPlace`** | `m = 12` / `20` / `68` | **49.3 / 81.6 / 319.8 ns** | Single-vector transform |
| **`HadamardTransform.ApplyKroneckerPaleyInPlace`** | `5120 (20*256)` / `17408 (68*256)` | **7.70 / 63.20 &mu;s** | **2.66 / 1.10 GB/s** |
| **`NariQTrellisCodec.DecodeStateSwar` vs `Scalar`** | All 65,536 states (`0x83DCD12Du`) | **0.608 vs 0.993 ns/state** | **1.63&times; SWAR speedup** |
| **`NariQTrellisCodec.DecodeTileInt8`** | `5120 x 256` (`TRL2` / `TRL3` / `TRL4`) | **4.04 / 3.85 / 4.06 ms** | **0.32 / 0.34 / 0.32 Gw/s** |
| **`NariQTrellisCodec.TileMatVecAdd`** | `5120 x 256` (`TRL2` / `TRL3` / `TRL4`) | **3.89 / 3.94 / 3.72 ms** | **0.34 / 0.33 / 0.35 Gw/s** |

## What's in v1.3.0

- **SIMD + parallel LittleBit-2 SVD & ADMM refinement (`LittleBit2Svd`, `LittleBit2Quantizer`, `LittleBit2Options.AdmmIterations`):** Dual-SVID, Joint-ITQ, and randomized truncated SVD paths vectorize via `System.Numerics.Tensors.TensorPrimitives` and parallelize across independent output rows/tiles on matrices above `262,144` operations. Pure-CPU `LittleBit2Quantizer.RefinePathAdmm` (enabled via `LittleBit2Options.AdmmIterations`) performs alternating binary coordinate descent and closed-form scale updates with a strict monotonic Frobenius guard, and `LittleBit2Quantizer.Quantize(..., columnScales, ...)` / `ApplyInverseColumnScales` fold activation weights back into `ColScales` at zero inference overhead.
- **Samsung LittleBit-2 checkpoint codec (`LittleBit2SamsungCodec`):** Converts SamsungLabs LSB-first `uint32` packed factors (`+1 -> 0, -1 -> 1`) to and from `LittleBit2Path` `ulong[]` (`+1 -> 1, -1 -> 0`) with bitwise-masked tails and log-space balanced FP16 scale packing (`PackSignedBalancedPath`, `FromSamsungCheckpointPath`).
- **Activation-weighted salient column selector (`ActivationSalientSelector`):** Ranks high-impact input columns (`ActivationWeightedL2`, `WeightRms`, `WeightMaxAbs`, `Auto`) from `imatrix` JSON profiles or SIMD weight scans and computes normalized column scaling multipliers without skewing pinned salient columns.
- **Large-margin KL / log-softmax stabilization (`KlDivergenceEvaluator`):** Replaces `Math.Log(1 + eps)` near the argmax logit with a `Log1p` formulation so `ComputeLogSoftmax` and `ComputeTokenKlDivergence` do not collapse to `0.0` at large logit margins (`>= 40`).

## Installation

```bash
dotnet add package Glacier.Quant
```

## Testing & Verification

```bash
dotnet test Glacier.Quant.slnx -c Release    # 237 / 237 passed
```

## License

MIT
