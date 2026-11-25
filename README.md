# Glacier.Quant

Cross-platform .NET 10 (Native AOT compatible) tensor quantization library covering `Q4_K_M`, `Q8_0`, `FP8`, 1-bit binary, and LittleBit-2 sub-1-bit kernels, PTQ calibration (`AWQ`, `GPTQ`, `imatrix`, activation-weighted salient column selection), and streaming GGUF/Safetensors I/O.

## What's in v1.3.0

- **SIMD + parallel LittleBit-2 SVD & ADMM refinement (`LittleBit2Svd`, `LittleBit2Quantizer`, `LittleBit2Options.AdmmIterations`):** Dual-SVID, Joint-ITQ, and randomized truncated SVD paths vectorize via `System.Numerics.Tensors.TensorPrimitives` and parallelize across independent output rows/tiles on matrices above `262,144` operations. Pure-CPU `LittleBit2Quantizer.RefinePathAdmm` (enabled via `LittleBit2Options.AdmmIterations`) performs alternating binary coordinate descent and closed-form scale updates with a strict monotonic Frobenius guard, and `LittleBit2Quantizer.Quantize(..., columnScales, ...)` / `ApplyInverseColumnScales` fold activation weights back into `ColScales` at zero inference overhead.
- **Samsung LittleBit-2 checkpoint codec (`LittleBit2SamsungCodec`):** Converts SamsungLabs LSB-first `uint32` packed factors (`+1 -> 0, -1 -> 1`) to and from `LittleBit2Path` `ulong[]` (`+1 -> 1, -1 -> 0`) with bitwise-masked tails and log-space balanced FP16 scale packing (`PackSignedBalancedPath`, `FromSamsungCheckpointPath`).
- **Activation-weighted salient column selector (`ActivationSalientSelector`):** Ranks high-impact input columns (`ActivationWeightedL2`, `WeightRms`, `WeightMaxAbs`, `Auto`) from `imatrix` JSON profiles or SIMD weight scans and computes normalized column scaling multipliers without skewing pinned salient columns.
- **Large-margin KL / log-softmax stabilization (`KlDivergenceEvaluator`):** Replaces `Math.Log(1 + eps)` near the argmax logit with a `Log1p` formulation so `ComputeLogSoftmax` and `ComputeTokenKlDivergence` do not collapse to `0.0` at large logit margins (`>= 40`).

## Installation

```bash
dotnet add package Glacier.Quant
```

## License

MIT
