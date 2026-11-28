# Glacier.Quant User Manual

This manual documents the public API in `Glacier.Quant` as implemented in `src/Glacier.Quant`.

---

## 1. Block & Scalar Quantization (`SimdQuantizer`, `Fp8DynamicScaler`)

Supported block types in `Glacier.Quant.Kernels.BlockTypes`:

- `QuantType` enum covering `Q8_0`, `Q4_K_M`, `FP8_E4M3` (canonical GGUF type ID `43u`, reading legacy `104u` transparently), 1-bit binary (`BlockBinary1Bit`), and trellis/sub-1-bit formats.
- `SimdQuantizer` provides vectorized `QuantizeRowQ8_0` / dequantization and `FP8_E4M3` lookup conversion over `Span<T>`.
- `Fp8DynamicScaler` computes dynamic tensor/channel scaling factors for FP8 quantization.

---

## 2. Hadamard Rotations (`HadamardTransform`)

`Glacier.Quant.Kernels.HadamardTransform` implements orthogonal transforms for outlier flattening before quantization:

- **Sylvester block FWHT:** Self-inverse normalized transform ($H \cdot H = I$, scaled by $1/\sqrt{\text{blockLen}}$) for any power-of-2 `blockLen >= 2`:
  ```csharp
  HadamardTransform.ApplySylvesterInPlace(span, blockLen: 512);
  ```
- **Paley Type-I & Kronecker transforms:** Exact orthogonal transforms for non-power-of-two hidden dimensions whose factor is `m in {12, 20, 68}` (`ApplyPaleyInPlace`, `ApplyPaleyInverseInPlace`, `ApplyKroneckerPaleyInPlace`, `ApplyKroneckerPaleyInverseInPlace`, validated with `IsSupportedPaleyOrder(m)`).
- **Randomized rotations (QuaRot / SpinQuant / QuIP#):** `GenerateRandomSigns`, `ApplyRandomizedRotation`, and `ApplyRandomizedRotationInverse`.

---

## 3. NariQ Trellis Codec (`NariQTrellisCodec`)

`NariQTrellisCodec` supports `TRL1..TRL6` (`1..6` bits per weight) over `16x16` (`256`-element) tiles:

- `GetTileWordCount(bits)` (`8 * bits`) and `GetTileByteCount(bits)` (`32 * bits`).
- `DecodeStateSwar(state)` / `DecodeStateScalar(state)` — Maps a 16-bit trellis state (`0..65535`) to `[-126, +126]`.
- `DecodeTileInt8(tileWords, bits, dstRowMajor)`, `DecodeTileFloat`, and `DecodeRows`.
- `DotTileRowSimd` and `TileMatVecAdd(tileWords, bits, scale, x, y)` — Direct AVX-512 / AVX2 tile matrix-vector multiply-accumulate.
- `InspectTile` — Computes `TileErrorMetrics(Rmse, Mae, MaxAbsError, SignalToNoiseDb, MinCode, MaxCode)`.

---

## 4. LittleBit-2 Sub-1-Bit Quantization (`LittleBit2Quantizer`)

Factorizes a weight matrix into signed binary matrices and tri-scales (`RowScales`, `ColScales`, `StepScales`) with optional residual path compensation:

```csharp
var options = new LittleBit2Options
{
    BitsPerWeight = 1.5,
    UseResidualCompensation = true,
    AdmmIterations = 4,
    RotationMode = LittleBit2RotationMode.Hadamard
};

LittleBit2Tensor quantized = LittleBit2Quantizer.Quantize(weights, rows, cols, options);
double bpw = LittleBit2Quantizer.ComputeEffectiveBpw(rows, cols, quantized.Primary.Rank, quantized.Residual?.Rank ?? 0);
```

Related helpers:

- `LittleBit2Quantizer.ComputeRankForBpw(rows, cols, bitsPerWeight, useResidualCompensation)`
- `LittleBit2Quantizer.RefinePathAdmm` — Alternating binary coordinate descent and closed-form scale refinement with monotonic Frobenius guard.
- `LittleBit2SamsungCodec` — Bidirectional conversion between SamsungLabs LSB-first `uint32` checkpoints and `LittleBit2Path`.

---

## 5. Post-Training Calibration (`Glacier.Quant.Calibration`)

Prepare activations in `CalibrationDataset` (`NumTokens`, `HiddenDim`, `Activations`):

- **`AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 20, targetBits: 4)`:** Computes per-channel activation magnitudes $s_X$ and searches $\alpha \in [0, 1]$ to minimize output reconstruction MSE $\lVert W X - \text{dequant}(\text{quant}(W \cdot s)) s^{-1} X \rVert_F^2$.
- **`GptqCalibrator.Calibrate(weights, rows, cols, dataset, damping: 0.01f, targetBits: 4, blockSize: 128, groupSize: 32, symmetric: true)`:** Computes Hessian $H = 2 X^T X + \lambda I$, Cholesky-factors with adaptive damping retries, and diffuses quantization error block-by-block.
- **`ImatrixCalibrator`:** Accumulates empirical activation importance vectors (`imatrix`) across token batches.
- **`ActivationSalientSelector`:** Selects top salient input columns using `SalientSelectionMethod` (`ActivationWeightedL2`, `WeightRms`, `WeightMaxAbs`, `Auto`).

---

## 6. Quality Evaluation (`Glacier.Quant.Evaluation`)

- `KlDivergenceEvaluator` (`KlDivergenceOptions`, `KlDivergenceResult`): Computes log-softmax via `Log1p` stabilization so KL divergence remains accurate at large logit margins (`>= 40`).
- `QuantQualityChecker`: Measures RMSE, SQNR, and max absolute error between reference and quantized tensors.

---

## 7. Streaming Checkpoint Writers & Readers (`Glacier.Quant.Formats`)

- **`GgufWriter` / `GgufReader`:** Streams aligned GGUF v3 binary files (`32`-byte default alignment) from `QuantTensorMetadata` and typed metadata dictionaries (`Uint8..Float64`, strings, arrays).
- **`SafetensorsWriter` / `SafetensorsReader`:** Writes and reads SafeTensors JSON headers and contiguous tensor payloads.
