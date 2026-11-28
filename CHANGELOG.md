# Changelog

All notable changes to `Glacier.Quant` are documented here.

## v1.4.2

- Aligned the `FP8_E4M3` GGUF tensor type ID to canonical `43u` while retaining legacy `104u` read compatibility.

## v1.4.0

- Added `HadamardTransform` with AVX-512F / AVX2 normalized Sylvester block FWHT, Paley Type-I (`m in {12, 20, 68}`), Kronecker Paley-Sylvester transforms (`m * 2^p`), and randomized sign rotations.
- Added `NariQTrellisCodec` supporting bit-exact `TRL1..TRL6` decoding (`DecodeStateSwar`), AVX-512/AVX2 `TileMatVecAdd`, and `TileErrorMetrics` diagnostics.

## v1.3.0

- Vectorized `LittleBit2Svd`, `LittleBit2Quantizer` ADMM refinement (`RefinePathAdmm`), and activation column scale folding (`ApplyInverseColumnScales`).
- Added `LittleBit2SamsungCodec` for bidirectional SamsungLabs LSB-first `uint32` checkpoint conversion.
- Added `ActivationSalientSelector` (`ActivationWeightedL2`, `WeightRms`, `WeightMaxAbs`, `Auto`) and `Log1p`-stabilized `KlDivergenceEvaluator`.
