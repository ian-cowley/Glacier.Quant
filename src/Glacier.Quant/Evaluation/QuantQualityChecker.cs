namespace Glacier.Quant.Evaluation;

using System;
using System.Buffers;
using System.Numerics.Tensors;
using Glacier.Quant.Calibration;
using Glacier.Quant.Gpu;
using Glacier.Quant.Kernels;

/// <summary>
/// End-to-end quantization quality checker that projects linear layers against reference
/// weights (FP32 or Q8_0) and evaluates output logit divergence with <see cref="KlDivergenceEvaluator"/>.
/// </summary>
public static class QuantQualityChecker
{
    /// <summary>
    /// Computes row-major linear projection logits[t, r] = dot(weights[r, :], activations[t, :]) + bias[r].
    /// </summary>
    public static void ProjectLogits(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> activations,
        Span<float> logitsOut,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(weights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);

        long requiredOut = (long)numTokens * vocabSize;
        if (logitsOut.Length < requiredOut)
            throw new ArgumentException($"Output logits span ({logitsOut.Length}) is smaller than required ({requiredOut}).", nameof(logitsOut));

        bool hasBias = !bias.IsEmpty;
        for (int t = 0; t < numTokens; t++)
        {
            var x = activations.Slice(t * hiddenDim, hiddenDim);
            int outRowOffset = t * vocabSize;
            for (int r = 0; r < vocabSize; r++)
            {
                var wRow = weights.Slice(r * hiddenDim, hiddenDim);
                float dot = TensorPrimitives.Dot(wRow, x);
                logitsOut[outRowOffset + r] = hasBias ? dot + bias[r] : dot;
            }
        }
    }

    /// <summary>
    /// Computes row-major linear projection logits[t, r] and returns a newly allocated [numTokens * vocabSize] array.
    /// </summary>
    public static float[] ProjectLogits(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        ReadOnlySpan<float> bias = default)
    {
        float[] logits = new float[checked(numTokens * vocabSize)];
        ProjectLogits(weights, activations, logits, numTokens, vocabSize, hiddenDim, bias);
        return logits;
    }

    /// <summary>
    /// Computes row-major linear projection logits[t, r] using activations from <paramref name="dataset"/>.
    /// </summary>
    public static float[] ProjectLogits(
        ReadOnlySpan<float> weights,
        CalibrationDataset dataset,
        int vocabSize,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        int hiddenDim = dataset.HiddenDim;
        int numTokens = dataset.NumTokens;
        var activations = dataset.Activations.AsSpan(0, checked(numTokens * hiddenDim));
        return ProjectLogits(weights, activations, numTokens, vocabSize, hiddenDim, bias);
    }

    /// <summary>
    /// Quantizes finite weights into <paramref name="quantType"/> and dequantizes back to FP32.
    /// </summary>
    public static void QuantizeAndDequantize(
        ReadOnlySpan<float> weights,
        QuantType quantType,
        Span<float> dequantizedOutput)
    {
        if (weights.IsEmpty)
            throw new ArgumentException("Weights span cannot be empty.", nameof(weights));
        if (dequantizedOutput.Length < weights.Length)
            throw new ArgumentException($"Destination span length ({dequantizedOutput.Length}) is smaller than weights length ({weights.Length}).", nameof(dequantizedOutput));

        EnsureAllFinite(weights, nameof(weights));

        if (quantType == QuantType.FP32)
        {
            weights.CopyTo(dequantizedOutput);
            return;
        }

        int byteCount = GpuQuantizationDispatcher.GetQuantizedByteCount(weights.Length, quantType);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(byteCount);
        try
        {
            var byteSpan = buffer.AsSpan(0, byteCount);
            GpuQuantizationDispatcher.QuantizeTensor(weights, quantType, byteSpan, out float dynamicScale);
            GpuQuantizationDispatcher.DequantizeTensor(byteSpan, quantType, dequantizedOutput.Slice(0, weights.Length), dynamicScale);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Quantizes finite weights into <paramref name="quantType"/> and returns a dequantized FP32 array.
    /// </summary>
    public static float[] QuantizeAndDequantize(ReadOnlySpan<float> weights, QuantType quantType)
    {
        float[] output = new float[weights.Length];
        QuantizeAndDequantize(weights, quantType, output);
        return output;
    }

    public static KlDivergenceResult EvaluateWeights(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<float> quantizedWeights,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        if (quantizedWeights.Length != referenceWeights.Length)
            throw new ArgumentException($"Quantized weights length ({quantizedWeights.Length}) must match reference weights length ({referenceWeights.Length}).", nameof(quantizedWeights));

        EnsureAllFinite(referenceWeights, nameof(referenceWeights));
        EnsureAllFinite(activations, nameof(activations));
        EnsureAllFinite(bias, nameof(bias));

        if (!AllFinite(quantizedWeights))
        {
            return CreateNonFiniteResult(
                numTokens,
                vocabSize,
                opts,
                "Quantized weights contain non-finite values (NaN or Infinity).");
        }

        int totalLogits = checked(numTokens * vocabSize);
        float[] refLogitsBuffer = ArrayPool<float>.Shared.Rent(totalLogits);
        float[] quantLogitsBuffer = ArrayPool<float>.Shared.Rent(totalLogits);
        try
        {
            var refLogits = refLogitsBuffer.AsSpan(0, totalLogits);
            var quantLogits = quantLogitsBuffer.AsSpan(0, totalLogits);

            ProjectLogits(referenceWeights, activations, refLogits, numTokens, vocabSize, hiddenDim, bias);
            if (!AllFinite(refLogits))
                throw new ArgumentException("Reference projection produced non-finite logits.", nameof(referenceWeights));

            ProjectLogits(quantizedWeights, activations, quantLogits, numTokens, vocabSize, hiddenDim, bias);
            if (!AllFinite(quantLogits))
            {
                return CreateNonFiniteResult(
                    numTokens,
                    vocabSize,
                    opts,
                    "Quantized projection logits contain non-finite values (NaN or Infinity).");
            }

            return KlDivergenceEvaluator.EvaluateLogits(refLogits, quantLogits, vocabSize, opts);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refLogitsBuffer);
            ArrayPool<float>.Shared.Return(quantLogitsBuffer);
        }
    }

    public static KlDivergenceResult EvaluateWeights(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<float> quantizedWeights,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        double threshold,
        ReadOnlySpan<float> bias = default,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement
        };
        return EvaluateWeights(referenceWeights, quantizedWeights, activations, numTokens, vocabSize, hiddenDim, opts, bias);
    }

    public static KlDivergenceResult EvaluateWeights(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<float> quantizedWeights,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateWeights(
            referenceWeights,
            quantizedWeights,
            activations,
            dataset.NumTokens,
            vocabSize,
            dataset.HiddenDim,
            options,
            bias);
    }

    public static KlDivergenceResult EvaluateWeights(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<float> quantizedWeights,
        CalibrationDataset dataset,
        int vocabSize,
        double threshold,
        ReadOnlySpan<float> bias = default,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement
        };
        return EvaluateWeights(referenceWeights, quantizedWeights, dataset, vocabSize, opts, bias);
    }

    public static KlDivergenceResult EvaluateQuantization(
        ReadOnlySpan<float> referenceWeights,
        QuantType targetType,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        QuantType referenceType = QuantType.FP32,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        EnsureAllFinite(referenceWeights, nameof(referenceWeights));
        EnsureAllFinite(activations, nameof(activations));
        EnsureAllFinite(bias, nameof(bias));

        int weightCount = referenceWeights.Length;
        float[] refBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        float[] targetBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        try
        {
            var refWeightsSpan = refBuffer.AsSpan(0, weightCount);
            var targetWeightsSpan = targetBuffer.AsSpan(0, weightCount);

            QuantizeAndDequantizeMatrix(referenceWeights, vocabSize, hiddenDim, referenceType, refWeightsSpan);
            if (!AllFinite(refWeightsSpan))
                throw new ArgumentException($"Reference quantization ({referenceType}) produced non-finite weights.", nameof(referenceWeights));

            QuantizeAndDequantizeMatrix(referenceWeights, vocabSize, hiddenDim, targetType, targetWeightsSpan);
            return EvaluateWeights(refWeightsSpan, targetWeightsSpan, activations, numTokens, vocabSize, hiddenDim, opts, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refBuffer);
            ArrayPool<float>.Shared.Return(targetBuffer);
        }
    }

    public static KlDivergenceResult EvaluateQuantization(
        ReadOnlySpan<float> referenceWeights,
        QuantType targetType,
        CalibrationDataset dataset,
        int vocabSize,
        QuantType referenceType = QuantType.FP32,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateQuantization(
            referenceWeights,
            targetType,
            activations,
            dataset.NumTokens,
            vocabSize,
            dataset.HiddenDim,
            referenceType,
            options,
            bias);
    }

    public static KlDivergenceResult EvaluatePtqResult(
        ReadOnlySpan<float> referenceWeights,
        PtqCalibrationResult ptqResult,
        CalibrationDataset dataset,
        int vocabSize,
        QuantType? postCalibrationQuantType = null,
        QuantType referenceType = QuantType.FP32,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(ptqResult);
        ArgumentNullException.ThrowIfNull(dataset);

        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        int hiddenDim = dataset.HiddenDim;
        int numTokens = dataset.NumTokens;
        var activations = dataset.Activations.AsSpan(0, checked(numTokens * hiddenDim));
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        EnsureAllFinite(referenceWeights, nameof(referenceWeights));
        EnsureAllFinite(activations, nameof(activations));
        EnsureAllFinite(bias, nameof(bias));

        if (ptqResult.CalibratedWeights is null || ptqResult.CalibratedWeights.Length != referenceWeights.Length)
            throw new ArgumentException("PTQ result CalibratedWeights length does not match reference weights length.", nameof(ptqResult));

        int weightCount = referenceWeights.Length;
        float[] refBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        float[] candBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        try
        {
            var refSpan = refBuffer.AsSpan(0, weightCount);
            var candSpan = candBuffer.AsSpan(0, weightCount);

            QuantizeAndDequantizeMatrix(referenceWeights, vocabSize, hiddenDim, referenceType, refSpan);
            if (!AllFinite(refSpan))
                throw new ArgumentException($"Reference quantization ({referenceType}) produced non-finite weights.", nameof(referenceWeights));

            if (postCalibrationQuantType is QuantType qt)
            {
                bool isAwqWithChannelScales =
                    ptqResult.OptimalScales is { Length: > 0 } &&
                    ptqResult.Method.StartsWith("AWQ", StringComparison.OrdinalIgnoreCase);

                if (isAwqWithChannelScales)
                {
                    var scales = ptqResult.OptimalScales!;
                    if (scales.Length != hiddenDim)
                        throw new ArgumentException($"AWQ OptimalScales length ({scales.Length}) must match hiddenDim ({hiddenDim}).", nameof(ptqResult));

                    if (!AllFinite(scales))
                    {
                        return CreateNonFiniteResult(
                            numTokens,
                            vocabSize,
                            opts,
                            "AWQ OptimalScales contain non-finite values.");
                    }

                    float[] scaledBuffer = ArrayPool<float>.Shared.Rent(weightCount);
                    try
                    {
                        var scaledSpan = scaledBuffer.AsSpan(0, weightCount);
                        for (int r = 0; r < vocabSize; r++)
                        {
                            int rowOffset = r * hiddenDim;
                            for (int c = 0; c < hiddenDim; c++)
                                scaledSpan[rowOffset + c] = referenceWeights[rowOffset + c] * scales[c];
                        }

                        if (!AllFinite(scaledSpan))
                        {
                            return CreateNonFiniteResult(
                                numTokens,
                                vocabSize,
                                opts,
                                "Scaled AWQ weights contain non-finite values.");
                        }

                        QuantizeAndDequantizeMatrix(scaledSpan, vocabSize, hiddenDim, qt, candSpan);

                        for (int r = 0; r < vocabSize; r++)
                        {
                            int rowOffset = r * hiddenDim;
                            for (int c = 0; c < hiddenDim; c++)
                                candSpan[rowOffset + c] /= scales[c];
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(scaledBuffer);
                    }
                }
                else
                {
                    if (!AllFinite(ptqResult.CalibratedWeights))
                    {
                        return CreateNonFiniteResult(
                            numTokens,
                            vocabSize,
                            opts,
                            "Calibrated PTQ weights contain non-finite values.");
                    }

                    QuantizeAndDequantizeMatrix(ptqResult.CalibratedWeights, vocabSize, hiddenDim, qt, candSpan);
                }

                return EvaluateWeights(refSpan, candSpan, activations, numTokens, vocabSize, hiddenDim, opts, bias);
            }

            return EvaluateWeights(refSpan, ptqResult.CalibratedWeights, activations, numTokens, vocabSize, hiddenDim, opts, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refBuffer);
            ArrayPool<float>.Shared.Return(candBuffer);
        }
    }

    public static KlDivergenceResult EvaluateAwqCalibrated(
        float[] referenceWeights,
        int vocabSize,
        CalibrationDataset dataset,
        QuantType? targetType = null,
        QuantType referenceType = QuantType.FP32,
        int gridSteps = 20,
        int targetBits = 4,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(referenceWeights);
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        EnsureAllFinite(referenceWeights, nameof(referenceWeights));
        EnsureAllFinite(dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim)), nameof(dataset));
        EnsureAllFinite(bias, nameof(bias));

        var ptqResult = AwqCalibrator.Calibrate(referenceWeights, vocabSize, dataset.HiddenDim, dataset, gridSteps, targetBits);
        return EvaluatePtqResult(referenceWeights, ptqResult, dataset, vocabSize, targetType, referenceType, options, bias);
    }

    public static KlDivergenceResult EvaluateBlocksQ8_0(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockQ8_0> quantizedBlocks,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        if (referenceWeights.Length % BlockQ8_0.BlockSize != 0)
            throw new ArgumentException($"Total weight count ({referenceWeights.Length}) must be a multiple of {BlockQ8_0.BlockSize}.", nameof(referenceWeights));

        int expectedBlocks = referenceWeights.Length / BlockQ8_0.BlockSize;
        if (quantizedBlocks.Length != expectedBlocks)
            throw new ArgumentException($"Expected {expectedBlocks} Q8_0 blocks, got {quantizedBlocks.Length}.", nameof(quantizedBlocks));

        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(referenceWeights.Length);
        try
        {
            var dequantSpan = dequantBuffer.AsSpan(0, referenceWeights.Length);
            SimdQuantizer.DequantizeRowQ8_0(quantizedBlocks, dequantSpan);
            return EvaluateWeights(referenceWeights, dequantSpan, activations, numTokens, vocabSize, hiddenDim, options, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateBlocksQ8_0(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockQ8_0> quantizedBlocks,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateBlocksQ8_0(referenceWeights, quantizedBlocks, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, options, bias);
    }

    public static KlDivergenceResult EvaluateBlocksQ4_K(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockQ4_K> quantizedBlocks,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        if (referenceWeights.Length % BlockQ4_K.BlockSize != 0)
            throw new ArgumentException($"Total weight count ({referenceWeights.Length}) must be a multiple of {BlockQ4_K.BlockSize}.", nameof(referenceWeights));

        int expectedBlocks = referenceWeights.Length / BlockQ4_K.BlockSize;
        if (quantizedBlocks.Length != expectedBlocks)
            throw new ArgumentException($"Expected {expectedBlocks} Q4_K blocks, got {quantizedBlocks.Length}.", nameof(quantizedBlocks));

        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(referenceWeights.Length);
        try
        {
            var dequantSpan = dequantBuffer.AsSpan(0, referenceWeights.Length);
            SimdQuantizer.DequantizeRowQ4_K(quantizedBlocks, dequantSpan);
            return EvaluateWeights(referenceWeights, dequantSpan, activations, numTokens, vocabSize, hiddenDim, options, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateBlocksQ4_K(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockQ4_K> quantizedBlocks,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateBlocksQ4_K(referenceWeights, quantizedBlocks, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, options, bias);
    }

    public static KlDivergenceResult EvaluateBlocks1Bit(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockBinary1Bit> quantizedBlocks,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        if (referenceWeights.Length % BlockBinary1Bit.BlockSize != 0)
            throw new ArgumentException($"Total weight count ({referenceWeights.Length}) must be a multiple of {BlockBinary1Bit.BlockSize}.", nameof(referenceWeights));

        int expectedBlocks = referenceWeights.Length / BlockBinary1Bit.BlockSize;
        if (quantizedBlocks.Length != expectedBlocks)
            throw new ArgumentException($"Expected {expectedBlocks} Binary1Bit blocks, got {quantizedBlocks.Length}.", nameof(quantizedBlocks));

        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(referenceWeights.Length);
        try
        {
            var dequantSpan = dequantBuffer.AsSpan(0, referenceWeights.Length);
            SimdQuantizer.DequantizeRow1Bit(quantizedBlocks, dequantSpan);
            return EvaluateWeights(referenceWeights, dequantSpan, activations, numTokens, vocabSize, hiddenDim, options, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateBlocks1Bit(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<BlockBinary1Bit> quantizedBlocks,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateBlocks1Bit(referenceWeights, quantizedBlocks, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, options, bias);
    }

    public static KlDivergenceResult EvaluateLittleBit2(
        ReadOnlySpan<float> referenceWeights,
        LittleBit2Tensor quantizedTensor,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(quantizedTensor);
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);

        if (quantizedTensor.Rows != vocabSize || quantizedTensor.Cols != hiddenDim)
            throw new ArgumentException($"LittleBit2 tensor dimensions ({quantizedTensor.Rows}x{quantizedTensor.Cols}) must match ({vocabSize}x{hiddenDim}).", nameof(quantizedTensor));

        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(referenceWeights.Length);
        try
        {
            var dequantSpan = dequantBuffer.AsSpan(0, referenceWeights.Length);
            LittleBit2Quantizer.Dequantize(quantizedTensor, dequantSpan);
            return EvaluateWeights(referenceWeights, dequantSpan, activations, numTokens, vocabSize, hiddenDim, options, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateLittleBit2(
        ReadOnlySpan<float> referenceWeights,
        LittleBit2Tensor quantizedTensor,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateLittleBit2(referenceWeights, quantizedTensor, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, options, bias);
    }

    public static KlDivergenceResult EvaluateLittleBit2(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        LittleBit2Options? littleBitOptions = null,
        KlDivergenceOptions? klOptions = null,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        EnsureAllFinite(referenceWeights, nameof(referenceWeights));

        LittleBit2Tensor quantized = LittleBit2Quantizer.Quantize(referenceWeights, vocabSize, hiddenDim, littleBitOptions);
        return EvaluateLittleBit2(referenceWeights, quantized, activations, numTokens, vocabSize, hiddenDim, klOptions, bias);
    }

    public static KlDivergenceResult EvaluateLittleBit2(
        ReadOnlySpan<float> referenceWeights,
        CalibrationDataset dataset,
        int vocabSize,
        LittleBit2Options? littleBitOptions = null,
        KlDivergenceOptions? klOptions = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateLittleBit2(referenceWeights, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, littleBitOptions, klOptions, bias);
    }

    public static KlDivergenceResult EvaluateFp8E4M3(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<byte> quantizedFp8,
        float dynamicScale,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        if (quantizedFp8.Length != referenceWeights.Length)
            throw new ArgumentException($"FP8 byte buffer length ({quantizedFp8.Length}) must match weight count ({referenceWeights.Length}).", nameof(quantizedFp8));

        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(referenceWeights.Length);
        try
        {
            var dequantSpan = dequantBuffer.AsSpan(0, referenceWeights.Length);
            SimdQuantizer.DequantizeRowFP8_E4M3(quantizedFp8, dynamicScale, dequantSpan);
            return EvaluateWeights(referenceWeights, dequantSpan, activations, numTokens, vocabSize, hiddenDim, options, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateFp8E4M3(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<byte> quantizedFp8,
        float dynamicScale,
        CalibrationDataset dataset,
        int vocabSize,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateFp8E4M3(referenceWeights, quantizedFp8, dynamicScale, activations, dataset.NumTokens, vocabSize, dataset.HiddenDim, options, bias);
    }

    public static KlDivergenceResult EvaluateQuantizedBuffer(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<byte> quantizedBuffer,
        QuantType quantType,
        ReadOnlySpan<float> activations,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        float dynamicScale = 1.0f,
        QuantType referenceType = QuantType.FP32,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateMatrixDimensions(referenceWeights.Length, activations.Length, numTokens, vocabSize, hiddenDim, bias);
        EnsureAllFinite(referenceWeights, nameof(referenceWeights));
        EnsureAllFinite(activations, nameof(activations));
        EnsureAllFinite(bias, nameof(bias));

        int weightCount = referenceWeights.Length;
        float[] refBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        float[] dequantBuffer = ArrayPool<float>.Shared.Rent(weightCount);
        try
        {
            var refSpan = refBuffer.AsSpan(0, weightCount);
            var dequantSpan = dequantBuffer.AsSpan(0, weightCount);

            QuantizeAndDequantizeMatrix(referenceWeights, vocabSize, hiddenDim, referenceType, refSpan);
            if (!AllFinite(refSpan))
                throw new ArgumentException($"Reference quantization ({referenceType}) produced non-finite weights.", nameof(referenceWeights));

            GpuQuantizationDispatcher.DequantizeTensor(quantizedBuffer, quantType, dequantSpan, dynamicScale);
            return EvaluateWeights(refSpan, dequantSpan, activations, numTokens, vocabSize, hiddenDim, opts, bias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refBuffer);
            ArrayPool<float>.Shared.Return(dequantBuffer);
        }
    }

    public static KlDivergenceResult EvaluateQuantizedBuffer(
        ReadOnlySpan<float> referenceWeights,
        ReadOnlySpan<byte> quantizedBuffer,
        QuantType quantType,
        CalibrationDataset dataset,
        int vocabSize,
        float dynamicScale = 1.0f,
        QuantType referenceType = QuantType.FP32,
        KlDivergenceOptions? options = null,
        ReadOnlySpan<float> bias = default)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        var activations = dataset.Activations.AsSpan(0, checked(dataset.NumTokens * dataset.HiddenDim));
        return EvaluateQuantizedBuffer(
            referenceWeights,
            quantizedBuffer,
            quantType,
            activations,
            dataset.NumTokens,
            vocabSize,
            dataset.HiddenDim,
            dynamicScale,
            referenceType,
            options,
            bias);
    }

    private static void QuantizeAndDequantizeMatrix(
        ReadOnlySpan<float> weights,
        int rows,
        int cols,
        QuantType quantType,
        Span<float> dequantizedOutput)
    {
        if (quantType == QuantType.LittleBit2)
        {
            if (dequantizedOutput.Length < weights.Length)
                throw new ArgumentException($"Destination span length ({dequantizedOutput.Length}) is smaller than weights length ({weights.Length}).", nameof(dequantizedOutput));
            LittleBit2Tensor tensor = LittleBit2Quantizer.Quantize(weights, rows, cols);
            LittleBit2Quantizer.Dequantize(tensor, dequantizedOutput);
            return;
        }

        QuantizeAndDequantize(weights, quantType, dequantizedOutput);
    }

    private static void ValidateMatrixDimensions(
        int weightLength,
        int activationLength,
        int numTokens,
        int vocabSize,
        int hiddenDim,
        ReadOnlySpan<float> bias)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);

        long expectedWeights = (long)vocabSize * hiddenDim;
        if (weightLength != expectedWeights)
            throw new ArgumentException($"Weights length ({weightLength}) must equal vocabSize * hiddenDim ({expectedWeights}).", nameof(weightLength));

        long expectedActivations = (long)numTokens * hiddenDim;
        if (activationLength != expectedActivations)
            throw new ArgumentException($"Activations length ({activationLength}) must equal numTokens * hiddenDim ({expectedActivations}).", nameof(activationLength));

        if (!bias.IsEmpty && bias.Length != vocabSize)
            throw new ArgumentException($"Bias length ({bias.Length}) must equal vocabSize ({vocabSize}).", nameof(bias));
    }

    private static void EnsureAllFinite(ReadOnlySpan<float> values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new ArgumentException($"Input '{paramName}' contains non-finite value ({values[i]}) at index {i}.", paramName);
        }
    }

    private static bool AllFinite(ReadOnlySpan<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                return false;
        }
        return true;
    }

    private static KlDivergenceResult CreateNonFiniteResult(
        int tokenCount,
        int vocabSize,
        KlDivergenceOptions opts,
        string reason)
    {
        return new KlDivergenceResult
        {
            MeanKlDivergence = double.PositiveInfinity,
            MaxKlDivergence = double.PositiveInfinity,
            MinKlDivergence = double.PositiveInfinity,
            Top1AgreementRate = 0.0,
            Top1Matches = 0,
            TokenCount = tokenCount,
            VocabSize = vocabSize,
            Threshold = opts.Threshold,
            MaxKlThreshold = opts.MaxKlThreshold,
            MinTop1Agreement = opts.MinTop1Agreement,
            HasNonFiniteQuantizedValues = true,
            Passed = false,
            FailureReason = reason
        };
    }
}
