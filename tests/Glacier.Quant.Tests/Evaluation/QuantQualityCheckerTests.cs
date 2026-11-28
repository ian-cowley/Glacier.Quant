namespace Glacier.Quant.Tests.Evaluation;

using System;
using System.Runtime.CompilerServices;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Gpu;
using Glacier.Quant.Kernels;
using Xunit;

public sealed class QuantQualityCheckerTests
{
    private static (float[] Weights, CalibrationDataset Dataset) CreateDeterministicFixture(
        int vocabSize,
        int hiddenDim,
        int numTokens,
        float weightScale = 0.35f,
        int seed = 42)
    {
        var rng = new Random(seed);
        float[] weights = new float[vocabSize * hiddenDim];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * weightScale;
        }

        float[] activations = new float[numTokens * hiddenDim];
        for (int i = 0; i < activations.Length; i++)
        {
            activations[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return (weights, new CalibrationDataset(numTokens, hiddenDim, activations));
    }

    [Fact]
    public void ProjectLogits_MatchesManualDotProductWithBias()
    {
        float[] weights =
        [
            1.0f, -2.0f, 3.0f,
            0.5f,  4.0f, -1.0f
        ];
        float[] activations =
        [
            2.0f, 1.0f, -1.0f,
            0.0f, 3.0f, 2.0f
        ];
        float[] bias = [0.25f, -0.5f];

        float[] logits = QuantQualityChecker.ProjectLogits(weights, activations, numTokens: 2, vocabSize: 2, hiddenDim: 3, bias);

        // Token 0: r0 = 2 - 2 - 3 + 0.25 = -2.75; r1 = 1 + 4 + 1 - 0.5 = 5.5
        // Token 1: r0 = 0 - 6 + 6 + 0.25 = 0.25; r1 = 0 + 12 - 2 - 0.5 = 9.5
        Assert.Equal(-2.75f, logits[0], 5);
        Assert.Equal(5.50f, logits[1], 5);
        Assert.Equal(0.25f, logits[2], 5);
        Assert.Equal(9.50f, logits[3], 5);
    }

    [Fact]
    public void QuantizeAndDequantize_UnsupportedAndMisalignedInputs_ThrowExpectedExceptions()
    {
        float[] unaligned = new float[33];
        Assert.Throws<ArgumentException>(() => QuantQualityChecker.QuantizeAndDequantize(unaligned, QuantType.Q8_0));
        Assert.Throws<ArgumentException>(() => QuantQualityChecker.QuantizeAndDequantize(new float[128], QuantType.Q4_K_M));
        Assert.Throws<ArgumentException>(() => QuantQualityChecker.QuantizeAndDequantize(new float[32], QuantType.Binary1Bit));

        float[] aligned = new float[256];
        Assert.Throws<NotSupportedException>(() => QuantQualityChecker.QuantizeAndDequantize(aligned, QuantType.FP16));
        Assert.Throws<NotSupportedException>(() => QuantQualityChecker.QuantizeAndDequantize(aligned, QuantType.FP8_E5M2));
    }

    [Fact]
    public void EvaluateQuantization_Q8_0_And_FP8_AgainstFp32_PassWithLowDivergence()
    {
        const int vocabSize = 8;
        const int hiddenDim = 256;
        const int numTokens = 16;
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 0.25f, seed: 101);

        var q8Result = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            QuantType.FP32,
            new KlDivergenceOptions { Threshold = 0.01, MinTop1Agreement = 0.95 });

        Assert.True(q8Result.Passed);
        Assert.False(q8Result.HasNonFiniteQuantizedValues);
        Assert.True(q8Result.MeanKlDivergence < 0.005);
        Assert.Equal(1.0, q8Result.Top1AgreementRate, 12);

        var fp8Result = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.FP8_E4M3,
            dataset,
            vocabSize,
            QuantType.FP32,
            new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.90 });

        Assert.True(fp8Result.Passed);
        Assert.True(fp8Result.MeanKlDivergence < 0.02);
    }

    [Fact]
    public void EvaluateQuantization_Q4_K_M_AgainstFp32AndQ8_0_PassesForWellScaledWeights()
    {
        const int vocabSize = 8;
        const int hiddenDim = 256;
        const int numTokens = 16;
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 0.20f, seed: 202);

        var vsFp32 = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q4_K_M,
            dataset,
            vocabSize,
            QuantType.FP32,
            new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.85 });

        var vsQ8 = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q4_K_M,
            dataset,
            vocabSize,
            QuantType.Q8_0,
            new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.85 });

        Assert.True(vsFp32.Passed);
        Assert.True(vsQ8.Passed);
    }

    [Fact]
    public void EvaluateCalibrators_AwqGptqImatrix_PassAgainstFp32Reference()
    {
        const int vocabSize = 8;
        const int hiddenDim = 64;
        const int numTokens = 16;

        // O(1) scale weights for PTQ calibrator evaluation (AWQ, GPTQ, imatrix)
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 1.0f, seed: 303);

        // Amplify two salient channels in activations to exercise AWQ channel scaling
        for (int t = 0; t < numTokens; t++)
        {
            dataset.Activations[t * hiddenDim + 3] *= 6.0f;
            dataset.Activations[t * hiddenDim + 17] *= 5.0f;
        }

        // 1. AWQ direct + AWQ preserving OptimalScales around Q4_K_M SimdQuantizer round-trip (8*64 = 512 % 256 == 0)
        var awqDirect = QuantQualityChecker.EvaluateAwqCalibrated(
            weights,
            vocabSize,
            dataset,
            targetType: null,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.15 });
        Assert.True(awqDirect.Passed);

        var awqQ4K = QuantQualityChecker.EvaluateAwqCalibrated(
            weights,
            vocabSize,
            dataset,
            targetType: QuantType.Q4_K_M,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.25 });
        Assert.True(awqQ4K.Passed);

        // 2. GPTQ (Cholesky blockwise calibration with full-rank calibration tokens)
        var (gptqWeights, gptqRawDataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens: 128, weightScale: 1.5f, seed: 304);
        float[] scaledActs = new float[gptqRawDataset.Activations.Length];
        for (int i = 0; i < scaledActs.Length; i++)
            scaledActs[i] = gptqRawDataset.Activations[i] * 0.25f;
        var gptqDataset = new CalibrationDataset(128, hiddenDim, scaledActs);

        var gptq = GptqCalibrator.Calibrate(gptqWeights, vocabSize, hiddenDim, gptqDataset);
        var gptqEval = QuantQualityChecker.EvaluatePtqResult(
            gptqWeights,
            gptq,
            gptqDataset,
            vocabSize,
            options: new KlDivergenceOptions { Threshold = 0.05 });
        Assert.True(gptqEval.Passed, gptqEval.FailureReason);

        // 3. Imatrix
        var imatrix = ImatrixCalibrator.Calibrate(weights, vocabSize, hiddenDim, dataset);
        var imatrixEval = QuantQualityChecker.EvaluatePtqResult(
            weights,
            imatrix,
            dataset,
            vocabSize,
            options: new KlDivergenceOptions { Threshold = 0.05 });
        Assert.True(imatrixEval.Passed);
    }

    [Fact]
    public void EvaluateBlocksQ8_0_CorruptedScalesAndNan_FailCleanly()
    {
        const int vocabSize = 8;
        const int hiddenDim = 64;
        const int numTokens = 12;
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 0.4f, seed: 404);

        int numBlocks = weights.Length / BlockQ8_0.BlockSize;
        var cleanBlocks = new BlockQ8_0[numBlocks];
        SimdQuantizer.QuantizeRowQ8_0(weights, cleanBlocks);

        var cleanResult = QuantQualityChecker.EvaluateBlocksQ8_0(
            weights,
            cleanBlocks,
            dataset,
            vocabSize,
            new KlDivergenceOptions { Threshold = 0.01 });
        Assert.True(cleanResult.Passed);

        // 1. Zeroed scale across half the blocks
        var zeroBlocks = (BlockQ8_0[])cleanBlocks.Clone();
        for (int b = 0; b < numBlocks / 2; b++)
            zeroBlocks[b].Scale = (Half)0.0f;
        var zeroResult = QuantQualityChecker.EvaluateBlocksQ8_0(weights, zeroBlocks, dataset, vocabSize, new KlDivergenceOptions { Threshold = 0.05 });
        Assert.False(zeroResult.Passed);
        Assert.False(zeroResult.HasNonFiniteQuantizedValues);
        Assert.NotNull(zeroResult.FailureReason);

        // 2. Negated scales
        var negatedBlocks = (BlockQ8_0[])cleanBlocks.Clone();
        for (int b = 0; b < numBlocks; b++)
            negatedBlocks[b].Scale = (Half)(-(float)negatedBlocks[b].Scale);
        var negatedResult = QuantQualityChecker.EvaluateBlocksQ8_0(weights, negatedBlocks, dataset, vocabSize, new KlDivergenceOptions { Threshold = 0.05 });
        Assert.False(negatedResult.Passed);

        // 3. Blown-up 50x scales
        var blownUpBlocks = (BlockQ8_0[])cleanBlocks.Clone();
        for (int b = 0; b < numBlocks; b++)
            blownUpBlocks[b].Scale = (Half)((float)blownUpBlocks[b].Scale * 50.0f);
        var blownUpResult = QuantQualityChecker.EvaluateBlocksQ8_0(weights, blownUpBlocks, dataset, vocabSize, new KlDivergenceOptions { Threshold = 0.05 });
        Assert.False(blownUpResult.Passed);

        // 4. Half.NaN and Half.PositiveInfinity scales -> failed KlDivergenceResult with HasNonFiniteQuantizedValues = true
        var nanBlocks = (BlockQ8_0[])cleanBlocks.Clone();
        nanBlocks[0].Scale = Half.NaN;
        var nanResult = QuantQualityChecker.EvaluateBlocksQ8_0(weights, nanBlocks, dataset, vocabSize);
        Assert.False(nanResult.Passed);
        Assert.True(nanResult.HasNonFiniteQuantizedValues);
        Assert.True(double.IsPositiveInfinity(nanResult.MeanKlDivergence));
        Assert.True(double.IsPositiveInfinity(nanResult.MaxKlDivergence));
        Assert.Equal(0.0, nanResult.Top1AgreementRate);
        Assert.NotNull(nanResult.FailureReason);

        var infBlocks = (BlockQ8_0[])cleanBlocks.Clone();
        infBlocks[0].Scale = Half.PositiveInfinity;
        var infResult = QuantQualityChecker.EvaluateBlocksQ8_0(weights, infBlocks, dataset, vocabSize);
        Assert.False(infResult.Passed);
        Assert.True(infResult.HasNonFiniteQuantizedValues);
    }

    [Fact]
    public void EvaluateBlocksQ4_K_And_1Bit_And_Fp8_CorruptionAndBufferRoundTrip_WorkCleanly()
    {
        const int vocabSize = 8;
        const int hiddenDim = 256;
        const int numTokens = 10;
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 0.3f, seed: 505);

        // Q4_K block corruption (D / DMin)
        int numQ4Blocks = weights.Length / BlockQ4_K.BlockSize;
        var q4Blocks = new BlockQ4_K[numQ4Blocks];
        SimdQuantizer.QuantizeRowQ4_K(weights, q4Blocks);

        var corruptedQ4 = (BlockQ4_K[])q4Blocks.Clone();
        for (int b = 0; b < numQ4Blocks; b++)
        {
            corruptedQ4[b].D = (Half)(-(float)corruptedQ4[b].D * 20.0f);
            corruptedQ4[b].DMin = (Half)(-(float)corruptedQ4[b].DMin * 20.0f);
        }
        var q4CorruptResult = QuantQualityChecker.EvaluateBlocksQ4_K(weights, corruptedQ4, dataset, vocabSize, new KlDivergenceOptions { Threshold = 0.05 });
        Assert.False(q4CorruptResult.Passed);

        corruptedQ4[0].D = Half.NaN;
        var q4NanResult = QuantQualityChecker.EvaluateBlocksQ4_K(weights, corruptedQ4, dataset, vocabSize);
        Assert.False(q4NanResult.Passed);
        Assert.True(q4NanResult.HasNonFiniteQuantizedValues);

        // 1-bit block corruption
        int num1BitBlocks = weights.Length / BlockBinary1Bit.BlockSize;
        var bitBlocks = new BlockBinary1Bit[num1BitBlocks];
        SimdQuantizer.QuantizeRow1Bit(weights, bitBlocks);
        bitBlocks[0].Scale = float.NaN;
        var bitNanResult = QuantQualityChecker.EvaluateBlocks1Bit(weights, bitBlocks, dataset, vocabSize);
        Assert.False(bitNanResult.Passed);
        Assert.True(bitNanResult.HasNonFiniteQuantizedValues);

        // FP8 corruption
        byte[] fp8Bytes = new byte[weights.Length];
        SimdQuantizer.QuantizeRowFP8_E4M3(weights, fp8Bytes, out float dynScale);
        var fp8BadScale = QuantQualityChecker.EvaluateFp8E4M3(weights, fp8Bytes, float.NaN, dataset, vocabSize);
        Assert.False(fp8BadScale.Passed);
        Assert.True(fp8BadScale.HasNonFiniteQuantizedValues);

        // Symmetric GpuQuantizationDispatcher buffer round-trip (Q8_0)
        int byteCount = GpuQuantizationDispatcher.GetQuantizedByteCount(weights.Length, QuantType.Q8_0);
        Assert.Equal(numQ4Blocks * 256 / 32 * Unsafe.SizeOf<BlockQ8_0>(), byteCount);
        byte[] q8Buffer = new byte[byteCount];
        GpuQuantizationDispatcher.QuantizeTensor(weights, QuantType.Q8_0, q8Buffer, out float q8Scale);
        var bufResult = QuantQualityChecker.EvaluateQuantizedBuffer(
            weights,
            q8Buffer,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            dynamicScale: q8Scale,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.01 });
        Assert.True(bufResult.Passed);
    }

    [Fact]
    public void EvaluateWeights_ZeroedAndNoisyWeights_FailThreshold_AndNonFiniteReferenceThrows()
    {
        const int vocabSize = 6;
        const int hiddenDim = 32;
        const int numTokens = 8;
        var (weights, dataset) = CreateDeterministicFixture(vocabSize, hiddenDim, numTokens, weightScale: 0.6f, seed: 606);

        // Zeroed weights collapse to uniform logits
        float[] zeroWeights = new float[weights.Length];
        var zeroResult = QuantQualityChecker.EvaluateWeights(weights, zeroWeights, dataset, vocabSize, threshold: 0.05);
        Assert.False(zeroResult.Passed);
        Assert.True(zeroResult.MeanKlDivergence > 0.1);

        // Excessive noise
        var rng = new Random(707);
        float[] noisyWeights = (float[])weights.Clone();
        for (int i = 0; i < noisyWeights.Length; i++)
            noisyWeights[i] += (float)(rng.NextDouble() * 4.0 - 2.0);
        var noisyResult = QuantQualityChecker.EvaluateWeights(weights, noisyWeights, dataset, vocabSize, threshold: 0.05);
        Assert.False(noisyResult.Passed);

        // Non-finite reference weights or activations must throw ArgumentException
        float[] badRefWeights = (float[])weights.Clone();
        badRefWeights[5] = float.NaN;
        Assert.Throws<ArgumentException>(() => QuantQualityChecker.EvaluateWeights(badRefWeights, weights, dataset, vocabSize));

        float[] badActs = (float[])dataset.Activations.Clone();
        badActs[2] = float.PositiveInfinity;
        var badDataset = new CalibrationDataset(numTokens, hiddenDim, badActs);
        Assert.Throws<ArgumentException>(() => QuantQualityChecker.EvaluateWeights(weights, weights, badDataset, vocabSize));
    }
}
