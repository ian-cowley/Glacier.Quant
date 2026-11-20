namespace Glacier.Quant.Tests.E2E;

using System;
using System.Runtime.CompilerServices;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Kernels;
using Xunit;

[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier2")]
public sealed class KlBoundaryE2ETests
{
    [Theory]
    [InlineData(10.0f)]
    [InlineData(1000.0f)]
    [InlineData(-1000.0f)]
    public void T2_F10_01_LogitTranslationInvariance_RemainsExactAndOverflowFree(float shift)
    {
        float[] refLogits = [2.0f, 1.0f, 0.0f, -1.0f];
        float[] shiftedLogits = new float[refLogits.Length];
        for (int i = 0; i < refLogits.Length; i++)
            shiftedLogits[i] = refLogits[i] + shift;

        KlDivergenceResult result = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refLogits, shiftedLogits);

        Assert.True(result.Passed);
        Assert.True(double.IsFinite(result.MeanKlDivergence));
        Assert.InRange(result.MeanKlDivergence, 0.0, 1e-5);
        Assert.Equal(1.0, result.Top1AgreementRate, precision: 12);

        Span<float> probs = stackalloc float[refLogits.Length];
        KlDivergenceEvaluator.ComputeSoftmax(shiftedLogits, probs);
        double probSum = 0.0;
        for (int i = 0; i < probs.Length; i++)
        {
            Assert.True(float.IsFinite(probs[i]) && probs[i] >= 0.0f);
            probSum += probs[i];
        }
        Assert.InRange(Math.Abs(probSum - 1.0), 0.0, 1e-5);
    }

    [Fact]
    public void T2_F10_02_NearZeroProbabilitiesAndMaskedNegativeInfinity_RemainFinite()
    {
        float[] refLogits = [100.0f, -100.0f, -100.0f, -100.0f];
        float[] quantLogits = [100.0f, -1000.0f, -1000.0f, -1000.0f];

        KlDivergenceResult extremeResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refLogits, quantLogits);
        Assert.True(double.IsFinite(extremeResult.MeanKlDivergence));
        Assert.InRange(extremeResult.MeanKlDivergence, 0.0, 1e-5);
        Assert.True(extremeResult.Passed);

        // Masked -Infinity token in Q where P also has -Infinity
        float[] maskedRef = [2.0f, float.NegativeInfinity, 0.0f, float.NegativeInfinity];
        float[] maskedQuant = [2.0f, float.NegativeInfinity, 0.0f, float.NegativeInfinity];
        double maskedKl = KlDivergenceEvaluator.ComputeTokenKlDivergence(maskedRef, maskedQuant);
        Assert.True(double.IsFinite(maskedKl));
        Assert.Equal(0.0, maskedKl, precision: 12);

        // Probability input with zero entry in Q where P > 0 uses Epsilon floor without NaN/Inf
        float[] p = [0.9f, 0.1f, 0.0f, 0.0f];
        float[] qZeroSupport = [1.0f, 0.0f, 0.0f, 0.0f];
        KlDivergenceResult probFloorResult = KlDivergenceEvaluator.EvaluateProbabilities(
            p,
            qZeroSupport,
            vocabSize: 4,
            new KlDivergenceOptions { Threshold = 100.0, Epsilon = 1e-12 });
        Assert.True(double.IsFinite(probFloorResult.MeanKlDivergence));
        Assert.True(probFloorResult.MeanKlDivergence > 1.0);
    }

    [Fact]
    public void T2_F10_03_SingleElementDistributionPassesAndEmptySpanThrows()
    {
        float[] singleRef = [3.5f];
        float[] singleQuant = [-99.0f];

        KlDivergenceResult singleResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(singleRef, singleQuant);
        Assert.True(singleResult.Passed);
        Assert.Equal(0.0, singleResult.MeanKlDivergence, precision: 12);
        Assert.Equal(1.0, singleResult.Top1AgreementRate, precision: 12);

        Assert.Throws<ArgumentException>(() =>
            KlDivergenceEvaluator.ComputeTokenKlDivergence(ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty));
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty));
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceEvaluator.EvaluateLogits(ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, vocabSize: 1));
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceEvaluator.EvaluateProbabilities(ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, vocabSize: 1));
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceEvaluator.ArgMax(ReadOnlySpan<float>.Empty));
    }

    [Fact]
    public void T2_F10_04_MismatchedLengthsAndInvalidShapesOrOptions_ThrowCleanly()
    {
        float[] three = [1.0f, 2.0f, 3.0f];
        float[] four = [1.0f, 2.0f, 3.0f, 4.0f];

        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence(three, four));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateSingleTokenLogits(three, four));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateLogits(three, four, vocabSize: 3));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateLogits(four, four, vocabSize: 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.EvaluateLogits(four, four, vocabSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.EvaluateLogits(four, four, vocabSize: -2));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(four, four, threshold: -0.01));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(four, four, epsilon: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(four, four, minTop1Agreement: 1.1));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void T2_F10_05_NonFiniteLogits_AreRejectedWithArgumentException(float badValue)
    {
        float[] valid = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] bad = [1.0f, badValue, 3.0f, 4.0f];

        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence(bad, valid));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence(valid, bad));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateSingleTokenLogits(valid, bad));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateLogits(valid, bad, vocabSize: 4));

        float[] allNegInf = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence(allNegInf, valid));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence(valid, allNegInf));
    }

    [Fact]
    public void T2_CorruptedQuantizedBlockScalesAndNonFiniteWeights_CaughtAsFailedQualityResult()
    {
        const int numTokens = 2;
        const int vocabSize = 2;
        const int hiddenDim = 256;
        int totalWeights = vocabSize * hiddenDim;

        float[] refWeights = new float[totalWeights];
        float[] activations = new float[numTokens * hiddenDim];
        for (int i = 0; i < totalWeights; i++)
            refWeights[i] = MathF.Sin(i * 0.11f) * 0.5f;
        for (int i = 0; i < activations.Length; i++)
            activations[i] = MathF.Cos(i * 0.07f) * 0.5f;

        // 1. Corrupted Q8_0 block scale (NaN)
        var q8Blocks = new BlockQ8_0[totalWeights / BlockQ8_0.BlockSize];
        SimdQuantizer.QuantizeRowQ8_0(refWeights, q8Blocks);
        q8Blocks[0].Scale = (Half)float.NaN;

        KlDivergenceResult q8Corrupt = QuantQualityChecker.EvaluateBlocksQ8_0(
            refWeights, q8Blocks, activations, numTokens, vocabSize, hiddenDim);
        Assert.False(q8Corrupt.Passed);
        Assert.True(q8Corrupt.HasNonFiniteQuantizedValues);
        Assert.True(double.IsPositiveInfinity(q8Corrupt.MeanKlDivergence));
        Assert.NotNull(q8Corrupt.FailureReason);

        // 2. Corrupted Q4_K super-block scale (+Infinity)
        var q4Blocks = new BlockQ4_K[totalWeights / BlockQ4_K.BlockSize];
        SimdQuantizer.QuantizeRowQ4_K(refWeights, q4Blocks);
        q4Blocks[1].D = (Half)float.PositiveInfinity;

        KlDivergenceResult q4Corrupt = QuantQualityChecker.EvaluateBlocksQ4_K(
            refWeights, q4Blocks, activations, numTokens, vocabSize, hiddenDim);
        Assert.False(q4Corrupt.Passed);
        Assert.True(q4Corrupt.HasNonFiniteQuantizedValues);
        Assert.NotNull(q4Corrupt.FailureReason);

        // 3. Corrupted 1-bit block scale (NaN)
        var bitBlocks = new BlockBinary1Bit[totalWeights / BlockBinary1Bit.BlockSize];
        SimdQuantizer.QuantizeRow1Bit(refWeights, bitBlocks);
        bitBlocks[0].Scale = float.NaN;

        KlDivergenceResult bitCorrupt = QuantQualityChecker.EvaluateBlocks1Bit(
            refWeights, bitBlocks, activations, numTokens, vocabSize, hiddenDim);
        Assert.False(bitCorrupt.Passed);
        Assert.True(bitCorrupt.HasNonFiniteQuantizedValues);

        // 4. Corrupted FP8 byte (NaN encoding 0x7F)
        byte[] fp8Bytes = new byte[totalWeights];
        SimdQuantizer.QuantizeRowFP8_E4M3(refWeights, fp8Bytes, out float dynamicScale);
        fp8Bytes[0] = 0x7F;

        KlDivergenceResult fp8Corrupt = QuantQualityChecker.EvaluateFp8E4M3(
            refWeights, fp8Bytes, dynamicScale, activations, numTokens, vocabSize, hiddenDim);
        Assert.False(fp8Corrupt.Passed);
        Assert.True(fp8Corrupt.HasNonFiniteQuantizedValues);

        // 5. Non-finite reference weights must throw ArgumentException instead of silently returning a result
        float[] badRef = (float[])refWeights.Clone();
        badRef[0] = float.NaN;
        Assert.Throws<ArgumentException>(() =>
            QuantQualityChecker.EvaluateWeights(badRef, refWeights, activations, numTokens, vocabSize, hiddenDim));
    }
}
