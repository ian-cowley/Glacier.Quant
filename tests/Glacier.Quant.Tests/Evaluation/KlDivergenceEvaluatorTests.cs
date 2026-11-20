namespace Glacier.Quant.Tests.Evaluation;

using System;
using Glacier.Quant.Evaluation;
using Xunit;

public sealed class KlDivergenceEvaluatorTests
{
    [Fact]
    public void EvaluateSingleTokenLogits_IdenticalLogits_ReturnsZeroKlAndPasses()
    {
        float[] logits = [1.5f, -0.5f, 3.2f, 0.0f, -2.1f];

        var result = KlDivergenceEvaluator.EvaluateSingleTokenLogits(logits, logits);

        Assert.True(result.Passed);
        Assert.False(result.HasNonFiniteQuantizedValues);
        Assert.Null(result.FailureReason);
        Assert.Equal(0.0, result.MeanKlDivergence, 12);
        Assert.Equal(0.0, result.MaxKlDivergence, 12);
        Assert.Equal(0.0, result.MinKlDivergence, 12);
        Assert.Equal(1.0, result.Top1AgreementRate, 12);
        Assert.Equal(1, result.Top1Matches);
        Assert.Equal(1, result.TokenCount);
        Assert.Equal(logits.Length, result.VocabSize);
    }

    [Theory]
    [InlineData(1000.0f)]
    [InlineData(-1000.0f)]
    [InlineData(10000.0f)]
    [InlineData(-10000.0f)]
    public void ComputeLogSoftmaxAndKl_ShiftInvariance_PreservesPrecision(float shift)
    {
        float[] refLogits = [2.5f, 1.0f, -0.5f, 0.25f];
        float[] quantLogits = [2.4f, 1.1f, -0.4f, 0.20f];

        float[] shiftedRef = new float[refLogits.Length];
        float[] shiftedQuant = new float[quantLogits.Length];
        for (int i = 0; i < refLogits.Length; i++)
        {
            shiftedRef[i] = refLogits[i] + shift;
            shiftedQuant[i] = quantLogits[i] + shift;
        }

        double[] logProbsBase = new double[refLogits.Length];
        double[] logProbsShifted = new double[refLogits.Length];
        KlDivergenceEvaluator.ComputeLogSoftmax(refLogits, logProbsBase);
        KlDivergenceEvaluator.ComputeLogSoftmax(shiftedRef, logProbsShifted);

        for (int i = 0; i < refLogits.Length; i++)
        {
            Assert.Equal(logProbsBase[i], logProbsShifted[i], 3);
        }

        double klBase = KlDivergenceEvaluator.ComputeTokenKlDivergence(refLogits, quantLogits);
        double klShifted = KlDivergenceEvaluator.ComputeTokenKlDivergence(shiftedRef, shiftedQuant);

        Assert.True(double.IsFinite(klBase));
        Assert.True(double.IsFinite(klShifted));
        Assert.Equal(klBase, klShifted, 3);
    }

    [Fact]
    public void ComputeSoftmaxAndLogSoftmax_ExtremeMagnitudes_DoNotOverflowOrUnderflowToNan()
    {
        float[] logits = [-1000.0f, -999.0f, 999.0f, 1000.0f];
        float[] probs = new float[logits.Length];
        double[] logProbs = new double[logits.Length];

        KlDivergenceEvaluator.ComputeSoftmax(logits, probs);
        KlDivergenceEvaluator.ComputeLogSoftmax(logits, logProbs);

        double sum = 0.0;
        for (int i = 0; i < probs.Length; i++)
        {
            Assert.True(float.IsFinite(probs[i]));
            Assert.True(double.IsFinite(logProbs[i]));
            Assert.True(logProbs[i] <= 0.0);
            sum += probs[i];
        }

        Assert.Equal(1.0, sum, 5);
        Assert.Equal(3, KlDivergenceEvaluator.ArgMax(logits));
    }

    [Fact]
    public void EvaluateLogits_NegativeInfinityMasks_HandleZeroMassAndEpsilonFloorSafely()
    {
        // Both masked at index 0 and 3
        float[] refMasked = [float.NegativeInfinity, 2.0f, 1.0f, float.NegativeInfinity];
        float[] quantSameMask = [float.NegativeInfinity, 2.0f, 1.0f, float.NegativeInfinity];

        var sameResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refMasked, quantSameMask);
        Assert.True(sameResult.Passed);
        Assert.Equal(0.0, sameResult.MeanKlDivergence, 12);

        // Quantized masks an active reference token -> floored via epsilon (large finite KL, no NaN/Inf)
        float[] quantExtraMask = [float.NegativeInfinity, 2.0f, float.NegativeInfinity, float.NegativeInfinity];
        var maskedDivergence = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refMasked, quantExtraMask, threshold: 0.05);
        Assert.True(double.IsFinite(maskedDivergence.MeanKlDivergence));
        Assert.True(maskedDivergence.MeanKlDivergence > 1.0);
        Assert.False(maskedDivergence.Passed);
        Assert.NotNull(maskedDivergence.FailureReason);
    }

    [Fact]
    public void EvaluateProbabilities_HandlesUnnormalizedAndZeroEntriesCleanly()
    {
        // Token 0: identical up to scale; Token 1: slight perturbation with zero entry in P
        float[] refProbs =
        [
            0.0f, 2.0f, 6.0f, 2.0f,
            0.0f, 0.7f, 0.3f, 0.0f
        ];
        float[] quantProbs =
        [
            0.0f, 0.2f, 0.6f, 0.2f,
            0.1f, 0.6f, 0.3f, 0.0f
        ];

        var result = KlDivergenceEvaluator.EvaluateProbabilities(refProbs, quantProbs, vocabSize: 4, threshold: 0.1);

        Assert.True(result.Passed);
        Assert.Equal(2, result.TokenCount);
        Assert.Equal(4, result.VocabSize);
        Assert.Equal(0.0, result.MinKlDivergence, 12);
        Assert.True(result.MaxKlDivergence > 0.0);
        Assert.Equal(1.0, result.Top1AgreementRate, 12);

        // P > 0 where Q == 0 triggers epsilon floor without NaN/Inf
        float[] zeroQ = [0.5f, 0.5f, 0.0f, 0.0f];
        float[] p = [0.5f, 0.0f, 0.5f, 0.0f];
        var floorResult = KlDivergenceEvaluator.EvaluateProbabilities(p, zeroQ, vocabSize: 4, threshold: 0.05);
        Assert.True(double.IsFinite(floorResult.MeanKlDivergence));
        Assert.False(floorResult.Passed);
    }

    [Fact]
    public void EvaluateLogits_BatchedAggregationAndThresholdGates_WorkAsExpected()
    {
        // 3 tokens, vocabSize 4
        // Token 0: identical (KL=0, top1 match)
        // Token 1: small drift (KL small, top1 match)
        // Token 2: top-1 flip (KL larger, top1 mismatch)
        float[] refLogits =
        [
            3.0f, 1.0f, 0.0f, -1.0f,
            2.0f, 1.0f, 0.0f, -1.0f,
            2.0f, 1.8f, 0.0f, -1.0f
        ];
        float[] quantLogits =
        [
            3.0f, 1.0f, 0.0f, -1.0f,
            1.9f, 1.1f, 0.0f, -1.0f,
            1.6f, 2.2f, 0.0f, -1.0f
        ];

        var lenient = KlDivergenceEvaluator.EvaluateLogits(
            refLogits,
            quantLogits,
            vocabSize: 4,
            new KlDivergenceOptions { Threshold = 0.5, MaxKlThreshold = 1.0, MinTop1Agreement = 0.5 });

        Assert.True(lenient.Passed);
        Assert.Equal(3, lenient.TokenCount);
        Assert.Equal(2, lenient.Top1Matches);
        Assert.Equal(2.0 / 3.0, lenient.Top1AgreementRate, 10);
        Assert.Equal(0.0, lenient.MinKlDivergence, 12);
        Assert.True(lenient.MaxKlDivergence >= lenient.MeanKlDivergence);

        // Fail via MaxKlThreshold even when MeanKl passes
        var failMaxKl = KlDivergenceEvaluator.EvaluateLogits(
            refLogits,
            quantLogits,
            vocabSize: 4,
            new KlDivergenceOptions { Threshold = 0.5, MaxKlThreshold = 0.01, MinTop1Agreement = 0.0 });
        Assert.False(failMaxKl.Passed);
        Assert.Contains("Max KL", failMaxKl.FailureReason);

        // Fail via MinTop1Agreement
        var failTop1 = KlDivergenceEvaluator.EvaluateLogits(
            refLogits,
            quantLogits,
            vocabSize: 4,
            new KlDivergenceOptions { Threshold = 0.5, MinTop1Agreement = 0.9 });
        Assert.False(failTop1.Passed);
        Assert.Contains("Top-1", failTop1.FailureReason);
    }

    [Fact]
    public void ArgMax_TieBreaking_SelectsLowestIndex()
    {
        float[] tied = [1.0f, 4.0f, 4.0f, 4.0f, 2.0f];
        Assert.Equal(1, KlDivergenceEvaluator.ArgMax(tied));

        float[] refLogits = [1.0f, 4.0f, 4.0f, 0.0f];
        float[] quantLogits = [1.0f, 4.0f, 4.0001f, 0.0f];
        var res = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refLogits, quantLogits, threshold: 0.5);
        Assert.Equal(0, res.Top1Matches); // ref argmax=1, quant argmax=2
        Assert.Equal(0.0, res.Top1AgreementRate, 12);
    }

    [Fact]
    public void InvalidInputs_ThrowArgumentExceptions()
    {
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([], []));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([1f, 2f], [1f]));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([float.NaN, 1f], [1f, 1f]));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([1f, 1f], [float.PositiveInfinity, 1f]));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([float.NegativeInfinity, float.NegativeInfinity], [1f, 1f]));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([1f, 1f], [float.NegativeInfinity, float.NegativeInfinity]));

        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.EvaluateLogits([1f, 2f], [1f, 2f], vocabSize: 0));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateLogits([1f, 2f, 3f], [1f, 2f, 3f], vocabSize: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.EvaluateSingleTokenLogits([1f, 2f], [1f, 2f], threshold: -0.01));
        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.EvaluateSingleTokenLogits([1f, 2f], [1f, 2f], threshold: double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => KlDivergenceEvaluator.ComputeTokenKlDivergence([1f, 2f], [1f, 2f], epsilon: 0.0));

        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateProbabilities([-0.1f, 1.1f], [0.5f, 0.5f], vocabSize: 2));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.EvaluateProbabilities([0.0f, 0.0f], [0.5f, 0.5f], vocabSize: 2));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ArgMax([]));
        Assert.Throws<ArgumentException>(() => KlDivergenceEvaluator.ArgMax([1.0f, float.NaN]));
    }
}
