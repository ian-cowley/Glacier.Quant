namespace Glacier.Quant.Tests.E2E;

using System;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Kernels;
using Xunit;

[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier1")]
public sealed class KlDivergenceE2ETests
{
    [Fact]
    public void T1_F10_01_IdenticalLogitsAndProbabilities_ProduceZeroKlAndFullTop1Agreement()
    {
        float[] refLogits = [2.0f, 1.0f, 0.0f, -1.0f];
        float[] quantLogits = [2.0f, 1.0f, 0.0f, -1.0f];

        double tokenKl = KlDivergenceEvaluator.ComputeTokenKlDivergence(refLogits, quantLogits);
        Assert.Equal(0.0, tokenKl, precision: 12);

        KlDivergenceResult logitResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(refLogits, quantLogits);
        Assert.True(logitResult.Passed);
        Assert.Null(logitResult.FailureReason);
        Assert.False(logitResult.HasNonFiniteQuantizedValues);
        Assert.Equal(1, logitResult.TokenCount);
        Assert.Equal(4, logitResult.VocabSize);
        Assert.Equal(0.0, logitResult.MeanKlDivergence, precision: 12);
        Assert.Equal(0.0, logitResult.MaxKlDivergence, precision: 12);
        Assert.Equal(0.0, logitResult.MinKlDivergence, precision: 12);
        Assert.Equal(1.0, logitResult.Top1AgreementRate, precision: 12);
        Assert.Equal(1, logitResult.Top1Matches);

        float[] probs = [0.5f, 0.25f, 0.15f, 0.10f];
        KlDivergenceResult probResult = KlDivergenceEvaluator.EvaluateProbabilities(probs, probs, vocabSize: 4);
        Assert.True(probResult.Passed);
        Assert.Equal(0.0, probResult.MeanKlDivergence, precision: 12);
        Assert.Equal(1.0, probResult.Top1AgreementRate, precision: 12);
    }

    [Fact]
    public void T1_F10_02_AnalyticalNonZeroDistribution_MatchesExactQuarterLn2()
    {
        // P = [0.5, 0.25, 0.125, 0.125], Q = [0.25, 0.25, 0.25, 0.25]
        // D_KL(P || Q) = 0.5*ln(2) + 0.25*0 + 2*0.125*(-ln(2)) = 0.25 * ln(2)
        double expectedKl = 0.25 * Math.Log(2.0);

        float[] refProbs = [0.5f, 0.25f, 0.125f, 0.125f];
        float[] quantProbs = [0.25f, 0.25f, 0.25f, 0.25f];

        KlDivergenceResult probResult = KlDivergenceEvaluator.EvaluateProbabilities(
            refProbs,
            quantProbs,
            vocabSize: 4,
            threshold: 0.20);

        Assert.InRange(Math.Abs(probResult.MeanKlDivergence - expectedKl), 0.0, 1e-5);
        Assert.Equal(1.0, probResult.Top1AgreementRate, precision: 12);
        Assert.True(probResult.Passed);

        float[] refLogits = [MathF.Log(4.0f), MathF.Log(2.0f), 0.0f, 0.0f];
        float[] quantLogits = [0.0f, 0.0f, 0.0f, 0.0f];

        double logitKl = KlDivergenceEvaluator.ComputeTokenKlDivergence(refLogits, quantLogits);
        Assert.InRange(Math.Abs(logitKl - expectedKl), 0.0, 1e-5);

        // Asymmetric logit pair to discriminate forward D_KL(P || Q) from reversed D_KL(Q || P):
        // P = [0.9, 0.1], Q = [0.5, 0.5]
        // Forward  D_KL(P || Q) = 0.9*ln(1.8) + 0.1*ln(0.2) ~= 0.3680642
        // Reversed D_KL(Q || P) = 0.5*ln(5/9) + 0.5*ln(5)   ~= 0.5108256
        float[] asymRefLogits = [MathF.Log(9.0f), 0.0f];
        float[] asymQuantLogits = [0.0f, 0.0f];
        double expectedAsymKl = 0.9 * Math.Log(1.8) + 0.1 * Math.Log(0.2);

        double asymLogitKl = KlDivergenceEvaluator.ComputeTokenKlDivergence(asymRefLogits, asymQuantLogits);
        Assert.InRange(Math.Abs(asymLogitKl - expectedAsymKl), 0.0, 1e-5);

        KlDivergenceResult asymLogitResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            asymRefLogits,
            asymQuantLogits,
            threshold: 0.40);
        Assert.True(asymLogitResult.Passed);
        Assert.InRange(Math.Abs(asymLogitResult.MeanKlDivergence - expectedAsymKl), 0.0, 1e-5);
    }

    [Fact]
    public void T1_F10_03_RegressionGate_PassesFaithfulDistributionWithinThreshold_AndEnforcesMaxKl()
    {
        float[] refLogits = [MathF.Log(4.0f), MathF.Log(2.0f), 0.0f, 0.0f];
        float[] quantLogits = [0.0f, 0.0f, 0.0f, 0.0f];

        KlDivergenceResult passResult = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            refLogits,
            quantLogits,
            new KlDivergenceOptions
            {
                Threshold = 0.20,
                MaxKlThreshold = 0.20,
                MinTop1Agreement = 1.0
            });

        Assert.True(passResult.Passed);
        Assert.Null(passResult.FailureReason);
        Assert.Equal(1.0, passResult.Top1AgreementRate, precision: 12);

        KlDivergenceResult strictMaxFail = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            refLogits,
            quantLogits,
            new KlDivergenceOptions
            {
                Threshold = 0.20,
                MaxKlThreshold = 0.17,
                MinTop1Agreement = 1.0
            });

        Assert.False(strictMaxFail.Passed);
        Assert.NotNull(strictMaxFail.FailureReason);
        Assert.Contains("Max KL", strictMaxFail.FailureReason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void T1_F10_04_RegressionGate_FailsDegradedLogitsWithTop1FlipAndHighKl()
    {
        float[] refLogits = [5.0f, 0.0f, 0.0f, 0.0f];
        float[] quantLogits = [0.0f, 5.0f, 0.0f, 0.0f];

        // Exact analytical KL for [5,0,0,0] vs [0,5,0,0]:
        // p0 = e^5 / (e^5 + 3), p1 = 1 / (e^5 + 3), KL = 5 * (e^5 - 1) / (e^5 + 3) ~= 4.86791
        double expectedKl = 5.0 * (Math.Exp(5.0) - 1.0) / (Math.Exp(5.0) + 3.0);

        KlDivergenceResult result = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            refLogits,
            quantLogits,
            new KlDivergenceOptions
            {
                Threshold = 0.05,
                MinTop1Agreement = 0.90
            });

        Assert.False(result.Passed);
        Assert.True(result.MeanKlDivergence > 4.0);
        Assert.InRange(Math.Abs(result.MeanKlDivergence - expectedKl), 0.0, 1e-4);
        Assert.Equal(0.0, result.Top1AgreementRate, precision: 12);
        Assert.Equal(0, result.Top1Matches);
        Assert.NotNull(result.FailureReason);
        Assert.Contains("Mean KL", result.FailureReason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Top-1", result.FailureReason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void T1_F10_05_MultiSampleBatch_AggregatesMeanMaxMinAndTop1RateAccurately()
    {
        // Sample 0: identical -> KL = 0.0, argmax 0 == 0 (match)
        // Sample 1: [ln(4), ln(2), 0, 0] vs [0, 0, 0, 0] -> KL = 0.25*ln(2), argmax 0 == 0 (match)
        // Sample 2: [5, 0, 0, 0] vs [0, 5, 0, 0] -> KL = 5*(e^5-1)/(e^5+3), argmax 0 != 1 (miss)
        float[] refBatch =
        [
            2.0f, 1.0f, 0.0f, -1.0f,
            MathF.Log(4.0f), MathF.Log(2.0f), 0.0f, 0.0f,
            5.0f, 0.0f, 0.0f, 0.0f
        ];
        float[] quantBatch =
        [
            2.0f, 1.0f, 0.0f, -1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 5.0f, 0.0f, 0.0f
        ];

        double kl0 = 0.0;
        double kl1 = 0.25 * Math.Log(2.0);
        double kl2 = 5.0 * (Math.Exp(5.0) - 1.0) / (Math.Exp(5.0) + 3.0);
        double expectedMean = (kl0 + kl1 + kl2) / 3.0;

        KlDivergenceResult batchResult = KlDivergenceEvaluator.EvaluateLogits(refBatch, quantBatch, vocabSize: 4);

        Assert.Equal(3, batchResult.TokenCount);
        Assert.Equal(4, batchResult.VocabSize);
        Assert.Equal(2, batchResult.Top1Matches);
        Assert.InRange(Math.Abs(batchResult.Top1AgreementRate - (2.0 / 3.0)), 0.0, 1e-12);
        Assert.InRange(Math.Abs(batchResult.MinKlDivergence - kl0), 0.0, 1e-6);
        Assert.InRange(Math.Abs(batchResult.MaxKlDivergence - kl2), 0.0, 1e-4);
        Assert.InRange(Math.Abs(batchResult.MeanKlDivergence - expectedMean), 0.0, 1e-4);
        Assert.False(batchResult.Passed);

        // Strictly positive asymmetric KL batch: verifies MinKlDivergence is initialized to +Infinity (not 0.0)
        // Sample A: P=[0.8, 0.2], Q=[0.5, 0.5] -> KL_A = 0.8*ln(1.6) + 0.2*ln(0.4) ~= 0.1927448
        // Sample B: P=[0.9, 0.1], Q=[0.5, 0.5] -> KL_B = 0.9*ln(1.8) + 0.1*ln(0.2) ~= 0.3680642
        float[] posRefBatch = [MathF.Log(4.0f), 0.0f, MathF.Log(9.0f), 0.0f];
        float[] posQuantBatch = [0.0f, 0.0f, 0.0f, 0.0f];
        double expectedKlA = 0.8 * Math.Log(1.6) + 0.2 * Math.Log(0.4);
        double expectedKlB = 0.9 * Math.Log(1.8) + 0.1 * Math.Log(0.2);

        KlDivergenceResult posBatchResult = KlDivergenceEvaluator.EvaluateLogits(
            posRefBatch,
            posQuantBatch,
            vocabSize: 2,
            threshold: 0.50);
        Assert.True(posBatchResult.Passed);
        Assert.InRange(Math.Abs(posBatchResult.MinKlDivergence - expectedKlA), 0.0, 1e-5);
        Assert.InRange(Math.Abs(posBatchResult.MaxKlDivergence - expectedKlB), 0.0, 1e-5);
        Assert.InRange(Math.Abs(posBatchResult.MeanKlDivergence - ((expectedKlA + expectedKlB) * 0.5)), 0.0, 1e-5);
        Assert.Equal(1.0, posBatchResult.Top1AgreementRate, precision: 12);

        float[] posRefProbs = [0.8f, 0.2f, 0.9f, 0.1f];
        float[] posQuantProbs = [0.5f, 0.5f, 0.5f, 0.5f];
        KlDivergenceResult posProbBatch = KlDivergenceEvaluator.EvaluateProbabilities(
            posRefProbs,
            posQuantProbs,
            vocabSize: 2,
            threshold: 0.50);
        Assert.True(posProbBatch.Passed);
        Assert.InRange(Math.Abs(posProbBatch.MinKlDivergence - expectedKlA), 0.0, 1e-5);
        Assert.InRange(Math.Abs(posProbBatch.MaxKlDivergence - expectedKlB), 0.0, 1e-5);
    }

    [Fact]
    public void T1_F10_06_ArgMaxAndKlEvaluation_EnforceLowestIndexTieBreaking()
    {
        Assert.Equal(0, KlDivergenceEvaluator.ArgMax([3.0f, 3.0f, 1.0f]));
        Assert.Equal(1, KlDivergenceEvaluator.ArgMax([1.0f, 3.0f, 3.0f]));

        KlDivergenceResult logitTieMiss = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            [3.0f, 3.0f, 1.0f],
            [3.0f, 3.01f, 1.0f],
            threshold: 1.0);
        Assert.Equal(0, logitTieMiss.Top1Matches);
        Assert.Equal(0.0, logitTieMiss.Top1AgreementRate, precision: 12);

        KlDivergenceResult logitTieMatch = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            [3.0f, 3.0f, 1.0f],
            [3.01f, 3.0f, 1.0f],
            threshold: 1.0);
        Assert.Equal(1, logitTieMatch.Top1Matches);
        Assert.Equal(1.0, logitTieMatch.Top1AgreementRate, precision: 12);

        KlDivergenceResult probTieMiss = KlDivergenceEvaluator.EvaluateProbabilities(
            [0.4f, 0.4f, 0.2f],
            [0.39f, 0.41f, 0.2f],
            vocabSize: 3,
            threshold: 1.0);
        Assert.Equal(0, probTieMiss.Top1Matches);
    }

    [Fact]
    public void T1_QuantQualityChecker_LinearProjectionAndQuantizedVsQ8AndFp32References_BehaveAsExpected()
    {
        const int numTokens = 4;
        const int vocabSize = 8;
        const int hiddenDim = 256; // Aligns cleanly to BlockQ8_0 (32) and BlockQ4_K (256)

        float[] weights = new float[vocabSize * hiddenDim];
        float[] activations = new float[numTokens * hiddenDim];
        float[] bias = new float[vocabSize];

        for (int r = 0; r < vocabSize; r++)
        {
            bias[r] = (r - 3) * 0.05f;
            for (int c = 0; c < hiddenDim; c++)
            {
                weights[r * hiddenDim + c] = MathF.Sin((r + 1) * 0.17f + c * 0.031f) * 0.25f;
            }
        }

        for (int t = 0; t < numTokens; t++)
        {
            for (int c = 0; c < hiddenDim; c++)
            {
                activations[t * hiddenDim + c] = MathF.Cos((t + 1) * 0.23f + c * 0.019f) * 0.5f;
            }
            // Give each token a distinct dominant signal row so top-1 logits are well-separated
            int targetRow = (t * 2) % vocabSize;
            for (int c = 0; c < hiddenDim; c++)
            {
                weights[targetRow * hiddenDim + c] += 0.04f * MathF.Sign(activations[t * hiddenDim + c]);
            }
        }

        // Verify ProjectLogits against independent double-precision reference matmul + bias
        float[] actualLogits = QuantQualityChecker.ProjectLogits(weights, activations, numTokens, vocabSize, hiddenDim, bias);
        for (int t = 0; t < numTokens; t++)
        {
            for (int r = 0; r < vocabSize; r++)
            {
                double expected = bias[r];
                for (int c = 0; c < hiddenDim; c++)
                {
                    expected += (double)weights[r * hiddenDim + c] * activations[t * hiddenDim + c];
                }
                Assert.InRange(Math.Abs(actualLogits[t * vocabSize + r] - expected), 0.0, 1e-4);
            }
        }

        var dataset = new CalibrationDataset(numTokens, hiddenDim, activations);

        // Q8_0 vs FP32 reference: near-zero (but strictly positive) KL and 100% top-1 agreement
        KlDivergenceResult q8VsFp32 = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.01, MinTop1Agreement = 1.0 },
            bias: bias);
        Assert.True(q8VsFp32.Passed);
        Assert.Equal(1.0, q8VsFp32.Top1AgreementRate, precision: 12);
        Assert.True(q8VsFp32.MeanKlDivergence > 0.0);
        Assert.True(q8VsFp32.MeanKlDivergence < 0.01);

        // Q8_0 vs Q8_0 reference: identical quantized reference weights must yield exact 0.0 KL
        KlDivergenceResult q8VsQ8 = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.01, MinTop1Agreement = 1.0 },
            bias: bias);
        Assert.True(q8VsQ8.Passed);
        Assert.Equal(0.0, q8VsQ8.MeanKlDivergence, precision: 12);
        Assert.Equal(0.0, q8VsQ8.MaxKlDivergence, precision: 12);
        Assert.Equal(1.0, q8VsQ8.Top1AgreementRate, precision: 12);

        // FP8_E4M3 vs Q8_0 reference: faithful
        KlDivergenceResult fp8VsQ8 = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.FP8_E4M3,
            dataset,
            vocabSize,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 1.0 },
            bias: bias);
        Assert.True(fp8VsQ8.Passed);
        Assert.Equal(1.0, fp8VsQ8.Top1AgreementRate, precision: 12);

        // Intentionally inverted weights fail the quality gate
        float[] invertedWeights = new float[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            invertedWeights[i] = -weights[i] * 4.0f;

        KlDivergenceResult degraded = QuantQualityChecker.EvaluateWeights(
            weights,
            invertedWeights,
            dataset,
            vocabSize,
            threshold: 0.05,
            bias: bias,
            minTop1Agreement: 1.0);
        Assert.False(degraded.Passed);
        Assert.True(degraded.MeanKlDivergence > 0.05);
        Assert.True(degraded.Top1AgreementRate < 1.0);
    }
}
