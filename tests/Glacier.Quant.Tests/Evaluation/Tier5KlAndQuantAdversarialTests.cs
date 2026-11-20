using System;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Kernels;
using Xunit;

namespace Glacier.Quant.Tests.Evaluation;

[Trait("Tier", "Tier5")]
public sealed class Tier5KlAndQuantAdversarialTests
{
    [Fact]
    public void T5_R5a_EvaluatePtqResult_AwqZeroAndNegativeOptimalScales_HandledSafely()
    {
        const int vocabSize = 2;
        const int hiddenDim = 32;
        const int numTokens = 2;
        float[] refW = new float[vocabSize * hiddenDim];
        float[] acts = new float[numTokens * hiddenDim];
        for (int i = 0; i < refW.Length; i++) refW[i] = MathF.Sin(i * 0.2f) * 0.5f;
        for (int i = 0; i < acts.Length; i++) acts[i] = MathF.Cos(i * 0.1f) * 0.5f;
        var dataset = new CalibrationDataset(numTokens, hiddenDim, acts);

        float[] zeroScales = new float[hiddenDim];
        Array.Fill(zeroScales, 1.0f);
        zeroScales[0] = 0.0f;

        var ptqZero = new PtqCalibrationResult
        {
            Method = "AWQ",
            CalibratedWeights = (float[])refW.Clone(),
            OptimalScales = zeroScales
        };

        KlDivergenceResult resZero = QuantQualityChecker.EvaluatePtqResult(refW, ptqZero, dataset, vocabSize, postCalibrationQuantType: QuantType.Q8_0);
        Assert.False(resZero.Passed);
        Assert.True(resZero.HasNonFiniteQuantizedValues);

        float[] negScales = new float[hiddenDim];
        Array.Fill(negScales, 1.0f);
        negScales[0] = -2.0f;
        var ptqNeg = new PtqCalibrationResult
        {
            Method = "AWQ",
            CalibratedWeights = (float[])refW.Clone(),
            OptimalScales = negScales
        };

        KlDivergenceResult resNeg = QuantQualityChecker.EvaluatePtqResult(refW, ptqNeg, dataset, vocabSize, postCalibrationQuantType: QuantType.Q8_0);
        Assert.True(resNeg.Passed);
        Assert.False(resNeg.HasNonFiniteQuantizedValues);
        Assert.InRange(resNeg.MeanKlDivergence, 0.0, 1e-4);

        var ptqWrongLen = new PtqCalibrationResult
        {
            Method = "AWQ",
            CalibratedWeights = (float[])refW.Clone(),
            OptimalScales = new float[hiddenDim - 1]
        };
        Assert.Throws<ArgumentException>(() =>
            QuantQualityChecker.EvaluatePtqResult(refW, ptqWrongLen, dataset, vocabSize, postCalibrationQuantType: QuantType.Q8_0));
    }

    [Fact]
    public void T5_R5a_KlDivergenceEvaluator_ZeroAndInfiniteThresholds_AndInvalidOptions()
    {
        float[] tied = { 2.0f, 2.0f, 1.0f, 0.0f };
        var zeroThresh = KlDivergenceEvaluator.EvaluateSingleTokenLogits(tied, tied, threshold: 0.0);
        Assert.True(zeroThresh.Passed);
        Assert.Equal(0.0, zeroThresh.MeanKlDivergence, 12);

        var infThresh = KlDivergenceEvaluator.EvaluateSingleTokenLogits(
            new[] { 10.0f, 0.0f, 0.0f, 0.0f },
            new[] { 0.0f, 0.0f, 0.0f, 10.0f },
            new KlDivergenceOptions { Threshold = double.PositiveInfinity, MaxKlThreshold = double.PositiveInfinity, MinTop1Agreement = 0.0 });
        Assert.True(infThresh.Passed);
        Assert.True(double.IsFinite(infThresh.MeanKlDivergence));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(tied, tied, new KlDivergenceOptions { Threshold = double.NegativeInfinity }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            KlDivergenceEvaluator.EvaluateSingleTokenLogits(tied, tied, new KlDivergenceOptions { MaxKlThreshold = -0.1 }));
    }

    [Fact]
    public void T5_R5a_QuantQualityChecker_ZeroActivationsAwq_AndSingleVocabBoundary()
    {
        const int vocabSize = 2;
        const int hiddenDim = 128;
        float[] refW = new float[vocabSize * hiddenDim];
        for (int i = 0; i < refW.Length; i++) refW[i] = ((i % 5) - 2) * 0.25f;
        var zeroDataset = new CalibrationDataset(2, hiddenDim, new float[2 * hiddenDim]);

        KlDivergenceResult awqZero = QuantQualityChecker.EvaluateAwqCalibrated(refW, vocabSize, zeroDataset, QuantType.Q4_K_M);
        Assert.False(awqZero.HasNonFiniteQuantizedValues);
        Assert.True(double.IsFinite(awqZero.MeanKlDivergence));

        float[] singleW = new float[32];
        Array.Fill(singleW, 0.1f);
        float[] singleA = new float[32];
        Array.Fill(singleA, 0.2f);
        KlDivergenceResult singleVocab = QuantQualityChecker.EvaluateWeights(singleW, singleW, singleA, numTokens: 1, vocabSize: 1, hiddenDim: 32);
        Assert.True(singleVocab.Passed);
        Assert.Equal(0.0, singleVocab.MeanKlDivergence, 12);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuantQualityChecker.EvaluateWeights(singleW, singleW, Array.Empty<float>(), numTokens: 0, vocabSize: 1, hiddenDim: 32));
    }
}
