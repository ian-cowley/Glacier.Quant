namespace Glacier.Quant.Calibration;

using System;
using System.Collections.Generic;

public sealed class CalibrationDataset
{
    public int NumTokens { get; }
    public int HiddenDim { get; }
    public float[] Activations { get; }

    public CalibrationDataset(int numTokens, int hiddenDim, float[] activations)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentNullException.ThrowIfNull(activations);

        if (activations.Length < numTokens * hiddenDim)
            throw new ArgumentException("Activations array length does not match token and dimension counts.");

        NumTokens = numTokens;
        HiddenDim = hiddenDim;
        Activations = activations;
    }

    public ReadOnlySpan<float> GetTokenVector(int tokenIndex)
    {
        return Activations.AsSpan(tokenIndex * HiddenDim, HiddenDim);
    }
}

public sealed class PtqCalibrationResult
{
    public string Method { get; init; } = "";
    public float[] CalibratedWeights { get; init; } = [];
    public float[]? OptimalScales { get; init; }
    public double BaselineRtnMse { get; init; }
    public double CalibratedMse { get; init; }
    public double MseReductionPercent => BaselineRtnMse > 1e-12
        ? Math.Max(0.0, (1.0 - (CalibratedMse / BaselineRtnMse)) * 100.0)
        : 0.0;
}
