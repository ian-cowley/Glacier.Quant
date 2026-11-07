namespace Glacier.Quant.Tests.Calibration;

using System;
using Glacier.Quant.Calibration;
using Xunit;

public class PtqCalibrationTests
{
    [Fact]
    public void AwqCalibrator_ReducesOutputMse_OverUncalibratedRtn()
    {
        int rows = 16;
        int cols = 32;
        int tokens = 30;

        // Weights with uniform sinusoidal distribution
        float[] weights = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                weights[r * cols + c] = MathF.Sin((r + 1) * (c + 1) * 0.1f) * 0.5f;
            }
        }

        // Activations where column 5 has salient outlier magnitude (25.0 vs 0.2)
        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                acts[t * cols + c] = (c == 5) ? (25.0f + (t * 0.2f)) : (0.2f + (c * 0.01f));
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 10, targetBits: 4);

        Assert.NotNull(result);
        Assert.NotNull(result.CalibratedWeights);
        Assert.NotNull(result.OptimalScales);
        Assert.Contains("AWQ", result.Method);

        // Optimal scale for column 5 should be larger than 1.0 to protect its precision
        Assert.True(result.OptimalScales![5] > 1.0f);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-9);
    }

    [Fact]
    public void GptqCalibrator_ExecutesHessianInversion_AndRank1ErrorCompensation()
    {
        int rows = 8;
        int cols = 16;
        int tokens = 20;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (i % 7) * 0.25f - 0.75f;

        float[] acts = new float[tokens * cols];
        for (int i = 0; i < acts.Length; i++) acts[i] = (float)Math.Sin(i * 0.2);

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.NotNull(result);
        Assert.Equal(weights.Length, result.CalibratedWeights.Length);
        Assert.Contains("GPTQ", result.Method);
    }

    [Fact]
    public void ImatrixCalibrator_MinimizesImportanceWeightedMse()
    {
        int rows = 8;
        int cols = 32;
        int tokens = 25;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)Math.Cos(i * 0.1) * 1.5f;

        // Create activations where columns 0..3 have massive second moments
        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                acts[t * cols + c] = (c < 4) ? 8.0f : 0.5f;
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.NotNull(result);
        Assert.NotNull(result.OptimalScales);
        Assert.Contains("imatrix", result.Method);

        // Columns 0..3 should have highest importance scores
        for (int c = 0; c < 4; c++)
        {
            Assert.True(result.OptimalScales![c] > 1.0f);
        }

        // Calibrated weighted MSE should be <= baseline
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-9);
    }
}
