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

    [Theory]
    [InlineData(1e-4f)]
    [InlineData(1e3f)]
    public void GptqCalibrator_TinyAndLargeWeightScales_BeatBaselineRtnMse(float weightScale)
    {
        const int rows = 8;
        const int cols = 64;
        const int tokens = 96;
        var rng = new Random(1337);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * weightScale;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            float shared = (float)(rng.NextDouble() * 2.0 - 1.0);
            for (int c = 0; c < cols; c++)
                acts[t * cols + c] = 0.6f * shared + 0.4f * (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse));
        Assert.True(double.IsFinite(result.CalibratedMse));
        Assert.True(result.BaselineRtnMse > 0.0);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-12);
    }

    [Theory]
    [InlineData(2, 4, 16, true)]
    [InlineData(3, 16, 32, false)]
    [InlineData(4, 128, -1, true)]
    [InlineData(8, 32, 32, false)]
    public void GptqCalibrator_ParameterizedTargetBitsAndBlockSizes_ValidatesAndConverges(
        int targetBits,
        int blockSize,
        int groupSize,
        bool symmetric)
    {
        const int rows = 6;
        const int cols = 48;
        const int tokens = 64;
        var rng = new Random(2026 + targetBits);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f + 0.1f;

        float[] acts = new float[tokens * cols];
        for (int i = 0; i < acts.Length; i++)
            acts[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset, damping: 0.01f, targetBits: targetBits, blockSize: blockSize, groupSize: groupSize, symmetric: symmetric);

        Assert.Contains("GPTQ", result.Method, StringComparison.Ordinal);
        Assert.Equal(weights.Length, result.CalibratedWeights.Length);
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-9);
    }

    [Fact]
    public void AwqCalibrator_MoreThan16Tokens_UsesTailTokensBeyond16()
    {
        const int rows = 12;
        const int cols = 32;
        const int tokens = 48;

        float[] weights = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                weights[r * cols + c] = MathF.Sin((r + 1) * (c + 2) * 0.17f) * 0.8f;
            }
        }

        // First 16 tokens are tiny uniform noise; tokens 16..47 carry a huge salient outlier on channel 11
        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (t < 16)
                    acts[t * cols + c] = 0.05f * (((t + c) % 3) - 1);
                else
                    acts[t * cols + c] = (c == 11) ? 45.0f + (t - 16) * 0.5f : 0.15f;
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 20, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.True(result.OptimalScales![11] > 1.0f, $"Expected channel 11 scale > 1.0, got {result.OptimalScales[11]}");
        Assert.True(result.CalibratedMse < result.BaselineRtnMse, $"Expected calibrated MSE ({result.CalibratedMse}) < baseline ({result.BaselineRtnMse})");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(8)]
    public void ImatrixCalibrator_ParameterizedTargetBits_BeatsUnweightedRtn(int targetBits)
    {
        const int rows = 8;
        const int cols = 64;
        const int tokens = 24;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = MathF.Sin(i * 0.13f) * 1.4f;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
                acts[t * cols + c] = (c < 8) ? 10.0f : 0.25f;
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset, targetBits);

        Assert.Contains($"{targetBits}-bit", result.Method, StringComparison.Ordinal);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-12);

        Assert.Throws<ArgumentOutOfRangeException>(() => ImatrixCalibrator.Calibrate(weights, rows, cols, dataset, targetBits: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImatrixCalibrator.Calibrate(weights, rows, cols, dataset, targetBits: 9));
    }

    [Theory]
    [InlineData(1.0f, 2.0f)]
    [InlineData(-2.0f, -1.0f)]
    public void GptqCalibrator_AsymmetricSingleSignGroup_PreservesFullPrecisionAndBeatsRtn(float wMin, float wMax)
    {
        const int rows = 4;
        const int cols = 64;
        const int tokens = 256;
        const int groupSize = 32;
        var rng = new Random(wMin > 0 ? 4101 : 4102);

        float[] weights = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int inGroup = c % groupSize;
                float t = inGroup / (float)(groupSize - 1);
                weights[r * cols + c] = wMin + t * (wMax - wMin);
            }
        }

        float[] acts = new float[tokens * cols];
        for (int i = 0; i < acts.Length; i++)
            acts[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(
            weights,
            rows,
            cols,
            dataset,
            damping: 0.01f,
            targetBits: 4,
            blockSize: 32,
            groupSize: groupSize,
            symmetric: false);

        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-12, $"GPTQ MSE ({result.CalibratedMse}) exceeded RTN ({result.BaselineRtnMse}).");

        // Verify each 32-element single-sign group retains rich quantization levels (no collapse to 1-2 levels)
        for (int r = 0; r < rows; r++)
        {
            for (int gStart = 0; gStart < cols; gStart += groupSize)
            {
                var distinct = new System.Collections.Generic.HashSet<float>();
                for (int c = gStart; c < gStart + groupSize; c++)
                {
                    float q = result.CalibratedWeights[r * cols + c];
                    Assert.InRange(q, wMin - 1e-4f, wMax + 1e-4f);
                    distinct.Add(MathF.Round(q, 5));
                }
                Assert.True(distinct.Count >= 12, $"Expected >= 12 distinct levels in [{wMin}, {wMax}] group, got {distinct.Count}.");
            }
        }
    }

    [Fact]
    public void GptqCalibrator_AsymmetricConstantNonZeroWeights_RemainsExact()
    {
        const int rows = 4;
        const int cols = 32;
        const int tokens = 64;
        const float constantW = 0.75f;

        float[] weights = new float[rows * cols];
        Array.Fill(weights, constantW);

        var rng = new Random(4103);
        float[] acts = new float[tokens * cols];
        for (int i = 0; i < acts.Length; i++)
            acts[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(
            weights,
            rows,
            cols,
            dataset,
            damping: 0.01f,
            targetBits: 4,
            blockSize: 32,
            groupSize: 32,
            symmetric: false);

        Assert.All(result.CalibratedWeights, w => Assert.Equal(constantW, w));
        Assert.Equal(0.0, result.BaselineRtnMse);
        Assert.Equal(0.0, result.CalibratedMse);
    }

    [Theory]
    [InlineData(1e7f)]
    [InlineData(1e18f)]
    [InlineData(1e20f)]
    [InlineData(1e-18f)]
    public void GptqCalibrator_ExtremeActivationScales_RemainFiniteAndBeatOrMatchRtn(float actScale)
    {
        const int rows = 4;
        const int cols = 64;
        const int tokens = 128;
        var rng = new Random(5101);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            float shared = (float)(rng.NextDouble() * 2.0 - 1.0);
            for (int c = 0; c < cols; c++)
                acts[t * cols + c] = (0.6f * shared + 0.4f * (float)(rng.NextDouble() * 2.0 - 1.0)) * actScale;
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset, damping: 0.01f, targetBits: 4, blockSize: 32, groupSize: 32, symmetric: true);

        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse > 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse * (1.0 + 1e-9), $"GPTQ MSE ({result.CalibratedMse}) exceeded RTN ({result.BaselineRtnMse}) at actScale={actScale}.");
    }

    [Fact]
    public void GptqCalibrator_MixedExtremeChannelScales_RemainsFiniteAndConverges()
    {
        const int rows = 4;
        const int cols = 64;
        const int tokens = 128;
        var rng = new Random(5102);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                float baseVal = (float)(rng.NextDouble() * 2.0 - 1.0);
                if (c == 0)
                    acts[t * cols + c] = baseVal * 1e18f;
                else if (c == 1)
                    acts[t * cols + c] = baseVal * 1e-18f;
                else
                    acts[t * cols + c] = baseVal;
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset, damping: 0.01f, targetBits: 4, blockSize: 32, groupSize: 32, symmetric: true);

        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse > 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse * (1.0 + 1e-9));
    }

    [Theory]
    [InlineData(1e18f)]
    [InlineData(1e20f)]
    [InlineData(1e-18f)]
    public void ImatrixCalibrator_ExtremeActivationScales_PreserveChannelImportanceRatios(float actScale)
    {
        const int rows = 4;
        const int cols = 32;
        const int tokens = 16;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = MathF.Sin(i * 0.19f) * 1.25f;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
                acts[t * cols + c] = (c < 4 ? 2.0f : 0.5f) * actScale;
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.All(result.OptimalScales!, s => Assert.True(float.IsFinite(s) && s > 0.0f));
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse >= 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);

        Assert.True(result.OptimalScales![0] > result.OptimalScales[8], $"Expected s[0] > s[8] at actScale={actScale}, got {result.OptimalScales[0]} vs {result.OptimalScales[8]}.");
        Assert.InRange(result.OptimalScales[0] / result.OptimalScales[8], 15.0f, 17.0f);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse * (1.0 + 1e-12));
    }

    [Fact]
    public void ImatrixCalibrator_AllZeroActivations_RemainsFiniteWithUniformFloorScales()
    {
        const int rows = 4;
        const int cols = 32;
        const int tokens = 8;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = MathF.Cos(i * 0.13f);

        float[] acts = new float[tokens * cols];
        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.All(result.OptimalScales!, s => Assert.Equal(1e-4f, s));
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse));
        Assert.True(double.IsFinite(result.CalibratedMse));
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse + 1e-12);
    }

    [Fact]
    public void AwqCalibrator_ExtremeLargeScale_1e37With128Tokens_PreservesChannelRatiosAndReducesMse()
    {
        const int rows = 4;
        const int cols = 32;
        const int tokens = 128;
        const float wScale = 0.5f;
        const float actScale = 1e37f;
        var rng = new Random(6101);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * wScale;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                float mag = (c < 4 ? 2.0f : 0.5f) * actScale;
                acts[t * cols + c] = (float)(0.5 + 0.5 * rng.NextDouble()) * mag * ((t + c) % 2 == 0 ? 1.0f : -1.0f);
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 20, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.All(result.OptimalScales!, s => Assert.True(float.IsFinite(s) && s > 0.0f));
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse > 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);
        Assert.NotEqual("AWQ (alpha=0.00)", result.Method);
        Assert.True(result.OptimalScales![0] > result.OptimalScales[8], $"Expected s[0] > s[8], got {result.OptimalScales[0]} vs {result.OptimalScales[8]}.");
        Assert.True(result.CalibratedMse < result.BaselineRtnMse, $"Expected calibrated MSE ({result.CalibratedMse}) < baseline ({result.BaselineRtnMse}).");
    }

    [Fact]
    public void AwqCalibrator_ExtremeProductOverflow_1e37WithWScale10AndCols64_ProducesFiniteMse()
    {
        const int rows = 4;
        const int cols = 64;
        const int tokens = 16;
        const float wScale = 10.0f;
        const float actScale = 1e37f;
        var rng = new Random(6102);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * wScale;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
                acts[t * cols + c] = (c < 4 ? 2.0f : 0.5f) * actScale;
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 20, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.All(result.OptimalScales!, s => Assert.True(float.IsFinite(s) && s > 0.0f));
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse > 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);
        Assert.True(result.CalibratedMse <= result.BaselineRtnMse);
    }

    [Theory]
    [InlineData(1e-18f)]
    [InlineData(1e-37f)]
    public void AwqCalibrator_ExtremeSmallScale_PreservesChannelRatiosAndReducesMse(float actScale)
    {
        const int rows = 4;
        const int cols = 32;
        const int tokens = 16;
        const float wScale = 0.5f;
        var rng = new Random(6103);

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * wScale;

        float[] acts = new float[tokens * cols];
        for (int t = 0; t < tokens; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                float mag = (c < 4 ? 2.0f : 0.5f) * actScale;
                acts[t * cols + c] = (float)(0.5 + 0.5 * rng.NextDouble()) * mag * ((t + c) % 2 == 0 ? 1.0f : -1.0f);
            }
        }

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 20, targetBits: 4);

        Assert.NotNull(result.OptimalScales);
        Assert.All(result.OptimalScales!, s => Assert.True(float.IsFinite(s) && s > 0.0f));
        Assert.All(result.CalibratedWeights, w => Assert.True(float.IsFinite(w)));
        Assert.True(double.IsFinite(result.BaselineRtnMse) && result.BaselineRtnMse > 0.0);
        Assert.True(double.IsFinite(result.CalibratedMse) && result.CalibratedMse >= 0.0);
        Assert.NotEqual("AWQ (alpha=0.00)", result.Method);
        Assert.True(result.OptimalScales![0] > result.OptimalScales[8], $"Expected s[0] > s[8] at actScale={actScale}, got {result.OptimalScales[0]} vs {result.OptimalScales[8]}.");
        Assert.True(result.CalibratedMse < result.BaselineRtnMse, $"Expected calibrated MSE ({result.CalibratedMse}) < baseline ({result.BaselineRtnMse}) at actScale={actScale}.");
    }
}
