namespace Glacier.Quant.Tests.Kernels;

using System;
using Glacier.Quant.Kernels;
using Xunit;

public class QuantizationKernelTests
{
    [Fact]
    public void Q8_0_QuantizeAndDequantize_AchievesAccurateRoundtrip()
    {
        float[] original = new float[64];
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = MathF.Sin(i * 0.1f) * 2.5f;
        }

        var blocks = new BlockQ8_0[2];
        SimdQuantizer.QuantizeRowQ8_0(original, blocks);

        float[] dequantized = new float[64];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequantized);

        // Verify MSE bound
        double mse = 0.0;
        for (int i = 0; i < original.Length; i++)
        {
            float diff = original[i] - dequantized[i];
            mse += diff * diff;
        }
        mse /= original.Length;

        Assert.True(mse < 0.001, $"Q8_0 MSE ({mse}) exceeded threshold 0.001");
    }

    [Fact]
    public void Q4_K_QuantizeAndDequantize_AchievesAccurateRoundtrip()
    {
        float[] original = new float[256];
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = (i - 128) * 0.02f;
        }

        var blocks = new BlockQ4_K[1];
        SimdQuantizer.QuantizeRowQ4_K(original, blocks);

        float[] dequantized = new float[256];
        SimdQuantizer.DequantizeRowQ4_K(blocks, dequantized);

        double mse = 0.0;
        for (int i = 0; i < original.Length; i++)
        {
            float diff = original[i] - dequantized[i];
            mse += diff * diff;
        }
        mse /= original.Length;

        Assert.True(mse < 0.05, $"Q4_K MSE ({mse}) exceeded threshold 0.05");

        // Verify non-zero scales are written (spec-compliant 8-subblock packing)
        bool hasNonZeroScale = false;
        unsafe
        {
            fixed (BlockQ4_K* pB = blocks)
            {
                for (int s = 0; s < 12; s++)
                {
                    if (pB[0].Scales[s] != 0) hasNonZeroScale = true;
                }
            }
        }
        Assert.True(hasNonZeroScale, "Q4_K Scales must be populated.");
    }

    [Fact]
    public void FP8_E4M3_QuantizeAndDequantize_AchievesAccurateRoundtrip()
    {
        float[] original = new float[128];
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = MathF.Cos(i * 0.05f) * 1.8f;
        }

        byte[] quants = new byte[128];
        SimdQuantizer.QuantizeRowFP8_E4M3(original, quants, out float scale);

        float[] dequantized = new float[128];
        SimdQuantizer.DequantizeRowFP8_E4M3(quants, scale, dequantized);

        double mse = 0.0;
        for (int i = 0; i < original.Length; i++)
        {
            float diff = original[i] - dequantized[i];
            mse += diff * diff;
        }
        mse /= original.Length;

        Assert.True(mse < 0.01, $"FP8 MSE ({mse}) exceeded threshold 0.01");
    }

    [Fact]
    public void Binary1Bit_QuantizeAndDequantize_AndHammingDistance()
    {
        float[] original = new float[64];
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }

        var blocks = new BlockBinary1Bit[1];
        SimdQuantizer.QuantizeRow1Bit(original, blocks);

        float[] dequantized = new float[64];
        SimdQuantizer.DequantizeRow1Bit(blocks, dequantized);

        // Sign should be exactly preserved
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(MathF.Sign(original[i]), MathF.Sign(dequantized[i]));
        }

        // Test Hamming distance
        ulong a = 0b10101010UL;
        ulong b = 0b10100000UL;
        int dist = SimdQuantizer.HammingDistance1Bit(a, b);
        Assert.Equal(2, dist);
    }

    [Fact]
    public void Fp8DynamicScaler_CentersChannelMeans()
    {
        int tokens = 4;
        int dim = 8;
        float[] activations = new float[tokens * dim];
        for (int t = 0; t < tokens; t++)
        {
            for (int d = 0; d < dim; d++)
            {
                activations[t * dim + d] = 10.0f + d * 2.0f + (t * 0.1f);
            }
        }

        float[] means = new float[dim];
        Fp8DynamicScaler.SmoothKeyChannelMeans(activations, tokens, dim, means);

        // Channel means should be close to 10.0 + d * 2.0 + 0.15
        for (int d = 0; d < dim; d++)
        {
            Assert.True(means[d] > 9.0f);
        }

        // Smoothed activations for each channel should sum to 0
        for (int d = 0; d < dim; d++)
        {
            float channelSum = 0.0f;
            for (int t = 0; t < tokens; t++)
            {
                channelSum += activations[t * dim + d];
            }
            Assert.True(MathF.Abs(channelSum) < 1e-4f, $"Channel {d} sum ({channelSum}) is not 0.");
        }
    }
}
