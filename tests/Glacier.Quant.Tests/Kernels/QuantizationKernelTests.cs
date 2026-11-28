namespace Glacier.Quant.Tests.Kernels;

using System;
using System.Runtime.InteropServices;
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

    [Fact]
    public void FP8_E4M3_ExhaustiveAll256Codes_RoundtripAndSubnormalsExact()
    {
        Assert.Equal(1.0f / 512.0f, SimdQuantizer.Fp8E4M3ToFloat(0x01));
        Assert.Equal(7.0f / 512.0f, SimdQuantizer.Fp8E4M3ToFloat(0x07));
        Assert.Equal(-1.0f / 512.0f, SimdQuantizer.Fp8E4M3ToFloat(0x81));
        Assert.Equal(1.0f / 64.0f, SimdQuantizer.Fp8E4M3ToFloat(0x08));
        Assert.Equal(448.0f, SimdQuantizer.Fp8E4M3ToFloat(0x7E));
        Assert.Equal(-448.0f, SimdQuantizer.Fp8E4M3ToFloat(0xFE));
        Assert.Equal(0x80000000u, BitConverter.SingleToUInt32Bits(SimdQuantizer.Fp8E4M3ToFloat(0x80)));
        Assert.True(float.IsNaN(SimdQuantizer.Fp8E4M3ToFloat(0x7F)));
        Assert.True(float.IsNaN(SimdQuantizer.Fp8E4M3ToFloat(0xFF)));

        for (int code = 0; code <= 255; code++)
        {
            byte b = (byte)code;
            float decoded = SimdQuantizer.Fp8E4M3ToFloat(b);
            if (b is 0x7F or 0xFF)
            {
                Assert.True(float.IsNaN(decoded));
                continue;
            }

            Assert.True(float.IsFinite(decoded));
            byte reEncoded = SimdQuantizer.FloatToFp8E4M3(decoded);
            Assert.Equal(b, reEncoded);
        }
    }

    [Fact]
    public void FP8_E4M3_OverflowAndRoundToNearestEven_SaturatesTo0x7E_NeverNaN()
    {
        float[] overflowVals = [448.0f, 464.0f, 480.0f, 496.0f, 511.9f, float.PositiveInfinity];
        foreach (float v in overflowVals)
        {
            Assert.Equal((byte)0x7E, SimdQuantizer.FloatToFp8E4M3(v));
            Assert.Equal((byte)0xFE, SimdQuantizer.FloatToFp8E4M3(-v));
        }

        // Subnormal RNE ties
        Assert.Equal((byte)0x00, SimdQuantizer.FloatToFp8E4M3(0.5f / 512.0f));
        Assert.Equal((byte)0x02, SimdQuantizer.FloatToFp8E4M3(1.5f / 512.0f));
        Assert.Equal((byte)0x02, SimdQuantizer.FloatToFp8E4M3(2.5f / 512.0f));
        Assert.Equal((byte)0x08, SimdQuantizer.FloatToFp8E4M3(7.5f / 512.0f));

        // Normal RNE ties: 1.0 -> 0x38 (even), 1.125 -> 0x39 (odd), 1.25 -> 0x3A (even)
        Assert.Equal((byte)0x38, SimdQuantizer.FloatToFp8E4M3(1.0625f));
        Assert.Equal((byte)0x3A, SimdQuantizer.FloatToFp8E4M3(1.1875f));
    }

    [Fact]
    public void SimdVsScalar_BitExactParity_AcrossQ8_Q4K_FP8_1Bit()
    {
        var rng = new Random(9001);

        // 1. Q8_0 (multi-block)
        float[] srcQ8 = new float[32 * 9];
        for (int i = 0; i < srcQ8.Length; i++)
            srcQ8[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 15.0f;

        var q8Simd = new BlockQ8_0[9];
        var q8Scalar = new BlockQ8_0[9];
        SimdQuantizer.QuantizeRowQ8_0(srcQ8, q8Simd);
        SimdQuantizer.QuantizeRowQ8_0Scalar(srcQ8, q8Scalar);
        Assert.Equal(
            MemoryMarshal.AsBytes(q8Scalar.AsSpan()).ToArray(),
            MemoryMarshal.AsBytes(q8Simd.AsSpan()).ToArray());

        var deqQ8Simd = new float[srcQ8.Length];
        var deqQ8Scalar = new float[srcQ8.Length];
        SimdQuantizer.DequantizeRowQ8_0(q8Simd, deqQ8Simd);
        SimdQuantizer.DequantizeRowQ8_0Scalar(q8Scalar, deqQ8Scalar);
        Assert.Equal(deqQ8Scalar, deqQ8Simd);

        // 1b. Q8_0 NaN parity (mixed NaNs in block 0, all-NaN block 1)
        float[] srcQ8NaN = new float[64];
        for (int i = 0; i < 32; i++)
            srcQ8NaN[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 10.0f;
        srcQ8NaN[5] = float.NaN;
        srcQ8NaN[19] = -float.NaN;
        for (int i = 32; i < 64; i++)
            srcQ8NaN[i] = float.NaN;

        var q8NaNSimd = new BlockQ8_0[2];
        var q8NaNScalar = new BlockQ8_0[2];
        SimdQuantizer.QuantizeRowQ8_0(srcQ8NaN, q8NaNSimd);
        SimdQuantizer.QuantizeRowQ8_0Scalar(srcQ8NaN, q8NaNScalar);
        Assert.Equal(
            MemoryMarshal.AsBytes(q8NaNScalar.AsSpan()).ToArray(),
            MemoryMarshal.AsBytes(q8NaNSimd.AsSpan()).ToArray());
        unsafe
        {
            Assert.Equal(0, q8NaNSimd[0].Quants[5]);
            Assert.Equal(0, q8NaNSimd[0].Quants[19]);
            Assert.Equal(0, (float)q8NaNSimd[1].Scale);
            Assert.Equal(0, q8NaNSimd[1].Quants[0]);
        }

        // 2. Q4_K (multi-superblock)
        float[] srcQ4 = new float[256 * 3];
        for (int i = 0; i < srcQ4.Length; i++)
            srcQ4[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 6.0f;

        var q4Simd = new BlockQ4_K[3];
        var q4Scalar = new BlockQ4_K[3];
        SimdQuantizer.QuantizeRowQ4_K(srcQ4, q4Simd);
        SimdQuantizer.QuantizeRowQ4_KScalar(srcQ4, q4Scalar);
        Assert.Equal(
            MemoryMarshal.AsBytes(q4Scalar.AsSpan()).ToArray(),
            MemoryMarshal.AsBytes(q4Simd.AsSpan()).ToArray());

        var deqQ4Simd = new float[srcQ4.Length];
        var deqQ4Scalar = new float[srcQ4.Length];
        SimdQuantizer.DequantizeRowQ4_K(q4Simd, deqQ4Simd);
        SimdQuantizer.DequantizeRowQ4_KScalar(q4Scalar, deqQ4Scalar);
        Assert.Equal(deqQ4Scalar, deqQ4Simd);

        // 3. FP8 E4M3 (non-aligned lengths including subnormal, overflow, tie, -0.0f)
        int[] fp8Lengths = [1, 7, 15, 16, 17, 31, 33, 64, 129, 513];
        foreach (int len in fp8Lengths)
        {
            float[] srcFp8 = new float[len];
            for (int i = 0; i < len; i++)
            {
                srcFp8[i] = i switch
                {
                    0 => -0.0f,
                    1 when len > 1 => 0.001f,
                    2 when len > 2 => 100.0f,
                    _ => (float)(rng.NextDouble() * 2.0 - 1.0) * 50.0f
                };
            }

            byte[] fp8Simd = new byte[len];
            byte[] fp8Scalar = new byte[len];
            SimdQuantizer.QuantizeRowFP8_E4M3(srcFp8, fp8Simd, out float sSimd);
            SimdQuantizer.QuantizeRowFP8_E4M3Scalar(srcFp8, fp8Scalar, out float sScalar);
            Assert.Equal(sScalar, sSimd);
            Assert.Equal(fp8Scalar, fp8Simd);

            float[] deqFp8Simd = new float[len];
            float[] deqFp8Scalar = new float[len];
            SimdQuantizer.DequantizeRowFP8_E4M3(fp8Simd, sSimd, deqFp8Simd);
            SimdQuantizer.DequantizeRowFP8_E4M3Scalar(fp8Scalar, sScalar, deqFp8Scalar);
            Assert.Equal(deqFp8Scalar, deqFp8Simd);
        }

        // 4. 1-Bit (multi-block)
        float[] src1Bit = new float[64 * 5];
        for (int i = 0; i < src1Bit.Length; i++)
            src1Bit[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 3.0f;

        var b1Simd = new BlockBinary1Bit[5];
        var b1Scalar = new BlockBinary1Bit[5];
        SimdQuantizer.QuantizeRow1Bit(src1Bit, b1Simd);
        SimdQuantizer.QuantizeRow1BitScalar(src1Bit, b1Scalar);
        Assert.Equal(b1Scalar, b1Simd);

        var deq1Simd = new float[src1Bit.Length];
        var deq1Scalar = new float[src1Bit.Length];
        SimdQuantizer.DequantizeRow1Bit(b1Simd, deq1Simd);
        SimdQuantizer.DequantizeRow1BitScalar(b1Scalar, deq1Scalar);
        Assert.Equal(deq1Scalar, deq1Simd);
    }

    [Fact]
    public void HammingDistance1Bit_SpanOverload_MatchesScalarPopCount()
    {
        ulong[] a = [0x0F0F0F0F0F0F0F0FUL, 0xAAAAAAAAAAAAAAAAUL, 0x123456789ABCDEF0UL, 0UL, ulong.MaxValue, 1UL, 2UL, 3UL, 4UL];
        ulong[] b = [0xF0F0F0F0F0F0F0F0UL, 0x5555555555555555UL, 0x0FEDCBA987654321UL, ulong.MaxValue, 0UL, 3UL, 6UL, 7UL, 12UL];

        int expected = 0;
        for (int i = 0; i < a.Length; i++)
            expected += SimdQuantizer.HammingDistance1Bit(a[i], b[i]);

        Assert.Equal(expected, SimdQuantizer.HammingDistance1Bit(a.AsSpan(), b.AsSpan()));
    }
}
