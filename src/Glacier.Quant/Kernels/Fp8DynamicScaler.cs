namespace Glacier.Quant.Kernels;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Dynamic FP8 scaling and outlier management engine with key channel-mean smoothing.
/// Reduces quantization noise in key-value caches and projection layers by centering
/// channel distributions: K_smoothed = K - mean(K).
/// </summary>
public static class Fp8DynamicScaler
{
    /// <summary>
    /// Applies channel-mean smoothing across input key activation rows:
    /// Computes channel means and subtracts them in-place, returning channel mean offsets.
    /// </summary>
    public static void SmoothKeyChannelMeans(
        Span<float> activations,
        int numTokens,
        int hiddenDim,
        Span<float> channelMeans)
    {
        if (activations.Length != numTokens * hiddenDim)
            throw new ArgumentException("Activations length does not match token and dimension counts.");
        if (channelMeans.Length < hiddenDim)
            throw new ArgumentException("Channel means buffer is too small.");

        channelMeans.Clear();

        // 1. Accumulate channel sums
        for (int t = 0; t < numTokens; t++)
        {
            int rowOffset = t * hiddenDim;
            for (int d = 0; d < hiddenDim; d++)
            {
                channelMeans[d] += activations[rowOffset + d];
            }
        }

        // 2. Compute means
        float invTokens = 1.0f / Math.Max(1, numTokens);
        for (int d = 0; d < hiddenDim; d++)
        {
            channelMeans[d] *= invTokens;
        }

        // 3. Center activations: x = x - mean
        for (int t = 0; t < numTokens; t++)
        {
            int rowOffset = t * hiddenDim;
            for (int d = 0; d < hiddenDim; d++)
            {
                activations[rowOffset + d] -= channelMeans[d];
            }
        }
    }

    /// <summary>
    /// Computes optimal dynamic scale factor for FP8 E4M3 quantization with percentile outlier suppression.
    /// </summary>
    public static float ComputeDynamicScaleE4M3(ReadOnlySpan<float> values, float percentile = 0.999f)
    {
        if (values.Length == 0) return 1.0f;

        // Sample up to 4096 values to compute robust percentile
        int sampleCount = Math.Min(values.Length, 4096);
        float[] samples = new float[sampleCount];
        int stride = Math.Max(1, values.Length / sampleCount);

        for (int i = 0; i < sampleCount; i++)
        {
            samples[i] = MathF.Abs(values[i * stride]);
        }

        Array.Sort(samples);
        int targetIdx = Math.Clamp((int)(sampleCount * percentile), 0, sampleCount - 1);
        float threshold = samples[targetIdx];

        const float maxFp8 = 448.0f;
        return threshold > 1e-12f ? threshold / maxFp8 : 1.0f;
    }
}
