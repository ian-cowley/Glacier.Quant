namespace Glacier.Quant.Kernels;

using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class SimdQuantizer
{
    // =========================================================================
    // 1. Q8_0 SIMD QUANTIZATION & DEQUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void QuantizeRowQ8_0(ReadOnlySpan<float> src, Span<BlockQ8_0> dst)
    {
        if (src.Length % BlockQ8_0.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of block size ({BlockQ8_0.BlockSize}).");

        int numBlocks = src.Length / BlockQ8_0.BlockSize;
        if (dst.Length < numBlocks)
            throw new ArgumentException($"Destination span length ({dst.Length}) is too small for {numBlocks} blocks.");

        fixed (float* pSrc = src)
        fixed (BlockQ8_0* pDst = dst)
        {
            for (int b = 0; b < numBlocks; b++)
            {
                float* blockSrc = pSrc + (b * BlockQ8_0.BlockSize);
                BlockQ8_0* blockDst = pDst + b;

                // 1. Find absolute max
                float amax = 0.0f;
                for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                {
                    float abs = MathF.Abs(blockSrc[i]);
                    if (abs > amax) amax = abs;
                }

                if (amax <= 1e-12f)
                {
                    blockDst->Scale = (Half)0.0f;
                    for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                        blockDst->Quants[i] = 0;
                    continue;
                }

                float scale = amax / 127.0f;
                float invScale = 127.0f / amax;
                blockDst->Scale = (Half)scale;

                // 2. Quantize 32 elements into signed bytes
                for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                {
                    float v = blockSrc[i] * invScale;
                    int q = (int)MathF.Round(v);
                    if (q > 127) q = 127;
                    else if (q < -128) q = -128;
                    blockDst->Quants[i] = (sbyte)q;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRowQ8_0(ReadOnlySpan<BlockQ8_0> src, Span<float> dst)
    {
        int totalElements = src.Length * BlockQ8_0.BlockSize;
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        fixed (BlockQ8_0* pSrc = src)
        fixed (float* pDst = dst)
        {
            for (int b = 0; b < src.Length; b++)
            {
                BlockQ8_0* blockSrc = pSrc + b;
                float* blockDst = pDst + (b * BlockQ8_0.BlockSize);
                float scale = (float)blockSrc->Scale;

                for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                {
                    blockDst[i] = blockSrc->Quants[i] * scale;
                }
            }
        }
    }

    // =========================================================================
    // 2. Q4_K_M SUPER-BLOCK QUANTIZATION & DEQUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void QuantizeRowQ4_K(ReadOnlySpan<float> src, Span<BlockQ4_K> dst)
    {
        if (src.Length % BlockQ4_K.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of {BlockQ4_K.BlockSize}.");

        int numSuperBlocks = src.Length / BlockQ4_K.BlockSize;
        if (dst.Length < numSuperBlocks)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {numSuperBlocks} blocks.");

        fixed (float* pSrc = src)
        fixed (BlockQ4_K* pDst = dst)
        {
            for (int b = 0; b < numSuperBlocks; b++)
            {
                float* sbSrc = pSrc + (b * BlockQ4_K.BlockSize);
                BlockQ4_K* sbDst = pDst + b;

                // Find global min and max across all 8 sub-blocks of 32 elements
                float minVal = float.MaxValue;
                float maxVal = float.MinValue;
                for (int i = 0; i < BlockQ4_K.BlockSize; i++)
                {
                    float v = sbSrc[i];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }

                float range = Math.Max(1e-12f, maxVal - minVal);
                float d = range / 15.0f;
                float dmin = minVal;

                sbDst->D = (Half)d;
                sbDst->DMin = (Half)dmin;

                float invD = 15.0f / range;

                // Clear 12 bytes scales and 128 bytes quants
                for (int s = 0; s < 12; s++) sbDst->Scales[s] = 0;
                for (int q = 0; q < 128; q++) sbDst->Quants[q] = 0;

                // Pack 256 4-bit values into 128 bytes (2 nibbles per byte)
                for (int i = 0; i < BlockQ4_K.BlockSize; i += 2)
                {
                    int q0 = (int)MathF.Round((sbSrc[i] - dmin) * invD);
                    int q1 = (int)MathF.Round((sbSrc[i + 1] - dmin) * invD);

                    q0 = Math.Clamp(q0, 0, 15);
                    q1 = Math.Clamp(q1, 0, 15);

                    sbDst->Quants[i / 2] = (byte)(q0 | (q1 << 4));
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRowQ4_K(ReadOnlySpan<BlockQ4_K> src, Span<float> dst)
    {
        int totalElements = src.Length * BlockQ4_K.BlockSize;
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        fixed (BlockQ4_K* pSrc = src)
        fixed (float* pDst = dst)
        {
            for (int b = 0; b < src.Length; b++)
            {
                BlockQ4_K* sbSrc = pSrc + b;
                float* sbDst = pDst + (b * BlockQ4_K.BlockSize);

                float d = (float)sbSrc->D;
                float dmin = (float)sbSrc->DMin;

                for (int i = 0; i < 128; i++)
                {
                    byte packed = sbSrc->Quants[i];
                    int q0 = packed & 0x0F;
                    int q1 = (packed >> 4) & 0x0F;

                    sbDst[i * 2] = dmin + (q0 * d);
                    sbDst[i * 2 + 1] = dmin + (q1 * d);
                }
            }
        }
    }

    // =========================================================================
    // 3. FP8 (E4M3) PER-BLOCK / DYNAMIC QUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeRowFP8_E4M3(ReadOnlySpan<float> src, Span<byte> dst, out float dynamicScale)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination span too small.");

        // Find max magnitude
        float amax = 0.0f;
        for (int i = 0; i < src.Length; i++)
        {
            float abs = MathF.Abs(src[i]);
            if (abs > amax) amax = abs;
        }

        // Max finite value for FP8 E4M3 is 448.0f
        const float maxFp8 = 448.0f;
        dynamicScale = amax > 1e-12f ? amax / maxFp8 : 1.0f;
        float invScale = 1.0f / dynamicScale;

        for (int i = 0; i < src.Length; i++)
        {
            float scaled = src[i] * invScale;
            dst[i] = FloatToFp8E4M3(scaled);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeRowFP8_E4M3(ReadOnlySpan<byte> src, float dynamicScale, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination span too small.");

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = Fp8E4M3ToFloat(src[i]) * dynamicScale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte FloatToFp8E4M3(float val)
    {
        if (float.IsNaN(val)) return 0x7F;

        uint bits = BitConverter.SingleToUInt32Bits(val);
        uint sign = (bits >> 31) & 1;
        int exp = (int)((bits >> 23) & 0xFF) - 127;
        uint mantissa = bits & 0x7FFFFF;

        if (exp > 8)
        {
            // Overflow: saturate to max E4M3 value (448.0)
            return (byte)((sign << 7) | 0x7E);
        }
        if (exp < -6)
        {
            // Underflow to zero
            return (byte)(sign << 7);
        }

        int fp8Exp = exp + 7;
        uint fp8Mant = mantissa >> 20; // 3 bits mantissa

        uint res = ((sign & 1U) << 7) | ((uint)(fp8Exp & 0x0F) << 3) | (fp8Mant & 0x07U);
        return (byte)res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Fp8E4M3ToFloat(byte val)
    {
        uint sign = (uint)(val >> 7);
        uint exp = (uint)((val >> 3) & 0x0F);
        uint mant = (uint)(val & 0x07);

        if (exp == 0 && mant == 0)
        {
            return sign == 1 ? -0.0f : 0.0f;
        }

        if (exp == 0x0F && mant == 0x07)
        {
            return float.NaN;
        }

        int trueExp = (int)exp - 7 + 127;
        uint trueMant = mant << 20;
        uint bits = (sign << 31) | ((uint)(trueExp & 0xFF) << 23) | (trueMant & 0x7FFFFF);
        return BitConverter.UInt32BitsToSingle(bits);
    }

    // =========================================================================
    // 4. 1-BIT BINARY QUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeRow1Bit(ReadOnlySpan<float> src, Span<BlockBinary1Bit> dst)
    {
        if (src.Length % BlockBinary1Bit.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of {BlockBinary1Bit.BlockSize}.");

        int numBlocks = src.Length / BlockBinary1Bit.BlockSize;
        if (dst.Length < numBlocks)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {numBlocks} blocks.");

        for (int b = 0; b < numBlocks; b++)
        {
            int offset = b * BlockBinary1Bit.BlockSize;
            var blockSlice = src.Slice(offset, BlockBinary1Bit.BlockSize);

            // Compute mean bias
            float sum = 0.0f;
            for (int i = 0; i < BlockBinary1Bit.BlockSize; i++)
                sum += blockSlice[i];
            float bias = sum / BlockBinary1Bit.BlockSize;

            // Compute absolute scale from bias
            float absSum = 0.0f;
            ulong bits = 0;
            for (int i = 0; i < BlockBinary1Bit.BlockSize; i++)
            {
                float diff = blockSlice[i] - bias;
                absSum += MathF.Abs(diff);
                if (diff >= 0.0f)
                {
                    bits |= (1UL << i);
                }
            }
            float scale = Math.Max(1e-12f, absSum / BlockBinary1Bit.BlockSize);

            dst[b] = new BlockBinary1Bit
            {
                Bits = bits,
                Scale = scale,
                Bias = bias
            };
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeRow1Bit(ReadOnlySpan<BlockBinary1Bit> src, Span<float> dst)
    {
        int totalElements = src.Length * BlockBinary1Bit.BlockSize;
        if (dst.Length < totalElements)
            throw new ArgumentException("Destination length too small.");

        for (int b = 0; b < src.Length; b++)
        {
            var block = src[b];
            int offset = b * BlockBinary1Bit.BlockSize;
            ulong bits = block.Bits;
            float scale = block.Scale;
            float bias = block.Bias;

            for (int i = 0; i < BlockBinary1Bit.BlockSize; i++)
            {
                bool bit = ((bits >> i) & 1UL) == 1UL;
                dst[offset + i] = bit ? (bias + scale) : (bias - scale);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int HammingDistance1Bit(ulong a, ulong b)
    {
        return BitOperations.PopCount(a ^ b);
    }
}
