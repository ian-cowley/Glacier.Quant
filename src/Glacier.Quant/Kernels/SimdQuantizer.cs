namespace Glacier.Quant.Kernels;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class SimdQuantizer
{
    private static readonly float[] s_fp8E4M3ToFloat = BuildFp8E4M3LookupTable();

    private static float[] BuildFp8E4M3LookupTable()
    {
        var table = new float[256];
        uint nanBits = BitConverter.SingleToUInt32Bits(float.NaN);

        for (int b = 0; b < 256; b++)
        {
            uint sign = (uint)(b >> 7);
            uint exp = (uint)((b >> 3) & 0x0F);
            uint mant = (uint)(b & 0x07);

            if (exp == 0 && mant == 0)
            {
                table[b] = sign == 1 ? -0.0f : 0.0f;
            }
            else if (exp == 0)
            {
                float mag = mant * (1.0f / 512.0f);
                table[b] = sign == 1 ? -mag : mag;
            }
            else if (exp == 0x0F && mant == 0x07)
            {
                table[b] = BitConverter.UInt32BitsToSingle(nanBits);
            }
            else
            {
                uint bits = (sign << 31) | ((exp + 120u) << 23) | (mant << 20);
                table[b] = BitConverter.UInt32BitsToSingle(bits);
            }
        }

        return table;
    }

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

        if (numBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockQ8_0* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                var vMinInt = Vector512.Create(-128);
                var vMaxInt = Vector512.Create(127);

                for (int b = 0; b < numBlocks; b++)
                {
                    float* blockSrc = pSrc + (b * BlockQ8_0.BlockSize);
                    BlockQ8_0* blockDst = pDst + b;

                    var v0 = Vector512.Load(blockSrc);
                    var v1 = Vector512.Load(blockSrc + 16);

                    v0 = Vector512.ConditionalSelect(Vector512.Equals(v0, v0), v0, Vector512<float>.Zero);
                    v1 = Vector512.ConditionalSelect(Vector512.Equals(v1, v1), v1, Vector512<float>.Zero);
                    float amax = HorizontalMax512(Vector512.Max(Vector512.Abs(v0), Vector512.Abs(v1)));

                    if (amax <= 1e-12f)
                    {
                        blockDst->Scale = (Half)0.0f;
                        Vector256<sbyte>.Zero.Store(blockDst->Quants);
                        continue;
                    }

                    float scale = amax / 127.0f;
                    float invScale = 127.0f / amax;
                    blockDst->Scale = (Half)scale;
                    var vInvScale = Vector512.Create(invScale);

                    var q0 = Avx512F.ConvertToVector512Int32(Vector512.Multiply(v0, vInvScale));
                    q0 = Vector512.Min(Vector512.Max(q0, vMinInt), vMaxInt);
                    Avx512F.ConvertToVector128SByte(q0).Store(blockDst->Quants);

                    var q1 = Avx512F.ConvertToVector512Int32(Vector512.Multiply(v1, vInvScale));
                    q1 = Vector512.Min(Vector512.Max(q1, vMinInt), vMaxInt);
                    Avx512F.ConvertToVector128SByte(q1).Store(blockDst->Quants + 16);
                }
            }
            else if (Avx2.IsSupported)
            {
                var vMinInt = Vector256.Create(-128);
                var vMaxInt = Vector256.Create(127);

                for (int b = 0; b < numBlocks; b++)
                {
                    float* blockSrc = pSrc + (b * BlockQ8_0.BlockSize);
                    BlockQ8_0* blockDst = pDst + b;

                    var v0 = Vector256.Load(blockSrc);
                    var v1 = Vector256.Load(blockSrc + 8);
                    var v2 = Vector256.Load(blockSrc + 16);
                    var v3 = Vector256.Load(blockSrc + 24);

                    v0 = Vector256.ConditionalSelect(Vector256.Equals(v0, v0), v0, Vector256<float>.Zero);
                    v1 = Vector256.ConditionalSelect(Vector256.Equals(v1, v1), v1, Vector256<float>.Zero);
                    v2 = Vector256.ConditionalSelect(Vector256.Equals(v2, v2), v2, Vector256<float>.Zero);
                    v3 = Vector256.ConditionalSelect(Vector256.Equals(v3, v3), v3, Vector256<float>.Zero);

                    var abs0 = Vector256.Abs(v0);
                    var abs1 = Vector256.Abs(v1);
                    var abs2 = Vector256.Abs(v2);
                    var abs3 = Vector256.Abs(v3);
                    float amax = HorizontalMax256(Vector256.Max(Vector256.Max(abs0, abs1), Vector256.Max(abs2, abs3)));

                    if (amax <= 1e-12f)
                    {
                        blockDst->Scale = (Half)0.0f;
                        Vector256<sbyte>.Zero.Store(blockDst->Quants);
                        continue;
                    }

                    float scale = amax / 127.0f;
                    float invScale = 127.0f / amax;
                    blockDst->Scale = (Half)scale;
                    var vInvScale = Vector256.Create(invScale);

                    var q0 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(v0, vInvScale)), vMinInt), vMaxInt);
                    var q1 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(v1, vInvScale)), vMinInt), vMaxInt);
                    var p01 = Sse2.PackSignedSaturate(q0.GetLower(), q0.GetUpper());
                    var p23 = Sse2.PackSignedSaturate(q1.GetLower(), q1.GetUpper());
                    Sse2.PackSignedSaturate(p01, p23).Store(blockDst->Quants);

                    var q2 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(v2, vInvScale)), vMinInt), vMaxInt);
                    var q3 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(v3, vInvScale)), vMinInt), vMaxInt);
                    var p45 = Sse2.PackSignedSaturate(q2.GetLower(), q2.GetUpper());
                    var p67 = Sse2.PackSignedSaturate(q3.GetLower(), q3.GetUpper());
                    Sse2.PackSignedSaturate(p45, p67).Store(blockDst->Quants + 16);
                }
            }
            else
            {
                QuantizeRowQ8_0ScalarCore(pSrc, pDst, numBlocks);
            }
        }
    }

    internal static unsafe void QuantizeRowQ8_0Scalar(ReadOnlySpan<float> src, Span<BlockQ8_0> dst)
    {
        if (src.Length % BlockQ8_0.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of block size ({BlockQ8_0.BlockSize}).");

        int numBlocks = src.Length / BlockQ8_0.BlockSize;
        if (dst.Length < numBlocks)
            throw new ArgumentException($"Destination span length ({dst.Length}) is too small for {numBlocks} blocks.");

        if (numBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockQ8_0* pDst = dst)
        {
            QuantizeRowQ8_0ScalarCore(pSrc, pDst, numBlocks);
        }
    }

    private static unsafe void QuantizeRowQ8_0ScalarCore(float* pSrc, BlockQ8_0* pDst, int numBlocks)
    {
        for (int b = 0; b < numBlocks; b++)
        {
            float* blockSrc = pSrc + (b * BlockQ8_0.BlockSize);
            BlockQ8_0* blockDst = pDst + b;

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

            for (int i = 0; i < BlockQ8_0.BlockSize; i++)
            {
                float raw = blockSrc[i];
                if (float.IsNaN(raw))
                {
                    blockDst->Quants[i] = 0;
                    continue;
                }
                float v = raw * invScale;
                int q = (int)MathF.Round(v);
                if (q > 127) q = 127;
                else if (q < -128) q = -128;
                blockDst->Quants[i] = (sbyte)q;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRowQ8_0(ReadOnlySpan<BlockQ8_0> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockQ8_0.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        if (src.IsEmpty)
            return;

        fixed (BlockQ8_0* pSrc = src)
        fixed (float* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                for (int b = 0; b < src.Length; b++)
                {
                    BlockQ8_0* blockSrc = pSrc + b;
                    float* blockDst = pDst + (b * BlockQ8_0.BlockSize);
                    var vScale = Vector512.Create((float)blockSrc->Scale);

                    var q0 = Vector128.Load(blockSrc->Quants);
                    var f0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(q0));
                    Vector512.Multiply(f0, vScale).Store(blockDst);

                    var q1 = Vector128.Load(blockSrc->Quants + 16);
                    var f1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(q1));
                    Vector512.Multiply(f1, vScale).Store(blockDst + 16);
                }
            }
            else if (Avx2.IsSupported)
            {
                for (int b = 0; b < src.Length; b++)
                {
                    BlockQ8_0* blockSrc = pSrc + b;
                    float* blockDst = pDst + (b * BlockQ8_0.BlockSize);
                    var vScale = Vector256.Create((float)blockSrc->Scale);

                    for (int i = 0; i < 32; i += 8)
                    {
                        var q8 = Vector128.CreateScalar(*(ulong*)(blockSrc->Quants + i)).AsSByte();
                        var f8 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(q8));
                        Vector256.Multiply(f8, vScale).Store(blockDst + i);
                    }
                }
            }
            else
            {
                DequantizeRowQ8_0ScalarCore(pSrc, pDst, src.Length);
            }
        }
    }

    internal static unsafe void DequantizeRowQ8_0Scalar(ReadOnlySpan<BlockQ8_0> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockQ8_0.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        if (src.IsEmpty)
            return;

        fixed (BlockQ8_0* pSrc = src)
        fixed (float* pDst = dst)
        {
            DequantizeRowQ8_0ScalarCore(pSrc, pDst, src.Length);
        }
    }

    private static unsafe void DequantizeRowQ8_0ScalarCore(BlockQ8_0* pSrc, float* pDst, int numBlocks)
    {
        for (int b = 0; b < numBlocks; b++)
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

        if (numSuperBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockQ4_K* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                QuantizeRowQ4_KAvx512(pSrc, pDst, numSuperBlocks);
            }
            else if (Avx2.IsSupported)
            {
                QuantizeRowQ4_KAvx2(pSrc, pDst, numSuperBlocks);
            }
            else
            {
                QuantizeRowQ4_KScalarCore(pSrc, pDst, numSuperBlocks);
            }
        }
    }

    internal static unsafe void QuantizeRowQ4_KScalar(ReadOnlySpan<float> src, Span<BlockQ4_K> dst)
    {
        if (src.Length % BlockQ4_K.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of {BlockQ4_K.BlockSize}.");

        int numSuperBlocks = src.Length / BlockQ4_K.BlockSize;
        if (dst.Length < numSuperBlocks)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {numSuperBlocks} blocks.");

        if (numSuperBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockQ4_K* pDst = dst)
        {
            QuantizeRowQ4_KScalarCore(pSrc, pDst, numSuperBlocks);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void QuantizeRowQ4_KAvx512(float* pSrc, BlockQ4_K* pDst, int numSuperBlocks)
    {
        float* subScale = stackalloc float[8];
        float* subMin = stackalloc float[8];
        byte* sc = stackalloc byte[8];
        byte* m = stackalloc byte[8];
        byte* L = stackalloc byte[256];

        var vPosInf = Vector512.Create(float.PositiveInfinity);
        var vNegInf = Vector512.Create(float.NegativeInfinity);
        var vZeroInt = Vector512<int>.Zero;
        var vFifteenInt = Vector512.Create(15);
        var vZeroByte = Vector256<byte>.Zero;

        for (int b = 0; b < numSuperBlocks; b++)
        {
            float* sbSrc = pSrc + (b * BlockQ4_K.BlockSize);
            BlockQ4_K* sbDst = pDst + b;

            float maxScale = 0f;
            float maxMin = 0f;

            for (int ib = 0; ib < 8; ib++)
            {
                float* bSrc = sbSrc + ib * 32;
                var v0 = Vector512.Load(bSrc);
                var v1 = Vector512.Load(bSrc + 16);

                var fin0 = Vector512.Equals(v0, v0);
                var fin1 = Vector512.Equals(v1, v1);

                var minVec = Vector512.Min(
                    Vector512.ConditionalSelect(fin0, v0, vPosInf),
                    Vector512.ConditionalSelect(fin1, v1, vPosInf));
                var maxVec = Vector512.Max(
                    Vector512.ConditionalSelect(fin0, v0, vNegInf),
                    Vector512.ConditionalSelect(fin1, v1, vNegInf));

                float bMin = HorizontalMin512(minVec);
                float bMax = HorizontalMax512(maxVec);

                if (float.IsPositiveInfinity(bMin) || float.IsNegativeInfinity(bMax))
                {
                    bMin = 0f;
                    bMax = 0f;
                }

                if (bMin > 0f) bMin = 0f;
                float range = bMax - bMin;
                float scale = range / 15.0f;
                subScale[ib] = scale;
                if (scale > maxScale) maxScale = scale;

                float minOffset = -bMin;
                subMin[ib] = minOffset;
                if (minOffset > maxMin) maxMin = minOffset;
            }

            ComputeQ4KSubScalesAndHeader(sbDst, subScale, subMin, sc, m, maxScale, maxMin);

            float dSuper = (float)sbDst->D;
            float dminSuper = (float)sbDst->DMin;

            for (int ib = 0; ib < 8; ib++)
            {
                float d = dSuper * sc[ib];
                float dm = dminSuper * m[ib];
                float invD = d > 0f ? 1.0f / d : 0f;
                byte* bL = L + ib * 32;

                if (invD <= 0f)
                {
                    vZeroByte.Store(bL);
                    continue;
                }

                float* bSrc = sbSrc + ib * 32;
                var vDm = Vector512.Create(dm);
                var vInvD = Vector512.Create(invD);

                var v0 = Vector512.Load(bSrc);
                var q0 = Avx512F.ConvertToVector512Int32(Vector512.Multiply(Vector512.Add(v0, vDm), vInvD));
                q0 = Vector512.Min(Vector512.Max(q0, vZeroInt), vFifteenInt);
                Avx512F.ConvertToVector128Byte(q0).Store(bL);

                var v1 = Vector512.Load(bSrc + 16);
                var q1 = Avx512F.ConvertToVector512Int32(Vector512.Multiply(Vector512.Add(v1, vDm), vInvD));
                q1 = Vector512.Min(Vector512.Max(q1, vZeroInt), vFifteenInt);
                Avx512F.ConvertToVector128Byte(q1).Store(bL + 16);
            }

            PackQ4KNibblesVector256(L, sbDst->Quants);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void QuantizeRowQ4_KAvx2(float* pSrc, BlockQ4_K* pDst, int numSuperBlocks)
    {
        float* subScale = stackalloc float[8];
        float* subMin = stackalloc float[8];
        byte* sc = stackalloc byte[8];
        byte* m = stackalloc byte[8];
        byte* L = stackalloc byte[256];

        var vPosInf = Vector256.Create(float.PositiveInfinity);
        var vNegInf = Vector256.Create(float.NegativeInfinity);
        var vZeroInt = Vector256<int>.Zero;
        var vFifteenInt = Vector256.Create(15);
        var vZeroByte = Vector256<byte>.Zero;

        for (int b = 0; b < numSuperBlocks; b++)
        {
            float* sbSrc = pSrc + (b * BlockQ4_K.BlockSize);
            BlockQ4_K* sbDst = pDst + b;

            float maxScale = 0f;
            float maxMin = 0f;

            for (int ib = 0; ib < 8; ib++)
            {
                float* bSrc = sbSrc + ib * 32;
                var v0 = Vector256.Load(bSrc);
                var v1 = Vector256.Load(bSrc + 8);
                var v2 = Vector256.Load(bSrc + 16);
                var v3 = Vector256.Load(bSrc + 24);

                var fin0 = Vector256.Equals(v0, v0);
                var fin1 = Vector256.Equals(v1, v1);
                var fin2 = Vector256.Equals(v2, v2);
                var fin3 = Vector256.Equals(v3, v3);

                var minVec = Vector256.Min(
                    Vector256.Min(Vector256.ConditionalSelect(fin0, v0, vPosInf), Vector256.ConditionalSelect(fin1, v1, vPosInf)),
                    Vector256.Min(Vector256.ConditionalSelect(fin2, v2, vPosInf), Vector256.ConditionalSelect(fin3, v3, vPosInf)));
                var maxVec = Vector256.Max(
                    Vector256.Max(Vector256.ConditionalSelect(fin0, v0, vNegInf), Vector256.ConditionalSelect(fin1, v1, vNegInf)),
                    Vector256.Max(Vector256.ConditionalSelect(fin2, v2, vNegInf), Vector256.ConditionalSelect(fin3, v3, vNegInf)));

                float bMin = HorizontalMin256(minVec);
                float bMax = HorizontalMax256(maxVec);

                if (float.IsPositiveInfinity(bMin) || float.IsNegativeInfinity(bMax))
                {
                    bMin = 0f;
                    bMax = 0f;
                }

                if (bMin > 0f) bMin = 0f;
                float range = bMax - bMin;
                float scale = range / 15.0f;
                subScale[ib] = scale;
                if (scale > maxScale) maxScale = scale;

                float minOffset = -bMin;
                subMin[ib] = minOffset;
                if (minOffset > maxMin) maxMin = minOffset;
            }

            ComputeQ4KSubScalesAndHeader(sbDst, subScale, subMin, sc, m, maxScale, maxMin);

            float dSuper = (float)sbDst->D;
            float dminSuper = (float)sbDst->DMin;

            for (int ib = 0; ib < 8; ib++)
            {
                float d = dSuper * sc[ib];
                float dm = dminSuper * m[ib];
                float invD = d > 0f ? 1.0f / d : 0f;
                byte* bL = L + ib * 32;

                if (invD <= 0f)
                {
                    vZeroByte.Store(bL);
                    continue;
                }

                float* bSrc = sbSrc + ib * 32;
                var vDm = Vector256.Create(dm);
                var vInvD = Vector256.Create(invD);

                var q0 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(Vector256.Add(Vector256.Load(bSrc), vDm), vInvD)), vZeroInt), vFifteenInt);
                var q1 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(Vector256.Add(Vector256.Load(bSrc + 8), vDm), vInvD)), vZeroInt), vFifteenInt);
                var p01 = Sse2.PackSignedSaturate(q0.GetLower(), q0.GetUpper());
                var p23 = Sse2.PackSignedSaturate(q1.GetLower(), q1.GetUpper());
                Sse2.PackUnsignedSaturate(p01, p23).Store(bL);

                var q2 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(Vector256.Add(Vector256.Load(bSrc + 16), vDm), vInvD)), vZeroInt), vFifteenInt);
                var q3 = Vector256.Min(Vector256.Max(Avx.ConvertToVector256Int32(Vector256.Multiply(Vector256.Add(Vector256.Load(bSrc + 24), vDm), vInvD)), vZeroInt), vFifteenInt);
                var p45 = Sse2.PackSignedSaturate(q2.GetLower(), q2.GetUpper());
                var p67 = Sse2.PackSignedSaturate(q3.GetLower(), q3.GetUpper());
                Sse2.PackUnsignedSaturate(p45, p67).Store(bL + 16);
            }

            PackQ4KNibblesVector256(L, sbDst->Quants);
        }
    }

    private static unsafe void QuantizeRowQ4_KScalarCore(float* pSrc, BlockQ4_K* pDst, int numSuperBlocks)
    {
        float* subScale = stackalloc float[8];
        float* subMin = stackalloc float[8];
        byte* sc = stackalloc byte[8];
        byte* m = stackalloc byte[8];
        byte* L = stackalloc byte[256];

        for (int b = 0; b < numSuperBlocks; b++)
        {
            float* sbSrc = pSrc + (b * BlockQ4_K.BlockSize);
            BlockQ4_K* sbDst = pDst + b;

            float maxScale = 0f;
            float maxMin = 0f;

            for (int ib = 0; ib < 8; ib++)
            {
                float* bSrc = sbSrc + ib * 32;
                float bMin = float.PositiveInfinity;
                float bMax = float.NegativeInfinity;
                for (int i = 0; i < 32; i++)
                {
                    float v = bSrc[i];
                    if (v < bMin) bMin = v;
                    if (v > bMax) bMax = v;
                }

                if (float.IsPositiveInfinity(bMin) || float.IsNegativeInfinity(bMax))
                {
                    bMin = 0f;
                    bMax = 0f;
                }

                if (bMin > 0f) bMin = 0f;
                float range = bMax - bMin;
                float scale = range / 15.0f;
                subScale[ib] = scale;
                if (scale > maxScale) maxScale = scale;

                float minOffset = -bMin;
                subMin[ib] = minOffset;
                if (minOffset > maxMin) maxMin = minOffset;
            }

            ComputeQ4KSubScalesAndHeader(sbDst, subScale, subMin, sc, m, maxScale, maxMin);

            float dSuper = (float)sbDst->D;
            float dminSuper = (float)sbDst->DMin;

            for (int ib = 0; ib < 8; ib++)
            {
                float d = dSuper * sc[ib];
                float dm = dminSuper * m[ib];
                float invD = d > 0f ? 1.0f / d : 0f;

                float* bSrc = sbSrc + ib * 32;
                byte* bL = L + ib * 32;

                for (int i = 0; i < 32; i++)
                {
                    int q = invD > 0f ? (int)MathF.Round((bSrc[i] + dm) * invD) : 0;
                    bL[i] = (byte)Math.Clamp(q, 0, 15);
                }
            }

            byte* qDst = sbDst->Quants;
            for (int g = 0; g < 256; g += 64)
            {
                for (int l = 0; l < 32; l++)
                {
                    qDst[l] = (byte)(L[g + l] | (L[g + l + 32] << 4));
                }
                qDst += 32;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ComputeQ4KSubScalesAndHeader(
        BlockQ4_K* sbDst,
        float* subScale,
        float* subMin,
        byte* sc,
        byte* m,
        float maxScale,
        float maxMin)
    {
        float invScale = maxScale > 0f ? 63.0f / maxScale : 0f;
        float invMin = maxMin > 0f ? 63.0f / maxMin : 0f;

        for (int ib = 0; ib < 8; ib++)
        {
            int scQuant = (int)MathF.Round(subScale[ib] * invScale);
            sc[ib] = (byte)Math.Clamp(scQuant, 0, 63);

            int mQuant = (int)MathF.Round(subMin[ib] * invMin);
            m[ib] = (byte)Math.Clamp(mQuant, 0, 63);
        }

        for (int s = 0; s < 12; s++) sbDst->Scales[s] = 0;
        for (int j = 0; j < 8; j++)
        {
            SetScaleMinK4(j, sbDst->Scales, sc[j], m[j]);
        }

        sbDst->D = (Half)(maxScale / 63.0f);
        sbDst->DMin = (Half)(maxMin / 63.0f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void PackQ4KNibblesVector256(byte* L, byte* qDst)
    {
        for (int g = 0; g < 256; g += 64)
        {
            var vLo = Vector256.Load(L + g);
            var vHi = Vector256.Load(L + g + 32);
            var vShifted = Vector256.ShiftLeft(vHi.AsUInt16(), 4).AsByte();
            Vector256.BitwiseOr(vLo, vShifted).Store(qDst);
            qDst += 32;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRowQ4_K(ReadOnlySpan<BlockQ4_K> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockQ4_K.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        if (src.IsEmpty)
            return;

        fixed (BlockQ4_K* pSrc = src)
        fixed (float* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                var vLowMask = Vector256.Create((byte)0x0F);

                for (int b = 0; b < src.Length; b++)
                {
                    BlockQ4_K* sbSrc = pSrc + b;
                    float* sbDst = pDst + (b * BlockQ4_K.BlockSize);

                    float d = (float)sbSrc->D;
                    float dmin = (float)sbSrc->DMin;

                    int isIdx = 0;
                    for (int g = 0; g < 256; g += 64)
                    {
                        GetScaleMinK4(isIdx, sbSrc->Scales, out byte sc0, out byte m0);
                        GetScaleMinK4(isIdx + 1, sbSrc->Scales, out byte sc1, out byte m1);

                        var vd0 = Vector512.Create(d * sc0);
                        var vdm0 = Vector512.Create(dmin * m0);
                        var vd1 = Vector512.Create(d * sc1);
                        var vdm1 = Vector512.Create(dmin * m1);

                        var packed = Vector256.Load(sbSrc->Quants + (g >> 1));
                        var vLo = Vector256.BitwiseAnd(packed, vLowMask);
                        var vHi = Vector256.BitwiseAnd(Vector256.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), vLowMask);

                        var f0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(vLo.GetLower()));
                        Avx512F.FusedMultiplySubtract(f0, vd0, vdm0).Store(sbDst + g);

                        var f1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(vLo.GetUpper()));
                        Avx512F.FusedMultiplySubtract(f1, vd0, vdm0).Store(sbDst + g + 16);

                        var f2 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(vHi.GetLower()));
                        Avx512F.FusedMultiplySubtract(f2, vd1, vdm1).Store(sbDst + g + 32);

                        var f3 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(vHi.GetUpper()));
                        Avx512F.FusedMultiplySubtract(f3, vd1, vdm1).Store(sbDst + g + 48);

                        isIdx += 2;
                    }
                }
            }
            else if (Avx2.IsSupported)
            {
                var vLowMask = Vector256.Create((byte)0x0F);

                for (int b = 0; b < src.Length; b++)
                {
                    BlockQ4_K* sbSrc = pSrc + b;
                    float* sbDst = pDst + (b * BlockQ4_K.BlockSize);

                    float d = (float)sbSrc->D;
                    float dmin = (float)sbSrc->DMin;

                    int isIdx = 0;
                    for (int g = 0; g < 256; g += 64)
                    {
                        GetScaleMinK4(isIdx, sbSrc->Scales, out byte sc0, out byte m0);
                        GetScaleMinK4(isIdx + 1, sbSrc->Scales, out byte sc1, out byte m1);

                        var vd0 = Vector256.Create(d * sc0);
                        var vdm0 = Vector256.Create(dmin * m0);
                        var vd1 = Vector256.Create(d * sc1);
                        var vdm1 = Vector256.Create(dmin * m1);

                        var packed = Vector256.Load(sbSrc->Quants + (g >> 1));
                        var vLo = Vector256.BitwiseAnd(packed, vLowMask);
                        var vHi = Vector256.BitwiseAnd(Vector256.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), vLowMask);

                        var lo0 = vLo.GetLower();
                        var lo1 = vLo.GetUpper();
                        var hi0 = vHi.GetLower();
                        var hi1 = vHi.GetUpper();

                        var f0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo0));
                        var f1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo0, 8)));
                        var f2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo1));
                        var f3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo1, 8)));

                        var g0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi0));
                        var g1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi0, 8)));
                        var g2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi1));
                        var g3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi1, 8)));

                        if (Fma.IsSupported)
                        {
                            Fma.MultiplySubtract(f0, vd0, vdm0).Store(sbDst + g);
                            Fma.MultiplySubtract(f1, vd0, vdm0).Store(sbDst + g + 8);
                            Fma.MultiplySubtract(f2, vd0, vdm0).Store(sbDst + g + 16);
                            Fma.MultiplySubtract(f3, vd0, vdm0).Store(sbDst + g + 24);

                            Fma.MultiplySubtract(g0, vd1, vdm1).Store(sbDst + g + 32);
                            Fma.MultiplySubtract(g1, vd1, vdm1).Store(sbDst + g + 40);
                            Fma.MultiplySubtract(g2, vd1, vdm1).Store(sbDst + g + 48);
                            Fma.MultiplySubtract(g3, vd1, vdm1).Store(sbDst + g + 56);
                        }
                        else
                        {
                            Avx.Subtract(Avx.Multiply(f0, vd0), vdm0).Store(sbDst + g);
                            Avx.Subtract(Avx.Multiply(f1, vd0), vdm0).Store(sbDst + g + 8);
                            Avx.Subtract(Avx.Multiply(f2, vd0), vdm0).Store(sbDst + g + 16);
                            Avx.Subtract(Avx.Multiply(f3, vd0), vdm0).Store(sbDst + g + 24);

                            Avx.Subtract(Avx.Multiply(g0, vd1), vdm1).Store(sbDst + g + 32);
                            Avx.Subtract(Avx.Multiply(g1, vd1), vdm1).Store(sbDst + g + 40);
                            Avx.Subtract(Avx.Multiply(g2, vd1), vdm1).Store(sbDst + g + 48);
                            Avx.Subtract(Avx.Multiply(g3, vd1), vdm1).Store(sbDst + g + 56);
                        }

                        isIdx += 2;
                    }
                }
            }
            else
            {
                DequantizeRowQ4_KScalarCore(pSrc, pDst, src.Length);
            }
        }
    }

    internal static unsafe void DequantizeRowQ4_KScalar(ReadOnlySpan<BlockQ4_K> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockQ4_K.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {totalElements} elements.");

        if (src.IsEmpty)
            return;

        fixed (BlockQ4_K* pSrc = src)
        fixed (float* pDst = dst)
        {
            DequantizeRowQ4_KScalarCore(pSrc, pDst, src.Length);
        }
    }

    private static unsafe void DequantizeRowQ4_KScalarCore(BlockQ4_K* pSrc, float* pDst, int numBlocks)
    {
        for (int b = 0; b < numBlocks; b++)
        {
            BlockQ4_K* sbSrc = pSrc + b;
            float* sbDst = pDst + (b * BlockQ4_K.BlockSize);

            float d = (float)sbSrc->D;
            float dmin = (float)sbSrc->DMin;

            int isIdx = 0;
            for (int g = 0; g < 256; g += 64)
            {
                GetScaleMinK4(isIdx, sbSrc->Scales, out byte sc0, out byte m0);
                GetScaleMinK4(isIdx + 1, sbSrc->Scales, out byte sc1, out byte m1);
                float d0 = d * sc0;
                float dm0 = dmin * m0;
                float d1 = d * sc1;
                float dm1 = dmin * m1;

                byte* q = sbSrc->Quants + (g >> 1);
                for (int l = 0; l < 32; l++)
                {
                    byte packed = q[l];
                    sbDst[g + l] = d0 * (packed & 0x0F) - dm0;
                    sbDst[g + l + 32] = d1 * ((packed >> 4) & 0x0F) - dm1;
                }
                isIdx += 2;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SetScaleMinK4(int j, byte* scales, byte sc, byte m)
    {
        if (j < 4)
        {
            scales[j] = (byte)((scales[j] & 0xC0) | (sc & 0x3F));
            scales[j + 4] = (byte)((scales[j + 4] & 0xC0) | (m & 0x3F));
        }
        else
        {
            scales[j + 4] = (byte)((sc & 0x0F) | ((m & 0x0F) << 4));
            scales[j - 4] = (byte)((scales[j - 4] & 0x3F) | ((sc >> 4) << 6));
            scales[j] = (byte)((scales[j] & 0x3F) | ((m >> 4) << 6));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    // =========================================================================
    // 3. FP8 (E4M3FN) PER-BLOCK / DYNAMIC QUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void QuantizeRowFP8_E4M3(ReadOnlySpan<float> src, Span<byte> dst, out float dynamicScale)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination span too small.");

        int len = src.Length;
        if (len == 0)
        {
            dynamicScale = 1.0f;
            return;
        }

        const float maxFp8 = 448.0f;

        fixed (float* pSrc = src)
        fixed (byte* pDst = dst)
        {
            float amax = ComputeMaxAbs(pSrc, len);
            dynamicScale = amax > 1e-12f ? amax / maxFp8 : 1.0f;
            float invScale = 1.0f / dynamicScale;

            int i = 0;

            if (Avx512F.IsSupported && len >= 16)
            {
                var vInvScale = Vector512.Create(invScale);
                var vAbsMask = Vector512.Create(0x7FFFFFFFu);
                var vSign8Mask = Vector512.Create(0x80u);
                var vOverflowThresh = Vector512.Create(0x43E00000u);
                var vSubThresh = Vector512.Create(0x3C800000u);
                var vInfBits = Vector512.Create(0x7F800000u);
                var vExpBias = Vector512.Create(120u);
                var vMant23Mask = Vector512.Create(0x7FFFFFu);
                var vRemMask = Vector512.Create(0xFFFFFu);
                var vHalfRem = Vector512.Create(0x80000u);
                var vOne = Vector512.Create(1u);
                var vEight = Vector512.Create(8u);
                var vMaxMagByte = Vector512.Create(0x7Eu);
                var vNanByte = Vector512.Create(0x7Fu);
                var v512 = Vector512.Create(512.0f);

                for (; i + 16 <= len; i += 16)
                {
                    var scaled = Vector512.Multiply(Vector512.Load(pSrc + i), vInvScale);
                    var bits = scaled.AsUInt32();
                    var sign8 = Vector512.BitwiseAnd(Vector512.ShiftRightLogical(bits, 24), vSign8Mask);
                    var absBits = Vector512.BitwiseAnd(bits, vAbsMask);

                    // Normal [2^-6, 448)
                    var e4 = Vector512.Subtract(Vector512.ShiftRightLogical(absBits, 23), vExpBias);
                    var mant23 = Vector512.BitwiseAnd(absBits, vMant23Mask);
                    var m3 = Vector512.ShiftRightLogical(mant23, 20);
                    var rem = Vector512.BitwiseAnd(mant23, vRemMask);

                    var gtHalf = Vector512.GreaterThan(rem, vHalfRem);
                    var eqHalf = Vector512.Equals(rem, vHalfRem);
                    var oddM3 = Vector512.Equals(Vector512.BitwiseAnd(m3, vOne), vOne);
                    var roundMask = Vector512.BitwiseOr(gtHalf, Vector512.BitwiseAnd(eqHalf, oddM3));

                    m3 = Vector512.Add(m3, Vector512.BitwiseAnd(roundMask, vOne));
                    var carryMask = Vector512.Equals(m3, vEight);
                    m3 = Vector512.ConditionalSelect(carryMask, Vector512<uint>.Zero, m3);
                    e4 = Vector512.Add(e4, Vector512.BitwiseAnd(carryMask, vOne));

                    var normByte = Vector512.Min(Vector512.BitwiseOr(Vector512.ShiftLeft(e4, 3), m3), vMaxMagByte);

                    // Subnormal [0, 2^-6)
                    var subF = Avx512F.RoundScale(Vector512.Multiply(absBits.AsSingle(), v512), 0x00);
                    var qSub = Avx512F.ConvertToVector512Int32(subF).AsUInt32();
                    var subByte = Vector512.Min(qSub, vEight);

                    var isSubMask = Vector512.LessThan(absBits, vSubThresh);
                    var magByte = Vector512.ConditionalSelect(isSubMask, subByte, normByte);

                    var isOverflowMask = Vector512.GreaterThanOrEqual(absBits, vOverflowThresh);
                    magByte = Vector512.ConditionalSelect(isOverflowMask, vMaxMagByte, magByte);

                    var res32 = Vector512.BitwiseOr(sign8, magByte);
                    var isNanMask = Vector512.GreaterThan(absBits, vInfBits);
                    res32 = Vector512.ConditionalSelect(isNanMask, vNanByte, res32);

                    Avx512F.ConvertToVector128Byte(res32).Store(pDst + i);
                }
            }
            else if (Avx2.IsSupported && len >= 8)
            {
                var vInvScale = Vector256.Create(invScale);
                var vAbsMask = Vector256.Create(0x7FFFFFFFu);
                var vSign8Mask = Vector256.Create(0x80u);
                var vOverflowThresh = Vector256.Create(0x43E00000u);
                var vSubThresh = Vector256.Create(0x3C800000u);
                var vInfBits = Vector256.Create(0x7F800000u);
                var vExpBias = Vector256.Create(120u);
                var vMant23Mask = Vector256.Create(0x7FFFFFu);
                var vRemMask = Vector256.Create(0xFFFFFu);
                var vHalfRem = Vector256.Create(0x80000u);
                var vOne = Vector256.Create(1u);
                var vEight = Vector256.Create(8u);
                var vMaxMagByte = Vector256.Create(0x7Eu);
                var vNanByte = Vector256.Create(0x7Fu);
                var v512 = Vector256.Create(512.0f);

                for (; i + 8 <= len; i += 8)
                {
                    var scaled = Vector256.Multiply(Vector256.Load(pSrc + i), vInvScale);
                    var bits = scaled.AsUInt32();
                    var sign8 = Vector256.BitwiseAnd(Vector256.ShiftRightLogical(bits, 24), vSign8Mask);
                    var absBits = Vector256.BitwiseAnd(bits, vAbsMask);

                    var e4 = Vector256.Subtract(Vector256.ShiftRightLogical(absBits, 23), vExpBias);
                    var mant23 = Vector256.BitwiseAnd(absBits, vMant23Mask);
                    var m3 = Vector256.ShiftRightLogical(mant23, 20);
                    var rem = Vector256.BitwiseAnd(mant23, vRemMask);

                    var gtHalf = Vector256.GreaterThan(rem, vHalfRem);
                    var eqHalf = Vector256.Equals(rem, vHalfRem);
                    var oddM3 = Vector256.Equals(Vector256.BitwiseAnd(m3, vOne), vOne);
                    var roundMask = Vector256.BitwiseOr(gtHalf, Vector256.BitwiseAnd(eqHalf, oddM3));

                    m3 = Vector256.Add(m3, Vector256.BitwiseAnd(roundMask, vOne));
                    var carryMask = Vector256.Equals(m3, vEight);
                    m3 = Vector256.ConditionalSelect(carryMask, Vector256<uint>.Zero, m3);
                    e4 = Vector256.Add(e4, Vector256.BitwiseAnd(carryMask, vOne));

                    var normByte = Vector256.Min(Vector256.BitwiseOr(Vector256.ShiftLeft(e4, 3), m3), vMaxMagByte);

                    var subF = Avx.RoundToNearestInteger(Vector256.Multiply(absBits.AsSingle(), v512));
                    var qSub = Avx.ConvertToVector256Int32(subF).AsUInt32();
                    var subByte = Vector256.Min(qSub, vEight);

                    var isSubMask = Vector256.LessThan(absBits, vSubThresh);
                    var magByte = Vector256.ConditionalSelect(isSubMask, subByte, normByte);

                    var isOverflowMask = Vector256.GreaterThanOrEqual(absBits, vOverflowThresh);
                    magByte = Vector256.ConditionalSelect(isOverflowMask, vMaxMagByte, magByte);

                    var res32 = Vector256.BitwiseOr(sign8, magByte).AsInt32();
                    var isNanMask = Vector256.GreaterThan(absBits, vInfBits).AsInt32();
                    res32 = Vector256.ConditionalSelect(isNanMask, vNanByte.AsInt32(), res32);

                    var p16 = Sse2.PackSignedSaturate(res32.GetLower(), res32.GetUpper());
                    var p8 = Sse2.PackUnsignedSaturate(p16, p16);
                    *(ulong*)(pDst + i) = p8.AsUInt64().ToScalar();
                }
            }

            for (; i < len; i++)
            {
                pDst[i] = FloatToFp8E4M3(pSrc[i] * invScale);
            }
        }
    }

    internal static void QuantizeRowFP8_E4M3Scalar(ReadOnlySpan<float> src, Span<byte> dst, out float dynamicScale)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination span too small.");

        float amax = 0.0f;
        for (int i = 0; i < src.Length; i++)
        {
            float abs = MathF.Abs(src[i]);
            if (abs > amax) amax = abs;
        }

        const float maxFp8 = 448.0f;
        dynamicScale = amax > 1e-12f ? amax / maxFp8 : 1.0f;
        float invScale = 1.0f / dynamicScale;

        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = FloatToFp8E4M3(src[i] * invScale);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRowFP8_E4M3(ReadOnlySpan<byte> src, float dynamicScale, Span<float> dst)
    {
        if (dst.Length < src.Length)
            throw new ArgumentException("Destination span too small.");

        int len = src.Length;
        if (len == 0)
            return;

        fixed (byte* pSrc = src)
        fixed (float* pDst = dst)
        {
            int i = 0;

            if (Avx512F.IsSupported && len >= 16)
            {
                var vScale = Vector512.Create(dynamicScale);
                var vSignMask = Vector512.Create(0x80u);
                var vAbsMask = Vector512.Create(0x7Fu);
                var vMantMask = Vector512.Create(0x07u);
                var vExpBias = Vector512.Create(120u);
                var vNanByte = Vector512.Create(0x7Fu);
                var vNanFloatBits = Vector512.Create(BitConverter.SingleToUInt32Bits(float.NaN));
                var vSubScale = Vector512.Create(1.0f / 512.0f);

                for (; i + 16 <= len; i += 16)
                {
                    var u = Avx512F.ConvertToVector512Int32(Vector128.Load(pSrc + i)).AsUInt32();
                    var sign = Vector512.ShiftLeft(Vector512.BitwiseAnd(u, vSignMask), 24);
                    var abs = Vector512.BitwiseAnd(u, vAbsMask);
                    var exp = Vector512.ShiftRightLogical(abs, 3);
                    var mant = Vector512.BitwiseAnd(abs, vMantMask);

                    var normBits = Vector512.BitwiseOr(
                        Vector512.ShiftLeft(Vector512.Add(exp, vExpBias), 23),
                        Vector512.ShiftLeft(mant, 20));

                    var subBits = Vector512.Multiply(Avx512F.ConvertToVector512Single(mant.AsInt32()), vSubScale).AsUInt32();
                    var isSub = Vector512.Equals(exp, Vector512<uint>.Zero);
                    var magBits = Vector512.ConditionalSelect(isSub, subBits, normBits);

                    var outBits = Vector512.BitwiseOr(sign, magBits);
                    var isNan = Vector512.Equals(abs, vNanByte);
                    outBits = Vector512.ConditionalSelect(isNan, vNanFloatBits, outBits);

                    Vector512.Multiply(outBits.AsSingle(), vScale).Store(pDst + i);
                }
            }
            else if (Avx2.IsSupported && len >= 8)
            {
                var vScale = Vector256.Create(dynamicScale);
                var vSignMask = Vector256.Create(0x80u);
                var vAbsMask = Vector256.Create(0x7Fu);
                var vMantMask = Vector256.Create(0x07u);
                var vExpBias = Vector256.Create(120u);
                var vNanByte = Vector256.Create(0x7Fu);
                var vNanFloatBits = Vector256.Create(BitConverter.SingleToUInt32Bits(float.NaN));
                var vSubScale = Vector256.Create(1.0f / 512.0f);

                for (; i + 8 <= len; i += 8)
                {
                    var vBytes = Vector128.CreateScalar(*(ulong*)(pSrc + i)).AsByte();
                    var u = Avx2.ConvertToVector256Int32(vBytes).AsUInt32();
                    var sign = Vector256.ShiftLeft(Vector256.BitwiseAnd(u, vSignMask), 24);
                    var abs = Vector256.BitwiseAnd(u, vAbsMask);
                    var exp = Vector256.ShiftRightLogical(abs, 3);
                    var mant = Vector256.BitwiseAnd(abs, vMantMask);

                    var normBits = Vector256.BitwiseOr(
                        Vector256.ShiftLeft(Vector256.Add(exp, vExpBias), 23),
                        Vector256.ShiftLeft(mant, 20));

                    var subBits = Vector256.Multiply(Avx.ConvertToVector256Single(mant.AsInt32()), vSubScale).AsUInt32();
                    var isSub = Vector256.Equals(exp, Vector256<uint>.Zero);
                    var magBits = Vector256.ConditionalSelect(isSub, subBits, normBits);

                    var outBits = Vector256.BitwiseOr(sign, magBits);
                    var isNan = Vector256.Equals(abs, vNanByte);
                    outBits = Vector256.ConditionalSelect(isNan, vNanFloatBits, outBits);

                    Vector256.Multiply(outBits.AsSingle(), vScale).Store(pDst + i);
                }
            }

            for (; i < len; i++)
            {
                pDst[i] = Fp8E4M3ToFloat(pSrc[i]) * dynamicScale;
            }
        }
    }

    internal static void DequantizeRowFP8_E4M3Scalar(ReadOnlySpan<byte> src, float dynamicScale, Span<float> dst)
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
        if (float.IsNaN(val))
            return 0x7F;

        uint bits = BitConverter.SingleToUInt32Bits(val);
        byte sign = (byte)((bits >> 24) & 0x80u);
        uint absBits = bits & 0x7FFFFFFFu;

        // Saturation for |val| >= 448.0f (including +Inf/-Inf and [448, 512)): never emit 0x7F/0xFF NaN
        if (absBits >= 0x43E00000u)
            return (byte)(sign | 0x7Eu);

        // Subnormal [0, 2^-6): mant / 512 with round-to-nearest-even
        if (absBits < 0x3C800000u)
        {
            float mag = BitConverter.UInt32BitsToSingle(absBits);
            int q = (int)MathF.Round(mag * 512.0f, MidpointRounding.ToEven);
            if (q <= 0)
                return sign; // Preserves -0.0f -> 0x80
            if (q >= 8)
                return (byte)(sign | 0x08u);
            return (byte)(sign | (byte)q);
        }

        // Normal [2^-6, 448.0f): round-to-nearest-even on 3-bit mantissa
        int e4 = (int)(absBits >> 23) - 120;
        uint mant23 = absBits & 0x7FFFFFu;
        uint m3 = mant23 >> 20;
        uint rem = mant23 & 0xFFFFFu;

        if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u) != 0u))
        {
            m3++;
            if (m3 == 8u)
            {
                m3 = 0u;
                e4++;
            }
        }

        if (e4 > 15 || (e4 == 15 && m3 >= 7u))
            return (byte)(sign | 0x7Eu);

        return (byte)(sign | (byte)((e4 << 3) | (int)m3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Fp8E4M3ToFloat(byte val)
    {
        return s_fp8E4M3ToFloat[val];
    }

    // =========================================================================
    // 4. 1-BIT BINARY QUANTIZATION
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void QuantizeRow1Bit(ReadOnlySpan<float> src, Span<BlockBinary1Bit> dst)
    {
        if (src.Length % BlockBinary1Bit.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of {BlockBinary1Bit.BlockSize}.");

        int numBlocks = src.Length / BlockBinary1Bit.BlockSize;
        if (dst.Length < numBlocks)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {numBlocks} blocks.");

        if (numBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockBinary1Bit* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                var vZero = Vector512<float>.Zero;

                for (int b = 0; b < numBlocks; b++)
                {
                    float* s = pSrc + (b * BlockBinary1Bit.BlockSize);
                    var v0 = Vector512.Load(s);
                    var v1 = Vector512.Load(s + 16);
                    var v2 = Vector512.Load(s + 32);
                    var v3 = Vector512.Load(s + 48);

                    var sum16 = Vector512.Add(Vector512.Add(v0, v1), Vector512.Add(v2, v3));
                    float sum = ReduceSum16Tree(sum16);
                    float bias = sum / BlockBinary1Bit.BlockSize;
                    var vBias = Vector512.Create(bias);

                    var d0 = Vector512.Subtract(v0, vBias);
                    var d1 = Vector512.Subtract(v1, vBias);
                    var d2 = Vector512.Subtract(v2, vBias);
                    var d3 = Vector512.Subtract(v3, vBias);

                    var absSum16 = Vector512.Add(
                        Vector512.Add(Vector512.Abs(d0), Vector512.Abs(d1)),
                        Vector512.Add(Vector512.Abs(d2), Vector512.Abs(d3)));
                    float absSum = ReduceSum16Tree(absSum16);
                    float scale = Math.Max(1e-12f, absSum / BlockBinary1Bit.BlockSize);

                    ulong m0 = Vector512.GreaterThanOrEqual(d0, vZero).ExtractMostSignificantBits();
                    ulong m1 = Vector512.GreaterThanOrEqual(d1, vZero).ExtractMostSignificantBits();
                    ulong m2 = Vector512.GreaterThanOrEqual(d2, vZero).ExtractMostSignificantBits();
                    ulong m3 = Vector512.GreaterThanOrEqual(d3, vZero).ExtractMostSignificantBits();
                    ulong bits = m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);

                    pDst[b] = new BlockBinary1Bit
                    {
                        Bits = bits,
                        Scale = scale,
                        Bias = bias
                    };
                }
            }
            else if (Avx2.IsSupported)
            {
                var vZero = Vector256<float>.Zero;

                for (int b = 0; b < numBlocks; b++)
                {
                    float* s = pSrc + (b * BlockBinary1Bit.BlockSize);
                    var v0Lo = Vector256.Load(s);
                    var v0Hi = Vector256.Load(s + 8);
                    var v1Lo = Vector256.Load(s + 16);
                    var v1Hi = Vector256.Load(s + 24);
                    var v2Lo = Vector256.Load(s + 32);
                    var v2Hi = Vector256.Load(s + 40);
                    var v3Lo = Vector256.Load(s + 48);
                    var v3Hi = Vector256.Load(s + 56);

                    var sumLo = Vector256.Add(Vector256.Add(v0Lo, v1Lo), Vector256.Add(v2Lo, v3Lo));
                    var sumHi = Vector256.Add(Vector256.Add(v0Hi, v1Hi), Vector256.Add(v2Hi, v3Hi));
                    float sum = ReduceSum8Tree(Vector256.Add(sumLo, sumHi));
                    float bias = sum / BlockBinary1Bit.BlockSize;
                    var vBias = Vector256.Create(bias);

                    var d0Lo = Vector256.Subtract(v0Lo, vBias);
                    var d0Hi = Vector256.Subtract(v0Hi, vBias);
                    var d1Lo = Vector256.Subtract(v1Lo, vBias);
                    var d1Hi = Vector256.Subtract(v1Hi, vBias);
                    var d2Lo = Vector256.Subtract(v2Lo, vBias);
                    var d2Hi = Vector256.Subtract(v2Hi, vBias);
                    var d3Lo = Vector256.Subtract(v3Lo, vBias);
                    var d3Hi = Vector256.Subtract(v3Hi, vBias);

                    var absLo = Vector256.Add(
                        Vector256.Add(Vector256.Abs(d0Lo), Vector256.Abs(d1Lo)),
                        Vector256.Add(Vector256.Abs(d2Lo), Vector256.Abs(d3Lo)));
                    var absHi = Vector256.Add(
                        Vector256.Add(Vector256.Abs(d0Hi), Vector256.Abs(d1Hi)),
                        Vector256.Add(Vector256.Abs(d2Hi), Vector256.Abs(d3Hi)));
                    float absSum = ReduceSum8Tree(Vector256.Add(absLo, absHi));
                    float scale = Math.Max(1e-12f, absSum / BlockBinary1Bit.BlockSize);

                    ulong b0 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d0Lo, vZero));
                    ulong b1 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d0Hi, vZero));
                    ulong b2 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d1Lo, vZero));
                    ulong b3 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d1Hi, vZero));
                    ulong b4 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d2Lo, vZero));
                    ulong b5 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d2Hi, vZero));
                    ulong b6 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d3Lo, vZero));
                    ulong b7 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(d3Hi, vZero));
                    ulong bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56);

                    pDst[b] = new BlockBinary1Bit
                    {
                        Bits = bits,
                        Scale = scale,
                        Bias = bias
                    };
                }
            }
            else
            {
                QuantizeRow1BitScalarCore(pSrc, pDst, numBlocks);
            }
        }
    }

    internal static unsafe void QuantizeRow1BitScalar(ReadOnlySpan<float> src, Span<BlockBinary1Bit> dst)
    {
        if (src.Length % BlockBinary1Bit.BlockSize != 0)
            throw new ArgumentException($"Source length ({src.Length}) must be a multiple of {BlockBinary1Bit.BlockSize}.");

        int numBlocks = src.Length / BlockBinary1Bit.BlockSize;
        if (dst.Length < numBlocks)
            throw new ArgumentException($"Destination length ({dst.Length}) is too small for {numBlocks} blocks.");

        if (numBlocks == 0)
            return;

        fixed (float* pSrc = src)
        fixed (BlockBinary1Bit* pDst = dst)
        {
            QuantizeRow1BitScalarCore(pSrc, pDst, numBlocks);
        }
    }

    private static unsafe void QuantizeRow1BitScalarCore(float* pSrc, BlockBinary1Bit* pDst, int numBlocks)
    {
        float* buf8 = stackalloc float[8];

        for (int b = 0; b < numBlocks; b++)
        {
            float* s = pSrc + (b * BlockBinary1Bit.BlockSize);

            // Identical pairwise tree reduction order as AVX-512 & AVX2 for bit-exact parity
            for (int k = 0; k < 8; k++)
            {
                float lo = (s[k] + s[k + 16]) + (s[k + 32] + s[k + 48]);
                float hi = (s[k + 8] + s[k + 24]) + (s[k + 40] + s[k + 56]);
                buf8[k] = lo + hi;
            }
            float sum = ReduceSum8Scalar(buf8);
            float bias = sum / BlockBinary1Bit.BlockSize;

            ulong bits = 0;
            for (int k = 0; k < 8; k++)
            {
                float d0 = s[k] - bias;
                float d16 = s[k + 16] - bias;
                float d32 = s[k + 32] - bias;
                float d48 = s[k + 48] - bias;
                float d8 = s[k + 8] - bias;
                float d24 = s[k + 24] - bias;
                float d40 = s[k + 40] - bias;
                float d56 = s[k + 56] - bias;

                if (d0 >= 0.0f) bits |= 1UL << k;
                if (d8 >= 0.0f) bits |= 1UL << (k + 8);
                if (d16 >= 0.0f) bits |= 1UL << (k + 16);
                if (d24 >= 0.0f) bits |= 1UL << (k + 24);
                if (d32 >= 0.0f) bits |= 1UL << (k + 32);
                if (d40 >= 0.0f) bits |= 1UL << (k + 40);
                if (d48 >= 0.0f) bits |= 1UL << (k + 48);
                if (d56 >= 0.0f) bits |= 1UL << (k + 56);

                float loAbs = (MathF.Abs(d0) + MathF.Abs(d16)) + (MathF.Abs(d32) + MathF.Abs(d48));
                float hiAbs = (MathF.Abs(d8) + MathF.Abs(d24)) + (MathF.Abs(d40) + MathF.Abs(d56));
                buf8[k] = loAbs + hiAbs;
            }

            float absSum = ReduceSum8Scalar(buf8);
            float scale = Math.Max(1e-12f, absSum / BlockBinary1Bit.BlockSize);

            pDst[b] = new BlockBinary1Bit
            {
                Bits = bits,
                Scale = scale,
                Bias = bias
            };
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DequantizeRow1Bit(ReadOnlySpan<BlockBinary1Bit> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockBinary1Bit.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException("Destination length too small.");

        if (src.IsEmpty)
            return;

        fixed (BlockBinary1Bit* pSrc = src)
        fixed (float* pDst = dst)
        {
            if (Avx512F.IsSupported)
            {
                var bitMasks = Vector512.Create(
                    1u << 0, 1u << 1, 1u << 2, 1u << 3,
                    1u << 4, 1u << 5, 1u << 6, 1u << 7,
                    1u << 8, 1u << 9, 1u << 10, 1u << 11,
                    1u << 12, 1u << 13, 1u << 14, 1u << 15);
                var zero = Vector512<uint>.Zero;

                for (int b = 0; b < src.Length; b++)
                {
                    BlockBinary1Bit block = pSrc[b];
                    ulong bits = block.Bits;
                    var vPos = Vector512.Create(block.Bias + block.Scale);
                    var vNeg = Vector512.Create(block.Bias - block.Scale);
                    float* d = pDst + (b * BlockBinary1Bit.BlockSize);

                    for (int chunk = 0; chunk < 64; chunk += 16)
                    {
                        var vBits = Vector512.Create((uint)((bits >> chunk) & 0xFFFFu));
                        var mask = Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), bitMasks).AsSingle();
                        Vector512.ConditionalSelect(mask, vPos, vNeg).Store(d + chunk);
                    }
                }
            }
            else if (Avx2.IsSupported)
            {
                var bitMasks = Vector256.Create(
                    1u << 0, 1u << 1, 1u << 2, 1u << 3,
                    1u << 4, 1u << 5, 1u << 6, 1u << 7);

                for (int b = 0; b < src.Length; b++)
                {
                    BlockBinary1Bit block = pSrc[b];
                    ulong bits = block.Bits;
                    var vPos = Vector256.Create(block.Bias + block.Scale);
                    var vNeg = Vector256.Create(block.Bias - block.Scale);
                    float* d = pDst + (b * BlockBinary1Bit.BlockSize);

                    for (int chunk = 0; chunk < 64; chunk += 8)
                    {
                        var vBits = Vector256.Create((uint)((bits >> chunk) & 0xFFu));
                        var mask = Vector256.Equals(Vector256.BitwiseAnd(vBits, bitMasks), bitMasks).AsSingle();
                        Vector256.ConditionalSelect(mask, vPos, vNeg).Store(d + chunk);
                    }
                }
            }
            else
            {
                DequantizeRow1BitScalarCore(pSrc, pDst, src.Length);
            }
        }
    }

    internal static unsafe void DequantizeRow1BitScalar(ReadOnlySpan<BlockBinary1Bit> src, Span<float> dst)
    {
        int totalElements = checked(src.Length * BlockBinary1Bit.BlockSize);
        if (dst.Length < totalElements)
            throw new ArgumentException("Destination length too small.");

        if (src.IsEmpty)
            return;

        fixed (BlockBinary1Bit* pSrc = src)
        fixed (float* pDst = dst)
        {
            DequantizeRow1BitScalarCore(pSrc, pDst, src.Length);
        }
    }

    private static unsafe void DequantizeRow1BitScalarCore(BlockBinary1Bit* pSrc, float* pDst, int numBlocks)
    {
        for (int b = 0; b < numBlocks; b++)
        {
            BlockBinary1Bit block = pSrc[b];
            float* d = pDst + (b * BlockBinary1Bit.BlockSize);
            ulong bits = block.Bits;
            float pos = block.Bias + block.Scale;
            float neg = block.Bias - block.Scale;

            for (int i = 0; i < BlockBinary1Bit.BlockSize; i++)
            {
                bool bit = ((bits >> i) & 1UL) == 1UL;
                d[i] = bit ? pos : neg;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int HammingDistance1Bit(ulong a, ulong b)
    {
        return BitOperations.PopCount(a ^ b);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int HammingDistance1Bit(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Span lengths must match ({a.Length} vs {b.Length}).");

        int total = 0;
        int i = 0;

        for (; i + 4 <= a.Length; i += 4)
        {
            total += BitOperations.PopCount(a[i] ^ b[i])
                   + BitOperations.PopCount(a[i + 1] ^ b[i + 1])
                   + BitOperations.PopCount(a[i + 2] ^ b[i + 2])
                   + BitOperations.PopCount(a[i + 3] ^ b[i + 3]);
        }

        for (; i < a.Length; i++)
            total += BitOperations.PopCount(a[i] ^ b[i]);

        return total;
    }

    // =========================================================================
    // SHARED SIMD HELPERS
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float ComputeMaxAbs(float* pSrc, int len)
    {
        float amax = 0.0f;
        int i = 0;

        if (Avx512F.IsSupported && len >= 16)
        {
            var vMax = Vector512<float>.Zero;
            for (; i + 16 <= len; i += 16)
            {
                var v = Vector512.Load(pSrc + i);
                var abs = Vector512.ConditionalSelect(Vector512.Equals(v, v), Vector512.Abs(v), Vector512<float>.Zero);
                vMax = Vector512.Max(vMax, abs);
            }
            amax = HorizontalMax512(vMax);
        }
        else if (Avx2.IsSupported && len >= 8)
        {
            var vMax = Vector256<float>.Zero;
            for (; i + 8 <= len; i += 8)
            {
                var v = Vector256.Load(pSrc + i);
                var abs = Vector256.ConditionalSelect(Vector256.Equals(v, v), Vector256.Abs(v), Vector256<float>.Zero);
                vMax = Vector256.Max(vMax, abs);
            }
            amax = HorizontalMax256(vMax);
        }

        for (; i < len; i++)
        {
            float abs = MathF.Abs(pSrc[i]);
            if (abs > amax) amax = abs;
        }

        return amax;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReduceSum16Tree(Vector512<float> v16)
    {
        return ReduceSum8Tree(Vector256.Add(v16.GetLower(), v16.GetUpper()));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReduceSum8Tree(Vector256<float> v8)
    {
        var v4 = Sse.Add(v8.GetLower(), v8.GetUpper());
        return (v4.GetElement(0) + v4.GetElement(2)) + (v4.GetElement(1) + v4.GetElement(3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float ReduceSum8Scalar(float* v8)
    {
        float s0 = v8[0] + v8[4];
        float s1 = v8[1] + v8[5];
        float s2 = v8[2] + v8[6];
        float s3 = v8[3] + v8[7];
        return (s0 + s2) + (s1 + s3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax512(Vector512<float> v)
    {
        return HorizontalMax256(Vector256.Max(v.GetLower(), v.GetUpper()));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax256(Vector256<float> v)
    {
        var v4 = Vector128.Max(v.GetLower(), v.GetUpper());
        v4 = Vector128.Max(v4, Sse.Shuffle(v4, v4, 0x4E));
        v4 = Vector128.Max(v4, Sse.Shuffle(v4, v4, 0xB1));
        return v4.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMin512(Vector512<float> v)
    {
        return HorizontalMin256(Vector256.Min(v.GetLower(), v.GetUpper()));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMin256(Vector256<float> v)
    {
        var v4 = Vector128.Min(v.GetLower(), v.GetUpper());
        v4 = Vector128.Min(v4, Sse.Shuffle(v4, v4, 0x4E));
        v4 = Vector128.Min(v4, Sse.Shuffle(v4, v4, 0xB1));
        return v4.ToScalar();
    }
}
