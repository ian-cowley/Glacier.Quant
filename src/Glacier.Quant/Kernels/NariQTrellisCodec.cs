namespace Glacier.Quant.Kernels;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Diagnostic error and code-range metrics for a decoded NariQ 16x16 trellis tile.
/// </summary>
public readonly record struct TileErrorMetrics(
    double Rmse,
    double Mae,
    float MaxAbsError,
    double SignalToNoiseDb,
    sbyte MinCode,
    sbyte MaxCode);

/// <summary>
/// Bit-exact NariQ Trellis-Coded Quantization codec (<c>TRL1..TRL6</c>, 1..6 bits/weight):
/// <list type="bullet">
///   <item>16x16 output-tile-major circular bitstream decoder using the <c>0x83DCD12Du</c> procedural codebook.</item>
///   <item>Portable 32-bit SWAR horizontal byte sum: <c>((((xv &gt;&gt; 2) &amp; 0x3F3F3F3Fu) * 0x01010101u) &gt;&gt; 24) - 126</c>.</item>
///   <item>AVX-512 / AVX2 SIMD tile row dot-product and 16x16 tile mat-vec kernels.</item>
///   <item>Tile inspection and quantization-error RMSE measurement helpers.</item>
/// </list>
/// </summary>
public static unsafe class NariQTrellisCodec
{
    public const int TileDim = 16;
    public const int TileElements = 256;
    public const int MinBits = 1;
    public const int MaxBits = 6;
    public const uint CodebookMultiplier = 0x83DCD12Du;

    /// <summary>
    /// Returns the number of 32-bit words per 16x16 tile for bitrate <paramref name="bits"/> (<c>1..6</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetTileWordCount(int bits)
    {
        ValidateBits(bits);
        return 8 * bits;
    }

    /// <summary>
    /// Returns the number of bytes per 16x16 tile for bitrate <paramref name="bits"/> (<c>1..6</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetTileByteCount(int bits)
    {
        ValidateBits(bits);
        return 32 * bits;
    }

    /// <summary>
    /// Decodes a 16-bit NariQ trellis state (<c>0..65535</c>) to its exact signed integer grid value in <c>[-126, +126]</c>
    /// using portable 32-bit SWAR horizontal byte summation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static sbyte DecodeStateSwar(uint state)
    {
        uint xv = unchecked((state & 0xFFFFu) * CodebookMultiplier);
        int v = (int)((((xv >> 2) & 0x3F3F3F3Fu) * 0x01010101u) >> 24) - 126;
        return (sbyte)v;
    }

    /// <summary>
    /// Scalar reference codebook decode for bit-exact SWAR verification.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static sbyte DecodeStateScalar(uint state)
    {
        uint xv = unchecked((state & 0xFFFFu) * CodebookMultiplier);
        int v = (int)(((xv >> 2) & 63u) + ((xv >> 10) & 63u) + ((xv >> 18) & 63u) + ((xv >> 26) & 63u)) - 126;
        return (sbyte)v;
    }

    /// <summary>
    /// Builds the 64 KB packed signed-int8 lookup table (16,384 uint32 words, 4 int8 values per word)
    /// mapping every 16-bit trellis state (<c>0..65535</c>) to its exact integer grid value in <c>[-126, +126]</c>.
    /// </summary>
    public static uint[] BuildPackedInt8Lut()
    {
        var lut = new uint[16384];
        for (uint s = 0; s < 65536; s++)
        {
            sbyte v = DecodeStateSwar(s);
            lut[s >> 2] |= (uint)(byte)v << (int)((s & 3u) << 3);
        }
        return lut;
    }

    // =========================================================================
    // 1. 16x16 TILE DECODERS (SWAR + SCALAR REFERENCE)
    // =========================================================================

    /// <summary>
    /// Decodes a single NariQ <c>TRL1..TRL6</c> 16x16 circular-bitstream tile into 256 signed int8 grid values
    /// in row-major order (<c>row * 16 + col</c>) using portable 32-bit SWAR codebook summation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DecodeTileInt8(ReadOnlySpan<uint> tileWords, int bits, Span<sbyte> dstRowMajor)
    {
        int nw = GetTileWordCount(bits);
        if (tileWords.Length < nw)
            throw new ArgumentException($"tileWords length ({tileWords.Length}) must be at least {nw} for TRL{bits}.", nameof(tileWords));
        if (dstRowMajor.Length < TileElements)
            throw new ArgumentException($"dstRowMajor length ({dstRowMajor.Length}) must be at least {TileElements}.", nameof(dstRowMajor));

        fixed (uint* t = tileWords)
        fixed (sbyte* dst = dstRowMajor)
        {
            DecodeTileInt8SwarPtr(t, bits, nw, dst);
        }
    }

    /// <summary>
    /// Scalar reference 16x16 tile decoder for bit-exact parity tests against <see cref="DecodeTileInt8"/>.
    /// </summary>
    internal static void DecodeTileInt8Scalar(ReadOnlySpan<uint> tileWords, int bits, Span<sbyte> dstRowMajor)
    {
        int nw = GetTileWordCount(bits);
        if (tileWords.Length < nw)
            throw new ArgumentException($"tileWords length ({tileWords.Length}) must be at least {nw} for TRL{bits}.", nameof(tileWords));
        if (dstRowMajor.Length < TileElements)
            throw new ArgumentException($"dstRowMajor length ({dstRowMajor.Length}) must be at least {TileElements}.", nameof(dstRowMajor));

        int total = TileElements * bits;
        fixed (uint* t = tileWords)
        fixed (sbyte* dst = dstRowMajor)
        {
            for (int p = 0; p < TileElements; p++)
            {
                int start = ((p + 1) * bits - 16 + total) % total;
                int w = start >> 5;
                int o = start & 31;
                ulong pair = ((ulong)t[w] << 32) | t[(w + 1) % nw];
                uint state = (uint)(pair >> (48 - o)) & 0xFFFFu;

                int a = p >> 3;
                int jj = p & 7;
                int row = (a >> 2) + 8 * ((jj >> 2) & 1);
                int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);

                dst[(row << 4) + col] = DecodeStateScalar(state);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodeTileInt8SwarPtr(uint* t, int bits, int nw, sbyte* dst)
    {
        int total = TileElements * bits;
        for (int p = 0; p < TileElements; p++)
        {
            int start = ((p + 1) * bits - 16 + total) % total;
            int w = start >> 5;
            int o = start & 31;
            int wNext = w + 1;
            if (wNext == nw) wNext = 0;

            ulong pair = ((ulong)t[w] << 32) | t[wNext];
            uint state = (uint)(pair >> (48 - o)) & 0xFFFFu;
            uint xv = unchecked(state * CodebookMultiplier);
            int v = (int)((((xv >> 2) & 0x3F3F3F3Fu) * 0x01010101u) >> 24) - 126;

            int a = p >> 3;
            int jj = p & 7;
            int row = (a >> 2) + 8 * ((jj >> 2) & 1);
            int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);

            dst[(row << 4) + col] = (sbyte)v;
        }
    }

    /// <summary>
    /// Decodes a single NariQ <c>TRL1..TRL6</c> 16x16 tile into 256 FP32 values scaled by <paramref name="scale"/>
    /// using SWAR decoding + AVX-512 / AVX2 int8-to-float widening.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DecodeTileFloat(ReadOnlySpan<uint> tileWords, int bits, float scale, Span<float> dstRowMajor)
    {
        int nw = GetTileWordCount(bits);
        if (tileWords.Length < nw)
            throw new ArgumentException($"tileWords length ({tileWords.Length}) must be at least {nw} for TRL{bits}.", nameof(tileWords));
        if (dstRowMajor.Length < TileElements)
            throw new ArgumentException($"dstRowMajor length ({dstRowMajor.Length}) must be at least {TileElements}.", nameof(dstRowMajor));

        sbyte* q = stackalloc sbyte[TileElements];
        fixed (uint* t = tileWords)
        {
            DecodeTileInt8SwarPtr(t, bits, nw, q);
        }

        fixed (float* dst = dstRowMajor)
        {
            ConvertAndScaleInt8Tile(q, scale, dst);
        }
    }

    /// <summary>
    /// Decodes <paramref name="nRows"/> rows (multiple of 16) starting at row <paramref name="n0"/> (multiple of 16)
    /// of a tile-major <c>TRL1..TRL6</c> matrix (<c>k</c> columns, multiple of 16) into row-major FP32 output.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DecodeRows(
        ReadOnlySpan<byte> codeData,
        int bits,
        int k,
        int n0,
        int nRows,
        ReadOnlySpan<Half> s16Scales,
        Span<float> output)
    {
        int nw = GetTileWordCount(bits);
        if (k <= 0 || (k % TileDim) != 0 || n0 < 0 || (n0 % TileDim) != 0 || nRows < 0 || (nRows % TileDim) != 0)
            throw new ArgumentException($"Trellis dimensions must be multiples of 16 (k={k}, n0={n0}, nRows={nRows}).");
        if (nRows == 0) return;

        int kt = k / TileDim;
        long requiredWords = (long)((n0 + nRows) / TileDim) * kt * nw;
        if (codeData.Length < requiredWords * sizeof(uint))
            throw new ArgumentException("codeData buffer too small for requested rows.", nameof(codeData));
        if (!s16Scales.IsEmpty && s16Scales.Length < kt)
            throw new ArgumentException($"s16Scales length ({s16Scales.Length}) must be at least k/16 ({kt}).", nameof(s16Scales));
        if (output.Length < (long)nRows * k)
            throw new ArgumentException("output buffer too small for requested rows.", nameof(output));

        ReadOnlySpan<uint> allWords = MemoryMarshal.Cast<byte, uint>(codeData);
        sbyte* qTile = stackalloc sbyte[TileElements];
        float* fTile = stackalloc float[TileElements];

        int jStart = n0 / TileDim;
        int jEnd = (n0 + nRows) / TileDim;

        fixed (uint* words = allWords)
        fixed (float* pOut = output)
        {
            for (int j = jStart; j < jEnd; j++)
            {
                int baseRow = TileDim * j - n0;
                for (int i = 0; i < kt; i++)
                {
                    uint* t = words + ((long)j * kt + i) * nw;
                    float scale = s16Scales.IsEmpty ? 1.0f : (float)s16Scales[i];

                    DecodeTileInt8SwarPtr(t, bits, nw, qTile);
                    ConvertAndScaleInt8Tile(qTile, scale, fTile);

                    int baseCol = TileDim * i;
                    for (int r = 0; r < TileDim; r++)
                    {
                        Buffer.MemoryCopy(
                            fTile + r * TileDim,
                            pOut + (long)(baseRow + r) * k + baseCol,
                            TileDim * sizeof(float),
                            TileDim * sizeof(float));
                    }
                }
            }
        }
    }

    // =========================================================================
    // 2. AVX-512 / AVX2 SIMD TILE DOT-PRODUCT & 16x16 MAT-VEC
    // =========================================================================

    /// <summary>
    /// Computes the dot product between a decoded 16-element signed int8 trellis row and 16 FP32 activations
    /// using AVX-512, AVX2/FMA, or scalar fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float DotTileRowSimd(ReadOnlySpan<sbyte> row16, ReadOnlySpan<float> x16)
    {
        if (row16.Length < TileDim)
            throw new ArgumentException($"row16 length must be at least {TileDim}.", nameof(row16));
        if (x16.Length < TileDim)
            throw new ArgumentException($"x16 length must be at least {TileDim}.", nameof(x16));

        fixed (sbyte* pRow = row16)
        fixed (float* pX = x16)
        {
            return DotTileRowPtr(pRow, pX);
        }
    }

    /// <summary>
    /// Decodes a NariQ <c>TRL1..TRL6</c> 16x16 tile via SWAR and accumulates
    /// <c>y16[r] += scale * dot(W_tile[r, 0..15], x16[0..15])</c> for <c>r = 0..15</c>
    /// using AVX-512 / AVX2 SIMD dot products.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void TileMatVecAdd(
        ReadOnlySpan<uint> tileWords,
        int bits,
        float scale,
        ReadOnlySpan<float> x16,
        Span<float> y16)
    {
        int nw = GetTileWordCount(bits);
        if (tileWords.Length < nw)
            throw new ArgumentException($"tileWords length ({tileWords.Length}) must be at least {nw} for TRL{bits}.", nameof(tileWords));
        if (x16.Length < TileDim)
            throw new ArgumentException($"x16 length must be at least {TileDim}.", nameof(x16));
        if (y16.Length < TileDim)
            throw new ArgumentException($"y16 length must be at least {TileDim}.", nameof(y16));

        sbyte* qTile = stackalloc sbyte[TileElements];
        fixed (uint* t = tileWords)
        {
            DecodeTileInt8SwarPtr(t, bits, nw, qTile);
        }

        fixed (float* pX = x16)
        fixed (float* pY = y16)
        {
            if (Avx512F.IsSupported)
            {
                var vx = Vector512.Load(pX);
                for (int r = 0; r < TileDim; r++)
                {
                    var q128 = Vector128.Load(qTile + (r << 4));
                    var wf = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(q128));
                    float dot = Vector512.Sum(Avx512F.Multiply(wf, vx));
                    pY[r] += dot * scale;
                }
            }
            else if (Avx2.IsSupported)
            {
                var vx0 = Vector256.Load(pX);
                var vx1 = Vector256.Load(pX + 8);
                for (int r = 0; r < TileDim; r++)
                {
                    sbyte* row = qTile + (r << 4);
                    var wf0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row));
                    var wf1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 8));
                    Vector256<float> acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(wf1, vx1, Avx.Multiply(wf0, vx0))
                        : Avx.Add(Avx.Multiply(wf0, vx0), Avx.Multiply(wf1, vx1));
                    pY[r] += Vector256.Sum(acc) * scale;
                }
            }
            else
            {
                for (int r = 0; r < TileDim; r++)
                {
                    sbyte* row = qTile + (r << 4);
                    float dot = 0f;
                    for (int c = 0; c < TileDim; c++)
                        dot += row[c] * pX[c];
                    pY[r] += dot * scale;
                }
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="TileMatVecAdd"/> for SIMD parity tests.
    /// </summary>
    internal static void TileMatVecAddScalar(
        ReadOnlySpan<uint> tileWords,
        int bits,
        float scale,
        ReadOnlySpan<float> x16,
        Span<float> y16)
    {
        Span<sbyte> qTile = stackalloc sbyte[TileElements];
        DecodeTileInt8Scalar(tileWords, bits, qTile);

        for (int r = 0; r < TileDim; r++)
        {
            float dot = 0f;
            int rowOff = r << 4;
            for (int c = 0; c < TileDim; c++)
                dot += qTile[rowOff + c] * x16[c];
            y16[r] += dot * scale;
        }
    }

    // =========================================================================
    // 3. TILE INSPECTION & QUANTIZATION-ERROR RMSE MEASUREMENT
    // =========================================================================

    /// <summary>
    /// Decodes a NariQ <c>TRL1..TRL6</c> 16x16 tile and computes the exact root-mean-square error (RMSE)
    /// against a 256-element row-major FP32 reference tile.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double MeasureTileRmse(
        ReadOnlySpan<float> referenceTile256,
        ReadOnlySpan<uint> tileWords,
        int bits,
        float scale)
    {
        if (referenceTile256.Length < TileElements)
            throw new ArgumentException($"referenceTile256 length must be at least {TileElements}.", nameof(referenceTile256));

        Span<float> decoded = stackalloc float[TileElements];
        DecodeTileFloat(tileWords, bits, scale, decoded);

        double sumSq = 0.0;
        for (int i = 0; i < TileElements; i++)
        {
            double diff = referenceTile256[i] - decoded[i];
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / TileElements);
    }

    /// <summary>
    /// Decodes a NariQ <c>TRL1..TRL6</c> 16x16 tile and returns full quantization quality metrics
    /// (RMSE, MAE, max absolute error, SNR in dB, and decoded int8 code bounds).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static TileErrorMetrics InspectTile(
        ReadOnlySpan<float> referenceTile256,
        ReadOnlySpan<uint> tileWords,
        int bits,
        float scale)
    {
        if (referenceTile256.Length < TileElements)
            throw new ArgumentException($"referenceTile256 length must be at least {TileElements}.", nameof(referenceTile256));

        Span<sbyte> qTile = stackalloc sbyte[TileElements];
        DecodeTileInt8(tileWords, bits, qTile);

        double sumSqErr = 0.0;
        double sumAbsErr = 0.0;
        double sumSqSig = 0.0;
        float maxAbsErr = 0f;
        sbyte minCode = sbyte.MaxValue;
        sbyte maxCode = sbyte.MinValue;

        for (int i = 0; i < TileElements; i++)
        {
            sbyte code = qTile[i];
            if (code < minCode) minCode = code;
            if (code > maxCode) maxCode = code;

            float refVal = referenceTile256[i];
            float decVal = code * scale;
            float absErr = MathF.Abs(refVal - decVal);
            if (absErr > maxAbsErr) maxAbsErr = absErr;

            double diff = refVal - decVal;
            sumSqErr += diff * diff;
            sumAbsErr += absErr;
            sumSqSig += (double)refVal * refVal;
        }

        double rmse = Math.Sqrt(sumSqErr / TileElements);
        double mae = sumAbsErr / TileElements;
        double snrDb = sumSqErr <= 0.0
            ? double.PositiveInfinity
            : (sumSqSig <= 0.0 ? double.NegativeInfinity : 10.0 * Math.Log10(sumSqSig / sumSqErr));

        return new TileErrorMetrics(rmse, mae, maxAbsErr, snrDb, minCode, maxCode);
    }

    // =========================================================================
    // 4. INTERNAL SIMD HELPERS
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ValidateBits(int bits)
    {
        if (bits < MinBits || bits > MaxBits)
            throw new ArgumentOutOfRangeException(nameof(bits), bits, $"NariQ trellis bitrate must be between {MinBits} and {MaxBits} (TRL1..TRL6).");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotTileRowPtr(sbyte* row16, float* x16)
    {
        if (Avx512F.IsSupported)
        {
            var q128 = Vector128.Load(row16);
            var wf = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(q128));
            return Vector512.Sum(Avx512F.Multiply(wf, Vector512.Load(x16)));
        }
        if (Avx2.IsSupported)
        {
            var wf0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row16));
            var wf1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row16 + 8));
            var vx0 = Vector256.Load(x16);
            var vx1 = Vector256.Load(x16 + 8);
            Vector256<float> acc = Fma.IsSupported
                ? Fma.MultiplyAdd(wf1, vx1, Avx.Multiply(wf0, vx0))
                : Avx.Add(Avx.Multiply(wf0, vx0), Avx.Multiply(wf1, vx1));
            return Vector256.Sum(acc);
        }

        float sum = 0f;
        for (int c = 0; c < TileDim; c++) sum += row16[c] * x16[c];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConvertAndScaleInt8Tile(sbyte* qTile, float scale, float* dst)
    {
        if (Avx512F.IsSupported)
        {
            var vScale = Vector512.Create(scale);
            for (int i = 0; i < TileElements; i += 16)
            {
                var q128 = Vector128.Load(qTile + i);
                var f = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(q128));
                Vector512.Store(Avx512F.Multiply(f, vScale), dst + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vScale = Vector256.Create(scale);
            for (int i = 0; i < TileElements; i += 8)
            {
                var f = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qTile + i));
                Vector256.Store(Avx.Multiply(f, vScale), dst + i);
            }
        }
        else
        {
            for (int i = 0; i < TileElements; i++)
                dst[i] = qTile[i] * scale;
        }
    }
}
