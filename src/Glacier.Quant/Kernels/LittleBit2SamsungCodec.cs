namespace Glacier.Quant.Kernels;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Cross-platform interoperability codec between SamsungLabs LittleBit-2 packed checkpoint tensors
/// (<c>uint32</c> LSB-first with <c>+1 -&gt; 0, -1 -&gt; 1</c>) and <see cref="LittleBit2Path"/>
/// (<c>uint64</c> LSB-first with <c>+1 -&gt; 1, -1 -&gt; 0</c>) with balanced FP16 signed scale packing.
/// </summary>
public static class LittleBit2SamsungCodec
{
    /// <summary>
    /// Converts SamsungLabs LSB-first <c>uint32</c> packed words (<c>+1 -&gt; 0, -1 -&gt; 1</c>)
    /// into <see cref="LittleBit2Path"/> <c>ulong[]</c> words (<c>+1 -&gt; 1, -1 -&gt; 0</c>), masking unused tail bits.
    /// </summary>
    public static ulong[] FromSamsungPackedBits(ReadOnlySpan<uint> srcU32, int rows, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);

        int words32 = (width + 31) >> 5;
        long expectedLen = (long)rows * words32;
        if (srcU32.Length != expectedLen)
            throw new ArgumentException($"Expected {expectedLen} uint32 words for {rows}x{width}, got {srcU32.Length}.", nameof(srcU32));

        int stride64 = (width + 63) >> 6;
        ulong[] dstU64 = new ulong[checked(rows * stride64)];

        for (int r = 0; r < rows; r++)
        {
            int srcRowOff = r * words32;
            int dstRowOff = r * stride64;
            for (int w64 = 0; w64 < stride64; w64++)
            {
                int w32Lo = w64 * 2;
                int w32Hi = w32Lo + 1;
                uint lo = ~srcU32[srcRowOff + w32Lo];
                uint hi = (w32Hi < words32) ? ~srcU32[srcRowOff + w32Hi] : 0u;
                dstU64[dstRowOff + w64] = (ulong)lo | ((ulong)hi << 32);
            }
        }

        LittleBit2Path.ClearUnusedTailBits(dstU64, rows, stride64, width);
        return dstU64;
    }

    /// <summary>
    /// Converts <see cref="LittleBit2Path"/> <c>ulong[]</c> packed words (<c>+1 -&gt; 1, -1 -&gt; 0</c>)
    /// back into SamsungLabs LSB-first <c>uint32</c> packed words (<c>+1 -&gt; 0, -1 -&gt; 1</c>), masking unused tail bits.
    /// </summary>
    public static uint[] ToSamsungPackedBits(ReadOnlySpan<ulong> srcU64, int rows, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);

        int stride64 = (width + 63) >> 6;
        long expectedLen = (long)rows * stride64;
        if (srcU64.Length != expectedLen)
            throw new ArgumentException($"Expected {expectedLen} uint64 words for {rows}x{width}, got {srcU64.Length}.", nameof(srcU64));

        int words32 = (width + 31) >> 5;
        uint[] dstU32 = new uint[checked(rows * words32)];
        int tail32 = width & 31;
        uint tailMask32 = tail32 == 0 ? uint.MaxValue : (1u << tail32) - 1u;

        for (int r = 0; r < rows; r++)
        {
            int srcRowOff = r * stride64;
            int dstRowOff = r * words32;
            for (int w32 = 0; w32 < words32; w32++)
            {
                ulong word64 = srcU64[srcRowOff + (w32 >> 1)];
                uint halfWord = (w32 & 1) == 0 ? (uint)word64 : (uint)(word64 >> 32);
                uint inverted = ~halfWord;
                if (w32 == words32 - 1)
                    inverted &= tailMask32;
                dstU32[dstRowOff + w32] = inverted;
            }
        }

        return dstU32;
    }

    /// <summary>
    /// Balances signed <c>rowScales</c>, <c>colScales</c>, and <c>latentScales</c> across log-space geometric mean
    /// and clamps them into finite <see cref="Half"/> values inside <see cref="LittleBit2Path"/>.
    /// </summary>
    public static LittleBit2Path PackSignedBalancedPath(
        int rows,
        int cols,
        int rank,
        ulong[] uBits,
        ulong[] vBits,
        ReadOnlySpan<double> rowScales,
        ReadOnlySpan<double> colScales,
        ReadOnlySpan<double> latentScales)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

        if (rowScales.Length != rows)
            throw new ArgumentException($"rowScales length ({rowScales.Length}) must equal rows ({rows}).", nameof(rowScales));
        if (colScales.Length != cols)
            throw new ArgumentException($"colScales length ({colScales.Length}) must equal cols ({cols}).", nameof(colScales));
        if (latentScales.Length != rank)
            throw new ArgumentException($"latentScales length ({latentScales.Length}) must equal rank ({rank}).", nameof(latentScales));

        Half[] hHalf = new Half[rows];
        Half[] gHalf = new Half[cols];
        Half[] lHalf = new Half[rank];

        double maxH = MaxAbsValidated(rowScales, nameof(rowScales));
        double maxG = MaxAbsValidated(colScales, nameof(colScales));
        double maxL = MaxAbsValidated(latentScales, nameof(latentScales));

        if (maxH > 0.0 && maxG > 0.0 && maxL > 0.0)
        {
            double logProd = Math.Log(maxH) + Math.Log(maxG) + Math.Log(maxL);
            if (double.IsFinite(logProd))
            {
                double logC = logProd / 3.0;
                double alpha = Math.Exp(logC - Math.Log(maxH));
                double beta = Math.Exp(logC - Math.Log(maxG));
                double gamma = Math.Exp(logC - Math.Log(maxL));

                if (double.IsFinite(alpha) && double.IsFinite(beta) && double.IsFinite(gamma))
                {
                    ConvertSignedToHalfClamped(rowScales, alpha, hHalf);
                    ConvertSignedToHalfClamped(colScales, beta, gHalf);
                    ConvertSignedToHalfClamped(latentScales, gamma, lHalf);
                }
            }
        }

        return new LittleBit2Path(rows, cols, rank, uBits, vBits, hHalf, gHalf, lHalf);
    }

    /// <summary>
    /// Converts a single SamsungLabs LittleBit-2 path (<c>U_packed</c>, <c>V_packed</c>, and <c>u1/u2/v1/v2</c> scales)
    /// into a balanced <see cref="LittleBit2Path"/> with <c>ell = v1 * u2</c>.
    /// </summary>
    public static LittleBit2Path FromSamsungCheckpointPath(
        int rows,
        int cols,
        int rank,
        ReadOnlySpan<uint> uPacked,
        ReadOnlySpan<uint> vPacked,
        ReadOnlySpan<double> u1,
        ReadOnlySpan<double> u2,
        ReadOnlySpan<double> v1,
        ReadOnlySpan<double> v2)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

        if (u2.Length != rank)
            throw new ArgumentException($"u2 length ({u2.Length}) must equal rank ({rank}).", nameof(u2));
        if (v1.Length != rank)
            throw new ArgumentException($"v1 length ({v1.Length}) must equal rank ({rank}).", nameof(v1));

        ulong[] uBits = FromSamsungPackedBits(uPacked, rows, rank);
        ulong[] vBits = FromSamsungPackedBits(vPacked, rank, cols);

        double[] ell = new double[rank];
        for (int k = 0; k < rank; k++)
            ell[k] = v1[k] * u2[k];

        return PackSignedBalancedPath(rows, cols, rank, uBits, vBits, u1, v2, ell);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double MaxAbsValidated(ReadOnlySpan<double> values, string paramName)
    {
        double max = 0.0;
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            if (!double.IsFinite(v))
                throw new ArgumentException($"Scale span '{paramName}' contains non-finite value at index {i}.", paramName);
            double a = Math.Abs(v);
            if (a > max)
                max = a;
        }
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConvertSignedToHalfClamped(ReadOnlySpan<double> src, double multiplier, Half[] dst)
    {
        const double maxHalf = 65504.0;
        for (int i = 0; i < src.Length; i++)
        {
            double scaled = Math.Clamp(src[i] * multiplier, -maxHalf, maxHalf);
            dst[i] = (Half)scaled;
        }
    }
}
