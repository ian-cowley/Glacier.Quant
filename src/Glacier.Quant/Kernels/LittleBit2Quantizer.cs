namespace Glacier.Quant.Kernels;

using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

/// <summary>
/// Clean-room LittleBit-2 sub-1-bit quantizer and multiplication-free SIMD inference engine
/// implementing SVD + Joint-ITQ orthogonal alignment, Dual-SVID tri-scale extraction,
/// 2-path residual compensation, and IEEE-754 sign-bit XOR GEMV/GEMM/Dequant kernels.
/// </summary>
public static class LittleBit2Quantizer
{
    private const int StackAllocFloatLimit = 512;
    private const int TokenTile = 32;
    private const int DequantColTile = 4096;
    private const long ParallelWorkThreshold = 131_072;

    public static int ComputeRankForBpw(
        int rows,
        int cols,
        double bitsPerWeight,
        bool useResidualCompensation = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        if (!double.IsFinite(bitsPerWeight) || bitsPerWeight <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(bitsPerWeight), bitsPerWeight, "BitsPerWeight must be finite and strictly positive.");

        double paths = useResidualCompensation ? 2.0 : 1.0;
        double dimSum = (double)rows + cols;
        double numerator = bitsPerWeight * rows * cols - 16.0 * paths * dimSum;
        double denominator = paths * (dimSum + 16.0);
        double rRaw = Math.Floor(numerator / denominator);

        int maxRank = Math.Min(rows, cols);
        if (rRaw <= 1.0)
            return 1;
        if (rRaw >= maxRank)
            return maxRank;
        return (int)rRaw;
    }

    public static double ComputeEffectiveBpw(
        int rows,
        int cols,
        int primaryRank,
        int residualRank = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(primaryRank);
        ArgumentOutOfRangeException.ThrowIfNegative(residualRank);

        long dimSum = (long)rows + cols;
        long totalBits = (long)primaryRank * (dimSum + 16L) + 16L * dimSum;
        if (residualRank > 0)
            totalBits += (long)residualRank * (dimSum + 16L) + 16L * dimSum;

        return (double)totalBits / ((double)rows * cols);
    }

    public static double ComputeEffectiveBpw(
        int rows,
        int cols,
        int rank,
        bool useResidualCompensation)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);
        return ComputeEffectiveBpw(rows, cols, rank, useResidualCompensation ? rank : 0);
    }

    public static LittleBit2Tensor Quantize(
        ReadOnlySpan<float> weights,
        int rows,
        int cols,
        LittleBit2Options? options = null)
    {
        var opts = options ?? new LittleBit2Options();
        opts.Validate(rows, cols);

        long expectedElements = (long)rows * cols;
        if (weights.Length != expectedElements)
            throw new ArgumentException($"Weights length ({weights.Length}) must equal rows * cols ({expectedElements}).", nameof(weights));

        EnsureAllFinite(weights, nameof(weights));

        int primaryRank = opts.Rank ?? ComputeRankForBpw(rows, cols, opts.BitsPerWeight, opts.UseResidualCompensation);
        LittleBit2Path primary = QuantizePath(weights, rows, cols, primaryRank, opts.Seed, opts);

        if (!opts.UseResidualCompensation)
            return new LittleBit2Tensor(rows, cols, primary, residual: null);

        int total = checked((int)expectedElements);
        float[] wHatPri = ArrayPool<float>.Shared.Rent(total);
        float[] wRes = ArrayPool<float>.Shared.Rent(total);
        try
        {
            var wHatSpan = wHatPri.AsSpan(0, total);
            DequantizePath(primary, wHatSpan, accumulate: false, opts.UseParallel);

            for (int i = 0; i < total; i++)
                wRes[i] = weights[i] - wHatSpan[i];

            int residualRank = opts.ResidualRank ?? primaryRank;
            LittleBit2Path residual = QuantizePath(wRes.AsSpan(0, total), rows, cols, residualRank, unchecked(opts.Seed + 1), opts);
            return new LittleBit2Tensor(rows, cols, primary, residual);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(wHatPri);
            ArrayPool<float>.Shared.Return(wRes);
        }
    }

    public static float[] Dequantize(LittleBit2Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        float[] dst = new float[checked(tensor.Rows * tensor.Cols)];
        Dequantize(tensor, dst);
        return dst;
    }

    public static void Dequantize(LittleBit2Tensor tensor, Span<float> dst)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        long expected = (long)tensor.Rows * tensor.Cols;
        if (dst.Length < expected)
            throw new ArgumentException($"Destination span length ({dst.Length}) is smaller than required ({expected}).", nameof(dst));

        var target = dst.Slice(0, checked((int)expected));
        DequantizePath(tensor.Primary, target, accumulate: false, useParallel: true);
        if (tensor.Residual is { } residual)
            DequantizePath(residual, target, accumulate: true, useParallel: true);
    }

    public static void MatVecMul(
        LittleBit2Tensor tensor,
        ReadOnlySpan<float> x,
        Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (x.Length < tensor.Cols)
            throw new ArgumentException($"Input vector length ({x.Length}) is smaller than tensor.Cols ({tensor.Cols}).", nameof(x));
        if (y.Length < tensor.Rows)
            throw new ArgumentException($"Output vector length ({y.Length}) is smaller than tensor.Rows ({tensor.Rows}).", nameof(y));

        var xSlice = x.Slice(0, tensor.Cols);
        EnsureAllFinite(xSlice, nameof(x));

        var ySlice = y.Slice(0, tensor.Rows);
        MatVecMulPath(tensor.Primary, xSlice, ySlice, accumulate: false, useParallel: true);
        if (tensor.Residual is { } residual)
            MatVecMulPath(residual, xSlice, ySlice, accumulate: true, useParallel: true);
    }

    public static void MatMul(
        LittleBit2Tensor tensor,
        ReadOnlySpan<float> xBatch,
        int batchSize,
        Span<float> yBatch)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegative(batchSize);

        if (batchSize == 0)
            return;

        long expectedX = (long)batchSize * tensor.Cols;
        long expectedY = (long)batchSize * tensor.Rows;
        if (xBatch.Length < expectedX)
            throw new ArgumentException($"Input batch length ({xBatch.Length}) is smaller than required ({expectedX}).", nameof(xBatch));
        if (yBatch.Length < expectedY)
            throw new ArgumentException($"Output batch length ({yBatch.Length}) is smaller than required ({expectedY}).", nameof(yBatch));

        var xSlice = xBatch.Slice(0, checked((int)expectedX));
        EnsureAllFinite(xSlice, nameof(xBatch));

        if (batchSize == 1)
        {
            MatVecMul(tensor, xSlice, yBatch.Slice(0, tensor.Rows));
            return;
        }

        var ySlice = yBatch.Slice(0, checked((int)expectedY));
        MatMulPath(tensor.Primary, xSlice, batchSize, ySlice, accumulate: false, useParallel: true);
        if (tensor.Residual is { } residual)
            MatMulPath(residual, xSlice, batchSize, ySlice, accumulate: true, useParallel: true);
    }

    internal static double[] ApplyJointItq(
        double[] aRowMajor,
        double[] bRowMajor,
        int rows,
        int cols,
        int rank,
        int iterations,
        int seed,
        out double[] aRot,
        out double[] bRot)
    {
        double[] rMat = LittleBit2Svd.CreateRandomOrthogonal(rank, seed);
        if (iterations == 0)
        {
            aRot = MultiplyDoubleMatrices(aRowMajor, rows, rank, rMat, rank);
            bRot = MultiplyDoubleTransposeFirst(rMat, rank, bRowMajor, cols);
            return rMat;
        }

        int totalRows = checked(rows + cols);
        int zLen = checked(totalRows * rank);
        double[] z = ArrayPool<double>.Shared.Rent(zLen);
        double[] zr = ArrayPool<double>.Shared.Rent(zLen);
        sbyte[] bSign = ArrayPool<sbyte>.Shared.Rent(zLen);
        double[] mMat = ArrayPool<double>.Shared.Rent(checked(rank * rank));

        try
        {
            // Z = [A; B^T] in row-major [(rows + cols) x rank]
            aRowMajor.AsSpan(0, rows * rank).CopyTo(z.AsSpan(0, rows * rank));
            int bBase = rows * rank;
            for (int j = 0; j < cols; j++)
            {
                int dstOff = bBase + j * rank;
                for (int k = 0; k < rank; k++)
                    z[dstOff + k] = bRowMajor[k * cols + j];
            }

            for (int iter = 0; iter < iterations; iter++)
            {
                // ZR = Z * R
                MultiplyDoubleMatricesInto(z, totalRows, rank, rMat, rank, zr);

                // Step A: Binary projection onto {-1, +1} (0.0 -> +1)
                int changedBits = 0;
                for (int idx = 0; idx < zLen; idx++)
                {
                    sbyte s = zr[idx] >= 0.0 ? (sbyte)1 : (sbyte)-1;
                    if (iter > 0 && bSign[idx] != s)
                        changedBits++;
                    bSign[idx] = s;
                }

                if (iter > 0 && changedBits == 0)
                    break;

                // Step B: M = B_sign^T * Z  [rank x rank]
                Array.Clear(mMat, 0, rank * rank);
                for (int row = 0; row < totalRows; row++)
                {
                    int rowOff = row * rank;
                    for (int p = 0; p < rank; p++)
                    {
                        double sp = bSign[rowOff + p];
                        int mRow = p * rank;
                        for (int q = 0; q < rank; q++)
                            mMat[mRow + q] += sp * z[rowOff + q];
                    }
                }

                // Step C: Exact SVD M = Phi * Omega * Psi^T
                LittleBit2Svd.ExactSvd(mMat.AsSpan(0, rank * rank), rank, rank, out double[] phi, out _, out double[] psi);

                // Step D: R = Psi * Phi^T
                for (int p = 0; p < rank; p++)
                {
                    int psiRow = p * rank;
                    int rRow = p * rank;
                    for (int q = 0; q < rank; q++)
                    {
                        int phiRow = q * rank;
                        double sum = 0.0;
                        for (int k = 0; k < rank; k++)
                            sum += psi[psiRow + k] * phi[phiRow + k];
                        rMat[rRow + q] = sum;
                    }
                }
            }

            aRot = MultiplyDoubleMatrices(aRowMajor, rows, rank, rMat, rank);
            bRot = MultiplyDoubleTransposeFirst(rMat, rank, bRowMajor, cols);
            return rMat;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(z);
            ArrayPool<double>.Shared.Return(zr);
            ArrayPool<sbyte>.Shared.Return(bSign);
            ArrayPool<double>.Shared.Return(mMat);
        }
    }

    private static LittleBit2Path QuantizePath(
        ReadOnlySpan<float> weights,
        int rows,
        int cols,
        int rank,
        int seed,
        LittleBit2Options opts)
    {
        int uStride = (rank + 63) >> 6;
        int vStride = (cols + 63) >> 6;

        double frobSq = 0.0;
        for (int i = 0; i < weights.Length; i++)
        {
            double v = weights[i];
            frobSq += v * v;
        }

        if (Math.Sqrt(frobSq) <= 1e-12)
        {
            ulong[] zeroUBits = new ulong[checked(rows * uStride)];
            ulong[] zeroVBits = new ulong[checked(rank * vStride)];
            Array.Fill(zeroUBits, ulong.MaxValue);
            Array.Fill(zeroVBits, ulong.MaxValue);
            return new LittleBit2Path(
                rows,
                cols,
                rank,
                zeroUBits,
                zeroVBits,
                new Half[rows],
                new Half[cols],
                new Half[rank]);
        }

        LittleBit2Svd.TruncatedSvd(
            weights,
            rows,
            cols,
            rank,
            opts.SvdOversampling,
            opts.PowerIterations,
            seed,
            out double[] a,
            out double[] b);

        double[] aRot;
        double[] bRot;
        switch (opts.RotationMode)
        {
            case LittleBit2RotationMode.None:
                aRot = a;
                bRot = b;
                break;

            case LittleBit2RotationMode.RandomOrthogonal:
            {
                double[] rMat = LittleBit2Svd.CreateRandomOrthogonal(rank, seed);
                aRot = MultiplyDoubleMatrices(a, rows, rank, rMat, rank);
                bRot = MultiplyDoubleTransposeFirst(rMat, rank, b, cols);
                break;
            }

            case LittleBit2RotationMode.JointItq:
                ApplyJointItq(a, b, rows, cols, rank, opts.JointItqIterations, seed, out aRot, out bRot);
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(opts.RotationMode), opts.RotationMode, "Unsupported rotation mode.");
        }

        ulong[] uBits = new ulong[checked(rows * uStride)];
        ulong[] vBits = new ulong[checked(rank * vStride)];

        for (int i = 0; i < rows; i++)
        {
            int srcRow = i * rank;
            int dstRow = i * uStride;
            for (int k = 0; k < rank; k++)
            {
                if (aRot[srcRow + k] >= 0.0)
                    uBits[dstRow + (k >> 6)] |= 1UL << (k & 63);
            }
        }

        for (int k = 0; k < rank; k++)
        {
            int srcRow = k * cols;
            int dstRow = k * vStride;
            for (int j = 0; j < cols; j++)
            {
                if (bRot[srcRow + j] >= 0.0)
                    vBits[dstRow + (j >> 6)] |= 1UL << (j & 63);
            }
        }

        // Dual-SVID rank-1 magnitude decomposition on |A_rot| and |B_rot|.
        double[] absA = ArrayPool<double>.Shared.Rent(aRot.Length);
        double[] absB = ArrayPool<double>.Shared.Rent(bRot.Length);
        try
        {
            for (int i = 0; i < aRot.Length; i++)
                absA[i] = Math.Abs(aRot[i]);
            for (int i = 0; i < bRot.Length; i++)
                absB[i] = Math.Abs(bRot[i]);

            LittleBit2Svd.RankOneNonNegativeSvd(absA.AsSpan(0, aRot.Length), rows, rank, opts.Rank1SvdIterations, out double[] h, out double[] ellU);
            LittleBit2Svd.RankOneNonNegativeSvd(absB.AsSpan(0, bRot.Length), rank, cols, opts.Rank1SvdIterations, out double[] ellV, out double[] g);

            double[] ell = new double[rank];
            for (int k = 0; k < rank; k++)
                ell[k] = ellU[k] * ellV[k];

            Half[] rowScales = new Half[rows];
            Half[] colScales = new Half[cols];
            Half[] latentScales = new Half[rank];

            double maxH = MaxPositive(h);
            double maxL = MaxPositive(ell);
            double maxG = MaxPositive(g);

            if (maxH > 1e-30 && maxL > 1e-30 && maxG > 1e-30)
            {
                // Overflow-safe geometric-mean balancing in log space before FP16 conversion.
                double logC = (Math.Log(maxH) + Math.Log(maxL) + Math.Log(maxG)) / 3.0;
                double c = Math.Exp(logC);
                double alpha = c / maxH;
                double beta = c / maxG;
                double gamma = 1.0 / (alpha * beta);

                ConvertToHalfClamped(h, alpha, rowScales);
                ConvertToHalfClamped(g, beta, colScales);
                ConvertToHalfClamped(ell, gamma, latentScales);
            }

            return new LittleBit2Path(rows, cols, rank, uBits, vBits, rowScales, colScales, latentScales);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(absA);
            ArrayPool<double>.Shared.Return(absB);
        }
    }

    private static unsafe void MatVecMulPath(
        LittleBit2Path path,
        ReadOnlySpan<float> x,
        Span<float> y,
        bool accumulate,
        bool useParallel)
    {
        int rows = path.Rows;
        int cols = path.Cols;
        int rank = path.Rank;
        int uStride = path.UStrideWords;
        int vStride = path.VStrideWords;

        float[]? rentedXScaled = cols > StackAllocFloatLimit ? ArrayPool<float>.Shared.Rent(cols) : null;
        float[]? rentedZ = rank > StackAllocFloatLimit ? ArrayPool<float>.Shared.Rent(rank) : null;
        float[]? rentedH = rows > StackAllocFloatLimit ? ArrayPool<float>.Shared.Rent(rows) : null;
        float[]? rentedEll = rank > StackAllocFloatLimit ? ArrayPool<float>.Shared.Rent(rank) : null;

        try
        {
            Span<float> xScaled = rentedXScaled is not null ? rentedXScaled.AsSpan(0, cols) : stackalloc float[cols];
            Span<float> z = rentedZ is not null ? rentedZ.AsSpan(0, rank) : stackalloc float[rank];
            Span<float> hFloat = rentedH is not null ? rentedH.AsSpan(0, rows) : stackalloc float[rows];
            Span<float> ellFloat = rentedEll is not null ? rentedEll.AsSpan(0, rank) : stackalloc float[rank];

            var colScales = path.ColScales;
            for (int j = 0; j < cols; j++)
                xScaled[j] = (float)colScales[j] * x[j];

            ConvertHalfToFloat(path.LatentScales, ellFloat);
            ConvertHalfToFloat(path.RowScales, hFloat);

            fixed (ulong* pVBitsFixed = path.VBits)
            fixed (ulong* pUBitsFixed = path.UBits)
            fixed (float* pXScaledFixed = xScaled)
            fixed (float* pZFixed = z)
            fixed (float* pEllFixed = ellFloat)
            fixed (float* pHFixed = hFloat)
            fixed (float* pYFixed = y)
            {
                ulong* pVBits = pVBitsFixed;
                ulong* pUBits = pUBitsFixed;
                float* pXScaled = pXScaledFixed;
                float* pZ = pZFixed;
                float* pEll = pEllFixed;
                float* pH = pHFixed;
                float* pY = pYFixed;

                int threads = Environment.ProcessorCount;
                if (useParallel && (long)rank * cols >= ParallelWorkThreshold && rank >= threads * 2)
                {
                    Parallel.For(0, rank, k =>
                    {
                        float dot = BinaryDot(pVBits + (long)k * vStride, pXScaled, cols);
                        pZ[k] = pEll[k] * dot;
                    });
                }
                else
                {
                    for (int k = 0; k < rank; k++)
                    {
                        float dot = BinaryDot(pVBits + (long)k * vStride, pXScaled, cols);
                        pZ[k] = pEll[k] * dot;
                    }
                }

                if (useParallel && (long)rows * rank >= ParallelWorkThreshold && rows >= threads * 2)
                {
                    Parallel.For(0, rows, i =>
                    {
                        float dot = BinaryDot(pUBits + (long)i * uStride, pZ, rank);
                        float val = pH[i] * dot;
                        if (accumulate)
                            pY[i] += val;
                        else
                            pY[i] = val;
                    });
                }
                else
                {
                    for (int i = 0; i < rows; i++)
                    {
                        float dot = BinaryDot(pUBits + (long)i * uStride, pZ, rank);
                        float val = pH[i] * dot;
                        if (accumulate)
                            pY[i] += val;
                        else
                            pY[i] = val;
                    }
                }
            }
        }
        finally
        {
            if (rentedXScaled is not null)
                ArrayPool<float>.Shared.Return(rentedXScaled);
            if (rentedZ is not null)
                ArrayPool<float>.Shared.Return(rentedZ);
            if (rentedH is not null)
                ArrayPool<float>.Shared.Return(rentedH);
            if (rentedEll is not null)
                ArrayPool<float>.Shared.Return(rentedEll);
        }
    }

    private static unsafe void MatMulPath(
        LittleBit2Path path,
        ReadOnlySpan<float> xBatch,
        int batchSize,
        Span<float> yBatch,
        bool accumulate,
        bool useParallel)
    {
        int rows = path.Rows;
        int cols = path.Cols;
        int rank = path.Rank;
        int uStride = path.UStrideWords;
        int vStride = path.VStrideWords;

        int tileCount = Math.Min(TokenTile, batchSize);
        float[] gFloatArr = ArrayPool<float>.Shared.Rent(cols);
        float[] ellFloatArr = ArrayPool<float>.Shared.Rent(rank);
        float[] hFloatArr = ArrayPool<float>.Shared.Rent(rows);
        float[] xScaledTileArr = ArrayPool<float>.Shared.Rent(checked(tileCount * cols));
        float[] zTileArr = ArrayPool<float>.Shared.Rent(checked(tileCount * rank));

        try
        {
            var gSpan = gFloatArr.AsSpan(0, cols);
            var ellSpan = ellFloatArr.AsSpan(0, rank);
            var hSpan = hFloatArr.AsSpan(0, rows);

            ConvertHalfToFloat(path.ColScales, gSpan);
            ConvertHalfToFloat(path.LatentScales, ellSpan);
            ConvertHalfToFloat(path.RowScales, hSpan);

            fixed (ulong* pVBitsFixed = path.VBits)
            fixed (ulong* pUBitsFixed = path.UBits)
            fixed (float* pGFixed = gSpan)
            fixed (float* pEllFixed = ellSpan)
            fixed (float* pHFixed = hSpan)
            fixed (float* pXScaledTileFixed = xScaledTileArr)
            fixed (float* pZTileFixed = zTileArr)
            fixed (float* pXBatchFixed = xBatch)
            fixed (float* pYBatchFixed = yBatch)
            {
                ulong* pVBits = pVBitsFixed;
                ulong* pUBits = pUBitsFixed;
                float* pEll = pEllFixed;
                float* pH = pHFixed;
                float* pXScaledTile = pXScaledTileFixed;
                float* pZTile = pZTileFixed;
                float* pYBatch = pYBatchFixed;
                int threads = Environment.ProcessorCount;

                for (int t0 = 0; t0 < batchSize; t0 += TokenTile)
                {
                    int count = Math.Min(TokenTile, batchSize - t0);

                    for (int t = 0; t < count; t++)
                    {
                        var xRow = xBatch.Slice((t0 + t) * cols, cols);
                        var dstRow = xScaledTileArr.AsSpan(t * cols, cols);
                        TensorPrimitives.Multiply(xRow, gSpan, dstRow);
                    }

                    // Stage 2: Z_tile[t, k] = ell[k] * BinaryDot(V_k, xScaled[t])
                    if (useParallel && (long)count * rank * cols >= ParallelWorkThreshold && rank >= threads * 2)
                    {
                        Parallel.For(0, rank, k =>
                        {
                            ulong* vRow = pVBits + (long)k * vStride;
                            float ellK = pEll[k];
                            int t = 0;
                            for (; t + 4 <= count; t += 4)
                            {
                                BinaryDotBatch4(
                                    vRow,
                                    pXScaledTile + (long)(t + 0) * cols,
                                    pXScaledTile + (long)(t + 1) * cols,
                                    pXScaledTile + (long)(t + 2) * cols,
                                    pXScaledTile + (long)(t + 3) * cols,
                                    cols,
                                    out float d0,
                                    out float d1,
                                    out float d2,
                                    out float d3);
                                pZTile[(t + 0) * rank + k] = ellK * d0;
                                pZTile[(t + 1) * rank + k] = ellK * d1;
                                pZTile[(t + 2) * rank + k] = ellK * d2;
                                pZTile[(t + 3) * rank + k] = ellK * d3;
                            }
                            for (; t < count; t++)
                            {
                                pZTile[t * rank + k] = ellK * BinaryDot(vRow, pXScaledTile + (long)t * cols, cols);
                            }
                        });
                    }
                    else
                    {
                        for (int k = 0; k < rank; k++)
                        {
                            ulong* vRow = pVBits + (long)k * vStride;
                            float ellK = pEll[k];
                            int t = 0;
                            for (; t + 4 <= count; t += 4)
                            {
                                BinaryDotBatch4(
                                    vRow,
                                    pXScaledTile + (long)(t + 0) * cols,
                                    pXScaledTile + (long)(t + 1) * cols,
                                    pXScaledTile + (long)(t + 2) * cols,
                                    pXScaledTile + (long)(t + 3) * cols,
                                    cols,
                                    out float d0,
                                    out float d1,
                                    out float d2,
                                    out float d3);
                                pZTile[(t + 0) * rank + k] = ellK * d0;
                                pZTile[(t + 1) * rank + k] = ellK * d1;
                                pZTile[(t + 2) * rank + k] = ellK * d2;
                                pZTile[(t + 3) * rank + k] = ellK * d3;
                            }
                            for (; t < count; t++)
                            {
                                pZTile[t * rank + k] = ellK * BinaryDot(vRow, pXScaledTile + (long)t * cols, cols);
                            }
                        }
                    }

                    // Stage 3: Y[t0 + t, i] (+)= h[i] * BinaryDot(U_i, Z_tile[t])
                    if (useParallel && (long)count * rows * rank >= ParallelWorkThreshold && rows >= threads * 2)
                    {
                        Parallel.For(0, rows, i =>
                        {
                            ulong* uRow = pUBits + (long)i * uStride;
                            float hI = pH[i];
                            int t = 0;
                            for (; t + 4 <= count; t += 4)
                            {
                                BinaryDotBatch4(
                                    uRow,
                                    pZTile + (long)(t + 0) * rank,
                                    pZTile + (long)(t + 1) * rank,
                                    pZTile + (long)(t + 2) * rank,
                                    pZTile + (long)(t + 3) * rank,
                                    rank,
                                    out float d0,
                                    out float d1,
                                    out float d2,
                                    out float d3);

                                long out0 = (long)(t0 + t + 0) * rows + i;
                                long out1 = (long)(t0 + t + 1) * rows + i;
                                long out2 = (long)(t0 + t + 2) * rows + i;
                                long out3 = (long)(t0 + t + 3) * rows + i;

                                if (accumulate)
                                {
                                    pYBatch[out0] += hI * d0;
                                    pYBatch[out1] += hI * d1;
                                    pYBatch[out2] += hI * d2;
                                    pYBatch[out3] += hI * d3;
                                }
                                else
                                {
                                    pYBatch[out0] = hI * d0;
                                    pYBatch[out1] = hI * d1;
                                    pYBatch[out2] = hI * d2;
                                    pYBatch[out3] = hI * d3;
                                }
                            }
                            for (; t < count; t++)
                            {
                                float val = hI * BinaryDot(uRow, pZTile + (long)t * rank, rank);
                                long outIdx = (long)(t0 + t) * rows + i;
                                if (accumulate)
                                    pYBatch[outIdx] += val;
                                else
                                    pYBatch[outIdx] = val;
                            }
                        });
                    }
                    else
                    {
                        for (int i = 0; i < rows; i++)
                        {
                            ulong* uRow = pUBits + (long)i * uStride;
                            float hI = pH[i];
                            int t = 0;
                            for (; t + 4 <= count; t += 4)
                            {
                                BinaryDotBatch4(
                                    uRow,
                                    pZTile + (long)(t + 0) * rank,
                                    pZTile + (long)(t + 1) * rank,
                                    pZTile + (long)(t + 2) * rank,
                                    pZTile + (long)(t + 3) * rank,
                                    rank,
                                    out float d0,
                                    out float d1,
                                    out float d2,
                                    out float d3);

                                long out0 = (long)(t0 + t + 0) * rows + i;
                                long out1 = (long)(t0 + t + 1) * rows + i;
                                long out2 = (long)(t0 + t + 2) * rows + i;
                                long out3 = (long)(t0 + t + 3) * rows + i;

                                if (accumulate)
                                {
                                    pYBatch[out0] += hI * d0;
                                    pYBatch[out1] += hI * d1;
                                    pYBatch[out2] += hI * d2;
                                    pYBatch[out3] += hI * d3;
                                }
                                else
                                {
                                    pYBatch[out0] = hI * d0;
                                    pYBatch[out1] = hI * d1;
                                    pYBatch[out2] = hI * d2;
                                    pYBatch[out3] = hI * d3;
                                }
                            }
                            for (; t < count; t++)
                            {
                                float val = hI * BinaryDot(uRow, pZTile + (long)t * rank, rank);
                                long outIdx = (long)(t0 + t) * rows + i;
                                if (accumulate)
                                    pYBatch[outIdx] += val;
                                else
                                    pYBatch[outIdx] = val;
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gFloatArr);
            ArrayPool<float>.Shared.Return(ellFloatArr);
            ArrayPool<float>.Shared.Return(hFloatArr);
            ArrayPool<float>.Shared.Return(xScaledTileArr);
            ArrayPool<float>.Shared.Return(zTileArr);
        }
    }

    private static unsafe void DequantizePath(
        LittleBit2Path path,
        Span<float> dst,
        bool accumulate,
        bool useParallel)
    {
        int rows = path.Rows;
        int cols = path.Cols;
        int rank = path.Rank;
        int uStride = path.UStrideWords;
        int vStride = path.VStrideWords;
        int maxTileLen = Math.Min(DequantColTile, cols);

        float[] gFloatArr = ArrayPool<float>.Shared.Rent(cols);
        float[] ellFloatArr = ArrayPool<float>.Shared.Rent(rank);
        float[] hFloatArr = ArrayPool<float>.Shared.Rent(rows);

        try
        {
            var gSpan = gFloatArr.AsSpan(0, cols);
            var ellSpan = ellFloatArr.AsSpan(0, rank);
            var hSpan = hFloatArr.AsSpan(0, rows);

            ConvertHalfToFloat(path.ColScales, gSpan);
            ConvertHalfToFloat(path.LatentScales, ellSpan);
            ConvertHalfToFloat(path.RowScales, hSpan);

            fixed (ulong* pUBitsFixed = path.UBits)
            fixed (ulong* pVBitsFixed = path.VBits)
            fixed (float* pGFixed = gSpan)
            fixed (float* pEllFixed = ellSpan)
            fixed (float* pHFixed = hSpan)
            fixed (float* pDstFixed = dst)
            {
                ulong* pUBits = pUBitsFixed;
                ulong* pVBits = pVBitsFixed;
                float* pG = pGFixed;
                float* pEll = pEllFixed;
                float* pH = pHFixed;
                float* pDst = pDstFixed;

                int threads = Environment.ProcessorCount;
                if (useParallel && (long)rows * rank * cols >= ParallelWorkThreshold && rows >= threads * 2)
                {
                    Parallel.For(
                        0,
                        rows,
                        () => (
                            ZRow: ArrayPool<float>.Shared.Rent(rank),
                            AccTile: ArrayPool<float>.Shared.Rent(maxTileLen)),
                        (i, _, scratch) =>
                        {
                            DequantizeRow(
                                i, rows, cols, rank, uStride, vStride,
                                pUBits, pVBits, pG, pEll, pH, pDst,
                                scratch.ZRow, scratch.AccTile, accumulate);
                            return scratch;
                        },
                        scratch =>
                        {
                            ArrayPool<float>.Shared.Return(scratch.ZRow);
                            ArrayPool<float>.Shared.Return(scratch.AccTile);
                        });
                }
                else
                {
                    float[] zRow = ArrayPool<float>.Shared.Rent(rank);
                    float[] accTile = ArrayPool<float>.Shared.Rent(maxTileLen);
                    try
                    {
                        for (int i = 0; i < rows; i++)
                        {
                            DequantizeRow(
                                i, rows, cols, rank, uStride, vStride,
                                pUBits, pVBits, pG, pEll, pH, pDst,
                                zRow, accTile, accumulate);
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(zRow);
                        ArrayPool<float>.Shared.Return(accTile);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gFloatArr);
            ArrayPool<float>.Shared.Return(ellFloatArr);
            ArrayPool<float>.Shared.Return(hFloatArr);
        }
    }

    private static unsafe void DequantizeRow(
        int rowIdx,
        int rows,
        int cols,
        int rank,
        int uStride,
        int vStride,
        ulong* pUBits,
        ulong* pVBits,
        float* pG,
        float* pEll,
        float* pH,
        float* pDst,
        float[] zRow,
        float[] accTile,
        bool accumulate)
    {
        _ = rows;
        ulong* uRow = pUBits + (long)rowIdx * uStride;
        for (int k = 0; k < rank; k++)
        {
            uint uBit = (uint)((uRow[k >> 6] >> (k & 63)) & 1UL);
            uint signMask = (uBit ^ 1u) << 31;
            uint ellBits = BitConverter.SingleToUInt32Bits(pEll[k]);
            zRow[k] = BitConverter.UInt32BitsToSingle(ellBits ^ signMask);
        }

        float hVal = pH[rowIdx];
        float* dstRow = pDst + (long)rowIdx * cols;

        fixed (float* pAcc = accTile)
        {
            for (int col0 = 0; col0 < cols; col0 += DequantColTile)
            {
                int tileLen = Math.Min(DequantColTile, cols - col0);
                int wordOffset = col0 >> 6;
                Array.Clear(accTile, 0, tileLen);

                for (int k = 0; k < rank; k++)
                {
                    uint zBits = BitConverter.SingleToUInt32Bits(zRow[k]);
                    ulong* vWords = pVBits + (long)k * vStride + wordOffset;
                    AccumulateSignedBroadcast(vWords, pAcc, tileLen, zBits);
                }

                float* dstTile = dstRow + col0;
                float* gTile = pG + col0;
                if (accumulate)
                {
                    for (int j = 0; j < tileLen; j++)
                        dstTile[j] += pAcc[j] * (hVal * gTile[j]);
                }
                else
                {
                    for (int j = 0; j < tileLen; j++)
                        dstTile[j] = pAcc[j] * (hVal * gTile[j]);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AccumulateSignedBroadcast(ulong* packedBits, float* acc, int length, uint scalarBits)
    {
        int fullWords = length >> 6;
        int tail = length & 63;

        if (Vector512.IsHardwareAccelerated)
        {
            var bitMasks = Vector512.Create(
                1u << 0, 1u << 1, 1u << 2, 1u << 3,
                1u << 4, 1u << 5, 1u << 6, 1u << 7,
                1u << 8, 1u << 9, 1u << 10, 1u << 11,
                1u << 12, 1u << 13, 1u << 14, 1u << 15);
            var signBit = Vector512.Create(0x8000_0000u);
            var zero = Vector512<uint>.Zero;
            var vScalar = Vector512.Create(scalarBits);

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                float* p = acc + (w << 6);
                for (int bit = 0; bit < 64; bit += 16)
                {
                    var vBits = Vector512.Create((uint)((word >> bit) & 0xFFFFu));
                    var isZero = Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), zero);
                    var vSign = Vector512.BitwiseAnd(isZero, signBit);
                    var vSigned = Vector512.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector512.Load(p + bit);
                    Vector512.Store(cur + vSigned, p + bit);
                }
            }

            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                float* p = acc + (fullWords << 6);
                int t = 0;
                while (t + 16 <= tail)
                {
                    var vBits = Vector512.Create((uint)((word >> t) & 0xFFFFu));
                    var isZero = Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), zero);
                    var vSign = Vector512.BitwiseAnd(isZero, signBit);
                    var vSigned = Vector512.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector512.Load(p + t);
                    Vector512.Store(cur + vSigned, p + t);
                    t += 16;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    p[t] += BitConverter.UInt32BitsToSingle(scalarBits ^ sign);
                }
            }
            return;
        }

        if (Avx2.IsSupported)
        {
            var bitMasks = Vector256.Create(1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u);
            var signBit = Vector256.Create(0x8000_0000u);
            var zero = Vector256<uint>.Zero;
            var vScalar = Vector256.Create(scalarBits);

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                float* p = acc + (w << 6);
                for (int bit = 0; bit < 64; bit += 8)
                {
                    var vBits = Vector256.Create((uint)((word >> bit) & 0xFFu));
                    var isZero = Avx2.CompareEqual(Avx2.And(vBits, bitMasks), zero);
                    var vSign = Avx2.And(isZero, signBit);
                    var vSigned = Avx2.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector256.Load(p + bit);
                    Vector256.Store(cur + vSigned, p + bit);
                }
            }

            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                float* p = acc + (fullWords << 6);
                int t = 0;
                while (t + 8 <= tail)
                {
                    var vBits = Vector256.Create((uint)((word >> t) & 0xFFu));
                    var isZero = Avx2.CompareEqual(Avx2.And(vBits, bitMasks), zero);
                    var vSign = Avx2.And(isZero, signBit);
                    var vSigned = Avx2.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector256.Load(p + t);
                    Vector256.Store(cur + vSigned, p + t);
                    t += 8;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    p[t] += BitConverter.UInt32BitsToSingle(scalarBits ^ sign);
                }
            }
            return;
        }

        if (AdvSimd.IsSupported)
        {
            var bitMasks = Vector128.Create(1u, 2u, 4u, 8u);
            var signBit = Vector128.Create(0x8000_0000u);
            var zero = Vector128<uint>.Zero;
            var vScalar = Vector128.Create(scalarBits);

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                float* p = acc + (w << 6);
                for (int bit = 0; bit < 64; bit += 4)
                {
                    var vBits = Vector128.Create((uint)((word >> bit) & 0xFu));
                    var isZero = AdvSimd.CompareEqual(AdvSimd.And(vBits, bitMasks), zero);
                    var vSign = AdvSimd.And(isZero, signBit);
                    var vSigned = AdvSimd.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector128.Load(p + bit);
                    Vector128.Store(AdvSimd.Add(cur, vSigned), p + bit);
                }
            }

            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                float* p = acc + (fullWords << 6);
                int t = 0;
                while (t + 4 <= tail)
                {
                    var vBits = Vector128.Create((uint)((word >> t) & 0xFu));
                    var isZero = AdvSimd.CompareEqual(AdvSimd.And(vBits, bitMasks), zero);
                    var vSign = AdvSimd.And(isZero, signBit);
                    var vSigned = AdvSimd.Xor(vScalar, vSign).AsSingle();
                    var cur = Vector128.Load(p + t);
                    Vector128.Store(AdvSimd.Add(cur, vSigned), p + t);
                    t += 4;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    p[t] += BitConverter.UInt32BitsToSingle(scalarBits ^ sign);
                }
            }
            return;
        }

        // Branchless scalar fallback
        for (int w = 0; w < fullWords; w++)
        {
            ulong word = packedBits[w];
            float* p = acc + (w << 6);
            for (int b = 0; b < 64; b++)
            {
                uint sign = ((uint)~word & 1u) << 31;
                p[b] += BitConverter.UInt32BitsToSingle(scalarBits ^ sign);
                word >>= 1;
            }
        }

        if (tail > 0)
        {
            ulong word = packedBits[fullWords];
            float* p = acc + (fullWords << 6);
            for (int t = 0; t < tail; t++)
            {
                uint sign = ((uint)~word & 1u) << 31;
                p[t] += BitConverter.UInt32BitsToSingle(scalarBits ^ sign);
                word >>= 1;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float BinaryDot(ulong* packedBits, float* vec, int length)
    {
        if (Vector512.IsHardwareAccelerated)
            return BinaryDotVector512(packedBits, vec, length);
        if (AdvSimd.IsSupported)
            return BinaryDotAdvSimd(packedBits, vec, length);
        if (Avx2.IsSupported)
            return BinaryDotAvx2(packedBits, vec, length);
        return BinaryDotScalar(packedBits, vec, length);
    }

    internal static unsafe float BinaryDotVector512(ulong* packedBits, float* vec, int length)
    {
        var bitMasks = Vector512.Create(
            1u << 0, 1u << 1, 1u << 2, 1u << 3,
            1u << 4, 1u << 5, 1u << 6, 1u << 7,
            1u << 8, 1u << 9, 1u << 10, 1u << 11,
            1u << 12, 1u << 13, 1u << 14, 1u << 15);
        var signBit = Vector512.Create(0x8000_0000u);
        var zero = Vector512<uint>.Zero;
        var acc0 = Vector512<float>.Zero;
        var acc1 = Vector512<float>.Zero;

        int fullWords = length >> 6;
        for (int w = 0; w < fullWords; w++)
        {
            ulong word = packedBits[w];
            float* p = vec + (w << 6);

            var vBits0 = Vector512.Create((uint)(word & 0xFFFFu));
            var vSign0 = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits0, bitMasks), zero), signBit);
            acc0 += Vector512.Xor(Vector512.Load(p + 0).AsUInt32(), vSign0).AsSingle();

            var vBits1 = Vector512.Create((uint)((word >> 16) & 0xFFFFu));
            var vSign1 = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits1, bitMasks), zero), signBit);
            acc1 += Vector512.Xor(Vector512.Load(p + 16).AsUInt32(), vSign1).AsSingle();

            var vBits2 = Vector512.Create((uint)((word >> 32) & 0xFFFFu));
            var vSign2 = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits2, bitMasks), zero), signBit);
            acc0 += Vector512.Xor(Vector512.Load(p + 32).AsUInt32(), vSign2).AsSingle();

            var vBits3 = Vector512.Create((uint)((word >> 48) & 0xFFFFu));
            var vSign3 = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits3, bitMasks), zero), signBit);
            acc1 += Vector512.Xor(Vector512.Load(p + 48).AsUInt32(), vSign3).AsSingle();
        }

        float scalarTail = 0.0f;
        int tail = length & 63;
        if (tail > 0)
        {
            ulong word = packedBits[fullWords];
            float* p = vec + (fullWords << 6);
            int t = 0;
            while (t + 16 <= tail)
            {
                var vBits = Vector512.Create((uint)((word >> t) & 0xFFFFu));
                var vSign = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), zero), signBit);
                acc0 += Vector512.Xor(Vector512.Load(p + t).AsUInt32(), vSign).AsSingle();
                t += 16;
            }
            for (; t < tail; t++)
            {
                uint sign = ((uint)~(word >> t) & 1u) << 31;
                uint xBits = BitConverter.SingleToUInt32Bits(p[t]);
                scalarTail += BitConverter.UInt32BitsToSingle(xBits ^ sign);
            }
        }

        return Vector512.Sum(acc0 + acc1) + scalarTail;
    }

    internal static unsafe float BinaryDotAvx2(ulong* packedBits, float* vec, int length)
    {
        var bitMasks = Vector256.Create(1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u);
        var signBit = Vector256.Create(0x8000_0000u);
        var zero = Vector256<uint>.Zero;
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;

        int fullWords = length >> 6;
        for (int w = 0; w < fullWords; w++)
        {
            ulong word = packedBits[w];
            float* p = vec + (w << 6);

            for (int bit = 0; bit < 64; bit += 16)
            {
                var vBits0 = Vector256.Create((uint)((word >> bit) & 0xFFu));
                var vSign0 = Avx2.And(Avx2.CompareEqual(Avx2.And(vBits0, bitMasks), zero), signBit);
                acc0 += Avx2.Xor(Vector256.Load(p + bit).AsUInt32(), vSign0).AsSingle();

                var vBits1 = Vector256.Create((uint)((word >> (bit + 8)) & 0xFFu));
                var vSign1 = Avx2.And(Avx2.CompareEqual(Avx2.And(vBits1, bitMasks), zero), signBit);
                acc1 += Avx2.Xor(Vector256.Load(p + bit + 8).AsUInt32(), vSign1).AsSingle();
            }
        }

        float scalarTail = 0.0f;
        int tail = length & 63;
        if (tail > 0)
        {
            ulong word = packedBits[fullWords];
            float* p = vec + (fullWords << 6);
            int t = 0;
            while (t + 8 <= tail)
            {
                var vBits = Vector256.Create((uint)((word >> t) & 0xFFu));
                var vSign = Avx2.And(Avx2.CompareEqual(Avx2.And(vBits, bitMasks), zero), signBit);
                acc0 += Avx2.Xor(Vector256.Load(p + t).AsUInt32(), vSign).AsSingle();
                t += 8;
            }
            for (; t < tail; t++)
            {
                uint sign = ((uint)~(word >> t) & 1u) << 31;
                uint xBits = BitConverter.SingleToUInt32Bits(p[t]);
                scalarTail += BitConverter.UInt32BitsToSingle(xBits ^ sign);
            }
        }

        return Vector256.Sum(acc0 + acc1) + scalarTail;
    }

    internal static unsafe float BinaryDotAdvSimd(ulong* packedBits, float* vec, int length)
    {
        var bitMasks = Vector128.Create(1u, 2u, 4u, 8u);
        var signBit = Vector128.Create(0x8000_0000u);
        var zero = Vector128<uint>.Zero;
        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;

        int fullWords = length >> 6;
        for (int w = 0; w < fullWords; w++)
        {
            ulong word = packedBits[w];
            float* p = vec + (w << 6);

            for (int bit = 0; bit < 64; bit += 8)
            {
                var vBits0 = Vector128.Create((uint)((word >> bit) & 0xFu));
                var vSign0 = AdvSimd.And(AdvSimd.CompareEqual(AdvSimd.And(vBits0, bitMasks), zero), signBit);
                acc0 = AdvSimd.Add(acc0, AdvSimd.Xor(Vector128.Load(p + bit).AsUInt32(), vSign0).AsSingle());

                var vBits1 = Vector128.Create((uint)((word >> (bit + 4)) & 0xFu));
                var vSign1 = AdvSimd.And(AdvSimd.CompareEqual(AdvSimd.And(vBits1, bitMasks), zero), signBit);
                acc1 = AdvSimd.Add(acc1, AdvSimd.Xor(Vector128.Load(p + bit + 4).AsUInt32(), vSign1).AsSingle());
            }
        }

        float scalarTail = 0.0f;
        int tail = length & 63;
        if (tail > 0)
        {
            ulong word = packedBits[fullWords];
            float* p = vec + (fullWords << 6);
            int t = 0;
            while (t + 4 <= tail)
            {
                var vBits = Vector128.Create((uint)((word >> t) & 0xFu));
                var vSign = AdvSimd.And(AdvSimd.CompareEqual(AdvSimd.And(vBits, bitMasks), zero), signBit);
                acc0 = AdvSimd.Add(acc0, AdvSimd.Xor(Vector128.Load(p + t).AsUInt32(), vSign).AsSingle());
                t += 4;
            }
            for (; t < tail; t++)
            {
                uint sign = ((uint)~(word >> t) & 1u) << 31;
                uint xBits = BitConverter.SingleToUInt32Bits(p[t]);
                scalarTail += BitConverter.UInt32BitsToSingle(xBits ^ sign);
            }
        }

        return Vector128.Sum(AdvSimd.Add(acc0, acc1)) + scalarTail;
    }

    internal static unsafe float BinaryDotScalar(ulong* packedBits, float* vec, int length)
    {
        float sum = 0.0f;
        int fullWords = length >> 6;
        for (int w = 0; w < fullWords; w++)
        {
            ulong wBits = packedBits[w];
            float* p = vec + (w << 6);
            for (int b = 0; b < 64; b++)
            {
                uint sign = ((uint)~wBits & 1u) << 31;
                uint xBits = BitConverter.SingleToUInt32Bits(p[b]);
                sum += BitConverter.UInt32BitsToSingle(xBits ^ sign);
                wBits >>= 1;
            }
        }

        int tail = length & 63;
        if (tail > 0)
        {
            ulong wBits = packedBits[fullWords];
            float* p = vec + (fullWords << 6);
            for (int t = 0; t < tail; t++)
            {
                uint sign = ((uint)~wBits & 1u) << 31;
                uint xBits = BitConverter.SingleToUInt32Bits(p[t]);
                sum += BitConverter.UInt32BitsToSingle(xBits ^ sign);
                wBits >>= 1;
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void BinaryDotBatch4(
        ulong* packedBits,
        float* v0,
        float* v1,
        float* v2,
        float* v3,
        int length,
        out float r0,
        out float r1,
        out float r2,
        out float r3)
    {
        int fullWords = length >> 6;
        int tail = length & 63;

        if (Vector512.IsHardwareAccelerated)
        {
            var bitMasks = Vector512.Create(
                1u << 0, 1u << 1, 1u << 2, 1u << 3,
                1u << 4, 1u << 5, 1u << 6, 1u << 7,
                1u << 8, 1u << 9, 1u << 10, 1u << 11,
                1u << 12, 1u << 13, 1u << 14, 1u << 15);
            var signBit = Vector512.Create(0x8000_0000u);
            var zero = Vector512<uint>.Zero;
            var a0 = Vector512<float>.Zero;
            var a1 = Vector512<float>.Zero;
            var a2 = Vector512<float>.Zero;
            var a3 = Vector512<float>.Zero;

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                int baseIdx = w << 6;
                for (int bit = 0; bit < 64; bit += 16)
                {
                    var vBits = Vector512.Create((uint)((word >> bit) & 0xFFFFu));
                    var vSign = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + bit;
                    a0 += Vector512.Xor(Vector512.Load(v0 + off).AsUInt32(), vSign).AsSingle();
                    a1 += Vector512.Xor(Vector512.Load(v1 + off).AsUInt32(), vSign).AsSingle();
                    a2 += Vector512.Xor(Vector512.Load(v2 + off).AsUInt32(), vSign).AsSingle();
                    a3 += Vector512.Xor(Vector512.Load(v3 + off).AsUInt32(), vSign).AsSingle();
                }
            }

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                int baseIdx = fullWords << 6;
                int t = 0;
                while (t + 16 <= tail)
                {
                    var vBits = Vector512.Create((uint)((word >> t) & 0xFFFFu));
                    var vSign = Vector512.BitwiseAnd(Vector512.Equals(Vector512.BitwiseAnd(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + t;
                    a0 += Vector512.Xor(Vector512.Load(v0 + off).AsUInt32(), vSign).AsSingle();
                    a1 += Vector512.Xor(Vector512.Load(v1 + off).AsUInt32(), vSign).AsSingle();
                    a2 += Vector512.Xor(Vector512.Load(v2 + off).AsUInt32(), vSign).AsSingle();
                    a3 += Vector512.Xor(Vector512.Load(v3 + off).AsUInt32(), vSign).AsSingle();
                    t += 16;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    int off = baseIdx + t;
                    s0 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v0[off]) ^ sign);
                    s1 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v1[off]) ^ sign);
                    s2 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v2[off]) ^ sign);
                    s3 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v3[off]) ^ sign);
                }
            }

            r0 = Vector512.Sum(a0) + s0;
            r1 = Vector512.Sum(a1) + s1;
            r2 = Vector512.Sum(a2) + s2;
            r3 = Vector512.Sum(a3) + s3;
            return;
        }

        if (AdvSimd.IsSupported)
        {
            var bitMasks = Vector128.Create(1u, 2u, 4u, 8u);
            var signBit = Vector128.Create(0x8000_0000u);
            var zero = Vector128<uint>.Zero;
            var a0 = Vector128<float>.Zero;
            var a1 = Vector128<float>.Zero;
            var a2 = Vector128<float>.Zero;
            var a3 = Vector128<float>.Zero;

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                int baseIdx = w << 6;
                for (int bit = 0; bit < 64; bit += 4)
                {
                    var vBits = Vector128.Create((uint)((word >> bit) & 0xFu));
                    var vSign = AdvSimd.And(AdvSimd.CompareEqual(AdvSimd.And(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + bit;
                    a0 = AdvSimd.Add(a0, AdvSimd.Xor(Vector128.Load(v0 + off).AsUInt32(), vSign).AsSingle());
                    a1 = AdvSimd.Add(a1, AdvSimd.Xor(Vector128.Load(v1 + off).AsUInt32(), vSign).AsSingle());
                    a2 = AdvSimd.Add(a2, AdvSimd.Xor(Vector128.Load(v2 + off).AsUInt32(), vSign).AsSingle());
                    a3 = AdvSimd.Add(a3, AdvSimd.Xor(Vector128.Load(v3 + off).AsUInt32(), vSign).AsSingle());
                }
            }

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                int baseIdx = fullWords << 6;
                int t = 0;
                while (t + 4 <= tail)
                {
                    var vBits = Vector128.Create((uint)((word >> t) & 0xFu));
                    var vSign = AdvSimd.And(AdvSimd.CompareEqual(AdvSimd.And(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + t;
                    a0 = AdvSimd.Add(a0, AdvSimd.Xor(Vector128.Load(v0 + off).AsUInt32(), vSign).AsSingle());
                    a1 = AdvSimd.Add(a1, AdvSimd.Xor(Vector128.Load(v1 + off).AsUInt32(), vSign).AsSingle());
                    a2 = AdvSimd.Add(a2, AdvSimd.Xor(Vector128.Load(v2 + off).AsUInt32(), vSign).AsSingle());
                    a3 = AdvSimd.Add(a3, AdvSimd.Xor(Vector128.Load(v3 + off).AsUInt32(), vSign).AsSingle());
                    t += 4;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    int off = baseIdx + t;
                    s0 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v0[off]) ^ sign);
                    s1 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v1[off]) ^ sign);
                    s2 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v2[off]) ^ sign);
                    s3 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v3[off]) ^ sign);
                }
            }

            r0 = Vector128.Sum(a0) + s0;
            r1 = Vector128.Sum(a1) + s1;
            r2 = Vector128.Sum(a2) + s2;
            r3 = Vector128.Sum(a3) + s3;
            return;
        }

        if (Avx2.IsSupported)
        {
            var bitMasks = Vector256.Create(1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u);
            var signBit = Vector256.Create(0x8000_0000u);
            var zero = Vector256<uint>.Zero;
            var a0 = Vector256<float>.Zero;
            var a1 = Vector256<float>.Zero;
            var a2 = Vector256<float>.Zero;
            var a3 = Vector256<float>.Zero;

            for (int w = 0; w < fullWords; w++)
            {
                ulong word = packedBits[w];
                int baseIdx = w << 6;
                for (int bit = 0; bit < 64; bit += 8)
                {
                    var vBits = Vector256.Create((uint)((word >> bit) & 0xFFu));
                    var vSign = Avx2.And(Avx2.CompareEqual(Avx2.And(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + bit;
                    a0 += Avx2.Xor(Vector256.Load(v0 + off).AsUInt32(), vSign).AsSingle();
                    a1 += Avx2.Xor(Vector256.Load(v1 + off).AsUInt32(), vSign).AsSingle();
                    a2 += Avx2.Xor(Vector256.Load(v2 + off).AsUInt32(), vSign).AsSingle();
                    a3 += Avx2.Xor(Vector256.Load(v3 + off).AsUInt32(), vSign).AsSingle();
                }
            }

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            if (tail > 0)
            {
                ulong word = packedBits[fullWords];
                int baseIdx = fullWords << 6;
                int t = 0;
                while (t + 8 <= tail)
                {
                    var vBits = Vector256.Create((uint)((word >> t) & 0xFFu));
                    var vSign = Avx2.And(Avx2.CompareEqual(Avx2.And(vBits, bitMasks), zero), signBit);
                    int off = baseIdx + t;
                    a0 += Avx2.Xor(Vector256.Load(v0 + off).AsUInt32(), vSign).AsSingle();
                    a1 += Avx2.Xor(Vector256.Load(v1 + off).AsUInt32(), vSign).AsSingle();
                    a2 += Avx2.Xor(Vector256.Load(v2 + off).AsUInt32(), vSign).AsSingle();
                    a3 += Avx2.Xor(Vector256.Load(v3 + off).AsUInt32(), vSign).AsSingle();
                    t += 8;
                }
                for (; t < tail; t++)
                {
                    uint sign = ((uint)~(word >> t) & 1u) << 31;
                    int off = baseIdx + t;
                    s0 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v0[off]) ^ sign);
                    s1 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v1[off]) ^ sign);
                    s2 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v2[off]) ^ sign);
                    s3 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v3[off]) ^ sign);
                }
            }

            r0 = Vector256.Sum(a0) + s0;
            r1 = Vector256.Sum(a1) + s1;
            r2 = Vector256.Sum(a2) + s2;
            r3 = Vector256.Sum(a3) + s3;
            return;
        }

        // Scalar fallback
        float sr0 = 0.0f, sr1 = 0.0f, sr2 = 0.0f, sr3 = 0.0f;
        for (int w = 0; w < fullWords; w++)
        {
            ulong wBits = packedBits[w];
            int baseIdx = w << 6;
            for (int b = 0; b < 64; b++)
            {
                uint sign = ((uint)~wBits & 1u) << 31;
                int off = baseIdx + b;
                sr0 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v0[off]) ^ sign);
                sr1 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v1[off]) ^ sign);
                sr2 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v2[off]) ^ sign);
                sr3 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v3[off]) ^ sign);
                wBits >>= 1;
            }
        }
        if (tail > 0)
        {
            ulong wBits = packedBits[fullWords];
            int baseIdx = fullWords << 6;
            for (int t = 0; t < tail; t++)
            {
                uint sign = ((uint)~wBits & 1u) << 31;
                int off = baseIdx + t;
                sr0 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v0[off]) ^ sign);
                sr1 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v1[off]) ^ sign);
                sr2 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v2[off]) ^ sign);
                sr3 += BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v3[off]) ^ sign);
                wBits >>= 1;
            }
        }

        r0 = sr0;
        r1 = sr1;
        r2 = sr2;
        r3 = sr3;
    }

    private static double[] MultiplyDoubleMatrices(double[] a, int m, int k, double[] b, int n)
    {
        double[] dst = new double[checked(m * n)];
        MultiplyDoubleMatricesInto(a, m, k, b, n, dst);
        return dst;
    }

    private static void MultiplyDoubleMatricesInto(double[] a, int m, int k, double[] b, int n, double[] dst)
    {
        Array.Clear(dst, 0, m * n);
        for (int i = 0; i < m; i++)
        {
            int aRow = i * k;
            int cRow = i * n;
            for (int t = 0; t < k; t++)
            {
                double ait = a[aRow + t];
                if (ait == 0.0)
                    continue;
                int bRow = t * n;
                for (int j = 0; j < n; j++)
                    dst[cRow + j] += ait * b[bRow + j];
            }
        }
    }

    private static double[] MultiplyDoubleTransposeFirst(double[] a, int rank, double[] b, int cols)
    {
        // Computes A^T * B where A is [rank x rank] and B is [rank x cols]
        double[] dst = new double[checked(rank * cols)];
        for (int t = 0; t < rank; t++)
        {
            int aRow = t * rank;
            int bRow = t * cols;
            for (int p = 0; p < rank; p++)
            {
                double atp = a[aRow + p];
                if (atp == 0.0)
                    continue;
                int cRow = p * cols;
                for (int j = 0; j < cols; j++)
                    dst[cRow + j] += atp * b[bRow + j];
            }
        }
        return dst;
    }

    private static double MaxPositive(double[] values)
    {
        double max = 0.0;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] > max)
                max = values[i];
        }
        return max;
    }

    private static void ConvertToHalfClamped(double[] src, double multiplier, Half[] dst)
    {
        const double maxHalf = 65504.0;
        for (int i = 0; i < src.Length; i++)
        {
            double scaled = Math.Clamp(src[i] * multiplier, 0.0, maxHalf);
            dst[i] = (Half)scaled;
        }
    }

    private static void ConvertHalfToFloat(Half[] src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = (float)src[i];
    }

    private static void EnsureAllFinite(ReadOnlySpan<float> values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new ArgumentException($"Input '{paramName}' contains non-finite value ({values[i]}) at index {i}.", paramName);
        }
    }
}
