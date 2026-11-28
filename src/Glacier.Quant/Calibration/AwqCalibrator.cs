namespace Glacier.Quant.Calibration;

using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

/// <summary>
/// Activation-aware Weight Quantization (AWQ) calibrator.
/// Protects salient weights by calculating per-channel activation magnitudes s_X
/// across all calibration tokens and performing a grid search over alpha in [0, 1]
/// to minimize output reconstruction MSE:
/// || W X - dequant(quant(W * s)) * s^-1 X ||_F^2.
/// </summary>
public static unsafe class AwqCalibrator
{
    private const long MaxCachedOutputElements = 16_777_216L;
    private const int StreamingTokenTile = 64;

    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset,
        int gridSteps = 20,
        int targetBits = 4)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegative(gridSteps);

        long totalWeights = (long)rows * cols;
        if (weights.Length != totalWeights)
            throw new ArgumentException($"Weights length ({weights.Length}) does not match rows * cols ({totalWeights}).", nameof(weights));
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).", nameof(dataset));
        if (targetBits is < 2 or > 8)
            throw new ArgumentOutOfRangeException(nameof(targetBits), targetBits, "Target bits must be in [2, 8].");

        int tokens = dataset.NumTokens;
        int actLen = checked(tokens * cols);
        EnsureAllFinite(weights, nameof(weights));
        EnsureAllFinite(dataset.Activations.AsSpan(0, actLen), nameof(dataset));

        long totalOut = (long)tokens * rows;
        bool cacheRefOut = totalOut <= MaxCachedOutputElements;
        int outBufLen = cacheRefOut
            ? checked((int)totalOut)
            : checked(Math.Min(tokens, StreamingTokenTile) * rows);

        double[] sxAcc = ArrayPool<double>.Shared.Rent(cols);
        float[] sx = ArrayPool<float>.Shared.Rent(cols);
        float[] candidateScales = ArrayPool<float>.Shared.Rent(cols);
        float[] invCandidateScales = ArrayPool<float>.Shared.Rent(cols);
        float[] scaledWeights = ArrayPool<float>.Shared.Rent(weights.Length);
        float[] quantized = ArrayPool<float>.Shared.Rent(weights.Length);
        float[]? refOut = null;
        float[]? candOut = null;
        double[]? refOutDouble = null;
        double[]? candOutDouble = null;

        try
        {
            Span<double> sxAccSpan = sxAcc.AsSpan(0, cols);
            Span<float> sxSpan = sx.AsSpan(0, cols);
            Span<float> candScaleSpan = candidateScales.AsSpan(0, cols);
            Span<float> invScaleSpan = invCandidateScales.AsSpan(0, cols);
            sxAccSpan.Clear();

            // 1. Compute per-channel activation magnitude across ALL tokens in double: s_X(j) = sum_k |X(k, j)|
            double maxAbsX = 0.0;
            for (int k = 0; k < tokens; k++)
            {
                ReadOnlySpan<float> row = dataset.GetTokenVector(k);
                for (int j = 0; j < cols; j++)
                {
                    double ax = Math.Abs((double)row[j]);
                    sxAccSpan[j] += ax;
                    if (ax > maxAbsX) maxAbsX = ax;
                }
            }

            double maxAbsW = 0.0;
            for (int i = 0; i < weights.Length; i++)
            {
                double aw = Math.Abs((double)weights[i]);
                if (aw > maxAbsW) maxAbsW = aw;
            }

            double sumSx = 0.0;
            for (int j = 0; j < cols; j++)
            {
                sumSx += sxAccSpan[j];
            }

            if (sumSx <= 0.0 || !double.IsFinite(sumSx))
            {
                sxSpan.Fill(1.0f);
            }
            else
            {
                double meanSx = sumSx / cols;
                for (int j = 0; j < cols; j++)
                {
                    sxSpan[j] = (float)Math.Max(1e-12, sxAccSpan[j] / meanSx);
                }
            }

            double maxProductBound = maxAbsX * maxAbsW * cols * Math.Max(1.0, (double)cols);
            bool useDoubleMse = maxProductBound > 1e33
                || (maxAbsX > 0.0 && maxAbsW > 0.0 && maxAbsX * maxAbsW < 1e-30);

            if (useDoubleMse)
            {
                refOutDouble = ArrayPool<double>.Shared.Rent(outBufLen);
                candOutDouble = ArrayPool<double>.Shared.Rent(outBufLen);
            }
            else
            {
                refOut = ArrayPool<float>.Shared.Rent(outBufLen);
                candOut = ArrayPool<float>.Shared.Rent(outBufLen);
            }

            // 2. Compute baseline Round-To-Nearest (RTN) weights & MSE
            float[] bestWeights = new float[weights.Length];
            float[] bestScales = new float[cols];
            Array.Fill(bestScales, 1.0f);

            if (maxAbsW > 1e37)
            {
                QuantizeRtnDouble(weights, bestWeights, ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, rows, cols, targetBits);
            }
            else
            {
                QuantizeRtnInto(weights, bestWeights, rows, cols, targetBits);
            }

            double baselineMse;
            if (useDoubleMse)
            {
                if (cacheRefOut)
                {
                    ComputeGemmNTDouble(dataset.Activations.AsSpan(0, actLen), tokens, cols, weights, rows, refOutDouble.AsSpan(0, outBufLen));
                }

                baselineMse = ComputeOutputMseDouble(
                    dataset.Activations.AsSpan(0, actLen),
                    weights,
                    bestWeights,
                    rows,
                    cols,
                    tokens,
                    cacheRefOut,
                    refOutDouble.AsSpan(0, outBufLen),
                    candOutDouble.AsSpan(0, outBufLen));
            }
            else
            {
                if (cacheRefOut)
                {
                    ComputeGemmNT(dataset.Activations.AsSpan(0, actLen), tokens, cols, weights, rows, refOut.AsSpan(0, outBufLen));
                }

                baselineMse = ComputeOutputMse(
                    dataset.Activations.AsSpan(0, actLen),
                    weights,
                    bestWeights,
                    rows,
                    cols,
                    tokens,
                    cacheRefOut,
                    refOut.AsSpan(0, outBufLen),
                    candOut.AsSpan(0, outBufLen));
            }

            // 3. Grid search over alpha in [0, 1]
            double bestMse = baselineMse;
            float bestAlpha = 0.0f;

            for (int step = 1; step <= gridSteps; step++)
            {
                float alpha = (float)step / gridSteps;

                // s(j) = s_X(j)^alpha in double
                double sumScales = 0.0;
                for (int j = 0; j < cols; j++)
                {
                    double s = Math.Pow(sxSpan[j], alpha);
                    sxAccSpan[j] = s;
                    sumScales += s;
                }

                // Normalize so mean scale is 1.0
                double meanScale = sumScales / cols;
                double invMean = (meanScale > 0.0 && double.IsFinite(meanScale)) ? (1.0 / meanScale) : 1.0;
                double maxCandScale = 0.0;
                for (int j = 0; j < cols; j++)
                {
                    double sNorm = Math.Max(1e-12, sxAccSpan[j] * invMean);
                    candScaleSpan[j] = (float)sNorm;
                    invScaleSpan[j] = (float)(1.0 / sNorm);
                    if (sNorm > maxCandScale) maxCandScale = sNorm;
                }

                if (maxAbsW * maxCandScale > 1e37)
                {
                    QuantizeRtnDouble(weights, quantized.AsSpan(0, weights.Length), candScaleSpan, invScaleSpan, rows, cols, targetBits);
                }
                else
                {
                    // Scale weights W'(i, j) = W(i, j) * s(j)
                    for (int i = 0; i < rows; i++)
                    {
                        int rOff = i * cols;
                        TensorPrimitives.Multiply(weights.AsSpan(rOff, cols), candScaleSpan, scaledWeights.AsSpan(rOff, cols));
                    }

                    // Quantize scaled weights into pooled buffer
                    QuantizeRtnInto(scaledWeights.AsSpan(0, weights.Length), quantized.AsSpan(0, weights.Length), rows, cols, targetBits);

                    // Unscale dequantized weights: W_hat(i, j) = W'_hat(i, j) * s(j)^-1
                    for (int i = 0; i < rows; i++)
                    {
                        int rOff = i * cols;
                        Span<float> qRow = quantized.AsSpan(rOff, cols);
                        TensorPrimitives.Multiply(qRow, invScaleSpan, qRow);
                    }
                }

                double candidateMse = useDoubleMse
                    ? ComputeOutputMseDouble(
                        dataset.Activations.AsSpan(0, actLen),
                        weights,
                        quantized.AsSpan(0, weights.Length),
                        rows,
                        cols,
                        tokens,
                        cacheRefOut,
                        refOutDouble.AsSpan(0, outBufLen),
                        candOutDouble.AsSpan(0, outBufLen))
                    : ComputeOutputMse(
                        dataset.Activations.AsSpan(0, actLen),
                        weights,
                        quantized.AsSpan(0, weights.Length),
                        rows,
                        cols,
                        tokens,
                        cacheRefOut,
                        refOut.AsSpan(0, outBufLen),
                        candOut.AsSpan(0, outBufLen));

                if (candidateMse < bestMse)
                {
                    bestMse = candidateMse;
                    bestAlpha = alpha;
                    Array.Copy(quantized, 0, bestWeights, 0, weights.Length);
                    candScaleSpan.CopyTo(bestScales);
                }
            }

            return new PtqCalibrationResult
            {
                Method = $"AWQ (alpha={bestAlpha:F2})",
                CalibratedWeights = bestWeights,
                OptimalScales = bestScales,
                BaselineRtnMse = baselineMse,
                CalibratedMse = bestMse
            };
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sxAcc);
            ArrayPool<float>.Shared.Return(sx);
            ArrayPool<float>.Shared.Return(candidateScales);
            ArrayPool<float>.Shared.Return(invCandidateScales);
            ArrayPool<float>.Shared.Return(scaledWeights);
            ArrayPool<float>.Shared.Return(quantized);
            if (refOut is not null) ArrayPool<float>.Shared.Return(refOut);
            if (candOut is not null) ArrayPool<float>.Shared.Return(candOut);
            if (refOutDouble is not null) ArrayPool<double>.Shared.Return(refOutDouble);
            if (candOutDouble is not null) ArrayPool<double>.Shared.Return(candOutDouble);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizeRtnInto(
        ReadOnlySpan<float> weights,
        Span<float> dst,
        int rows,
        int cols,
        int targetBits)
    {
        float maxQ = (1 << (targetBits - 1)) - 1;
        float minQ = -(1 << (targetBits - 1));

        fixed (float* pW = weights)
        fixed (float* pDst = dst)
        {
            if ((long)rows * cols >= 262_144L && rows >= 8)
            {
                nint wAddr = (nint)pW;
                nint dstAddr = (nint)pDst;
                Parallel.For(0, rows, i =>
                {
                    QuantizeRowRtn((float*)wAddr + (long)i * cols, (float*)dstAddr + (long)i * cols, cols, minQ, maxQ);
                });
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    QuantizeRowRtn(pW + (long)i * cols, pDst + (long)i * cols, cols, minQ, maxQ);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizeRowRtn(float* srcRow, float* dstRow, int cols, float minQ, float maxQ)
    {
        const int blockSize = 32;
        int b = 0;

        if (Avx512F.IsSupported)
        {
            var vMinQ = Vector512.Create(minQ);
            var vMaxQ = Vector512.Create(maxQ);
            for (; b + blockSize <= cols; b += blockSize)
            {
                float* s = srcRow + b;
                float* d = dstRow + b;
                var v0 = Vector512.Load(s);
                var v1 = Vector512.Load(s + 16);
                var vMax = Vector512.Max(Vector512.Abs(v0), Vector512.Abs(v1));
                float amax = HorizontalMax512(vMax);

                if (amax <= 1e-12f)
                {
                    Vector512<float>.Zero.Store(d);
                    Vector512<float>.Zero.Store(d + 16);
                    continue;
                }

                float scale = amax / maxQ;
                float invScale = maxQ / amax;
                var vScale = Vector512.Create(scale);
                var vInvScale = Vector512.Create(invScale);

                var q0 = Avx512F.RoundScale(Vector512.Multiply(v0, vInvScale), 0x00);
                q0 = Vector512.Min(Vector512.Max(q0, vMinQ), vMaxQ);
                Vector512.Multiply(q0, vScale).Store(d);

                var q1 = Avx512F.RoundScale(Vector512.Multiply(v1, vInvScale), 0x00);
                q1 = Vector512.Min(Vector512.Max(q1, vMinQ), vMaxQ);
                Vector512.Multiply(q1, vScale).Store(d + 16);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vMinQ = Vector256.Create(minQ);
            var vMaxQ = Vector256.Create(maxQ);
            for (; b + blockSize <= cols; b += blockSize)
            {
                float* s = srcRow + b;
                float* d = dstRow + b;
                var v0 = Vector256.Load(s);
                var v1 = Vector256.Load(s + 8);
                var v2 = Vector256.Load(s + 16);
                var v3 = Vector256.Load(s + 24);
                var vMax01 = Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1));
                var vMax23 = Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3));
                float amax = HorizontalMax256(Vector256.Max(vMax01, vMax23));

                if (amax <= 1e-12f)
                {
                    Vector256<float>.Zero.Store(d);
                    Vector256<float>.Zero.Store(d + 8);
                    Vector256<float>.Zero.Store(d + 16);
                    Vector256<float>.Zero.Store(d + 24);
                    continue;
                }

                float scale = amax / maxQ;
                float invScale = maxQ / amax;
                var vScale = Vector256.Create(scale);
                var vInvScale = Vector256.Create(invScale);

                var q0 = Vector256.Min(Vector256.Max(Avx.RoundToNearestInteger(Vector256.Multiply(v0, vInvScale)), vMinQ), vMaxQ);
                Vector256.Multiply(q0, vScale).Store(d);

                var q1 = Vector256.Min(Vector256.Max(Avx.RoundToNearestInteger(Vector256.Multiply(v1, vInvScale)), vMinQ), vMaxQ);
                Vector256.Multiply(q1, vScale).Store(d + 8);

                var q2 = Vector256.Min(Vector256.Max(Avx.RoundToNearestInteger(Vector256.Multiply(v2, vInvScale)), vMinQ), vMaxQ);
                Vector256.Multiply(q2, vScale).Store(d + 16);

                var q3 = Vector256.Min(Vector256.Max(Avx.RoundToNearestInteger(Vector256.Multiply(v3, vInvScale)), vMinQ), vMaxQ);
                Vector256.Multiply(q3, vScale).Store(d + 24);
            }
        }

        for (; b < cols; b += blockSize)
        {
            int curBlock = Math.Min(blockSize, cols - b);
            float amax = 0.0f;
            for (int j = 0; j < curBlock; j++)
            {
                float a = MathF.Abs(srcRow[b + j]);
                if (a > amax) amax = a;
            }

            if (amax <= 1e-12f)
            {
                for (int j = 0; j < curBlock; j++)
                    dstRow[b + j] = 0.0f;
                continue;
            }

            float scale = amax / maxQ;
            float invScale = maxQ / amax;

            for (int j = 0; j < curBlock; j++)
            {
                float q = MathF.Round(srcRow[b + j] * invScale);
                q = Math.Clamp(q, minQ, maxQ);
                dstRow[b + j] = q * scale;
            }
        }
    }

    private static void QuantizeRtnDouble(
        ReadOnlySpan<float> weights,
        Span<float> dst,
        ReadOnlySpan<float> candScales,
        ReadOnlySpan<float> invScales,
        int rows,
        int cols,
        int targetBits)
    {
        double maxQ = (1 << (targetBits - 1)) - 1;
        double minQ = -(1 << (targetBits - 1));
        bool hasScales = !candScales.IsEmpty;
        const int blockSize = 32;

        for (int r = 0; r < rows; r++)
        {
            int rOff = r * cols;
            for (int b = 0; b < cols; b += blockSize)
            {
                int curBlock = Math.Min(blockSize, cols - b);
                double amax = 0.0;
                for (int j = 0; j < curBlock; j++)
                {
                    double w = weights[rOff + b + j];
                    if (hasScales) w *= candScales[b + j];
                    double a = Math.Abs(w);
                    if (a > amax) amax = a;
                }

                if (amax <= 1e-12)
                {
                    for (int j = 0; j < curBlock; j++)
                        dst[rOff + b + j] = 0.0f;
                    continue;
                }

                double scale = amax / maxQ;
                double invScale = maxQ / amax;

                for (int j = 0; j < curBlock; j++)
                {
                    double w = weights[rOff + b + j];
                    if (hasScales) w *= candScales[b + j];
                    double q = Math.Round(w * invScale);
                    q = Math.Clamp(q, minQ, maxQ);
                    double dequant = q * scale;
                    if (hasScales) dequant *= invScales[b + j];
                    dst[rOff + b + j] = (float)Math.Clamp(dequant, -(double)float.MaxValue, (double)float.MaxValue);
                }
            }
        }
    }

    private static double ComputeOutputMse(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<float> originalW,
        ReadOnlySpan<float> quantW,
        int rows,
        int cols,
        int tokens,
        bool useCachedRef,
        Span<float> refOut,
        Span<float> candOut)
    {
        if (useCachedRef)
        {
            ComputeGemmNT(activations, tokens, cols, quantW, rows, candOut);
            double sse = SumSquaredDifference(refOut, candOut);
            double mse = sse / ((double)tokens * rows);
            return double.IsFinite(mse)
                ? mse
                : ComputeOutputMseDoubleStreaming(activations, originalW, quantW, rows, cols, tokens);
        }

        double totalSse = 0.0;
        for (int tStart = 0; tStart < tokens; tStart += StreamingTokenTile)
        {
            int curTokens = Math.Min(StreamingTokenTile, tokens - tStart);
            int curActLen = curTokens * cols;
            int curOutLen = curTokens * rows;
            ReadOnlySpan<float> actSlice = activations.Slice(tStart * cols, curActLen);
            Span<float> refTile = refOut.Slice(0, curOutLen);
            Span<float> candTile = candOut.Slice(0, curOutLen);

            ComputeGemmNT(actSlice, curTokens, cols, originalW, rows, refTile);
            ComputeGemmNT(actSlice, curTokens, cols, quantW, rows, candTile);
            totalSse += SumSquaredDifference(refTile, candTile);
            if (!double.IsFinite(totalSse))
                return ComputeOutputMseDoubleStreaming(activations, originalW, quantW, rows, cols, tokens);
        }

        double streamingMse = totalSse / ((double)tokens * rows);
        return double.IsFinite(streamingMse)
            ? streamingMse
            : ComputeOutputMseDoubleStreaming(activations, originalW, quantW, rows, cols, tokens);
    }

    private static double ComputeOutputMseDoubleStreaming(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<float> originalW,
        ReadOnlySpan<float> quantW,
        int rows,
        int cols,
        int tokens)
    {
        int tileOutLen = checked(Math.Min(tokens, StreamingTokenTile) * rows);
        double[] refTile = ArrayPool<double>.Shared.Rent(tileOutLen);
        double[] candTile = ArrayPool<double>.Shared.Rent(tileOutLen);
        try
        {
            return ComputeOutputMseDouble(
                activations,
                originalW,
                quantW,
                rows,
                cols,
                tokens,
                useCachedRef: false,
                refTile.AsSpan(0, tileOutLen),
                candTile.AsSpan(0, tileOutLen));
        }
        finally
        {
            ArrayPool<double>.Shared.Return(refTile);
            ArrayPool<double>.Shared.Return(candTile);
        }
    }

    private static double ComputeOutputMseDouble(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<float> originalW,
        ReadOnlySpan<float> quantW,
        int rows,
        int cols,
        int tokens,
        bool useCachedRef,
        Span<double> refOut,
        Span<double> candOut)
    {
        if (useCachedRef)
        {
            ComputeGemmNTDouble(activations, tokens, cols, quantW, rows, candOut);
            double sse = SumSquaredDifferenceDouble(refOut, candOut);
            return sse / ((double)tokens * rows);
        }

        double totalSse = 0.0;
        for (int tStart = 0; tStart < tokens; tStart += StreamingTokenTile)
        {
            int curTokens = Math.Min(StreamingTokenTile, tokens - tStart);
            int curActLen = curTokens * cols;
            int curOutLen = curTokens * rows;
            ReadOnlySpan<float> actSlice = activations.Slice(tStart * cols, curActLen);
            Span<double> refTile = refOut.Slice(0, curOutLen);
            Span<double> candTile = candOut.Slice(0, curOutLen);

            ComputeGemmNTDouble(actSlice, curTokens, cols, originalW, rows, refTile);
            ComputeGemmNTDouble(actSlice, curTokens, cols, quantW, rows, candTile);
            totalSse += SumSquaredDifferenceDouble(refTile, candTile);
        }

        return totalSse / ((double)tokens * rows);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeGemmNTDouble(
        ReadOnlySpan<float> a,
        int m,
        int k,
        ReadOnlySpan<float> b,
        int n,
        Span<double> c)
    {
        fixed (float* pA = a)
        fixed (float* pB = b)
        fixed (double* pC = c)
        {
            if ((long)m * k * n >= 262_144L && n >= 8)
            {
                nint aAddr = (nint)pA;
                nint bAddr = (nint)pB;
                nint cAddr = (nint)pC;
                Parallel.For(0, n, r =>
                {
                    GemmNtRowDouble((float*)aAddr, m, k, (float*)bAddr + (long)r * k, (double*)cAddr, n, r);
                });
            }
            else
            {
                for (int r = 0; r < n; r++)
                {
                    GemmNtRowDouble(pA, m, k, pB + (long)r * k, pC, n, r);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmNtRowDouble(float* pA, int m, int k, float* bRow, double* pC, int n, int r)
    {
        int t = 0;
        for (; t + 4 <= m; t += 4)
        {
            float* a0 = pA + (long)t * k;
            float* a1 = a0 + k;
            float* a2 = a1 + k;
            float* a3 = a2 + k;

            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;

            for (int c = 0; c < k; c++)
            {
                double bv = bRow[c];
                s0 += (double)a0[c] * bv;
                s1 += (double)a1[c] * bv;
                s2 += (double)a2[c] * bv;
                s3 += (double)a3[c] * bv;
            }

            pC[(long)t * n + r] = s0;
            pC[(long)(t + 1) * n + r] = s1;
            pC[(long)(t + 2) * n + r] = s2;
            pC[(long)(t + 3) * n + r] = s3;
        }

        for (; t < m; t++)
        {
            float* a0 = pA + (long)t * k;
            double sum = 0.0;
            for (int c = 0; c < k; c++)
                sum += (double)a0[c] * (double)bRow[c];
            pC[(long)t * n + r] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumSquaredDifferenceDouble(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeGemmNT(
        ReadOnlySpan<float> a,
        int m,
        int k,
        ReadOnlySpan<float> b,
        int n,
        Span<float> c)
    {
        fixed (float* pA = a)
        fixed (float* pB = b)
        fixed (float* pC = c)
        {
            if ((long)m * k * n >= 262_144L && n >= 8)
            {
                nint aAddr = (nint)pA;
                nint bAddr = (nint)pB;
                nint cAddr = (nint)pC;
                Parallel.For(0, n, r =>
                {
                    GemmNtRow((float*)aAddr, m, k, (float*)bAddr + (long)r * k, (float*)cAddr, n, r);
                });
            }
            else
            {
                for (int r = 0; r < n; r++)
                {
                    GemmNtRow(pA, m, k, pB + (long)r * k, pC, n, r);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmNtRow(float* pA, int m, int k, float* bRow, float* pC, int n, int r)
    {
        int t = 0;

        if (Avx512F.IsSupported)
        {
            for (; t + 4 <= m; t += 4)
            {
                float* a0 = pA + (long)t * k;
                float* a1 = a0 + k;
                float* a2 = a1 + k;
                float* a3 = a2 + k;

                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;

                int c = 0;
                for (; c + 16 <= k; c += 16)
                {
                    var vb = Vector512.Load(bRow + c);
                    acc0 = Avx512F.FusedMultiplyAdd(Vector512.Load(a0 + c), vb, acc0);
                    acc1 = Avx512F.FusedMultiplyAdd(Vector512.Load(a1 + c), vb, acc1);
                    acc2 = Avx512F.FusedMultiplyAdd(Vector512.Load(a2 + c), vb, acc2);
                    acc3 = Avx512F.FusedMultiplyAdd(Vector512.Load(a3 + c), vb, acc3);
                }

                float s0 = Vector512.Sum(acc0);
                float s1 = Vector512.Sum(acc1);
                float s2 = Vector512.Sum(acc2);
                float s3 = Vector512.Sum(acc3);

                for (; c < k; c++)
                {
                    float bv = bRow[c];
                    s0 += a0[c] * bv;
                    s1 += a1[c] * bv;
                    s2 += a2[c] * bv;
                    s3 += a3[c] * bv;
                }

                pC[(long)t * n + r] = s0;
                pC[(long)(t + 1) * n + r] = s1;
                pC[(long)(t + 2) * n + r] = s2;
                pC[(long)(t + 3) * n + r] = s3;
            }

            for (; t < m; t++)
            {
                float* a0 = pA + (long)t * k;
                var acc = Vector512<float>.Zero;
                int c = 0;
                for (; c + 16 <= k; c += 16)
                    acc = Avx512F.FusedMultiplyAdd(Vector512.Load(a0 + c), Vector512.Load(bRow + c), acc);
                float sum = Vector512.Sum(acc);
                for (; c < k; c++)
                    sum += a0[c] * bRow[c];
                pC[(long)t * n + r] = sum;
            }
            return;
        }

        if (Avx2.IsSupported)
        {
            for (; t + 4 <= m; t += 4)
            {
                float* a0 = pA + (long)t * k;
                float* a1 = a0 + k;
                float* a2 = a1 + k;
                float* a3 = a2 + k;

                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;

                int c = 0;
                if (Fma.IsSupported)
                {
                    for (; c + 8 <= k; c += 8)
                    {
                        var vb = Vector256.Load(bRow + c);
                        acc0 = Fma.MultiplyAdd(Vector256.Load(a0 + c), vb, acc0);
                        acc1 = Fma.MultiplyAdd(Vector256.Load(a1 + c), vb, acc1);
                        acc2 = Fma.MultiplyAdd(Vector256.Load(a2 + c), vb, acc2);
                        acc3 = Fma.MultiplyAdd(Vector256.Load(a3 + c), vb, acc3);
                    }
                }
                else
                {
                    for (; c + 8 <= k; c += 8)
                    {
                        var vb = Vector256.Load(bRow + c);
                        acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Load(a0 + c), vb));
                        acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Load(a1 + c), vb));
                        acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Load(a2 + c), vb));
                        acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Load(a3 + c), vb));
                    }
                }

                float s0 = Vector256.Sum(acc0);
                float s1 = Vector256.Sum(acc1);
                float s2 = Vector256.Sum(acc2);
                float s3 = Vector256.Sum(acc3);

                for (; c < k; c++)
                {
                    float bv = bRow[c];
                    s0 += a0[c] * bv;
                    s1 += a1[c] * bv;
                    s2 += a2[c] * bv;
                    s3 += a3[c] * bv;
                }

                pC[(long)t * n + r] = s0;
                pC[(long)(t + 1) * n + r] = s1;
                pC[(long)(t + 2) * n + r] = s2;
                pC[(long)(t + 3) * n + r] = s3;
            }

            for (; t < m; t++)
            {
                float* a0 = pA + (long)t * k;
                var acc = Vector256<float>.Zero;
                int c = 0;
                if (Fma.IsSupported)
                {
                    for (; c + 8 <= k; c += 8)
                        acc = Fma.MultiplyAdd(Vector256.Load(a0 + c), Vector256.Load(bRow + c), acc);
                }
                else
                {
                    for (; c + 8 <= k; c += 8)
                        acc = Avx.Add(acc, Avx.Multiply(Vector256.Load(a0 + c), Vector256.Load(bRow + c)));
                }
                float sum = Vector256.Sum(acc);
                for (; c < k; c++)
                    sum += a0[c] * bRow[c];
                pC[(long)t * n + r] = sum;
            }
            return;
        }

        for (; t < m; t++)
        {
            float* a0 = pA + (long)t * k;
            float sum = 0.0f;
            for (int c = 0; c < k; c++)
                sum += a0[c] * bRow[c];
            pC[(long)t * n + r] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumSquaredDifference(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = a.Length;
        double sum = 0.0;
        int i = 0;

        fixed (float* pA = a)
        fixed (float* pB = b)
        {
            if (Avx512F.IsSupported && len >= 16)
            {
                var vSum = Vector512<double>.Zero;
                for (; i + 16 <= len; i += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pA + i), Vector512.Load(pB + i));
                    (Vector512<double> dLo, Vector512<double> dHi) = Vector512.Widen(diff);
                    vSum = Avx512F.FusedMultiplyAdd(dLo, dLo, vSum);
                    vSum = Avx512F.FusedMultiplyAdd(dHi, dHi, vSum);
                }
                sum += Vector512.Sum(vSum);
            }
            else if (Avx2.IsSupported && len >= 8)
            {
                var vSum = Vector256<double>.Zero;
                for (; i + 8 <= len; i += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pA + i), Vector256.Load(pB + i));
                    (Vector256<double> dLo, Vector256<double> dHi) = Vector256.Widen(diff);
                    if (Fma.IsSupported)
                    {
                        vSum = Fma.MultiplyAdd(dLo, dLo, vSum);
                        vSum = Fma.MultiplyAdd(dHi, dHi, vSum);
                    }
                    else
                    {
                        vSum = Avx.Add(vSum, Avx.Multiply(dLo, dLo));
                        vSum = Avx.Add(vSum, Avx.Multiply(dHi, dHi));
                    }
                }
                sum += Vector256.Sum(vSum);
            }

            for (; i < len; i++)
            {
                double d = pA[i] - pB[i];
                sum += d * d;
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax512(Vector512<float> v)
    {
        var v8 = Vector256.Max(v.GetLower(), v.GetUpper());
        return HorizontalMax256(v8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax256(Vector256<float> v)
    {
        var v4 = Vector128.Max(v.GetLower(), v.GetUpper());
        v4 = Vector128.Max(v4, Sse.Shuffle(v4, v4, 0x4E));
        v4 = Vector128.Max(v4, Sse.Shuffle(v4, v4, 0xB1));
        return v4.ToScalar();
    }

    private static void EnsureAllFinite(ReadOnlySpan<float> values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new ArgumentException($"Input contains non-finite value at index {i}.", paramName);
        }
    }
}
