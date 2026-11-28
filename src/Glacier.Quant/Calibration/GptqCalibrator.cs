namespace Glacier.Quant.Calibration;

using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

/// <summary>
/// Generalized Post-Training Quantization (GPTQ) calibrator.
/// Computes the empirical Hessian H = 2 * X^T * X + lambda * I, factors H = L * L^T via
/// Cholesky decomposition (with adaptive damping retry), inverts L -> L^-1 -> H^-1 = (L^-1)^T * L^-1,
/// extracts the upper-triangular Cholesky Schur-complement factor U (U^T * U = H^-1),
/// and performs blockwise quantized error diffusion with lazy trailing-block updates.
/// </summary>
public static class GptqCalibrator
{
    private const int MaxCholeskyRetries = 6;
    private const double MinCholeskyPivot = 1e-15;

    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset,
        float damping = 0.01f)
        => Calibrate(weights, rows, cols, dataset, damping, targetBits: 4, blockSize: 128, groupSize: 32, symmetric: true);

    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset,
        float damping,
        int targetBits,
        int blockSize = 128,
        int groupSize = 32,
        bool symmetric = true)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);

        long totalWeights = (long)rows * cols;
        if (weights.Length != totalWeights)
            throw new ArgumentException($"Weights length ({weights.Length}) does not match rows * cols ({totalWeights}).", nameof(weights));
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).", nameof(dataset));
        if (damping < 0.0f || !float.IsFinite(damping))
            throw new ArgumentOutOfRangeException(nameof(damping), damping, "Damping must be finite and non-negative.");
        if (targetBits is < 2 or > 8)
            throw new ArgumentOutOfRangeException(nameof(targetBits), targetBits, "Target bits must be in [2, 8].");

        EnsureAllFinite(weights, nameof(weights));
        EnsureAllFinite(dataset.Activations.AsSpan(0, checked(dataset.NumTokens * cols)), nameof(dataset));

        int effectiveBlockSize = Math.Min(blockSize, cols);
        int effectiveGroupSize = groupSize <= 0 ? cols : Math.Min(groupSize, cols);
        int numGroups = (cols + effectiveGroupSize - 1) / effectiveGroupSize;

        int hLen = checked(cols * cols);
        int groupParamLen = checked(rows * numGroups);
        int errBlockLen = checked(rows * effectiveBlockSize);

        double[] hOrig = ArrayPool<double>.Shared.Rent(hLen);
        double[] hWork = ArrayPool<double>.Shared.Rent(hLen);
        double[] lInv = ArrayPool<double>.Shared.Rent(hLen);
        double[] xDouble = ArrayPool<double>.Shared.Rent(cols);
        float[] uFloat = ArrayPool<float>.Shared.Rent(hLen);
        float[] scales = ArrayPool<float>.Shared.Rent(groupParamLen);
        float[] offsets = ArrayPool<float>.Shared.Rent(groupParamLen);
        float[] baselineQuant = ArrayPool<float>.Shared.Rent(weights.Length);
        float[] errBlock = ArrayPool<float>.Shared.Rent(errBlockLen);

        try
        {
            Array.Clear(hOrig, 0, hLen);

            // 1. Accumulate empirical Hessian upper triangle: H = 2 * X^T * X
            for (int t = 0; t < dataset.NumTokens; t++)
            {
                ReadOnlySpan<float> x = dataset.GetTokenVector(t);
                for (int c = 0; c < cols; c++)
                    xDouble[c] = x[c];

                for (int r = 0; r < cols; r++)
                {
                    double twoXr = 2.0 * xDouble[r];
                    if (twoXr == 0.0)
                        continue;

                    int len = cols - r;
                    Span<double> hRow = hOrig.AsSpan(r * cols + r, len);
                    TensorPrimitives.MultiplyAdd(xDouble.AsSpan(r, len), twoXr, hRow, hRow);
                }
            }

            // Mirror upper triangle to lower triangle
            for (int r = 0; r < cols; r++)
            {
                int rOff = r * cols;
                for (int c = r + 1; c < cols; c++)
                {
                    hOrig[c * cols + r] = hOrig[rOff + c];
                }
            }

            // 2. Normalize Hessian by max diagonal so dampening, Cholesky, and H^-1 stay scale-invariant
            double maxDiag = 0.0;
            for (int i = 0; i < cols; i++)
            {
                double d = hOrig[i * cols + i];
                if (d > maxDiag)
                    maxDiag = d;
            }

            double hScale = maxDiag > 0.0 && double.IsFinite(maxDiag) ? maxDiag : 1.0;
            if (hScale != 1.0)
            {
                double invHScale = 1.0 / hScale;
                TensorPrimitives.Multiply(hOrig.AsSpan(0, hLen), invHScale, hOrig.AsSpan(0, hLen));
            }

            double diagSumScaled = 0.0;
            for (int i = 0; i < cols; i++)
                diagSumScaled += hOrig[i * cols + i];

            double diagMeanScaled = diagSumScaled / cols;
            double baseLambda = diagMeanScaled > 0.0 && double.IsFinite(diagMeanScaled)
                ? Math.Max(1e-12, diagMeanScaled * damping)
                : 1.0;
            bool factored = false;
            double lambda = baseLambda;

            for (int attempt = 0; attempt <= MaxCholeskyRetries; attempt++)
            {
                Array.Copy(hOrig, hWork, hLen);
                for (int i = 0; i < cols; i++)
                    hWork[i * cols + i] += lambda;

                if (TryCholeskyLowerInPlace(hWork, cols))
                {
                    factored = true;
                    break;
                }

                lambda = Math.Max(lambda * 10.0, 1e-6);
            }

            if (!factored)
                throw new InvalidOperationException("GPTQ Hessian Cholesky factorization failed even after adaptive damping retries.");

            // Invert lower-triangular L -> L^-1
            InvertLowerTriangular(hWork, lInv, cols);

            // Compute H_scaled^-1 = (L^-1)^T * (L^-1) into hOrig
            ComputeLtTimesL(lInv, hOrig, cols);

            // Compute upper-triangular Cholesky factor U of H_scaled^-1 (U^T * U = H_scaled^-1).
            // Save a clean copy in hWork because TryCholeskyLowerInPlace mutates in-place on failure.
            Array.Copy(hOrig, hWork, hLen);
            if (!TryCholeskyLowerInPlace(hOrig, cols))
            {
                double hInvMaxDiag = 0.0;
                for (int i = 0; i < cols; i++)
                {
                    double d = hWork[i * cols + i];
                    if (d > hInvMaxDiag && double.IsFinite(d))
                        hInvMaxDiag = d;
                }

                double jitter = Math.Max(1e-10, hInvMaxDiag * 1e-10);
                bool uFactored = false;
                for (int retry = 0; retry < 4; retry++)
                {
                    Array.Copy(hWork, hOrig, hLen);
                    for (int i = 0; i < cols; i++)
                        hOrig[i * cols + i] = Math.Max(jitter, hOrig[i * cols + i]) + jitter;

                    if (TryCholeskyLowerInPlace(hOrig, cols))
                    {
                        uFactored = true;
                        break;
                    }

                    jitter *= 100.0;
                }

                if (!uFactored)
                    throw new InvalidOperationException("Failed to compute Schur-complement Cholesky factor U of H^-1.");
            }

            Array.Clear(lInv, 0, hLen);
            Array.Clear(uFloat, 0, hLen);
            for (int j = 0; j < cols; j++)
            {
                int uRow = j * cols;
                for (int k = j; k < cols; k++)
                {
                    double ujk = hOrig[k * cols + j];
                    lInv[uRow + k] = ujk;
                    uFloat[uRow + k] = (float)ujk;
                }
            }

            // 3. Precompute per-row, per-group quantization parameters & baseline RTN MSE
            int qMin;
            int qMax;
            if (symmetric)
            {
                qMax = (1 << (targetBits - 1)) - 1;
                qMin = -(1 << (targetBits - 1));
            }
            else
            {
                qMin = 0;
                qMax = (1 << targetBits) - 1;
            }

            for (int r = 0; r < rows; r++)
            {
                int rOff = r * cols;
                int gBase = r * numGroups;
                for (int g = 0; g < numGroups; g++)
                {
                    int cStart = g * effectiveGroupSize;
                    int cEnd = Math.Min(cols, cStart + effectiveGroupSize);

                    if (symmetric)
                    {
                        float amax = 0.0f;
                        for (int c = cStart; c < cEnd; c++)
                        {
                            float abs = MathF.Abs(weights[rOff + c]);
                            if (abs > amax) amax = abs;
                        }

                        scales[gBase + g] = amax > 1e-12f ? amax / qMax : 0.0f;
                        offsets[gBase + g] = 0.0f;
                    }
                    else
                    {
                        float wMin = weights[rOff + cStart];
                        float wMax = wMin;
                        for (int c = cStart + 1; c < cEnd; c++)
                        {
                            float w = weights[rOff + c];
                            if (w < wMin) wMin = w;
                            if (w > wMax) wMax = w;
                        }

                        double span = (double)wMax - wMin;
                        if (span <= 1e-12)
                        {
                            scales[gBase + g] = 0.0f;
                            offsets[gBase + g] = wMin;
                        }
                        else
                        {
                            scales[gBase + g] = (float)(span / qMax);
                            offsets[gBase + g] = wMin;
                        }
                    }
                }
            }

            for (int r = 0; r < rows; r++)
            {
                int rOff = r * cols;
                int gBase = r * numGroups;
                for (int c = 0; c < cols; c++)
                {
                    int g = c / effectiveGroupSize;
                    baselineQuant[rOff + c] = QuantizeElement(weights[rOff + c], scales[gBase + g], offsets[gBase + g], qMin, qMax, symmetric);
                }
            }

            double baselineMse = ComputeOutputMse(weights, baselineQuant.AsSpan(0, weights.Length), rows, cols, dataset);

            // 4. Blockwise GPTQ quantization with lazy trailing-block updates
            float[] gptqWeights = (float[])weights.Clone();

            for (int i1 = 0; i1 < cols; i1 += effectiveBlockSize)
            {
                int i2 = Math.Min(cols, i1 + effectiveBlockSize);
                int count = i2 - i1;

                for (int j = i1; j < i2; j++)
                {
                    double ujj = lInv[j * cols + j];
                    double invUjj = ujj > MinCholeskyPivot ? 1.0 / ujj : 0.0;
                    int g = j / effectiveGroupSize;
                    int inBlockCol = j - i1;
                    int innerRem = i2 - j - 1;
                    ReadOnlySpan<float> uRowInner = innerRem > 0
                        ? uFloat.AsSpan(j * cols + j + 1, innerRem)
                        : ReadOnlySpan<float>.Empty;

                    for (int r = 0; r < rows; r++)
                    {
                        int rOff = r * cols;
                        int gIdx = r * numGroups + g;
                        float origW = gptqWeights[rOff + j];
                        float quantW = QuantizeElement(origW, scales[gIdx], offsets[gIdx], qMin, qMax, symmetric);
                        gptqWeights[rOff + j] = quantW;

                        float scaledErr = (float)((origW - quantW) * invUjj);
                        errBlock[r * count + inBlockCol] = scaledErr;

                        if (innerRem > 0 && scaledErr != 0.0f)
                        {
                            float negErr = -scaledErr;
                            if (innerRem >= 8)
                            {
                                Span<float> target = gptqWeights.AsSpan(rOff + j + 1, innerRem);
                                TensorPrimitives.MultiplyAdd(uRowInner, negErr, target, target);
                            }
                            else
                            {
                                int baseK = rOff + j + 1;
                                for (int k = 0; k < innerRem; k++)
                                    gptqWeights[baseK + k] += negErr * uRowInner[k];
                            }
                        }
                    }
                }

                // Lazy trailing-block update across columns i2 .. cols - 1
                int trailingRem = cols - i2;
                if (trailingRem > 0)
                {
                    if ((long)rows * count * trailingRem >= 262_144L && rows >= 8)
                    {
                        Parallel.For(0, rows, r =>
                        {
                            int rOff = r * cols + i2;
                            int errOff = r * count;
                            Span<float> target = gptqWeights.AsSpan(rOff, trailingRem);
                            for (int m = 0; m < count; m++)
                            {
                                float negErr = -errBlock[errOff + m];
                                if (negErr == 0.0f)
                                    continue;
                                ReadOnlySpan<float> uRow = uFloat.AsSpan((i1 + m) * cols + i2, trailingRem);
                                TensorPrimitives.MultiplyAdd(uRow, negErr, target, target);
                            }
                        });
                    }
                    else
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            int rOff = r * cols + i2;
                            int errOff = r * count;
                            Span<float> target = gptqWeights.AsSpan(rOff, trailingRem);
                            for (int m = 0; m < count; m++)
                            {
                                float negErr = -errBlock[errOff + m];
                                if (negErr == 0.0f)
                                    continue;
                                ReadOnlySpan<float> uRow = uFloat.AsSpan((i1 + m) * cols + i2, trailingRem);
                                TensorPrimitives.MultiplyAdd(uRow, negErr, target, target);
                            }
                        }
                    }
                }
            }

            double calibratedMse = ComputeOutputMse(weights, gptqWeights, rows, cols, dataset);

            return new PtqCalibrationResult
            {
                Method = $"GPTQ (Cholesky Blockwise {targetBits}-bit)",
                CalibratedWeights = gptqWeights,
                BaselineRtnMse = baselineMse,
                CalibratedMse = calibratedMse
            };
        }
        finally
        {
            ArrayPool<double>.Shared.Return(hOrig);
            ArrayPool<double>.Shared.Return(hWork);
            ArrayPool<double>.Shared.Return(lInv);
            ArrayPool<double>.Shared.Return(xDouble);
            ArrayPool<float>.Shared.Return(uFloat);
            ArrayPool<float>.Shared.Return(scales);
            ArrayPool<float>.Shared.Return(offsets);
            ArrayPool<float>.Shared.Return(baselineQuant);
            ArrayPool<float>.Shared.Return(errBlock);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float QuantizeElement(float w, float scale, float offset, int qMin, int qMax, bool symmetric)
    {
        if (scale <= 0.0f)
            return symmetric ? 0.0f : offset;

        if (symmetric)
        {
            float q = Math.Clamp(MathF.Round(w / scale), qMin, qMax);
            return q * scale;
        }
        else
        {
            float q = Math.Clamp(MathF.Round((w - offset) / scale), qMin, qMax);
            return offset + q * scale;
        }
    }

    private static bool TryCholeskyLowerInPlace(double[] matrix, int n)
    {
        for (int i = 0; i < n; i++)
        {
            int iRow = i * n;
            for (int j = 0; j < i; j++)
            {
                int jRow = j * n;
                double dot = j >= 8
                    ? TensorPrimitives.Dot(matrix.AsSpan(iRow, j), matrix.AsSpan(jRow, j))
                    : DotScalar(matrix, iRow, jRow, j);
                double sum = matrix[iRow + j] - dot;
                matrix[iRow + j] = sum / matrix[jRow + j];
            }

            double diagSumSq = i >= 8
                ? TensorPrimitives.SumOfSquares(matrix.AsSpan(iRow, i))
                : SumOfSquaresScalar(matrix, iRow, i);
            double diag = matrix[iRow + i] - diagSumSq;
            if (diag <= MinCholeskyPivot || !double.IsFinite(diag))
                return false;

            matrix[iRow + i] = Math.Sqrt(diag);
        }

        // Zero strict upper triangle
        for (int i = 0; i < n; i++)
        {
            int iRow = i * n;
            for (int j = i + 1; j < n; j++)
                matrix[iRow + j] = 0.0;
        }

        return true;
    }

    private static void InvertLowerTriangular(double[] l, double[] lInv, int n)
    {
        Array.Clear(lInv, 0, n * n);

        for (int i = 0; i < n; i++)
        {
            int iRow = i * n;
            double invLii = 1.0 / l[iRow + i];
            lInv[iRow + i] = invLii;

            for (int j = 0; j < i; j++)
            {
                double sum = 0.0;
                for (int k = j; k < i; k++)
                    sum += l[iRow + k] * lInv[k * n + j];
                lInv[iRow + j] = -sum * invLii;
            }
        }
    }

    private static void ComputeLtTimesL(double[] lInv, double[] hInv, int n)
    {
        for (int r = 0; r < n; r++)
        {
            int rOff = r * n;
            for (int c = r; c < n; c++)
            {
                double sum = 0.0;
                for (int k = c; k < n; k++)
                    sum += lInv[k * n + r] * lInv[k * n + c];
                hInv[rOff + c] = sum;
                if (r != c)
                    hInv[c * n + r] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double DotScalar(double[] m, int rowA, int rowB, int len)
    {
        double sum = 0.0;
        for (int k = 0; k < len; k++)
            sum += m[rowA + k] * m[rowB + k];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SumOfSquaresScalar(double[] m, int row, int len)
    {
        double sum = 0.0;
        for (int k = 0; k < len; k++)
        {
            double v = m[row + k];
            sum += v * v;
        }
        return sum;
    }

    private static double ComputeOutputMse(
        ReadOnlySpan<float> originalW,
        ReadOnlySpan<float> quantW,
        int rows,
        int cols,
        CalibrationDataset dataset)
    {
        int tokens = dataset.NumTokens;
        double sumSq = 0.0;
        for (int t = 0; t < tokens; t++)
        {
            ReadOnlySpan<float> x = dataset.GetTokenVector(t);
            for (int r = 0; r < rows; r++)
            {
                int rOff = r * cols;
                double dotRef = 0.0;
                double dotQuant = 0.0;
                for (int c = 0; c < cols; c++)
                {
                    double xc = x[c];
                    dotRef += (double)originalW[rOff + c] * xc;
                    dotQuant += (double)quantW[rOff + c] * xc;
                }
                double diff = dotRef - dotQuant;
                sumSq += diff * diff;
            }
        }
        return sumSq / ((double)tokens * rows);
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
