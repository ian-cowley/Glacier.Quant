namespace Glacier.Quant.Calibration;

using System;
using System.Buffers;

/// <summary>
/// Importance Matrix (imatrix) PTQ calibrator.
/// Weights quantization scale optimization by activation second moments:
/// I_j = sum_k X_{k, j}^2, minimizing sum_j I_j * (w_j - w_hat_j)^2.
/// </summary>
public static class ImatrixCalibrator
{
    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset)
        => Calibrate(weights, rows, cols, dataset, targetBits: 8);

    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset,
        int targetBits)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);

        long totalWeights = (long)rows * cols;
        if (weights.Length != totalWeights)
            throw new ArgumentException($"Weights length ({weights.Length}) does not match rows * cols ({totalWeights}).", nameof(weights));
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).", nameof(dataset));
        if (targetBits is < 2 or > 8)
            throw new ArgumentOutOfRangeException(nameof(targetBits), targetBits, "Target bits must be in [2, 8].");

        EnsureAllFinite(weights, nameof(weights));
        EnsureAllFinite(dataset.Activations.AsSpan(0, checked(dataset.NumTokens * cols)), nameof(dataset));

        float maxQ = (1 << (targetBits - 1)) - 1;
        float minQ = -(1 << (targetBits - 1));

        // 1. Compute second moment diagonal vector in double: I_j = sum_k X_{k, j}^2
        float[] imatrix = new float[cols];
        double[] imatrixAcc = ArrayPool<double>.Shared.Rent(cols);
        const int blockSize = 32;
        float[] baselineQuant = ArrayPool<float>.Shared.Rent(weights.Length);
        try
        {
            Array.Clear(imatrixAcc, 0, cols);
            for (int k = 0; k < dataset.NumTokens; k++)
            {
                ReadOnlySpan<float> row = dataset.GetTokenVector(k);
                for (int j = 0; j < cols; j++)
                {
                    double x = row[j];
                    imatrixAcc[j] += x * x;
                }
            }

            // Scale-invariant normalization so mean importance is 1.0 across any finite activation scale
            double sumI = 0.0;
            for (int j = 0; j < cols; j++)
                sumI += imatrixAcc[j];

            if (sumI <= 0.0 || !double.IsFinite(sumI))
            {
                Array.Fill(imatrix, 1e-4f);
            }
            else
            {
                double meanI = sumI / cols;
                for (int j = 0; j < cols; j++)
                    imatrix[j] = (float)Math.Clamp(imatrixAcc[j] / meanI, 1e-4, float.MaxValue);
            }

            // 2. Compute baseline unweighted RTN using pooled scratch
            QuantizeBlockRtnInto(weights, baselineQuant.AsSpan(0, weights.Length), rows, cols, blockSize, minQ, maxQ);
            double baselineWeightedMse = ComputeWeightedMse(weights, baselineQuant.AsSpan(0, weights.Length), rows, cols, imatrix);

            // 3. Optimize scale per block using importance weights
            float[] calibrated = (float[])weights.Clone();

            for (int r = 0; r < rows; r++)
            {
                int rOffset = r * cols;
                for (int b = 0; b < cols; b += blockSize)
                {
                    int curBlock = Math.Min(blockSize, cols - b);

                    float amax = 0.0f;
                    for (int j = 0; j < curBlock; j++)
                    {
                        float a = MathF.Abs(weights[rOffset + b + j]);
                        if (a > amax) amax = a;
                    }

                    if (amax <= 1e-12f)
                    {
                        for (int j = 0; j < curBlock; j++)
                            calibrated[rOffset + b + j] = 0.0f;
                        continue;
                    }

                    float baseScale = amax / maxQ;
                    float bestScale = baseScale;

                    // Compute loss at baseScale
                    double bestBlockLoss = 0.0;
                    float invBase = 1.0f / baseScale;
                    for (int j = 0; j < curBlock; j++)
                    {
                        int c = b + j;
                        float origVal = weights[rOffset + c];
                        float q = MathF.Round(origVal * invBase);
                        q = Math.Clamp(q, minQ, maxQ);
                        double diff = (double)origVal - (double)(q * baseScale);
                        bestBlockLoss += (double)imatrix[c] * diff * diff;
                    }

                    // Try 15 scale candidates around base scale [0.7x .. 1.3x]
                    for (int sIdx = 0; sIdx < 15; sIdx++)
                    {
                        float candScale = baseScale * (0.7f + (sIdx * 0.0428f));
                        float invCandScale = 1.0f / candScale;
                        double candLoss = 0.0;

                        for (int j = 0; j < curBlock; j++)
                        {
                            int c = b + j;
                            float origVal = weights[rOffset + c];
                            float q = MathF.Round(origVal * invCandScale);
                            q = Math.Clamp(q, minQ, maxQ);
                            float dequant = q * candScale;

                            double diff = (double)origVal - (double)dequant;
                            candLoss += (double)imatrix[c] * diff * diff;
                        }

                        if (candLoss < bestBlockLoss)
                        {
                            bestBlockLoss = candLoss;
                            bestScale = candScale;
                        }
                    }

                    // Apply best scale
                    float invBest = 1.0f / bestScale;
                    for (int j = 0; j < curBlock; j++)
                    {
                        int c = b + j;
                        float q = MathF.Round(weights[rOffset + c] * invBest);
                        q = Math.Clamp(q, minQ, maxQ);
                        calibrated[rOffset + c] = q * bestScale;
                    }
                }
            }

            double calibratedWeightedMse = ComputeWeightedMse(weights, calibrated, rows, cols, imatrix);

            return new PtqCalibrationResult
            {
                Method = $"imatrix ({targetBits}-bit Importance-Weighted Least Squares)",
                CalibratedWeights = calibrated,
                OptimalScales = imatrix,
                BaselineRtnMse = baselineWeightedMse,
                CalibratedMse = calibratedWeightedMse
            };
        }
        finally
        {
            ArrayPool<double>.Shared.Return(imatrixAcc);
            ArrayPool<float>.Shared.Return(baselineQuant);
        }
    }

    private static void QuantizeBlockRtnInto(
        ReadOnlySpan<float> weights,
        Span<float> result,
        int rows,
        int cols,
        int blockSize,
        float minQ,
        float maxQ)
    {
        for (int r = 0; r < rows; r++)
        {
            int rOffset = r * cols;
            for (int b = 0; b < cols; b += blockSize)
            {
                int curBlock = Math.Min(blockSize, cols - b);
                float amax = 0.0f;
                for (int j = 0; j < curBlock; j++)
                {
                    float a = MathF.Abs(weights[rOffset + b + j]);
                    if (a > amax) amax = a;
                }

                if (amax <= 1e-12f)
                {
                    result.Slice(rOffset + b, curBlock).Clear();
                    continue;
                }

                float scale = amax / maxQ;
                float invScale = maxQ / amax;
                for (int j = 0; j < curBlock; j++)
                {
                    float q = MathF.Round(weights[rOffset + b + j] * invScale);
                    q = Math.Clamp(q, minQ, maxQ);
                    result[rOffset + b + j] = q * scale;
                }
            }
        }
    }

    private static double ComputeWeightedMse(
        ReadOnlySpan<float> orig,
        ReadOnlySpan<float> quant,
        int rows,
        int cols,
        ReadOnlySpan<float> imatrix)
    {
        double sum = 0.0;
        for (int r = 0; r < rows; r++)
        {
            int rOffset = r * cols;
            for (int c = 0; c < cols; c++)
            {
                double diff = (double)orig[rOffset + c] - (double)quant[rOffset + c];
                sum += (double)imatrix[c] * diff * diff;
            }
        }
        return sum / ((double)rows * cols);
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
