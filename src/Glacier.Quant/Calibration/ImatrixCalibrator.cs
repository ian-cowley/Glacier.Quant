namespace Glacier.Quant.Calibration;

using System;

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
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(dataset);
        if (weights.Length != rows * cols)
            throw new ArgumentException("Weights length does not match rows * cols.");
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).");

        // 1. Compute second moment diagonal vector: I_j = sum_k X_{k, j}^2
        float[] imatrix = new float[cols];
        for (int k = 0; k < dataset.NumTokens; k++)
        {
            var row = dataset.GetTokenVector(k);
            for (int j = 0; j < cols; j++)
            {
                imatrix[j] += row[j] * row[j];
            }
        }

        // Normalize so mean importance is 1.0
        float sumI = 0.0f;
        for (int j = 0; j < cols; j++) sumI += imatrix[j];
        float invMeanI = cols / Math.Max(1e-12f, sumI);
        for (int j = 0; j < cols; j++)
        {
            imatrix[j] = Math.Max(1e-4f, imatrix[j] * invMeanI);
        }

        // 2. Compute baseline unweighted RTN
        const int blockSize = 32;
        float[] baselineQuant = QuantizeBlockRtn(weights, rows, cols, blockSize);
        double baselineWeightedMse = ComputeWeightedMse(weights, baselineQuant, rows, cols, imatrix);

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
                    for (int j = 0; j < curBlock; j++) calibrated[rOffset + b + j] = 0.0f;
                    continue;
                }

                float baseScale = amax / 127.0f;
                float bestScale = baseScale;
                
                // Compute loss at baseScale
                double bestBlockLoss = 0.0;
                float invBase = 1.0f / baseScale;
                for (int j = 0; j < curBlock; j++)
                {
                    int c = b + j;
                    float origVal = weights[rOffset + c];
                    float q = MathF.Round(origVal * invBase);
                    q = Math.Clamp(q, -128.0f, 127.0f);
                    float diff = origVal - (q * baseScale);
                    bestBlockLoss += imatrix[c] * diff * diff;
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
                        q = Math.Clamp(q, -128.0f, 127.0f);
                        float dequant = q * candScale;

                        float diff = origVal - dequant;
                        candLoss += imatrix[c] * diff * diff;
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
                    q = Math.Clamp(q, -128.0f, 127.0f);
                    calibrated[rOffset + c] = q * bestScale;
                }
            }
        }

        double calibratedWeightedMse = ComputeWeightedMse(weights, calibrated, rows, cols, imatrix);

        return new PtqCalibrationResult
        {
            Method = "imatrix (Importance-Weighted Least Squares)",
            CalibratedWeights = calibrated,
            OptimalScales = imatrix,
            BaselineRtnMse = baselineWeightedMse,
            CalibratedMse = calibratedWeightedMse
        };
    }

    private static float[] QuantizeBlockRtn(float[] weights, int rows, int cols, int blockSize)
    {
        float[] result = new float[weights.Length];
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
                if (amax <= 1e-12f) continue;
                float scale = amax / 127.0f;
                float invScale = 1.0f / scale;
                for (int j = 0; j < curBlock; j++)
                {
                    float q = MathF.Round(weights[rOffset + b + j] * invScale);
                    q = Math.Clamp(q, -128.0f, 127.0f);
                    result[rOffset + b + j] = q * scale;
                }
            }
        }
        return result;
    }

    private static double ComputeWeightedMse(float[] orig, float[] quant, int rows, int cols, float[] imatrix)
    {
        double sum = 0.0;
        for (int r = 0; r < rows; r++)
        {
            int rOffset = r * cols;
            for (int c = 0; c < cols; c++)
            {
                float diff = orig[rOffset + c] - quant[rOffset + c];
                sum += imatrix[c] * diff * diff;
            }
        }
        return sum / (rows * cols);
    }
}
