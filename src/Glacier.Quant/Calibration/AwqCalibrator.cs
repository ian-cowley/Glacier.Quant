namespace Glacier.Quant.Calibration;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Activation-aware Weight Quantization (AWQ) calibrator.
/// Protects salient weights by calculating per-channel activation magnitudes s_X
/// and performing a grid search over alpha in [0, 1] to minimize output reconstruction MSE:
/// || W X - dequant(quant(W * s)) * s^-1 X ||_F^2.
/// </summary>
public static class AwqCalibrator
{
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
        if (weights.Length != rows * cols)
            throw new ArgumentException("Weights length does not match rows * cols.");
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).");

        // 1. Compute per-channel activation magnitude: s_X(j) = 1/N * sum_k |X(k, j)|
        float[] sx = new float[cols];
        for (int k = 0; k < dataset.NumTokens; k++)
        {
            var row = dataset.GetTokenVector(k);
            for (int j = 0; j < cols; j++)
            {
                sx[j] += MathF.Abs(row[j]);
            }
        }
        float invTokens = 1.0f / Math.Max(1, dataset.NumTokens);
        for (int j = 0; j < cols; j++)
        {
            sx[j] = Math.Max(1e-6f, sx[j] * invTokens);
        }

        // 2. Compute baseline Round-To-Nearest (RTN) MSE
        float[] uncalibratedQuant = QuantizeRtn(weights, rows, cols, targetBits);
        double baselineMse = ComputeOutputMse(weights, uncalibratedQuant, rows, cols, dataset);

        // 3. Grid search over alpha in [0, 1]
        double bestMse = baselineMse;
        float bestAlpha = 0.0f;
        float[] bestWeights = (float[])uncalibratedQuant.Clone();
        float[] bestScales = new float[cols];
        Array.Fill(bestScales, 1.0f);

        float[] candidateScales = new float[cols];
        float[] scaledWeights = new float[weights.Length];

        for (int step = 1; step <= gridSteps; step++)
        {
            float alpha = (float)step / gridSteps;

            // s(j) = s_X(j)^alpha
            float sumScales = 0.0f;
            for (int j = 0; j < cols; j++)
            {
                candidateScales[j] = MathF.Pow(sx[j], alpha);
                sumScales += candidateScales[j];
            }

            // Normalize so mean scale is 1.0
            float meanScale = sumScales / cols;
            float invMean = 1.0f / Math.Max(1e-12f, meanScale);
            for (int j = 0; j < cols; j++)
            {
                candidateScales[j] *= invMean;
            }

            // Scale weights W'(i, j) = W(i, j) * s(j)
            for (int i = 0; i < rows; i++)
            {
                int rOffset = i * cols;
                for (int j = 0; j < cols; j++)
                {
                    scaledWeights[rOffset + j] = weights[rOffset + j] * candidateScales[j];
                }
            }

            // Quantize scaled weights
            float[] quantized = QuantizeRtn(scaledWeights, rows, cols, targetBits);

            // Unscale dequantized weights: W_hat(i, j) = W'_hat(i, j) / s(j)
            for (int i = 0; i < rows; i++)
            {
                int rOffset = i * cols;
                for (int j = 0; j < cols; j++)
                {
                    quantized[rOffset + j] /= candidateScales[j];
                }
            }

            double candidateMse = ComputeOutputMse(weights, quantized, rows, cols, dataset);
            if (candidateMse < bestMse)
            {
                bestMse = candidateMse;
                bestAlpha = alpha;
                Array.Copy(quantized, bestWeights, weights.Length);
                Array.Copy(candidateScales, bestScales, cols);
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

    private static float[] QuantizeRtn(float[] weights, int rows, int cols, int targetBits = 4)
    {
        float[] result = new float[weights.Length];
        const int blockSize = 32;
        float maxQ = (1 << (targetBits - 1)) - 1;
        float minQ = -(1 << (targetBits - 1));

        for (int i = 0; i < rows; i++)
        {
            int rOffset = i * cols;
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
                    for (int j = 0; j < curBlock; j++) result[rOffset + b + j] = 0.0f;
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
        return result;
    }

    private static double ComputeOutputMse(
        float[] originalW,
        float[] quantW,
        int rows,
        int cols,
        CalibrationDataset dataset)
    {
        double totalError = 0.0;
        int tokens = Math.Min(dataset.NumTokens, 16); // Evaluate on subset for high performance

        float[] originalOut = new float[rows];
        float[] quantOut = new float[rows];

        for (int t = 0; t < tokens; t++)
        {
            var x = dataset.GetTokenVector(t);

            // Matmul: out = W * x
            for (int r = 0; r < rows; r++)
            {
                int rOffset = r * cols;
                float sumOrig = 0.0f;
                float sumQuant = 0.0f;

                for (int c = 0; c < cols; c++)
                {
                    float val = x[c];
                    sumOrig += originalW[rOffset + c] * val;
                    sumQuant += quantW[rOffset + c] * val;
                }

                originalOut[r] = sumOrig;
                quantOut[r] = sumQuant;

                float diff = sumOrig - sumQuant;
                totalError += diff * diff;
            }
        }

        return totalError / (tokens * rows);
    }
}
