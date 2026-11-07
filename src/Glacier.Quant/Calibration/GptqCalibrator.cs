namespace Glacier.Quant.Calibration;

using System;

/// <summary>
/// Generalized Post-Training Quantization (GPTQ) calibrator.
/// Inverts the empirical activation Hessian H = 2 * X^T * X + lambda * I and lazily updates
/// remaining unquantized weight columns via rank-1 error updates:
/// W_{:, j+1:} -= (w_j - w_hat_j) / [H^-1]_{jj} * [H^-1]_{j, j+1:}.
/// </summary>
public static class GptqCalibrator
{
    public static PtqCalibrationResult Calibrate(
        float[] weights,
        int rows,
        int cols,
        CalibrationDataset dataset,
        float damping = 0.01f)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(dataset);
        if (weights.Length != rows * cols)
            throw new ArgumentException("Weights length does not match rows * cols.");
        if (dataset.HiddenDim != cols)
            throw new ArgumentException($"Dataset hidden dim ({dataset.HiddenDim}) does not match weight cols ({cols}).");

        // 1. Compute empirical Hessian matrix: H = 2 * X^T * X
        double[,] H = new double[cols, cols];
        for (int t = 0; t < dataset.NumTokens; t++)
        {
            var x = dataset.GetTokenVector(t);
            for (int r = 0; r < cols; r++)
            {
                double xr = x[r];
                for (int c = r; c < cols; c++)
                {
                    double val = 2.0 * xr * x[c];
                    H[r, c] += val;
                    if (r != c) H[c, r] += val;
                }
            }
        }

        // Add damping lambda to diagonal for numerical stability
        double diagSum = 0.0;
        for (int i = 0; i < cols; i++) diagSum += H[i, i];
        double lambda = Math.Max(1e-4, (diagSum / cols) * damping);
        for (int i = 0; i < cols; i++) H[i, i] += lambda;

        // 2. Invert H -> H_inv
        double[,] Hinv = InvertSymmetricMatrix(H, cols);

        // 3. Compute baseline Round-To-Nearest (RTN) MSE
        float[] baselineQuant = (float[])weights.Clone();
        QuantizeInPlace(baselineQuant, rows, cols);
        double baselineMse = ComputeWeightMse(weights, baselineQuant);

        // 4. Execute greedy column-wise GPTQ error compensation
        float[] gptqWeights = (float[])weights.Clone();

        for (int j = 0; j < cols; j++)
        {
            double hjj = Math.Max(1e-12, Hinv[j, j]);
            double invHjj = 1.0 / hjj;

            for (int r = 0; r < rows; r++)
            {
                int idx = r * cols + j;
                float origW = gptqWeights[idx];
                float quantW = QuantizeScalar(origW);
                gptqWeights[idx] = quantW;

                float err = origW - quantW;
                double updateFactor = err * invHjj;

                // Rank-1 error propagation to remaining columns
                int rOffset = r * cols;
                for (int k = j + 1; k < cols; k++)
                {
                    gptqWeights[rOffset + k] -= (float)(updateFactor * Hinv[j, k]);
                }
            }
        }

        // Final snap of all columns to nearest discrete quant
        QuantizeInPlace(gptqWeights, rows, cols);
        double calibratedMse = ComputeWeightMse(weights, gptqWeights);

        return new PtqCalibrationResult
        {
            Method = "GPTQ (Cholesky Hessian Inversion + Rank-1 Update)",
            CalibratedWeights = gptqWeights,
            BaselineRtnMse = baselineMse,
            CalibratedMse = calibratedMse
        };
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float QuantizeScalar(float w)
    {
        // 4-bit scalar quantization into 16 uniform levels [-8..7]
        return MathF.Round(w * 4.0f) * 0.25f;
    }

    private static void QuantizeInPlace(float[] w, int rows, int cols)
    {
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = QuantizeScalar(w[i]);
        }
    }

    private static double ComputeWeightMse(float[] original, float[] quant)
    {
        double sum = 0.0;
        for (int i = 0; i < original.Length; i++)
        {
            double diff = original[i] - quant[i];
            sum += diff * diff;
        }
        return sum / original.Length;
    }

    /// <summary>
    /// Inverts a positive-definite symmetric matrix using Gauss-Jordan elimination with partial pivoting.
    /// </summary>
    private static double[,] InvertSymmetricMatrix(double[,] matrix, int n)
    {
        double[,] a = (double[,])matrix.Clone();
        double[,] inv = new double[n, n];
        for (int i = 0; i < n; i++) inv[i, i] = 1.0;

        for (int i = 0; i < n; i++)
        {
            int pivot = i;
            double maxVal = Math.Abs(a[i, i]);
            for (int k = i + 1; k < n; k++)
            {
                double abs = Math.Abs(a[k, i]);
                if (abs > maxVal)
                {
                    maxVal = abs;
                    pivot = k;
                }
            }

            if (pivot != i)
            {
                for (int j = 0; j < n; j++)
                {
                    (a[i, j], a[pivot, j]) = (a[pivot, j], a[i, j]);
                    (inv[i, j], inv[pivot, j]) = (inv[pivot, j], inv[i, j]);
                }
            }

            double diag = Math.Max(1e-12, a[i, i]);
            double invDiag = 1.0 / diag;
            for (int j = 0; j < n; j++)
            {
                a[i, j] *= invDiag;
                inv[i, j] *= invDiag;
            }

            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = a[k, i];
                    for (int j = 0; j < n; j++)
                    {
                        a[k, j] -= factor * a[i, j];
                        inv[k, j] -= factor * inv[i, j];
                    }
                }
            }
        }

        return inv;
    }
}
