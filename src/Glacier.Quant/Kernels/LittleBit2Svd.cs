namespace Glacier.Quant.Kernels;

using System;
using System.Buffers;

/// <summary>
/// Pure-C# AOT-compatible double-precision linear algebra kernels for LittleBit-2:
/// twice-iterated MGS orthonormalization, Householder QR random orthogonal sampling,
/// one-sided Hestenes-Jacobi exact SVD, tall/wide randomized truncated SVD, and
/// Perron-Frobenius rank-1 non-negative magnitude SVD.
/// </summary>
internal static class LittleBit2Svd
{
    internal static void OrthonormalizeColumns(double[] matrixColMajor, int rows, int cols)
    {
        for (int j = 0; j < cols; j++)
        {
            int colJ = j * rows;

            // Two passes of Modified Gram-Schmidt for full machine-precision orthogonality.
            for (int pass = 0; pass < 2; pass++)
            {
                for (int i = 0; i < j; i++)
                {
                    int colI = i * rows;
                    double dot = Dot(matrixColMajor, colI, colJ, rows);
                    Axpy(-dot, matrixColMajor, colI, colJ, rows);
                }
            }

            double norm = Math.Sqrt(NormSq(matrixColMajor, colJ, rows));
            if (norm <= 1e-14)
            {
                // Rank-deficient completion using standard basis vectors.
                int bestBasis = -1;
                double bestNorm = -1.0;

                for (int s = 0; s < rows; s++)
                {
                    Array.Clear(matrixColMajor, colJ, rows);
                    matrixColMajor[colJ + s] = 1.0;

                    for (int pass = 0; pass < 2; pass++)
                    {
                        for (int i = 0; i < j; i++)
                        {
                            int colI = i * rows;
                            double dot = Dot(matrixColMajor, colI, colJ, rows);
                            Axpy(-dot, matrixColMajor, colI, colJ, rows);
                        }
                    }

                    double candidateNorm = Math.Sqrt(NormSq(matrixColMajor, colJ, rows));
                    if (candidateNorm > 1e-10)
                    {
                        Scale(matrixColMajor, colJ, rows, 1.0 / candidateNorm);
                        bestBasis = s;
                        break;
                    }

                    if (candidateNorm > bestNorm)
                    {
                        bestNorm = candidateNorm;
                        bestBasis = s;
                    }
                }

                if (bestBasis >= 0 && Math.Sqrt(NormSq(matrixColMajor, colJ, rows)) <= 1e-10 && bestNorm > 0.0)
                {
                    Array.Clear(matrixColMajor, colJ, rows);
                    matrixColMajor[colJ + bestBasis] = 1.0;
                    for (int pass = 0; pass < 2; pass++)
                    {
                        for (int i = 0; i < j; i++)
                        {
                            int colI = i * rows;
                            double dot = Dot(matrixColMajor, colI, colJ, rows);
                            Axpy(-dot, matrixColMajor, colI, colJ, rows);
                        }
                    }
                    double nrm = Math.Sqrt(NormSq(matrixColMajor, colJ, rows));
                    if (nrm > 0.0)
                        Scale(matrixColMajor, colJ, rows, 1.0 / nrm);
                }
            }
            else
            {
                Scale(matrixColMajor, colJ, rows, 1.0 / norm);
            }
        }
    }

    internal static void OrthonormalizeRowMajorColumns(double[] matrixRowMajor, int rows, int cols)
    {
        int total = checked(rows * cols);
        double[] colMajor = ArrayPool<double>.Shared.Rent(total);
        try
        {
            for (int i = 0; i < rows; i++)
            {
                int rowOff = i * cols;
                for (int j = 0; j < cols; j++)
                    colMajor[j * rows + i] = matrixRowMajor[rowOff + j];
            }

            OrthonormalizeColumns(colMajor, rows, cols);

            for (int i = 0; i < rows; i++)
            {
                int rowOff = i * cols;
                for (int j = 0; j < cols; j++)
                    matrixRowMajor[rowOff + j] = colMajor[j * rows + i];
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(colMajor);
        }
    }

    internal static double[] CreateRandomOrthogonal(int size, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        if (size == 1)
            return [1.0];

        int total = checked(size * size);
        double[] g = new double[total]; // column-major
        double[] q = new double[total]; // column-major

        var rng = new SplitMix64(seed);
        for (int idx = 0; idx < total; idx++)
            g[idx] = rng.NextStandardNormal();

        for (int j = 0; j < size; j++)
            q[j * size + j] = 1.0;

        double[] v = new double[size];
        for (int k = 0; k < size - 1; k++)
        {
            int len = size - k;
            int colK = k * size;
            double normSq = 0.0;
            for (int i = 0; i < len; i++)
            {
                double val = g[colK + k + i];
                normSq += val * val;
            }

            double norm = Math.Sqrt(normSq);
            if (norm <= 1e-30)
                continue;

            double gkk = g[colK + k];
            double alpha = gkk >= 0.0 ? -norm : norm;
            v[0] = gkk - alpha;
            for (int i = 1; i < len; i++)
                v[i] = g[colK + k + i];

            double vNormSq = normSq - gkk * gkk + v[0] * v[0];
            if (vNormSq <= 1e-30)
                continue;

            double beta = 2.0 / vNormSq;

            // Apply H = I - beta * v * v^T from the left to G[k..size-1, k..size-1].
            for (int j = k; j < size; j++)
            {
                int colJ = j * size + k;
                double dot = 0.0;
                for (int i = 0; i < len; i++)
                    dot += v[i] * g[colJ + i];
                double scale = beta * dot;
                for (int i = 0; i < len; i++)
                    g[colJ + i] -= scale * v[i];
            }

            // Accumulate Q <- Q * H from the right across all rows of Q[:, k..size-1].
            for (int row = 0; row < size; row++)
            {
                double dot = 0.0;
                for (int i = 0; i < len; i++)
                    dot += q[(k + i) * size + row] * v[i];
                double scale = beta * dot;
                for (int i = 0; i < len; i++)
                    q[(k + i) * size + row] -= scale * v[i];
            }
        }

        // Mezzadri Haar sign correction using signs of R's diagonal.
        bool needsFallbackOrthonorm = false;
        for (int k = 0; k < size; k++)
        {
            double diag = g[k * size + k];
            if (Math.Abs(diag) <= 1e-14)
                needsFallbackOrthonorm = true;
            if (diag < 0.0)
            {
                int colK = k * size;
                for (int row = 0; row < size; row++)
                    q[colK + row] = -q[colK + row];
            }
        }

        if (needsFallbackOrthonorm)
            OrthonormalizeColumns(q, size, size);

        // Convert column-major Q to row-major output.
        double[] rMat = new double[total];
        for (int i = 0; i < size; i++)
        {
            int rowOff = i * size;
            for (int j = 0; j < size; j++)
                rMat[rowOff + j] = q[j * size + i];
        }

        return rMat;
    }

    internal static void ExactSvd(
        ReadOnlySpan<double> matrixRowMajor,
        int m,
        int n,
        out double[] uRowMajor,
        out double[] singularValues,
        out double[] vRowMajor)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);

        if (m >= n)
        {
            ExactSvdTall(matrixRowMajor, m, n, out uRowMajor, out singularValues, out vRowMajor);
            return;
        }

        // Wide dispatch: A^T = U_t * Sigma * V_t^T => A = V_t * Sigma * U_t^T.
        double[] transposed = ArrayPool<double>.Shared.Rent(checked(m * n));
        try
        {
            for (int i = 0; i < m; i++)
            {
                int rowOff = i * n;
                for (int j = 0; j < n; j++)
                    transposed[j * m + i] = matrixRowMajor[rowOff + j];
            }

            ExactSvdTall(transposed.AsSpan(0, m * n), n, m, out double[] uT, out singularValues, out double[] vT);
            uRowMajor = vT;
            vRowMajor = uT;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(transposed);
        }
    }

    private static void ExactSvdTall(
        ReadOnlySpan<double> aRowMajor,
        int m,
        int n,
        out double[] uRowMajor,
        out double[] singularValues,
        out double[] vRowMajor)
    {
        int wLen = checked(m * n);
        int vLen = checked(n * n);
        double[] wCol = new double[wLen];
        double[] vCol = new double[vLen];
        double[] colNormSq = new double[n];

        for (int i = 0; i < m; i++)
        {
            int rowOff = i * n;
            for (int j = 0; j < n; j++)
                wCol[j * m + i] = aRowMajor[rowOff + j];
        }

        for (int j = 0; j < n; j++)
        {
            vCol[j * n + j] = 1.0;
            colNormSq[j] = NormSq(wCol, j * m, m);
        }

        const double tol = 1e-13;
        const int maxSweeps = 60;

        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            bool rotated = false;
            for (int p = 0; p < n - 1; p++)
            {
                int pOffW = p * m;
                int pOffV = p * n;
                for (int q = p + 1; q < n; q++)
                {
                    double app = colNormSq[p];
                    double aqq = colNormSq[q];
                    if (app == 0.0 || aqq == 0.0)
                        continue;

                    int qOffW = q * m;
                    double apq = Dot(wCol, pOffW, qOffW, m);
                    if (Math.Abs(apq) <= tol * Math.Sqrt(app * aqq))
                        continue;

                    rotated = true;

                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau == 0.0)
                        ? 1.0
                        : Math.CopySign(1.0, tau) / (Math.Abs(tau) + Math.Sqrt(1.0 + tau * tau));
                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    for (int i = 0; i < m; i++)
                    {
                        double wp = wCol[pOffW + i];
                        double wq = wCol[qOffW + i];
                        wCol[pOffW + i] = c * wp - s * wq;
                        wCol[qOffW + i] = s * wp + c * wq;
                    }

                    int qOffV = q * n;
                    for (int i = 0; i < n; i++)
                    {
                        double vp = vCol[pOffV + i];
                        double vq = vCol[qOffV + i];
                        vCol[pOffV + i] = c * vp - s * vq;
                        vCol[qOffV + i] = s * vp + c * vq;
                    }

                    colNormSq[p] = NormSq(wCol, pOffW, m);
                    colNormSq[q] = NormSq(wCol, qOffW, m);
                }
            }

            if (!rotated)
                break;
        }

        int[] order = new int[n];
        singularValues = new double[n];
        for (int j = 0; j < n; j++)
        {
            order[j] = j;
            singularValues[j] = Math.Sqrt(Math.Max(0.0, colNormSq[j]));
        }

        Array.Sort(singularValues, order, Comparer<double>.Create((a, b) => b.CompareTo(a)));

        vRowMajor = new double[vLen];
        for (int k = 0; k < n; k++)
        {
            int srcCol = order[k] * n;
            for (int i = 0; i < n; i++)
                vRowMajor[i * n + k] = vCol[srcCol + i];
        }

        double[] uCol = new double[wLen];
        double sigmaThresh = singularValues[0] > 0.0 ? 1e-14 * singularValues[0] : 0.0;
        bool hasNullColumns = false;

        for (int k = 0; k < n; k++)
        {
            double sk = singularValues[k];
            if (sk > sigmaThresh && sk > 0.0)
            {
                int srcCol = order[k] * m;
                int dstCol = k * m;
                double invS = 1.0 / sk;
                for (int i = 0; i < m; i++)
                    uCol[dstCol + i] = wCol[srcCol + i] * invS;
            }
            else
            {
                hasNullColumns = true;
            }
        }

        if (hasNullColumns)
        {
            // Complete zero-singular-value columns via MGS2 so U^T U = I_n unconditionally.
            for (int k = 0; k < n; k++)
            {
                double sk = singularValues[k];
                if (sk > sigmaThresh && sk > 0.0)
                    continue;

                int colK = k * m;
                bool completed = false;
                for (int s = 0; s < m; s++)
                {
                    Array.Clear(uCol, colK, m);
                    uCol[colK + s] = 1.0;

                    for (int pass = 0; pass < 2; pass++)
                    {
                        for (int i = 0; i < k; i++)
                        {
                            int colI = i * m;
                            double dot = Dot(uCol, colI, colK, m);
                            Axpy(-dot, uCol, colI, colK, m);
                        }
                    }

                    double nrm = Math.Sqrt(NormSq(uCol, colK, m));
                    if (nrm > 1e-10)
                    {
                        Scale(uCol, colK, m, 1.0 / nrm);
                        completed = true;
                        break;
                    }
                }

                if (!completed)
                    OrthonormalizeColumns(uCol, m, k + 1);
            }
        }

        uRowMajor = new double[wLen];
        for (int i = 0; i < m; i++)
        {
            int rowOff = i * n;
            for (int k = 0; k < n; k++)
                uRowMajor[rowOff + k] = uCol[k * m + i];
        }
    }

    internal static void TruncatedSvd(
        ReadOnlySpan<float> matrixRowMajor,
        int m,
        int n,
        int rank,
        int oversampling,
        int powerIterations,
        int seed,
        out double[] aRowMajor,
        out double[] bRowMajor)
    {
        int dMin = Math.Min(m, n);
        int r = Math.Clamp(rank, 1, dMin);
        int k = Math.Min(dMin, checked(r + oversampling));

        if (dMin <= 128 || k >= dMin)
        {
            double[] wDouble = ArrayPool<double>.Shared.Rent(checked(m * n));
            try
            {
                int total = m * n;
                for (int i = 0; i < total; i++)
                    wDouble[i] = matrixRowMajor[i];

                ExactSvd(wDouble.AsSpan(0, total), m, n, out double[] u, out double[] sigma, out double[] v);
                BuildSymmetricFactors(u, dMin, sigma, v, dMin, m, n, r, out aRowMajor, out bRowMajor);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(wDouble);
            }
            return;
        }

        var rng = new SplitMix64(seed);

        if (m >= n)
        {
            // Tall Randomized SVD (m >= n)
            double[] omega = new double[checked(n * k)];
            FillStandardNormal(omega, ref rng);

            double[] y = new double[checked(m * k)];
            MultiplyFloatByDoubleRowMajor(matrixRowMajor, m, n, omega, k, y);
            OrthonormalizeRowMajorColumns(y, m, k);
            double[] q = y;

            if (powerIterations > 0)
            {
                double[] z = new double[checked(n * k)];
                for (int iter = 0; iter < powerIterations; iter++)
                {
                    MultiplyFloatTByDoubleRowMajor(matrixRowMajor, m, n, q, k, z);
                    OrthonormalizeRowMajorColumns(z, n, k);
                    MultiplyFloatByDoubleRowMajor(matrixRowMajor, m, n, z, k, q);
                    OrthonormalizeRowMajorColumns(q, m, k);
                }
            }

            double[] bProj = new double[checked(k * n)];
            MultiplyDoubleTByFloatRowMajor(q, m, k, matrixRowMajor, n, bProj);

            ExactSvd(bProj, k, n, out double[] uHat, out double[] sigma, out double[] v);

            // U_r = Q * uHat[:, 0:r]
            double[] uR = new double[checked(m * r)];
            for (int i = 0; i < m; i++)
            {
                int qRow = i * k;
                int uRow = i * r;
                for (int t = 0; t < k; t++)
                {
                    double qit = q[qRow + t];
                    if (qit == 0.0)
                        continue;
                    int hatRow = t * k;
                    for (int c = 0; c < r; c++)
                        uR[uRow + c] += qit * uHat[hatRow + c];
                }
            }

            BuildSymmetricFactors(uR, r, sigma, v, k, m, n, r, out aRowMajor, out bRowMajor);
        }
        else
        {
            // Wide Randomized SVD (m < n)
            double[] omega = new double[checked(m * k)];
            FillStandardNormal(omega, ref rng);

            double[] y = new double[checked(n * k)];
            MultiplyFloatTByDoubleRowMajor(matrixRowMajor, m, n, omega, k, y);
            OrthonormalizeRowMajorColumns(y, n, k);
            double[] q = y;

            if (powerIterations > 0)
            {
                double[] z = new double[checked(m * k)];
                for (int iter = 0; iter < powerIterations; iter++)
                {
                    MultiplyFloatByDoubleRowMajor(matrixRowMajor, m, n, q, k, z);
                    OrthonormalizeRowMajorColumns(z, m, k);
                    MultiplyFloatTByDoubleRowMajor(matrixRowMajor, m, n, z, k, q);
                    OrthonormalizeRowMajorColumns(q, n, k);
                }
            }

            double[] bProj = new double[checked(m * k)];
            MultiplyFloatByDoubleRowMajor(matrixRowMajor, m, n, q, k, bProj);

            ExactSvd(bProj, m, k, out double[] u, out double[] sigma, out double[] vHat);

            // V_r = Q * vHat[:, 0:r]
            double[] vR = new double[checked(n * r)];
            for (int j = 0; j < n; j++)
            {
                int qRow = j * k;
                int vRow = j * r;
                for (int t = 0; t < k; t++)
                {
                    double qjt = q[qRow + t];
                    if (qjt == 0.0)
                        continue;
                    int hatRow = t * k;
                    for (int c = 0; c < r; c++)
                        vR[vRow + c] += qjt * vHat[hatRow + c];
                }
            }

            BuildSymmetricFactors(u, k, sigma, vR, r, m, n, r, out aRowMajor, out bRowMajor);
        }
    }

    internal static void RankOneNonNegativeSvd(
        ReadOnlySpan<double> absMatrixRowMajor,
        int m,
        int n,
        int maxIterations,
        out double[] uVec,
        out double[] vVec)
    {
        uVec = new double[m];
        vVec = new double[n];

        double frobSq = 0.0;
        for (int i = 0; i < absMatrixRowMajor.Length; i++)
        {
            double val = absMatrixRowMajor[i];
            frobSq += val * val;
        }

        if (Math.Sqrt(frobSq) <= 1e-15)
            return;

        double[] u = new double[m];
        double[] v = new double[n];
        double[] vNext = new double[n];

        double initV = 1.0 / Math.Sqrt(n);
        for (int j = 0; j < n; j++)
            v[j] = initV;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // u = M * v
            double uNormSq = 0.0;
            for (int i = 0; i < m; i++)
            {
                int rowOff = i * n;
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    sum += absMatrixRowMajor[rowOff + j] * v[j];
                u[i] = sum;
                uNormSq += sum * sum;
            }

            double uNorm = Math.Sqrt(uNormSq);
            if (uNorm <= 1e-30)
                return;

            double invUNorm = 1.0 / uNorm;
            for (int i = 0; i < m; i++)
                u[i] *= invUNorm;

            // vNext = M^T * u
            Array.Clear(vNext);
            for (int i = 0; i < m; i++)
            {
                double ui = u[i];
                if (ui == 0.0)
                    continue;
                int rowOff = i * n;
                for (int j = 0; j < n; j++)
                    vNext[j] += absMatrixRowMajor[rowOff + j] * ui;
            }

            double vNormSq = 0.0;
            for (int j = 0; j < n; j++)
                vNormSq += vNext[j] * vNext[j];

            double vNorm = Math.Sqrt(vNormSq);
            if (vNorm <= 1e-30)
                return;

            double invVNorm = 1.0 / vNorm;
            double diffSq = 0.0;
            for (int j = 0; j < n; j++)
            {
                double vn = vNext[j] * invVNorm;
                double d = vn - v[j];
                diffSq += d * d;
                v[j] = vn;
            }

            if (diffSq < 1e-20)
                break;
        }

        // Rayleigh quotient sigma_0 = u^T (M v)
        double sigma = 0.0;
        for (int i = 0; i < m; i++)
        {
            int rowOff = i * n;
            double mv = 0.0;
            for (int j = 0; j < n; j++)
                mv += absMatrixRowMajor[rowOff + j] * v[j];
            sigma += u[i] * mv;
        }

        double s0 = Math.Sqrt(Math.Max(0.0, sigma));
        for (int i = 0; i < m; i++)
            uVec[i] = s0 * u[i];
        for (int j = 0; j < n; j++)
            vVec[j] = s0 * v[j];
    }

    private static void BuildSymmetricFactors(
        double[] u,
        int uCols,
        double[] sigma,
        double[] v,
        int vCols,
        int m,
        int n,
        int r,
        out double[] aRowMajor,
        out double[] bRowMajor)
    {
        aRowMajor = new double[checked(m * r)];
        bRowMajor = new double[checked(r * n)];

        Span<double> sqrtSigma = r <= 512 ? stackalloc double[r] : new double[r];
        for (int c = 0; c < r; c++)
            sqrtSigma[c] = Math.Sqrt(Math.Max(0.0, sigma[c]));

        for (int i = 0; i < m; i++)
        {
            int uOff = i * uCols;
            int aOff = i * r;
            for (int c = 0; c < r; c++)
                aRowMajor[aOff + c] = u[uOff + c] * sqrtSigma[c];
        }

        for (int c = 0; c < r; c++)
        {
            double sc = sqrtSigma[c];
            int bOff = c * n;
            for (int j = 0; j < n; j++)
                bRowMajor[bOff + j] = sc * v[j * vCols + c];
        }
    }

    private static void MultiplyFloatByDoubleRowMajor(
        ReadOnlySpan<float> w,
        int m,
        int n,
        double[] q,
        int k,
        double[] y)
    {
        Array.Clear(y, 0, m * k);
        for (int i = 0; i < m; i++)
        {
            int wRow = i * n;
            int yRow = i * k;
            for (int j = 0; j < n; j++)
            {
                double wij = w[wRow + j];
                if (wij == 0.0)
                    continue;
                int qRow = j * k;
                for (int c = 0; c < k; c++)
                    y[yRow + c] += wij * q[qRow + c];
            }
        }
    }

    private static void MultiplyFloatTByDoubleRowMajor(
        ReadOnlySpan<float> w,
        int m,
        int n,
        double[] q,
        int k,
        double[] z)
    {
        Array.Clear(z, 0, n * k);
        for (int i = 0; i < m; i++)
        {
            int wRow = i * n;
            int qRow = i * k;
            for (int j = 0; j < n; j++)
            {
                double wij = w[wRow + j];
                if (wij == 0.0)
                    continue;
                int zRow = j * k;
                for (int c = 0; c < k; c++)
                    z[zRow + c] += wij * q[qRow + c];
            }
        }
    }

    private static void MultiplyDoubleTByFloatRowMajor(
        double[] q,
        int m,
        int k,
        ReadOnlySpan<float> w,
        int n,
        double[] bProj)
    {
        Array.Clear(bProj, 0, k * n);
        for (int i = 0; i < m; i++)
        {
            int qRow = i * k;
            int wRow = i * n;
            for (int c = 0; c < k; c++)
            {
                double qic = q[qRow + c];
                if (qic == 0.0)
                    continue;
                int bRow = c * n;
                for (int j = 0; j < n; j++)
                    bProj[bRow + j] += qic * w[wRow + j];
            }
        }
    }

    private static void FillStandardNormal(double[] dst, ref SplitMix64 rng)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = rng.NextStandardNormal();
    }

    private static double Dot(double[] data, int offsetA, int offsetB, int length)
    {
        double sum = 0.0;
        for (int i = 0; i < length; i++)
            sum += data[offsetA + i] * data[offsetB + i];
        return sum;
    }

    private static double NormSq(double[] data, int offset, int length)
    {
        double sum = 0.0;
        for (int i = 0; i < length; i++)
        {
            double val = data[offset + i];
            sum += val * val;
        }
        return sum;
    }

    private static void Axpy(double alpha, double[] data, int offsetX, int offsetY, int length)
    {
        if (alpha == 0.0)
            return;
        for (int i = 0; i < length; i++)
            data[offsetY + i] += alpha * data[offsetX + i];
    }

    private static void Scale(double[] data, int offset, int length, double scale)
    {
        for (int i = 0; i < length; i++)
            data[offset + i] *= scale;
    }

    private struct SplitMix64
    {
        private ulong _state;
        private bool _hasSpare;
        private double _spare;

        public SplitMix64(int seed)
        {
            _state = unchecked((ulong)(long)seed) ^ 0x9E3779B97F4A7C15UL;
            _hasSpare = false;
            _spare = 0.0;
        }

        private ulong NextUInt64()
        {
            ulong z = (_state += 0x9E3779B97F4A7C15UL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        private double NextUniform01()
        {
            // Strict (0, 1] via 53-bit mantissa to guarantee Math.Log(u1) is finite.
            ulong bits = NextUInt64() >> 11;
            return (bits + 1.0) * (1.0 / 9007199254740992.0);
        }

        public double NextStandardNormal()
        {
            if (_hasSpare)
            {
                _hasSpare = false;
                return _spare;
            }

            double u1 = NextUniform01();
            double u2 = NextUniform01();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = 2.0 * Math.PI * u2;
            _spare = radius * Math.Sin(angle);
            _hasSpare = true;
            return radius * Math.Cos(angle);
        }
    }
}
