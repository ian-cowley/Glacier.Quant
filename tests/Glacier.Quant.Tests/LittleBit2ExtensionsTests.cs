namespace Glacier.Quant.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Kernels;
using Xunit;

public sealed class LittleBit2ExtensionsTests
{
    [Theory]
    [InlineData(5, 64)]
    [InlineData(7, 37)]
    [InlineData(4, 95)]
    public void SamsungCodec_PackedBits_RoundTripsAcrossAlignedAndUnalignedWidths(int rows, int width)
    {
        var rng = new Random(1000 + rows * 31 + width);
        int words32 = (width + 31) >> 5;
        uint[] origSamsung = new uint[rows * words32];
        for (int i = 0; i < origSamsung.Length; i++)
            origSamsung[i] = unchecked((uint)rng.Next());

        // Mask canonical tail bits in origSamsung so comparison after round-trip is exact.
        int tail32 = width & 31;
        if (tail32 != 0)
        {
            uint mask32 = (1u << tail32) - 1u;
            for (int r = 0; r < rows; r++)
                origSamsung[r * words32 + words32 - 1] &= mask32;
        }

        ulong[] ltbBits = LittleBit2SamsungCodec.FromSamsungPackedBits(origSamsung, rows, width);
        uint[] roundTrip = LittleBit2SamsungCodec.ToSamsungPackedBits(ltbBits, rows, width);
        Assert.Equal(origSamsung, roundTrip);

        // Verify reverse direction and bitwise inversion on every active bit position.
        int stride64 = (width + 63) >> 6;
        Assert.Equal(rows * stride64, ltbBits.Length);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < width; c++)
            {
                bool ltbOne = ((ltbBits[r * stride64 + (c >> 6)] >> (c & 63)) & 1UL) == 1UL;
                bool samOne = ((origSamsung[r * words32 + (c >> 5)] >> (c & 31)) & 1U) == 1U;
                Assert.NotEqual(samOne, ltbOne);
            }
        }
    }

    [Fact]
    public void SamsungCodec_FromCheckpointPath_MatchesExplicitDoubleForward_AndBalancesExtremeScales()
    {
        const int rows = 41;
        const int cols = 73;
        const int rank = 19;
        var rng = new Random(4242);

        int uWords32 = (rank + 31) >> 5;
        int vWords32 = (cols + 31) >> 5;
        uint[] uPacked = new uint[rows * uWords32];
        uint[] vPacked = new uint[rank * vWords32];
        for (int i = 0; i < uPacked.Length; i++) uPacked[i] = unchecked((uint)rng.Next());
        for (int i = 0; i < vPacked.Length; i++) vPacked[i] = unchecked((uint)rng.Next());

        // Deliberately skew scales into ranges that would overflow/underflow naive FP16 if unbalanced.
        double[] u1 = new double[rows];
        double[] v2 = new double[cols];
        double[] u2 = new double[rank];
        double[] v1 = new double[rank];
        for (int r = 0; r < rows; r++) u1[r] = (rng.NextDouble() * 2.0 - 1.0) * 1e4;
        for (int c = 0; c < cols; c++) v2[c] = (rng.NextDouble() * 2.0 - 1.0) * 1e-5;
        for (int k = 0; k < rank; k++)
        {
            u2[k] = (rng.NextDouble() * 2.0 - 1.0) * 1e2;
            v1[k] = (rng.NextDouble() * 2.0 - 1.0) * 0.5;
        }

        LittleBit2Path path = LittleBit2SamsungCodec.FromSamsungCheckpointPath(
            rows, cols, rank, uPacked, vPacked, u1, u2, v1, v2);

        foreach (Half h in path.RowScales) Assert.True(Half.IsFinite(h));
        foreach (Half g in path.ColScales) Assert.True(Half.IsFinite(g));
        foreach (Half l in path.LatentScales) Assert.True(Half.IsFinite(l));

        float[] x = new float[cols];
        for (int c = 0; c < cols; c++)
            x[c] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] actualY = new float[rows];
        LittleBit2Quantizer.MatVecMul(new LittleBit2Tensor(rows, cols, path), x, actualY);

        // Compute explicit double-precision reference from Samsung packed bits (+1 if bit==0, -1 if bit==1).
        double[] z = new double[rank];
        for (int k = 0; k < rank; k++)
        {
            double acc = 0.0;
            for (int c = 0; c < cols; c++)
            {
                bool samBit = ((vPacked[k * vWords32 + (c >> 5)] >> (c & 31)) & 1U) == 1U;
                double signV = samBit ? -1.0 : 1.0;
                acc += (x[c] * v2[c]) * signV;
            }
            z[k] = acc * (v1[k] * u2[k]);
        }

        double maxRelDiff = 0.0;
        for (int r = 0; r < rows; r++)
        {
            double acc = 0.0;
            for (int k = 0; k < rank; k++)
            {
                bool samBit = ((uPacked[r * uWords32 + (k >> 5)] >> (k & 31)) & 1U) == 1U;
                double signU = samBit ? -1.0 : 1.0;
                acc += z[k] * signU;
            }
            double refY = acc * u1[r];
            double err = Math.Abs(actualY[r] - refY) / Math.Max(1.0, Math.Abs(refY));
            if (err > maxRelDiff) maxRelDiff = err;
        }

        Assert.True(maxRelDiff < 5e-3, $"Samsung codec MatVecMul FP16 balanced relative error too high: {maxRelDiff}");
    }

    [Fact]
    public void ActivationSalientSelector_JsonLoad_Strategies_Scaling_And_LargeSimdParity()
    {
        const int rows = 256;
        const int cols = 256;
        var rng = new Random(777);
        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.25f;

        // Boost columns 17 and 193 so WeightRms and WeightMaxAbs reliably find them,
        // and boost column 5 via imatrix so ActivationWeightedL2 prefers 5.
        for (int r = 0; r < rows; r++)
        {
            weights[r * cols + 17] *= 15.0f;
            weights[r * cols + 193] *= 12.0f;
        }
        weights[11 * cols + 88] = 10.0f; // single spike: highest MaxAbs (10 > 3.75) while keeping RMS (~0.63) below cols 17/193

        float[] imat = new float[cols];
        Array.Fill(imat, 0.1f);
        imat[5] = 10_000.0f;
        imat[17] = 100.0f;

        string tmpDir = Path.Combine(Path.GetTempPath(), $"glacier_quant_salient_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tmpDir);
        try
        {
            string jsonPath = Path.Combine(tmpDir, "imatrix.json");
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(new Dictionary<string, float[]>
            {
                ["blk.0.attn_q.weight"] = imat
            }));

            var loaded = ActivationSalientSelector.Load(jsonPath);
            Assert.Equal(1, loaded.LoadedTensorCount);
            Assert.Equal(0, ActivationSalientSelector.Load(null).LoadedTensorCount);

            uint[] autoKnown = loaded.SelectSalientColumns("blk.0.attn_q.weight", weights, rows, cols, 2.0 / cols, SalientSelectionMethod.Auto);
            Assert.Equal([5u, 17u], autoKnown);

            uint[] autoUnknown = loaded.SelectSalientColumns("blk.99.missing.weight", weights, rows, cols, 2.0 / cols, SalientSelectionMethod.Auto);
            Assert.Equal([17u, 193u], autoUnknown);

            uint[] maxAbsTop = loaded.SelectSalientColumns("blk.0.attn_q.weight", weights, rows, cols, 1.0 / cols, SalientSelectionMethod.WeightMaxAbs);
            Assert.Equal([88u], maxAbsTop);

            // Verify large parallel SIMD path matches explicit scalar RMS scores.
            float[] uniform = new float[rows * cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    uniform[r * cols + c] = 0.01f * ((c * 13 + r) % 7 + 1);

            uint[] salientParallel = loaded.SelectSalientColumns("other.weight", uniform, rows, cols, 0.05, SalientSelectionMethod.WeightRms);
            Assert.Equal(13, salientParallel.Length);
            double[] scalarRms = new double[cols];
            for (int c = 0; c < cols; c++)
            {
                double ssq = 0.0;
                for (int r = 0; r < rows; r++)
                {
                    double v = uniform[r * cols + c];
                    ssq += v * v;
                }
                scalarRms[c] = Math.Sqrt(ssq / rows);
            }
            int[] scalarOrder = new int[cols];
            for (int c = 0; c < cols; c++) scalarOrder[c] = c;
            Array.Sort(scalarOrder, (a, b) => { int cmp = scalarRms[b].CompareTo(scalarRms[a]); return cmp != 0 ? cmp : a.CompareTo(b); });
            uint[] expectedSalient = new uint[13];
            for (int i = 0; i < 13; i++) expectedSalient[i] = (uint)scalarOrder[i];
            Array.Sort(expectedSalient);
            Assert.Equal(expectedSalient, salientParallel);

            Assert.Null(loaded.GetColumnScalingFactors("blk.0.attn_q.weight", weights, rows, cols, alpha: 0.0f));
            float[]? scales = loaded.GetColumnScalingFactors("blk.0.attn_q.weight", weights, rows, cols, alpha: 0.5f, salientCols: autoKnown);
            Assert.NotNull(scales);
            Assert.Equal(cols, scales!.Length);
            Assert.Equal(1.0f, scales[5]);
            Assert.Equal(1.0f, scales[17]);

            var fromDict = ActivationSalientSelector.FromDictionary(new Dictionary<string, float[]> { ["t"] = [4.0f, 1.0f, 1.0f, 1.0f] });
            Assert.Equal(1, fromDict.LoadedTensorCount);
        }
        finally
        {
            Directory.Delete(tmpDir, recursive: true);
        }
    }

    [Fact]
    public void KlDivergenceEvaluator_LargeMargin_AvoidsSoftmaxCollapse()
    {
        float[] refLogits = [42f, 0f, 0f, 0f];
        float[] quantLogits = [40f, 0f, 0f, 0f];
        double[] logProbs = new double[4];

        KlDivergenceEvaluator.ComputeLogSoftmax(quantLogits, logProbs);
        double expectedArgMaxLogProb = -3.0 * Math.Exp(-40.0);
        Assert.True(logProbs[0] < 0.0 && double.IsFinite(logProbs[0]), $"Expected negative finite logProb[0], got {logProbs[0]}");
        Assert.True(Math.Abs(logProbs[0] - expectedArgMaxLogProb) / Math.Abs(expectedArgMaxLogProb) < 1e-4,
            $"LogSoftmax[0] {logProbs[0]} diverged from {expectedArgMaxLogProb}");

        double kl = KlDivergenceEvaluator.ComputeTokenKlDivergence(refLogits, quantLogits, epsilon: 1e-30);
        Assert.True(kl > 0.0 && double.IsFinite(kl), $"Expected positive finite KL divergence at +42/+40 margin, got {kl}");
    }

    [Theory]
    [InlineData(384, 256, 32)]
    [InlineData(256, 384, 32)]
    public void LittleBit2Svd_LargeParallelSimd_TruncatedSvd_AccurateAndDeterministic(int m, int n, int rank)
    {
        var rng = new Random(m * 100 + n);
        double[] uTrue = new double[m * rank];
        double[] vTrue = new double[rank * n];
        for (int i = 0; i < uTrue.Length; i++) uTrue[i] = rng.NextDouble() * 2.0 - 1.0;
        for (int i = 0; i < vTrue.Length; i++) vTrue[i] = rng.NextDouble() * 2.0 - 1.0;

        float[] w = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int r = 0; r < rank; r++)
                    sum += uTrue[i * rank + r] * vTrue[r * n + j];
                w[i * n + j] = (float)(sum * 0.05 + (rng.NextDouble() * 2.0 - 1.0) * 0.01);
            }
        }

        LittleBit2Svd.TruncatedSvd(w, m, n, rank, oversampling: 16, powerIterations: 2, seed: 42, out double[] a1, out double[] b1);
        LittleBit2Svd.TruncatedSvd(w, m, n, rank, oversampling: 16, powerIterations: 2, seed: 42, out double[] a2, out double[] b2);

        Assert.Equal(a1, a2);
        Assert.Equal(b1, b2);

        double diffSq = 0.0;
        double normSq = 0.0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double hat = 0.0;
                for (int r = 0; r < rank; r++)
                    hat += a1[i * rank + r] * b1[r * n + j];
                double orig = w[i * n + j];
                double d = orig - hat;
                diffSq += d * d;
                normSq += orig * orig;
            }
        }

        double relErr = Math.Sqrt(diffSq) / Math.Sqrt(normSq);
        Assert.True(relErr < 0.08, $"Parallel TruncatedSvd ({m}x{n}, r={rank}) relErr too high: {relErr}");
    }

    [Fact]
    public void LittleBit2Svd_LargeParallelSimd_RankOneNonNegativeSvd_RecoversExactOuterProduct()
    {
        const int m = 512;
        const int n = 520;
        double[] trueU = new double[m];
        double[] trueV = new double[n];
        for (int i = 0; i < m; i++) trueU[i] = 0.5 + ((i * 7) % 13) * 0.1;
        for (int j = 0; j < n; j++) trueV[j] = 0.3 + ((j * 11) % 17) * 0.08;

        double[] absMat = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                absMat[i * n + j] = trueU[i] * trueV[j];

        LittleBit2Svd.RankOneNonNegativeSvd(absMat, m, n, maxIterations: 50, out double[] uVec, out double[] vVec);

        double maxRelErr = 0.0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double target = absMat[i * n + j];
                double err = Math.Abs(uVec[i] * vVec[j] - target) / target;
                if (err > maxRelErr) maxRelErr = err;
            }
        }

        Assert.True(maxRelErr < 1e-10, $"Parallel RankOneNonNegativeSvd error too high: {maxRelErr}");
    }

    [Fact]
    public void LittleBit2Quantizer_RefinePathAdmm_MonotonicallyImprovesOrMaintainsError_AndMatchesSequential()
    {
        const int rows = 48;
        const int cols = 64;
        var rng = new Random(909);
        float[] weights = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                weights[i * cols + j] = (float)(Math.Sin(i * 0.25) * Math.Cos(j * 0.19) * 1.4 +
                                                (rng.NextDouble() * 2.0 - 1.0) * 0.2);
            }
        }

        var baseOpts = new LittleBit2Options
        {
            Rank = 4,
            UseResidualCompensation = false,
            AdmmIterations = 0,
            UseParallel = true,
            Seed = 42
        };
        var admmParOpts = baseOpts with { AdmmIterations = 5, UseParallel = true };
        var admmSeqOpts = baseOpts with { AdmmIterations = 5, UseParallel = false };

        LittleBit2Tensor tBase = LittleBit2Quantizer.Quantize(weights, rows, cols, baseOpts);
        LittleBit2Tensor tAdmmPar = LittleBit2Quantizer.Quantize(weights, rows, cols, admmParOpts);
        LittleBit2Tensor tAdmmSeq = LittleBit2Quantizer.Quantize(weights, rows, cols, admmSeqOpts);

        double baseErr = RelativeFrobError(weights, tBase);
        double admmParErr = RelativeFrobError(weights, tAdmmPar);
        double admmSeqErr = RelativeFrobError(weights, tAdmmSeq);

        Assert.True(admmParErr <= baseErr, $"ADMM regressed error: {admmParErr} > {baseErr}");
        Assert.True(admmParErr < baseErr, $"Expected ADMM to improve reconstruction error: base={baseErr}, admm={admmParErr}");
        Assert.Equal(tAdmmPar.Primary.UBits, tAdmmSeq.Primary.UBits);
        Assert.Equal(tAdmmPar.Primary.VBits, tAdmmSeq.Primary.VBits);
        Assert.True(Math.Abs(admmParErr - admmSeqErr) < 1e-12);
    }

    [Fact]
    public void LittleBit2Quantizer_ColumnScalesOverload_PrioritizesHotColumns_AndValidatesInputs()
    {
        const int rows = 64;
        const int cols = 128;
        var rng = new Random(1313);
        float[] weights = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                weights[r * cols + c] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;

        float[] colScales = new float[cols];
        Array.Fill(colScales, 0.25f);
        for (int c = 0; c < 12; c++)
            colScales[c] = 10.0f;

        var opts = new LittleBit2Options
        {
            Rank = 4,
            ResidualRank = 4,
            UseResidualCompensation = true,
            AdmmIterations = 2,
            UseParallel = true,
            Seed = 77
        };

        LittleBit2Tensor unweighted = LittleBit2Quantizer.Quantize(weights, rows, cols, opts);
        LittleBit2Tensor weighted = LittleBit2Quantizer.Quantize(weights, rows, cols, colScales, opts);

        double unweightedHotErr = WeightedColumnError(weights, unweighted, rows, cols, colScales);
        double weightedHotErr = WeightedColumnError(weights, weighted, rows, cols, colScales);
        Assert.True(weightedHotErr < unweightedHotErr,
            $"Expected activation-scaled quantization to lower weighted error: weighted={weightedHotErr}, unweighted={unweightedHotErr}");

        // Validation guards
        Assert.Throws<ArgumentOutOfRangeException>(() => new LittleBit2Options { AdmmIterations = -1 }.Validate(rows, cols));
        Assert.Throws<ArgumentException>(() => LittleBit2Quantizer.Quantize(weights, rows, cols, new float[cols - 1], opts));

        float[] badScale = (float[])colScales.Clone();
        badScale[3] = 0.0f;
        Assert.Throws<ArgumentException>(() => LittleBit2Quantizer.Quantize(weights, rows, cols, badScale, opts));
        badScale[3] = float.NaN;
        Assert.Throws<ArgumentException>(() => LittleBit2Quantizer.ApplyInverseColumnScales(weighted, badScale));
    }

    private static double RelativeFrobError(float[] orig, LittleBit2Tensor tensor)
    {
        float[] recon = LittleBit2Quantizer.Dequantize(tensor);
        double diff = 0.0;
        double norm = 0.0;
        for (int i = 0; i < orig.Length; i++)
        {
            double d = orig[i] - recon[i];
            diff += d * d;
            norm += (double)orig[i] * orig[i];
        }
        return Math.Sqrt(diff / norm);
    }

    private static double WeightedColumnError(float[] orig, LittleBit2Tensor tensor, int rows, int cols, float[] colScales)
    {
        float[] recon = LittleBit2Quantizer.Dequantize(tensor);
        double err = 0.0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                double d = (orig[r * cols + c] - recon[r * cols + c]) * colScales[c];
                err += d * d;
            }
        }
        return err;
    }
}
