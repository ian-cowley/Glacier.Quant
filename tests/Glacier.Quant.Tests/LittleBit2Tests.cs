namespace Glacier.Quant.Tests;

using System;
using System.IO;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Formats;
using Glacier.Quant.Gpu;
using Glacier.Quant.Kernels;
using Xunit;

public sealed class LittleBit2Tests
{
    [Theory]
    [InlineData(64, 64)]
    [InlineData(200, 64)]
    [InlineData(64, 200)]
    [InlineData(1, 37)]
    [InlineData(37, 1)]
    public void ExactSvd_ProducesOrthonormalFactorsAndAccurateReconstruction(int m, int n)
    {
        double[] w = CreateDeterministicDoubleMatrix(m, n, seed: 101 + m * 7 + n);
        LittleBit2Svd.ExactSvd(w, m, n, out double[] u, out double[] sigma, out double[] v);

        int d = Math.Min(m, n);
        Assert.Equal(d, sigma.Length);
        for (int i = 1; i < d; i++)
            Assert.True(sigma[i - 1] >= sigma[i] - 1e-12);

        double uOrthoErr = ComputeOrthogonalityError(u, m, d);
        double vOrthoErr = ComputeOrthogonalityError(v, n, d);
        Assert.True(uOrthoErr < 1e-11, $"U orthogonality error too high: {uOrthoErr}");
        Assert.True(vOrthoErr < 1e-11, $"V orthogonality error too high: {vOrthoErr}");

        double relReconErr = ComputeSvdReconstructionRelativeError(w, u, sigma, v, m, n, d);
        Assert.True(relReconErr < 1e-11, $"SVD reconstruction relative error too high: {relReconErr}");
    }

    [Fact]
    public void ExactSvd_RankDeficientMatrix_CompletesOrthonormalNullspace()
    {
        const int m = 48;
        const int n = 32;
        const int trueRank = 6;
        double[] w = CreateLowRankDoubleMatrix(m, n, trueRank, seed: 77);

        LittleBit2Svd.ExactSvd(w, m, n, out double[] u, out double[] sigma, out double[] v);

        Assert.True(ComputeOrthogonalityError(u, m, n) < 1e-11);
        Assert.True(ComputeOrthogonalityError(v, n, n) < 1e-11);
        Assert.True(ComputeSvdReconstructionRelativeError(w, u, sigma, v, m, n, n) < 1e-11);
        for (int k = trueRank; k < n; k++)
            Assert.True(sigma[k] < 1e-10);
    }

    [Theory]
    [InlineData(200, 160, 24)]
    [InlineData(160, 200, 24)]
    [InlineData(64, 96, 16)]
    public void TruncatedSvd_CapturesDominantSubspaceAcrossTallWideAndSmallMatrices(int m, int n, int rank)
    {
        float[] w = CreateLowRankFloatMatrixWithTail(m, n, trueRank: rank, tailScale: 0.02f, seed: 2026);
        LittleBit2Svd.TruncatedSvd(w, m, n, rank, oversampling: 16, powerIterations: 2, seed: 42, out double[] a, out double[] b);

        double[] recon = MultiplyRowMajor(a, m, rank, b, n);
        double relErr = ComputeRelativeFrobeniusError(w, recon);
        Assert.True(relErr < 0.08, $"TruncatedSvd relative error too high ({m}x{n}, r={rank}): {relErr}");
    }

    [Fact]
    public void JointItq_PreservesOrthogonalityAndReducesQuantizationDistortion()
    {
        const int m = 96;
        const int n = 80;
        const int rank = 24;

        float[] w = CreateHeavyTailedFloatMatrix(m, n, gamma: 1.5, seed: 314);
        LittleBit2Svd.TruncatedSvd(w, m, n, rank, oversampling: 16, powerIterations: 2, seed: 42, out double[] a, out double[] b);

        double[] rMat = LittleBit2Quantizer.ApplyJointItq(
            a, b, m, n, rank, iterations: 50, seed: 42,
            out double[] aItq, out double[] bItq);

        // Orthogonality: ||R^T R - I_r||_F < 1e-12
        Assert.True(ComputeMatrixOrthogonalityError(rMat, rank) < 1e-12);

        // Product invariance: ||(A R)(R^T B) - A B||_F / ||A B||_F < 1e-12
        double[] ab = MultiplyRowMajor(a, m, rank, b, n);
        double[] abRot = MultiplyRowMajor(aItq, m, rank, bItq, n);
        double diffSq = 0.0;
        double normSq = 0.0;
        for (int i = 0; i < ab.Length; i++)
        {
            double d = ab[i] - abRot[i];
            diffSq += d * d;
            normSq += ab[i] * ab[i];
        }
        Assert.True(Math.Sqrt(diffSq) / Math.Sqrt(normSq) < 1e-12);

        // Distortion lambda(u) = 1 - (1/r) * (||u||_1 / ||u||_2)^2
        double[] rRand = LittleBit2Svd.CreateRandomOrthogonal(rank, seed: 42);
        double[] aRand = MultiplyRowMajor(a, m, rank, rRand, rank);

        double lambdaNone = ComputeMeanRowDistortion(a, m, rank);
        double lambdaRand = ComputeMeanRowDistortion(aRand, m, rank);
        double lambdaItq = ComputeMeanRowDistortion(aItq, m, rank);

        Assert.True(lambdaItq <= lambdaRand + 1e-9, $"Expected lambdaItq ({lambdaItq}) <= lambdaRand ({lambdaRand})");
        Assert.True(lambdaRand < lambdaNone, $"Expected lambdaRand ({lambdaRand}) < lambdaNone ({lambdaNone})");
    }

    [Theory]
    [InlineData(1e-5f)]
    [InlineData(1e4f)]
    public void DualSvid_BalancedFp16Scales_AvoidUnderflowAndOverflowAtExtremeMagnitudes(float magnitudeScale)
    {
        const int rows = 64;
        const int cols = 64;
        float[] baseWeights = CreateHeavyTailedFloatMatrix(rows, cols, gamma: 1.2, seed: 55);
        float[] scaled = new float[baseWeights.Length];
        for (int i = 0; i < baseWeights.Length; i++)
            scaled[i] = baseWeights[i] * magnitudeScale;

        var opts = new LittleBit2Options
        {
            Rank = 24,
            UseResidualCompensation = true,
            RotationMode = LittleBit2RotationMode.JointItq
        };

        LittleBit2Tensor unitTensor = LittleBit2Quantizer.Quantize(baseWeights, rows, cols, opts);
        float[] unitDequant = LittleBit2Quantizer.Dequantize(unitTensor);
        double unitRelErr = ComputeRelativeFrobeniusError(baseWeights, unitDequant);

        LittleBit2Tensor tensor = LittleBit2Quantizer.Quantize(scaled, rows, cols, opts);
        float[] dequant = LittleBit2Quantizer.Dequantize(tensor);

        for (int i = 0; i < dequant.Length; i++)
            Assert.True(float.IsFinite(dequant[i]));

        double relErr = ComputeRelativeFrobeniusError(scaled, dequant);
        Assert.True(relErr < 0.35, $"Extreme scale {magnitudeScale} relative error too large: {relErr}");
        Assert.True(Math.Abs(relErr - unitRelErr) < 0.03, $"Scale-invariance broken at {magnitudeScale}: {relErr} vs unit {unitRelErr}");

        Assert.Contains(tensor.Primary.RowScales, s => (float)s > 0.0f);
        Assert.Contains(tensor.Primary.ColScales, s => (float)s > 0.0f);
        Assert.Contains(tensor.Primary.LatentScales, s => (float)s > 0.0f);
    }

    [Fact]
    public void TwoPathResidualCompensation_ReducesReconstructionErrorComparedToSinglePath()
    {
        const int rows = 96;
        const int cols = 80;
        float[] weights = CreateHeavyTailedFloatMatrix(rows, cols, gamma: 0.25, seed: 88);

        LittleBit2Tensor onePath = LittleBit2Quantizer.Quantize(weights, rows, cols, new LittleBit2Options
        {
            Rank = 16,
            UseResidualCompensation = false
        });
        LittleBit2Tensor twoPath = LittleBit2Quantizer.Quantize(weights, rows, cols, new LittleBit2Options
        {
            Rank = 16,
            UseResidualCompensation = true
        });

        Assert.Equal(1, onePath.PathCount);
        Assert.False(onePath.HasResidual);
        Assert.Equal(2, twoPath.PathCount);
        Assert.True(twoPath.HasResidual);

        float[] d1 = LittleBit2Quantizer.Dequantize(onePath);
        float[] d2 = LittleBit2Quantizer.Dequantize(twoPath);

        double err1 = ComputeRelativeFrobeniusError(weights, d1);
        double err2 = ComputeRelativeFrobeniusError(weights, d2);
        Assert.True(err2 < err1, $"Expected 2-path error ({err2}) < 1-path error ({err1})");
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 17, 3)]
    [InlineData(63, 65, 19)]
    [InlineData(64, 64, 64)]
    [InlineData(65, 133, 65)]
    public void BitPackingAndTailBits_HandleNonMultipleOf64DimensionsAndZeroTailInvariant(int rows, int cols, int rank)
    {
        int uStride = (rank + 63) >> 6;
        int vStride = (cols + 63) >> 6;

        // Dirty unused tail bits to verify LittleBit2Path constructor clears them.
        ulong[] dirtyU = new ulong[rows * uStride];
        ulong[] dirtyV = new ulong[rank * vStride];
        Array.Fill(dirtyU, ulong.MaxValue);
        Array.Fill(dirtyV, ulong.MaxValue);

        Half[] h = new Half[rows];
        Half[] g = new Half[cols];
        Half[] ell = new Half[rank];
        Array.Fill(h, (Half)1.0f);
        Array.Fill(g, (Half)1.0f);
        Array.Fill(ell, (Half)1.0f);

        var path = new LittleBit2Path(rows, cols, rank, dirtyU, dirtyV, h, g, ell);

        int uTail = rank & 63;
        if (uTail != 0)
        {
            ulong uValidMask = (1UL << uTail) - 1UL;
            for (int r = 0; r < rows; r++)
                Assert.Equal(0UL, path.UBits[r * uStride + (uStride - 1)] & ~uValidMask);
        }

        int vTail = cols & 63;
        if (vTail != 0)
        {
            ulong vValidMask = (1UL << vTail) - 1UL;
            for (int k = 0; k < rank; k++)
                Assert.Equal(0UL, path.VBits[k * vStride + (vStride - 1)] & ~vValidMask);
        }

        var tensor = new LittleBit2Tensor(rows, cols, path);
        float[] x = new float[cols];
        for (int j = 0; j < cols; j++)
            x[j] = (j % 5) - 2.0f;

        float[] yMatVec = new float[rows];
        LittleBit2Quantizer.MatVecMul(tensor, x, yMatVec);

        float[] dequant = LittleBit2Quantizer.Dequantize(tensor);
        float[] yRef = DenseMatVec(dequant, rows, cols, x);

        for (int i = 0; i < rows; i++)
            Assert.True(Math.Abs(yMatVec[i] - yRef[i]) < 1e-3f, $"Row {i}: {yMatVec[i]} != {yRef[i]}");
    }

    [Fact]
    public unsafe void BitPacking_ZeroAndNegativeZeroEncodeAsPlusOneAndNegativeAsMinusOne()
    {
        float[] weights =
        [
            +0.0f, -0.0f, +2.0f, -3.0f
        ];

        // Quantize with RotationMode.None and Rank=1 on a simple 2x2 matrix containing +0.0f/-0.0f
        float[] mat =
        [
            0.0f, -0.0f,
            1.0f, 1.0f
        ];
        var tensor = LittleBit2Quantizer.Quantize(mat, 2, 2, new LittleBit2Options
        {
            Rank = 1,
            UseResidualCompensation = false,
            RotationMode = LittleBit2RotationMode.None
        });
        Assert.NotNull(tensor);

        // Direct sign-bit contract verification on BinaryDotScalar for +1 (bit 1) vs -1 (bit 0)
        ulong packed = 0b0111UL; // bit 0=1 (+), bit 1=1 (+), bit 2=1 (+), bit 3=0 (-)
        fixed (float* pW = weights)
        {
            float dot = LittleBit2Quantizer.BinaryDotScalar(&packed, pW, 4);
            // (+0.0f) + (-0.0f) + (+2.0f) - (-3.0f) = 5.0f
            Assert.Equal(5.0f, dot, precision: 5);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(259)]
    public unsafe void BinaryDot_AllSupportedIsaPathsMatchScalarReference(int length)
    {
        int stride = (length + 63) >> 6;
        ulong[] bits = new ulong[stride];
        float[] vec = new float[length];

        var rng = new Random(9001 + length);
        for (int w = 0; w < stride; w++)
            bits[w] = ((ulong)(uint)rng.Next() << 32) | (uint)rng.Next();

        // Intentionally flip unused tail bits to 1 to verify tail-safety
        int tail = length & 63;
        if (tail != 0)
            bits[stride - 1] |= ~((1UL << tail) - 1UL);

        for (int i = 0; i < length; i++)
            vec[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

        fixed (ulong* pBits = bits)
        fixed (float* pVec = vec)
        {
            float scalar = LittleBit2Quantizer.BinaryDotScalar(pBits, pVec, length);
            float dispatched = LittleBit2Quantizer.BinaryDot(pBits, pVec, length);
            Assert.True(Math.Abs(scalar - dispatched) < 1e-4f, $"Dispatched mismatch at len={length}: {dispatched} vs {scalar}");

            if (Vector512.IsHardwareAccelerated)
            {
                float v512 = LittleBit2Quantizer.BinaryDotVector512(pBits, pVec, length);
                Assert.True(Math.Abs(scalar - v512) < 1e-4f, $"Vector512 mismatch at len={length}: {v512} vs {scalar}");
            }

            if (Avx2.IsSupported)
            {
                float avx2 = LittleBit2Quantizer.BinaryDotAvx2(pBits, pVec, length);
                Assert.True(Math.Abs(scalar - avx2) < 1e-4f, $"Avx2 mismatch at len={length}: {avx2} vs {scalar}");
            }

            if (AdvSimd.IsSupported)
            {
                float neon = LittleBit2Quantizer.BinaryDotAdvSimd(pBits, pVec, length);
                Assert.True(Math.Abs(scalar - neon) < 1e-4f, $"AdvSimd mismatch at len={length}: {neon} vs {scalar}");
            }
        }
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(4, false)]
    [InlineData(7, false)]
    [InlineData(32, false)]
    [InlineData(33, false)]
    [InlineData(65, false)]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(4, true)]
    [InlineData(7, true)]
    [InlineData(33, true)]
    [InlineData(65, true)]
    public void MatVecMul_MatMul_AndDequantize_AreEquivalentAcrossBatchSizes(int batchSize, bool useResidual)
    {
        const int rows = 73;
        const int cols = 131;
        const int rank = 67;

        float[] weights = CreateHeavyTailedFloatMatrix(rows, cols, gamma: 0.35, seed: 1234);
        LittleBit2Tensor tensor = LittleBit2Quantizer.Quantize(weights, rows, cols, new LittleBit2Options
        {
            Rank = rank,
            UseResidualCompensation = useResidual,
            JointItqIterations = 15
        });

        float[] dequant = LittleBit2Quantizer.Dequantize(tensor);

        var rng = new Random(4321 + batchSize);
        float[] xBatch = new float[batchSize * cols];
        for (int i = 0; i < xBatch.Length; i++)
            xBatch[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] yBatch = new float[batchSize * rows];
        LittleBit2Quantizer.MatMul(tensor, xBatch, batchSize, yBatch);

        for (int t = 0; t < batchSize; t++)
        {
            ReadOnlySpan<float> xRow = xBatch.AsSpan(t * cols, cols);
            float[] ySingle = new float[rows];
            LittleBit2Quantizer.MatVecMul(tensor, xRow, ySingle);
            float[] yDense = DenseMatVec(dequant, rows, cols, xRow);

            for (int r = 0; r < rows; r++)
            {
                float yBatchVal = yBatch[t * rows + r];
                Assert.True(Math.Abs(yBatchVal - ySingle[r]) < 1e-3f, $"MatMul vs MatVecMul mismatch at ({t},{r}): {yBatchVal} vs {ySingle[r]}");
                Assert.True(Math.Abs(yBatchVal - yDense[r]) < 1e-3f, $"MatMul vs Dequant*x mismatch at ({t},{r}): {yBatchVal} vs {yDense[r]}");
            }
        }
    }

    [Fact]
    public void RankAndBpwFormulas_MatchPaperBenchmarksAndClampTinyMatrices()
    {
        int rSquare = LittleBit2Quantizer.ComputeRankForBpw(4096, 4096, 0.55, useResidualCompensation: true);
        Assert.Equal(546, rSquare);
        double bpwSquare = LittleBit2Quantizer.ComputeEffectiveBpw(4096, 4096, rSquare, useResidualCompensation: true);
        Assert.InRange(bpwSquare, 0.549, 0.550);

        int rRect = LittleBit2Quantizer.ComputeRankForBpw(4096, 11008, 0.10, useResidualCompensation: true);
        Assert.Equal(133, rRect);
        double bpwRect = LittleBit2Quantizer.ComputeEffectiveBpw(4096, 11008, rRect, useResidualCompensation: true);
        Assert.InRange(bpwRect, 0.099, 0.100);

        // Overload disambiguation test: int vs bool 4th argument
        double bpwSingle = LittleBit2Quantizer.ComputeEffectiveBpw(4096, 4096, 546);
        double bpwSingleExplicit = LittleBit2Quantizer.ComputeEffectiveBpw(4096, 4096, 546, 0);
        double bpwSingleBool = LittleBit2Quantizer.ComputeEffectiveBpw(4096, 4096, 546, false);
        Assert.Equal(bpwSingle, bpwSingleExplicit, precision: 12);
        Assert.Equal(bpwSingle, bpwSingleBool, precision: 12);
        Assert.True(bpwSquare > bpwSingle);

        // Tiny matrix clamps to rank >= 1
        int rTiny = LittleBit2Quantizer.ComputeRankForBpw(64, 64, 0.10, useResidualCompensation: true);
        Assert.Equal(1, rTiny);
    }

    [Fact]
    public void EdgeCases_ZeroMatrixRectangularAndNonFiniteValidation()
    {
        // All-zero matrix
        float[] zeros = new float[17 * 23];
        LittleBit2Tensor zeroTensor = LittleBit2Quantizer.Quantize(zeros, 17, 23);
        float[] zeroDequant = LittleBit2Quantizer.Dequantize(zeroTensor);
        Assert.All(zeroDequant, v => Assert.Equal(0.0f, v));

        float[] x = new float[23];
        Array.Fill(x, 1.5f);
        float[] y = new float[17];
        LittleBit2Quantizer.MatVecMul(zeroTensor, x, y);
        Assert.All(y, v => Assert.Equal(0.0f, v));

        // Rectangular 1xN and Nx1
        float[] rowVec = [1.0f, -2.0f, 3.0f, -4.0f, 5.0f];
        LittleBit2Tensor t1x5 = LittleBit2Quantizer.Quantize(rowVec, 1, 5);
        Assert.Equal(1, t1x5.Rank);
        Assert.True(ComputeRelativeFrobeniusError(rowVec, LittleBit2Quantizer.Dequantize(t1x5)) < 0.5);

        LittleBit2Tensor t5x1 = LittleBit2Quantizer.Quantize(rowVec, 5, 1);
        Assert.Equal(1, t5x1.Rank);
        Assert.True(ComputeRelativeFrobeniusError(rowVec, LittleBit2Quantizer.Dequantize(t5x1)) < 0.5);

        // Non-finite rejection
        float[] nanWeights = [1.0f, float.NaN, 2.0f, 3.0f];
        Assert.Throws<ArgumentException>(() => LittleBit2Quantizer.Quantize(nanWeights, 2, 2));

        float[] infWeights = [1.0f, float.PositiveInfinity, 2.0f, 3.0f];
        Assert.Throws<ArgumentException>(() => LittleBit2Quantizer.Quantize(infWeights, 2, 2));
    }

    [Fact]
    public void BinarySerialization_ToBytesFromBytesAndStreamRoundTripPreserveExactTensor()
    {
        const int rows = 67;
        const int cols = 71;
        float[] weights = CreateHeavyTailedFloatMatrix(rows, cols, gamma: 0.3, seed: 999);
        LittleBit2Tensor original = LittleBit2Quantizer.Quantize(weights, rows, cols, new LittleBit2Options
        {
            Rank = 21,
            ResidualRank = 13,
            UseResidualCompensation = true
        });

        byte[] payload = original.ToBytes();
        LittleBit2Tensor fromBytes = LittleBit2Tensor.FromBytes(payload);

        using var ms = new MemoryStream();
        original.WriteTo(ms);
        ms.Position = 0;
        LittleBit2Tensor fromStream = LittleBit2Tensor.ReadFrom(ms);

        float[] origDequant = LittleBit2Quantizer.Dequantize(original);
        float[] bytesDequant = LittleBit2Quantizer.Dequantize(fromBytes);
        float[] streamDequant = LittleBit2Quantizer.Dequantize(fromStream);

        Assert.Equal(original.Rows, fromBytes.Rows);
        Assert.Equal(original.Cols, fromBytes.Cols);
        Assert.Equal(original.Rank, fromBytes.Rank);
        Assert.Equal(original.Residual!.Rank, fromBytes.Residual!.Rank);
        Assert.Equal(original.EffectiveBitsPerWeight, fromBytes.EffectiveBitsPerWeight, precision: 12);
        Assert.Equal(original.CompressedByteSize, fromBytes.CompressedByteSize);
        Assert.Equal(origDequant, bytesDequant);
        Assert.Equal(origDequant, streamDequant);

        // Corrupt magic header throws InvalidDataException
        byte[] badPayload = (byte[])payload.Clone();
        badPayload[0] ^= 0xFF;
        Assert.Throws<InvalidDataException>(() => LittleBit2Tensor.FromBytes(badPayload));

        // Truncated payload throws InvalidDataException
        Assert.Throws<InvalidDataException>(() => LittleBit2Tensor.FromBytes(payload.AsSpan(0, payload.Length - 3)));
    }

    [Theory]
    [InlineData(100_000_000, 65, 8)]
    [InlineData(300_000_000, 65, 8)]
    [InlineData(1_100_000_000, 65, 65)]
    public void BinaryDeserialization_CraftedHugeHeaders_ThrowInvalidDataExceptionWithBoundedAllocation(int rows, int cols, int rank)
    {
        byte[] crafted24 = new byte[24];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(crafted24.AsSpan(0, 4), 0x3242544Cu); // "LTB2"
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(crafted24.AsSpan(4, 4), 1u);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(crafted24.AsSpan(8, 4), rows);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(crafted24.AsSpan(12, 4), cols);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(crafted24.AsSpan(16, 4), 1);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(crafted24.AsSpan(20, 4), rank);

        Assert.Throws<InvalidDataException>(() => LittleBit2Tensor.FromBytes(crafted24));

        long seekableAllocBefore = GC.GetAllocatedBytesForCurrentThread();
        using (var seekable = new MemoryStream(crafted24, writable: false))
        {
            Assert.Throws<InvalidDataException>(() => LittleBit2Tensor.ReadFrom(seekable));
        }
        long seekableAllocBytes = GC.GetAllocatedBytesForCurrentThread() - seekableAllocBefore;
        Assert.True(seekableAllocBytes < 1024 * 1024, $"Seekable truncated stream allocated {seekableAllocBytes} bytes.");

        long nonSeekableAllocBefore = GC.GetAllocatedBytesForCurrentThread();
        using (var nonSeekable = new NonSeekableReadStream(crafted24))
        {
            Assert.Throws<InvalidDataException>(() => LittleBit2Tensor.ReadFrom(nonSeekable));
        }
        long nonSeekableAllocBytes = GC.GetAllocatedBytesForCurrentThread() - nonSeekableAllocBefore;
        Assert.True(nonSeekableAllocBytes < 2 * 1024 * 1024, $"Non-seekable truncated stream allocated {nonSeekableAllocBytes} bytes.");
    }

    [Fact]
    public void BinarySerialization_MultiChunkNonSeekableStream_RoundTripsCorrectly()
    {
        const int rows = 100_000;
        const int cols = 65;
        const int rank = 1;
        ulong[] uBits = new ulong[rows];
        for (int i = 0; i < rows; i++) uBits[i] = (ulong)(i & 1);
        ulong[] vBits = [0x5555555555555555UL, 1UL];
        Half[] rowScales = new Half[rows];
        Array.Fill(rowScales, (Half)1.0f);
        Half[] colScales = new Half[cols];
        Array.Fill(colScales, (Half)1.0f);
        Half[] latentScales = [(Half)1.0f];

        var path = new LittleBit2Path(rows, cols, rank, uBits, vBits, rowScales, colScales, latentScales);
        var tensor = new LittleBit2Tensor(rows, cols, path);

        using var ms = new MemoryStream();
        tensor.WriteTo(ms);
        byte[] serialized = ms.ToArray();

        using var nonSeekable = new NonSeekableReadStream(serialized);
        LittleBit2Tensor roundTripped = LittleBit2Tensor.ReadFrom(nonSeekable);
        Assert.Equal(rows, roundTripped.Rows);
        Assert.Equal(cols, roundTripped.Cols);
        Assert.Equal(rank, roundTripped.Rank);
        Assert.Equal(uBits, roundTripped.Primary.UBits);
        Assert.Equal(vBits, roundTripped.Primary.VBits);
    }

    private sealed class NonSeekableReadStream : Stream
    {
        private readonly MemoryStream _inner;
        public NonSeekableReadStream(byte[] data) => _inner = new MemoryStream(data, writable: false);
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override int Read(Span<byte> buffer) => _inner.Read(buffer);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing)
        {
            if (disposing) _inner.Dispose();
            base.Dispose(disposing);
        }
    }

    [Fact]
    public void QualityCheckerAndIntegration_EvaluateLittleBit2DispatcherAndSafetensorsWorkCleanly()
    {
        const int vocabSize = 32;
        const int hiddenDim = 256;
        const int numTokens = 16;

        float[] weights = CreateHeavyTailedFloatMatrix(vocabSize, hiddenDim, gamma: 0.7, seed: 777);
        float[] activations = new float[numTokens * hiddenDim];
        var rng = new Random(778);
        for (int t = 0; t < numTokens; t++)
        {
            int targetClass = (t * 3) % vocabSize;
            int wOff = targetClass * hiddenDim;
            int aOff = t * hiddenDim;
            for (int c = 0; c < hiddenDim; c++)
                activations[aOff + c] = 0.5f * weights[wOff + c] + (float)(rng.NextDouble() * 0.04 - 0.02);
        }

        var dataset = new CalibrationDataset(numTokens, hiddenDim, activations);
        var lbOptions = new LittleBit2Options
        {
            Rank = 32,
            UseResidualCompensation = true,
            RotationMode = LittleBit2RotationMode.JointItq
        };
        LittleBit2Tensor quantized = LittleBit2Quantizer.Quantize(weights, vocabSize, hiddenDim, lbOptions);

        var passingKl = new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.75 };
        KlDivergenceResult res1 = QuantQualityChecker.EvaluateLittleBit2(weights, quantized, activations, numTokens, vocabSize, hiddenDim, passingKl);
        KlDivergenceResult res2 = QuantQualityChecker.EvaluateLittleBit2(weights, quantized, dataset, vocabSize, passingKl);
        KlDivergenceResult res3 = QuantQualityChecker.EvaluateLittleBit2(weights, activations, numTokens, vocabSize, hiddenDim, lbOptions, passingKl);
        KlDivergenceResult res4 = QuantQualityChecker.EvaluateLittleBit2(weights, dataset, vocabSize, lbOptions, passingKl);

        Assert.True(res1.Passed, $"Expected KL gate pass, reason: {res1.FailureReason}");
        Assert.Equal(res1.MeanKlDivergence, res2.MeanKlDivergence, precision: 12);
        Assert.Equal(res1.MeanKlDivergence, res3.MeanKlDivergence, precision: 12);
        Assert.Equal(res1.MeanKlDivergence, res4.MeanKlDivergence, precision: 12);

        // Strict threshold triggers deterministic failure
        var impossibleKl = new KlDivergenceOptions { Threshold = 1e-12 };
        KlDivergenceResult failRes = QuantQualityChecker.EvaluateLittleBit2(weights, quantized, dataset, vocabSize, impossibleKl);
        Assert.False(failRes.Passed);

        // EvaluateQuantization 2D routing for QuantType.LittleBit2
        KlDivergenceResult routeRes = QuantQualityChecker.EvaluateQuantization(
            weights, QuantType.LittleBit2, dataset, vocabSize, options: new KlDivergenceOptions { Threshold = 1.0 });
        Assert.False(routeRes.HasNonFiniteQuantizedValues);
        Assert.True(double.IsFinite(routeRes.MeanKlDivergence));

        // GpuQuantizationDispatcher explicit NotSupportedException for 2D LittleBit2
        Assert.Throws<NotSupportedException>(() => GpuQuantizationDispatcher.GetQuantizedByteCount(64 * 64, QuantType.LittleBit2));
        Assert.Throws<NotSupportedException>(() =>
        {
            Span<byte> buf = stackalloc byte[64];
            GpuQuantizationDispatcher.QuantizeTensor(weights, QuantType.LittleBit2, buf, out _);
        });
        Assert.Throws<NotSupportedException>(() =>
        {
            Span<byte> buf = stackalloc byte[64];
            Span<float> dst = new float[64];
            GpuQuantizationDispatcher.DequantizeTensor(buf, QuantType.LittleBit2, dst);
        });

        // Safetensors dtype mapping
        Assert.Equal("U8", SafetensorsWriter.MapQuantTypeToSafetensorsDtype(QuantType.LittleBit2));
        Assert.Equal("U64", SafetensorsWriter.MapQuantTypeToSafetensorsDtype(QuantType.Binary1Bit));
    }

    private static double[] CreateDeterministicDoubleMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        double[] m = new double[rows * cols];
        for (int i = 0; i < m.Length; i++)
            m[i] = rng.NextDouble() * 2.0 - 1.0;
        return m;
    }

    private static double[] CreateLowRankDoubleMatrix(int rows, int cols, int rank, int seed)
    {
        var rng = new Random(seed);
        double[] a = new double[rows * rank];
        double[] b = new double[rank * cols];
        for (int i = 0; i < a.Length; i++)
            a[i] = rng.NextDouble() * 2.0 - 1.0;
        for (int i = 0; i < b.Length; i++)
            b[i] = rng.NextDouble() * 2.0 - 1.0;
        return MultiplyRowMajor(a, rows, rank, b, cols);
    }

    private static float[] CreateLowRankFloatMatrixWithTail(int rows, int cols, int trueRank, float tailScale, int seed)
    {
        double[] lr = CreateLowRankDoubleMatrix(rows, cols, trueRank, seed);
        var rng = new Random(seed + 17);
        float[] res = new float[rows * cols];
        for (int i = 0; i < res.Length; i++)
            res[i] = (float)lr[i] + (float)(rng.NextDouble() * 2.0 - 1.0) * tailScale;
        return res;
    }

    private static float[] CreateHeavyTailedFloatMatrix(int rows, int cols, double gamma, int seed)
    {
        int d = Math.Min(rows, cols);
        var rng = new Random(seed);
        double[] u = new double[rows * d];
        double[] v = new double[cols * d];
        for (int i = 0; i < u.Length; i++)
            u[i] = rng.NextDouble() * 2.0 - 1.0;
        for (int i = 0; i < v.Length; i++)
            v[i] = rng.NextDouble() * 2.0 - 1.0;
        LittleBit2Svd.OrthonormalizeRowMajorColumns(u, rows, d);
        LittleBit2Svd.OrthonormalizeRowMajorColumns(v, cols, d);

        float[] w = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < d; k++)
                {
                    double sk = Math.Pow(k + 1, -gamma);
                    sum += u[i * d + k] * sk * v[j * d + k];
                }
                w[i * cols + j] = (float)sum;
            }
        }
        return w;
    }

    private static double[] MultiplyRowMajor(double[] a, int m, int k, double[] b, int n)
    {
        double[] c = new double[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int t = 0; t < k; t++)
            {
                double ait = a[i * k + t];
                for (int j = 0; j < n; j++)
                    c[i * n + j] += ait * b[t * n + j];
            }
        }
        return c;
    }

    private static float[] DenseMatVec(float[] w, int rows, int cols, ReadOnlySpan<float> x)
    {
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0.0;
            int rowOff = r * cols;
            for (int c = 0; c < cols; c++)
                sum += (double)w[rowOff + c] * x[c];
            y[r] = (float)sum;
        }
        return y;
    }

    private static double ComputeOrthogonalityError(double[] qRowMajor, int rows, int cols)
    {
        double errSq = 0.0;
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double dot = 0.0;
                for (int r = 0; r < rows; r++)
                    dot += qRowMajor[r * cols + i] * qRowMajor[r * cols + j];
                double expected = i == j ? 1.0 : 0.0;
                double d = dot - expected;
                errSq += d * d;
            }
        }
        return Math.Sqrt(errSq);
    }

    private static double ComputeMatrixOrthogonalityError(double[] rMat, int size) =>
        ComputeOrthogonalityError(rMat, size, size);

    private static double ComputeSvdReconstructionRelativeError(
        double[] w, double[] u, double[] sigma, double[] v, int m, int n, int d)
    {
        double diffSq = 0.0;
        double normSq = 0.0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double recon = 0.0;
                for (int k = 0; k < d; k++)
                    recon += u[i * d + k] * sigma[k] * v[j * d + k];
                double orig = w[i * n + j];
                double diff = orig - recon;
                diffSq += diff * diff;
                normSq += orig * orig;
            }
        }
        return normSq > 0.0 ? Math.Sqrt(diffSq) / Math.Sqrt(normSq) : Math.Sqrt(diffSq);
    }

    private static double ComputeRelativeFrobeniusError(float[] reference, float[] approx)
    {
        double diffSq = 0.0;
        double normSq = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            double diff = reference[i] - approx[i];
            diffSq += diff * diff;
            normSq += (double)reference[i] * reference[i];
        }
        return normSq > 0.0 ? Math.Sqrt(diffSq) / Math.Sqrt(normSq) : Math.Sqrt(diffSq);
    }

    private static double ComputeRelativeFrobeniusError(float[] reference, double[] approx)
    {
        double diffSq = 0.0;
        double normSq = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            double diff = reference[i] - approx[i];
            diffSq += diff * diff;
            normSq += (double)reference[i] * reference[i];
        }
        return normSq > 0.0 ? Math.Sqrt(diffSq) / Math.Sqrt(normSq) : Math.Sqrt(diffSq);
    }

    private static double ComputeMeanRowDistortion(double[] mat, int rows, int rank)
    {
        double total = 0.0;
        int valid = 0;
        for (int i = 0; i < rows; i++)
        {
            double l1 = 0.0;
            double l2Sq = 0.0;
            int rowOff = i * rank;
            for (int k = 0; k < rank; k++)
            {
                double v = mat[rowOff + k];
                l1 += Math.Abs(v);
                l2Sq += v * v;
            }
            if (l2Sq > 1e-24)
            {
                total += 1.0 - (l1 * l1) / (rank * l2Sq);
                valid++;
            }
        }
        return valid > 0 ? total / valid : 0.0;
    }
}
