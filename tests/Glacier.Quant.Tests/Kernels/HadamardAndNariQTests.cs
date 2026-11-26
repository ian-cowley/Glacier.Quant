namespace Glacier.Quant.Tests.Kernels;

using System;
using System.Runtime.InteropServices;
using Glacier.Quant.Kernels;
using Xunit;

public class HadamardAndNariQTests
{
    // =========================================================================
    // 1. SYLVESTER FWHT TESTS
    // =========================================================================

    [Theory]
    [InlineData(2, 16)]
    [InlineData(4, 16)]
    [InlineData(8, 32)]
    [InlineData(16, 64)]
    [InlineData(32, 128)]
    [InlineData(64, 256)]
    [InlineData(128, 256)]
    [InlineData(256, 512)]
    [InlineData(512, 1024)]
    [InlineData(1024, 2048)]
    [InlineData(4096, 4096)]
    public void Sylvester_SelfInverseOrthogonality_H_of_H_Equals_Identity(int blockLen, int totalLen)
    {
        var rng = new Random(1337 + blockLen);
        float[] original = new float[totalLen];
        float[] work = new float[totalLen];
        for (int i = 0; i < totalLen; i++)
        {
            original[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            work[i] = original[i];
        }

        HadamardTransform.ApplySylvesterInPlace(work, blockLen);
        // Verify L2 norm preservation (Parseval)
        double origNormSq = 0.0, rotNormSq = 0.0;
        for (int i = 0; i < totalLen; i++)
        {
            origNormSq += (double)original[i] * original[i];
            rotNormSq += (double)work[i] * work[i];
        }
        Assert.True(Math.Abs(origNormSq - rotNormSq) / Math.Max(1.0, origNormSq) < 1e-5,
            $"L2 norm drift too large for blockLen={blockLen}: orig={origNormSq}, rot={rotNormSq}");

        // Second application recovers original (H * H = I)
        HadamardTransform.ApplySylvesterInPlace(work, blockLen);

        float maxErr = 0f;
        for (int i = 0; i < totalLen; i++)
        {
            float err = MathF.Abs(work[i] - original[i]);
            if (err > maxErr) maxErr = err;
        }
        Assert.True(maxErr < 1e-4f, $"H(H(x)) max error {maxErr} exceeded 1e-4 for blockLen={blockLen}");
    }

    [Theory]
    [InlineData(16, 64)]
    [InlineData(32, 128)]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    [InlineData(256, 1024)]
    [InlineData(1024, 2048)]
    public void Sylvester_SimdVsScalar_IsBitExact(int blockLen, int totalLen)
    {
        var rng = new Random(4242 + blockLen);
        float[] simd = new float[totalLen];
        float[] scalar = new float[totalLen];
        for (int i = 0; i < totalLen; i++)
        {
            float v = (float)(rng.NextDouble() * 10.0 - 5.0);
            if (i % 37 == 0) v = -0.0f;
            if (i % 41 == 0) v = 0.0f;
            simd[i] = v;
            scalar[i] = v;
        }

        HadamardTransform.ApplySylvesterInPlace(simd, blockLen);
        HadamardTransform.ApplySylvesterScalarInPlace(scalar, blockLen);

        for (int i = 0; i < totalLen; i++)
        {
            int bitsSimd = BitConverter.SingleToInt32Bits(simd[i]);
            int bitsScalar = BitConverter.SingleToInt32Bits(scalar[i]);
            Assert.Equal(bitsScalar, bitsSimd);
        }
    }

    [Fact]
    public void Sylvester_PreservesInnerProduct_Parseval()
    {
        const int blockLen = 256;
        const int len = 1024;
        var rng = new Random(9001);
        float[] x = new float[len];
        float[] y = new float[len];
        for (int i = 0; i < len; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            y[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        double dotBefore = 0.0;
        for (int i = 0; i < len; i++) dotBefore += (double)x[i] * y[i];

        HadamardTransform.ApplySylvesterInPlace(x, blockLen);
        HadamardTransform.ApplySylvesterInPlace(y, blockLen);

        double dotAfter = 0.0;
        for (int i = 0; i < len; i++) dotAfter += (double)x[i] * y[i];

        Assert.True(Math.Abs(dotBefore - dotAfter) < 1e-3,
            $"Inner product drift: before={dotBefore}, after={dotAfter}");
    }

    // =========================================================================
    // 2. PALEY HADAMARD TESTS (m in {12, 20, 68})
    // =========================================================================

    [Theory]
    [InlineData(12, 1)]
    [InlineData(12, 4)]
    [InlineData(20, 1)]
    [InlineData(20, 3)]
    [InlineData(68, 1)]
    [InlineData(68, 3)]
    public void Paley_Orthogonality_HT_of_H_And_H_of_HT_Equal_Identity(int m, int numBlocks)
    {
        Assert.True(HadamardTransform.IsSupportedPaleyOrder(m));
        int len = m * numBlocks;
        var rng = new Random(777 + m * 10 + numBlocks);
        float[] original = new float[len];
        float[] fwdInv = new float[len];
        float[] invFwd = new float[len];

        for (int i = 0; i < len; i++)
        {
            original[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
            fwdInv[i] = original[i];
            invFwd[i] = original[i];
        }

        // H^T(H(x)) == x
        HadamardTransform.ApplyPaleyInPlace(fwdInv, m);

        // Verify norm preservation
        double origNormSq = 0.0, rotNormSq = 0.0;
        for (int i = 0; i < len; i++)
        {
            origNormSq += (double)original[i] * original[i];
            rotNormSq += (double)fwdInv[i] * fwdInv[i];
        }
        Assert.True(Math.Abs(origNormSq - rotNormSq) / Math.Max(1.0, origNormSq) < 1e-5);

        HadamardTransform.ApplyPaleyInverseInPlace(fwdInv, m);

        // H(H^T(x)) == x
        HadamardTransform.ApplyPaleyInverseInPlace(invFwd, m);
        HadamardTransform.ApplyPaleyInPlace(invFwd, m);

        for (int i = 0; i < len; i++)
        {
            Assert.True(MathF.Abs(fwdInv[i] - original[i]) < 1e-4f,
                $"Paley H^T(H(x)) mismatch at {i} (m={m}): got {fwdInv[i]}, expected {original[i]}");
            Assert.True(MathF.Abs(invFwd[i] - original[i]) < 1e-4f,
                $"Paley H(H^T(x)) mismatch at {i} (m={m}): got {invFwd[i]}, expected {original[i]}");
        }
    }

    [Theory]
    [InlineData(12, 16)]  // 192
    [InlineData(20, 32)]  // 640
    [InlineData(68, 16)]  // 1088
    public void KroneckerPaley_FullSlice_RoundtripIsOrthogonal(int m, int run)
    {
        int len = m * run;
        var rng = new Random(8080 + m + run);
        float[] original = new float[len];
        float[] work = new float[len];
        for (int i = 0; i < len; i++)
        {
            original[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            work[i] = original[i];
        }

        HadamardTransform.ApplyKroneckerPaleyInPlace(work, m);
        HadamardTransform.ApplyKroneckerPaleyInverseInPlace(work, m);

        float maxErr = 0f;
        for (int i = 0; i < len; i++)
        {
            float err = MathF.Abs(work[i] - original[i]);
            if (err > maxErr) maxErr = err;
        }
        Assert.True(maxErr < 1e-4f, $"Kronecker Paley roundtrip maxErr={maxErr} for m={m}, run={run}");
    }

    // =========================================================================
    // 3. RANDOMIZED HADAMARD ROTATION (QuaRot / SpinQuant / QuIP# / NariQ)
    // =========================================================================

    [Theory]
    [InlineData(64, 256, true)]
    [InlineData(64, 256, false)]
    [InlineData(256, 1024, true)]
    [InlineData(256, 1024, false)]
    public void RandomizedRotation_RoundtripAndOutlierSpreading(int blockLen, int totalLen, bool perElementSigns)
    {
        float[] signs = new float[perElementSigns ? totalLen : blockLen];
        HadamardTransform.GenerateRandomSigns(signs, seed: 0xC0FFEE_12345678UL);

        // Verify all signs are strictly +1 or -1
        for (int i = 0; i < signs.Length; i++)
        {
            Assert.True(signs[i] == 1.0f || signs[i] == -1.0f);
        }

        // Test outlier suppression: single spike of magnitude 100 spreads to 100 / sqrt(blockLen)
        float[] spike = new float[totalLen];
        spike[0] = 100.0f;
        HadamardTransform.ApplyRandomizedRotation(spike, signs, blockLen);

        float expectedSpread = 100.0f / MathF.Sqrt(blockLen);
        for (int i = 0; i < blockLen; i++)
        {
            Assert.True(MathF.Abs(MathF.Abs(spike[i]) - expectedSpread) < 1e-4f,
                $"Expected uniform spread {expectedSpread} at {i}, got {spike[i]}");
        }

        // Exact inverse recovers the spike
        HadamardTransform.ApplyRandomizedRotationInverse(spike, signs, blockLen);
        Assert.True(MathF.Abs(spike[0] - 100.0f) < 1e-3f);
        for (int i = 1; i < totalLen; i++)
        {
            Assert.True(MathF.Abs(spike[i]) < 1e-4f);
        }
    }

    // =========================================================================
    // 4. NARIQ TRELLIS CODEC TESTS (TRL1..TRL6, SWAR vs SCALAR, SIMD DOT, RMSE)
    // =========================================================================

    [Fact]
    public void NariQ_SwarCodebook_MatchesScalarExhaustivelyForAll65536States()
    {
        uint[] packedLut = NariQTrellisCodec.BuildPackedInt8Lut();
        Assert.Equal(16384, packedLut.Length);

        sbyte minVal = sbyte.MaxValue;
        sbyte maxVal = sbyte.MinValue;

        for (uint state = 0; state < 65536; state++)
        {
            sbyte swar = NariQTrellisCodec.DecodeStateSwar(state);
            sbyte scalar = NariQTrellisCodec.DecodeStateScalar(state);
            sbyte fromLut = unchecked((sbyte)(packedLut[state >> 2] >> (int)((state & 3u) << 3)));

            Assert.Equal(scalar, swar);
            Assert.Equal(scalar, fromLut);
            Assert.InRange(swar, (sbyte)-126, (sbyte)126);

            if (swar < minVal) minVal = swar;
            if (swar > maxVal) maxVal = swar;
        }

        Assert.True(minVal >= -126 && maxVal <= 126);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void NariQ_DecodeTileInt8_SwarMatchesScalarBitExactly_ForTRL1ToTRL6(int bits)
    {
        int nw = NariQTrellisCodec.GetTileWordCount(bits);
        Assert.Equal(8 * bits, nw);
        Assert.Equal(32 * bits, NariQTrellisCodec.GetTileByteCount(bits));

        var rng = new Random(202600 + bits);
        uint[] words = new uint[nw];
        sbyte[] swarOut = new sbyte[NariQTrellisCodec.TileElements];
        sbyte[] scalarOut = new sbyte[NariQTrellisCodec.TileElements];

        for (int trial = 0; trial < 8; trial++)
        {
            for (int w = 0; w < nw; w++)
                words[w] = (uint)rng.Next() ^ ((uint)rng.Next() << 16);

            NariQTrellisCodec.DecodeTileInt8(words, bits, swarOut);
            NariQTrellisCodec.DecodeTileInt8Scalar(words, bits, scalarOut);

            Assert.Equal(scalarOut, swarOut);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void NariQ_TileMatVecAdd_SimdMatchesScalar_ForTRL1ToTRL6(int bits)
    {
        int nw = NariQTrellisCodec.GetTileWordCount(bits);
        var rng = new Random(55500 + bits);

        uint[] words = new uint[nw];
        for (int w = 0; w < nw; w++)
            words[w] = (uint)rng.Next() ^ ((uint)rng.Next() << 16);

        float[] x16 = new float[16];
        for (int i = 0; i < 16; i++)
            x16[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float scale = 0.015625f; // exact power of 2
        float[] ySimd = new float[16];
        float[] yScalar = new float[16];

        NariQTrellisCodec.TileMatVecAdd(words, bits, scale, x16, ySimd);
        NariQTrellisCodec.TileMatVecAddScalar(words, bits, scale, x16, yScalar);

        for (int r = 0; r < 16; r++)
        {
            Assert.True(MathF.Abs(ySimd[r] - yScalar[r]) < 1e-4f,
                $"TRL{bits} row {r} mismatch: simd={ySimd[r]}, scalar={yScalar[r]}");
        }

        // Also check DotTileRowSimd directly against decoded int8 row
        sbyte[] qTile = new sbyte[256];
        NariQTrellisCodec.DecodeTileInt8(words, bits, qTile);
        for (int r = 0; r < 16; r++)
        {
            float dotSimd = NariQTrellisCodec.DotTileRowSimd(qTile.AsSpan(r * 16, 16), x16);
            Assert.True(MathF.Abs(dotSimd * scale - yScalar[r]) < 1e-4f);
        }
    }

    [Fact]
    public void NariQ_DecodeRows_MatchesIndividualTileDecodeWithFp16Scales()
    {
        const int bits = 4;
        const int k = 32;
        const int nRows = 32;
        int kt = k / 16;
        int nt = nRows / 16;
        int nw = NariQTrellisCodec.GetTileWordCount(bits);

        var rng = new Random(9999);
        uint[] allWords = new uint[nt * kt * nw];
        for (int i = 0; i < allWords.Length; i++)
            allWords[i] = (uint)rng.Next() ^ ((uint)rng.Next() << 16);

        Half[] scales = new Half[] { (Half)0.025f, (Half)0.05f };
        float[] fullOut = new float[nRows * k];

        NariQTrellisCodec.DecodeRows(
            MemoryMarshal.AsBytes(allWords.AsSpan()),
            bits, k, n0: 0, nRows: nRows,
            scales, fullOut);

        float[] singleTile = new float[256];
        for (int j = 0; j < nt; j++)
        {
            for (int i = 0; i < kt; i++)
            {
                ReadOnlySpan<uint> tileWords = allWords.AsSpan((j * kt + i) * nw, nw);
                NariQTrellisCodec.DecodeTileFloat(tileWords, bits, (float)scales[i], singleTile);

                for (int r = 0; r < 16; r++)
                {
                    for (int c = 0; c < 16; c++)
                    {
                        float expected = singleTile[r * 16 + c];
                        float actual = fullOut[(j * 16 + r) * k + (i * 16 + c)];
                        Assert.Equal(expected, actual);
                    }
                }
            }
        }
    }

    [Fact]
    public void NariQ_MeasureTileRmseAndInspectTile_ComputesExactMetrics()
    {
        const int bits = 3;
        int nw = NariQTrellisCodec.GetTileWordCount(bits);
        var rng = new Random(314159);
        uint[] words = new uint[nw];
        for (int i = 0; i < nw; i++)
            words[i] = (uint)rng.Next() ^ ((uint)rng.Next() << 16);

        const float scale = 0.01f;
        float[] decoded = new float[256];
        NariQTrellisCodec.DecodeTileFloat(words, bits, scale, decoded);

        // Zero error against exact decoded reference
        double zeroRmse = NariQTrellisCodec.MeasureTileRmse(decoded, words, bits, scale);
        Assert.Equal(0.0, zeroRmse, 12);

        // Uniform +0.05f perturbation -> exact RMSE == 0.05, MAE == 0.05, MaxAbsError == 0.05
        float[] perturbed = new float[256];
        for (int i = 0; i < 256; i++) perturbed[i] = decoded[i] + 0.05f;

        double rmse = NariQTrellisCodec.MeasureTileRmse(perturbed, words, bits, scale);
        TileErrorMetrics metrics = NariQTrellisCodec.InspectTile(perturbed, words, bits, scale);

        Assert.InRange(rmse, 0.04999, 0.05001);
        Assert.InRange(metrics.Rmse, 0.04999, 0.05001);
        Assert.InRange(metrics.Mae, 0.04999, 0.05001);
        Assert.InRange(metrics.MaxAbsError, 0.0499f, 0.0501f);
        Assert.True(double.IsFinite(metrics.SignalToNoiseDb));
        Assert.True(metrics.MinCode >= -126 && metrics.MaxCode <= 126 && metrics.MinCode < metrics.MaxCode);
    }

    [Fact]
    public void HadamardAndNariQ_InvalidArguments_ThrowExpectedExceptions()
    {
        float[] buf = new float[20];
        Assert.Throws<ArgumentOutOfRangeException>(() => HadamardTransform.ApplySylvesterInPlace(buf, 12)); // not pow2
        Assert.Throws<ArgumentException>(() => HadamardTransform.ApplySylvesterInPlace(buf, 16)); // 20 % 16 != 0
        Assert.Throws<ArgumentOutOfRangeException>(() => HadamardTransform.ApplyPaleyInPlace(buf, 16)); // unsupported m
        Assert.Throws<ArgumentException>(() => HadamardTransform.ApplyPaleyInPlace(buf, 12)); // 20 % 12 != 0

        Assert.Throws<ArgumentOutOfRangeException>(() => NariQTrellisCodec.GetTileWordCount(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => NariQTrellisCodec.GetTileWordCount(7));
    }
}
