namespace Glacier.Quant.Tests.Kernels;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Quant.Calibration;
using Glacier.Quant.Formats;
using Glacier.Quant.Kernels;
using Xunit;

public class AdversarialQuantStressTests
{
    // =========================================================================
    // 1. Q8_0 STRESS & ADVERSARIAL TESTS
    // =========================================================================

    [Fact]
    public void Q8_0_AllZeros_QuantizesToZeroScale_WithoutNaNOrCrash()
    {
        float[] zeros = new float[BlockQ8_0.BlockSize * 4];
        var blocks = new BlockQ8_0[4];

        SimdQuantizer.QuantizeRowQ8_0(zeros, blocks);

        unsafe
        {
            fixed (BlockQ8_0* pb = blocks)
            {
                for (int b = 0; b < blocks.Length; b++)
                {
                    Assert.Equal((Half)0.0f, pb[b].Scale);
                    for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                    {
                        Assert.Equal(0, pb[b].Quants[i]);
                    }
                }
            }
        }

        float[] dequant = new float[zeros.Length];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.Equal(0.0f, dequant[i]);
            Assert.False(float.IsNaN(dequant[i]));
        }
    }

    [Fact]
    public void Q8_0_SubnormalFloats_HandledCleanly()
    {
        // Subnormal values near float.Epsilon (~1.4e-45f)
        float[] subnormals = new float[BlockQ8_0.BlockSize];
        for (int i = 0; i < subnormals.Length; i++)
        {
            subnormals[i] = 1e-40f * (i % 2 == 0 ? 1.0f : -1.0f);
        }

        var blocks = new BlockQ8_0[1];
        SimdQuantizer.QuantizeRowQ8_0(subnormals, blocks);

        // amax <= 1e-12f branch triggers, scale is 0.0f
        Assert.Equal((Half)0.0f, blocks[0].Scale);

        float[] dequant = new float[subnormals.Length];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.Equal(0.0f, dequant[i]);
        }
    }

    [Fact]
    public void Q8_0_ExtremeScaleDifferences_SurvivesWithoutOverflow()
    {
        float[] extreme = new float[BlockQ8_0.BlockSize];
        extreme[0] = 65000.0f; // near Half.MaxValue (65504)
        extreme[1] = -65000.0f;
        for (int i = 2; i < extreme.Length; i++)
        {
            extreme[i] = 1e-10f;
        }

        var blocks = new BlockQ8_0[1];
        SimdQuantizer.QuantizeRowQ8_0(extreme, blocks);

        float scale = (float)blocks[0].Scale;
        Assert.False(float.IsInfinity(scale));
        Assert.False(float.IsNaN(scale));
        Assert.True(scale > 0.0f);

        float[] dequant = new float[extreme.Length];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequant);

        Assert.False(float.IsNaN(dequant[0]));
        Assert.True(MathF.Abs(dequant[0] - 65000.0f) < 600.0f);
    }

    // =========================================================================
    // 2. Q4_K_M SUPER-BLOCK STRESS & ADVERSARIAL TESTS
    // =========================================================================

    [Fact]
    public void Q4_K_AllZeros_ProducesFiniteD_AndZeroDequant()
    {
        float[] zeros = new float[BlockQ4_K.BlockSize];
        var blocks = new BlockQ4_K[1];

        SimdQuantizer.QuantizeRowQ4_K(zeros, blocks);

        float d = (float)blocks[0].D;
        float dmin = (float)blocks[0].DMin;

        Assert.False(float.IsNaN(d));
        Assert.False(float.IsInfinity(d));
        Assert.False(float.IsNaN(dmin));
        Assert.False(float.IsInfinity(dmin));

        float[] dequant = new float[zeros.Length];
        SimdQuantizer.DequantizeRowQ4_K(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.True(MathF.Abs(dequant[i]) < 1e-4f, $"Dequantized element {i} was {dequant[i]}");
        }
    }

    [Fact]
    public void Q4_K_SubnormalsAndExtremeDynamicRange()
    {
        float[] vals = new float[BlockQ4_K.BlockSize];
        vals[0] = -500.0f;
        vals[1] = 500.0f;
        for (int i = 2; i < vals.Length; i++)
        {
            vals[i] = 1e-35f;
        }

        var blocks = new BlockQ4_K[1];
        SimdQuantizer.QuantizeRowQ4_K(vals, blocks);

        float[] dequant = new float[vals.Length];
        SimdQuantizer.DequantizeRowQ4_K(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.False(float.IsInfinity(dequant[i]));
        }
        // Check extrema reconstruction
        Assert.True(MathF.Abs(dequant[0] - (-500.0f)) < 40.0f);
        Assert.True(MathF.Abs(dequant[1] - 500.0f) < 40.0f);
    }

    [Fact]
    public void Q8_0_NaNsHandledGracefully_WithoutCrashing()
    {
        float[] withNans = new float[BlockQ8_0.BlockSize];
        Array.Fill(withNans, float.NaN);

        var blocks = new BlockQ8_0[1];
        SimdQuantizer.QuantizeRowQ8_0(withNans, blocks);

        // amax stays 0.0f, scale is 0.0f
        Assert.Equal((Half)0.0f, blocks[0].Scale);

        float[] dequant = new float[withNans.Length];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.Equal(0.0f, dequant[i]);
        }
    }

    [Fact]
    public void Q4_K_NaNsHandledGracefully_WithoutCrashing()
    {
        float[] withNans = new float[BlockQ4_K.BlockSize];
        withNans[0] = 10.0f;
        withNans[1] = -10.0f;
        for (int i = 2; i < withNans.Length; i++)
        {
            withNans[i] = float.NaN;
        }

        var blocks = new BlockQ4_K[1];
        SimdQuantizer.QuantizeRowQ4_K(withNans, blocks);

        float[] dequant = new float[withNans.Length];
        SimdQuantizer.DequantizeRowQ4_K(blocks, dequant);

        // Does not throw, non-NaN elements are reconstructed
        Assert.True(MathF.Abs(dequant[0] - 10.0f) < 2.0f);
        Assert.True(MathF.Abs(dequant[1] - (-10.0f)) < 2.0f);
    }

    // =========================================================================
    // 3. FP8_E4M3 STRESS & ADVERSARIAL TESTS
    // =========================================================================

    [Fact]
    public void FP8_E4M3_AllZeros_DynamicScaleIsOne_QuantsZero()
    {
        float[] zeros = new float[128];
        byte[] quants = new byte[128];

        SimdQuantizer.QuantizeRowFP8_E4M3(zeros, quants, out float scale);

        Assert.Equal(1.0f, scale);
        for (int i = 0; i < quants.Length; i++)
        {
            Assert.Equal(0, quants[i]);
        }

        float[] dequant = new float[128];
        SimdQuantizer.DequantizeRowFP8_E4M3(quants, scale, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.Equal(0.0f, dequant[i]);
        }
    }

    [Fact]
    public void FP8_E4M3_NaNAndInfinityHandling()
    {
        // NaN should convert to 0x7F
        byte nanFp8 = SimdQuantizer.FloatToFp8E4M3(float.NaN);
        Assert.Equal(0x7F, nanFp8);
        float dequantNan = SimdQuantizer.Fp8E4M3ToFloat(nanFp8);
        Assert.True(float.IsNaN(dequantNan));

        // +Infinity saturates to max E4M3 (0x7E)
        byte infFp8 = SimdQuantizer.FloatToFp8E4M3(float.PositiveInfinity);
        Assert.Equal(0x7E, infFp8);
        float dequantInf = SimdQuantizer.Fp8E4M3ToFloat(infFp8);
        Assert.False(float.IsInfinity(dequantInf));
        Assert.True(dequantInf > 400.0f);

        // -Infinity saturates to negative max E4M3 (0xFE)
        byte negInfFp8 = SimdQuantizer.FloatToFp8E4M3(float.NegativeInfinity);
        Assert.Equal(0xFE, negInfFp8);
        float dequantNegInf = SimdQuantizer.Fp8E4M3ToFloat(negInfFp8);
        Assert.True(dequantNegInf < -400.0f);
    }

    [Fact]
    public void FP8_E4M3_SubnormalUnderflow()
    {
        // Subnormals smaller than 2^-6 underflow to 0
        float tiny = 1e-10f;
        byte tinyFp8 = SimdQuantizer.FloatToFp8E4M3(tiny);
        Assert.Equal(0, tinyFp8);
        Assert.Equal(0.0f, SimdQuantizer.Fp8E4M3ToFloat(tinyFp8));
    }

    // =========================================================================
    // 4. 1-BIT BINARY QUANTIZATION STRESS TESTS
    // =========================================================================

    [Fact]
    public void Binary1Bit_AllZeros_ScaleClampedTo1e12_NoDivideByZero()
    {
        float[] zeros = new float[BlockBinary1Bit.BlockSize];
        var blocks = new BlockBinary1Bit[1];

        SimdQuantizer.QuantizeRow1Bit(zeros, blocks);

        Assert.Equal(0.0f, blocks[0].Bias);
        Assert.Equal(1e-12f, blocks[0].Scale);
        Assert.Equal(ulong.MaxValue, blocks[0].Bits); // 0.0f >= 0.0f sets bit

        float[] dequant = new float[zeros.Length];
        SimdQuantizer.DequantizeRow1Bit(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.False(float.IsNaN(dequant[i]));
            Assert.True(MathF.Abs(dequant[i]) < 1e-10f);
        }
    }

    [Fact]
    public void Binary1Bit_ConstantValues_CorrectBiasAndClampedScale()
    {
        float[] constVals = new float[BlockBinary1Bit.BlockSize];
        Array.Fill(constVals, 42.0f);

        var blocks = new BlockBinary1Bit[1];
        SimdQuantizer.QuantizeRow1Bit(constVals, blocks);

        Assert.Equal(42.0f, blocks[0].Bias);
        Assert.Equal(1e-12f, blocks[0].Scale);

        float[] dequant = new float[constVals.Length];
        SimdQuantizer.DequantizeRow1Bit(blocks, dequant);

        for (int i = 0; i < dequant.Length; i++)
        {
            Assert.True(MathF.Abs(dequant[i] - 42.0f) < 1e-5f);
        }
    }

    // =========================================================================
    // 5. GGUF V3 ALIGNMENT & NATIVEAOT TESTS
    // =========================================================================

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    public void GgufWriter_EnforcesExactAlignmentBoundary_ForConfiguredAlignments(int alignment)
    {
        var metadata = new Dictionary<string, string>
        {
            ["test.key"] = "test.value"
        };

        // 3 tensors of irregular sizes
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "t1", Shape = [17], Type = QuantType.FP32 }, new byte[17]),
            (new QuantTensorMetadata { Name = "t2", Shape = [33], Type = QuantType.Q8_0 }, new byte[33]),
            (new QuantTensorMetadata { Name = "t3", Shape = [7], Type = QuantType.FP16 }, new byte[7])
        };

        using var ms = new MemoryStream();
        GgufWriter.WriteToStream(ms, metadata, tensors, alignment);

        ms.Position = 0;
        var (readMeta, readTensors, dataStart) = GgufReader.ReadHeader(ms);

        // Header padding must align dataStart to alignment boundary
        Assert.Equal(0L, dataStart % 32); // Default header reader aligns to 32
        Assert.Equal(3, readTensors.Count);

        for (int i = 0; i < readTensors.Count; i++)
        {
            Assert.Equal(0UL, readTensors[i].DataOffset % (ulong)alignment);
        }
    }

    [Fact]
    public void GgufWriter_EmptyTensorsAndMetadata_RoundtripsSuccessfully()
    {
        var metadata = new Dictionary<string, string>();
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>();

        using var ms = new MemoryStream();
        GgufWriter.WriteToStream(ms, metadata, tensors, alignment: 32);

        ms.Position = 0;
        var (readMeta, readTensors, dataStart) = GgufReader.ReadHeader(ms);

        Assert.Empty(readMeta);
        Assert.Empty(readTensors);
        Assert.True(dataStart >= 16);
    }

    [Fact]
    public void SafetensorsWriter_EmptyTensors_WritesValidJsonHeaderPrefix()
    {
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>();

        using var ms = new MemoryStream();
        SafetensorsWriter.WriteToStream(ms, tensors);

        ms.Position = 0;
        var (readDict, dataStart) = SafetensorsReader.ReadHeader(ms);

        Assert.Empty(readDict);
        Assert.True(dataStart > 8);
    }

    // =========================================================================
    // 6. PTQ CALIBRATION EDGE CASES (SINGULAR HESSIAN, EMPTY DATASETS)
    // =========================================================================

    [Fact]
    public void CalibrationDataset_ZeroOrNegativeArguments_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CalibrationDataset(0, 16, new float[16]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CalibrationDataset(-1, 16, new float[16]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CalibrationDataset(10, 0, new float[16]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CalibrationDataset(10, -5, new float[16]));
        Assert.Throws<ArgumentNullException>(() => new CalibrationDataset(10, 16, null!));
        Assert.Throws<ArgumentException>(() => new CalibrationDataset(10, 16, new float[10])); // too small
    }

    [Fact]
    public void GptqCalibrator_SingularAllZeroActivations_DoesNotCrashOrProduceNaN()
    {
        int rows = 4;
        int cols = 8;
        int tokens = 5;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (i + 1) * 0.1f;

        // All zero activations -> empirical Hessian X^T * X is all zeros
        float[] zeroActs = new float[tokens * cols];
        var dataset = new CalibrationDataset(tokens, cols, zeroActs);

        // Due to diagonal damping lambda, Hessian is positive definite (lambda * I)
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.NotNull(result);
        Assert.Equal(weights.Length, result.CalibratedWeights.Length);
        for (int i = 0; i < result.CalibratedWeights.Length; i++)
        {
            Assert.False(float.IsNaN(result.CalibratedWeights[i]), $"Calibrated weight {i} is NaN");
            Assert.False(float.IsInfinity(result.CalibratedWeights[i]), $"Calibrated weight {i} is Infinity");
        }
        Assert.False(double.IsNaN(result.CalibratedMse));
    }

    [Fact]
    public void GptqCalibrator_RankDeficientUnderDeterminedActivations_InvertsStably()
    {
        // 1 token for 16 dimensions (rank at most 1, heavily singular)
        int rows = 4;
        int cols = 16;
        int tokens = 1;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = MathF.Sin(i);

        float[] acts = new float[tokens * cols];
        Array.Fill(acts, 1.0f); // rank-1 collinear vector

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = GptqCalibrator.Calibrate(weights, rows, cols, dataset, damping: 0.05f);

        Assert.NotNull(result);
        for (int i = 0; i < result.CalibratedWeights.Length; i++)
        {
            Assert.False(float.IsNaN(result.CalibratedWeights[i]));
            Assert.False(float.IsInfinity(result.CalibratedWeights[i]));
        }
    }

    [Fact]
    public void AwqCalibrator_AllZeroActivations_FallsBackGracefullyWithoutNaN()
    {
        int rows = 8;
        int cols = 16;
        int tokens = 10;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (i % 5) * 0.2f - 0.4f;

        float[] zeroActs = new float[tokens * cols];
        var dataset = new CalibrationDataset(tokens, cols, zeroActs);

        var result = AwqCalibrator.Calibrate(weights, rows, cols, dataset, gridSteps: 5);

        Assert.NotNull(result);
        Assert.NotNull(result.OptimalScales);
        for (int i = 0; i < result.OptimalScales.Length; i++)
        {
            Assert.False(float.IsNaN(result.OptimalScales[i]));
            Assert.False(float.IsInfinity(result.OptimalScales[i]));
        }
        for (int i = 0; i < result.CalibratedWeights.Length; i++)
        {
            Assert.False(float.IsNaN(result.CalibratedWeights[i]));
        }
    }

    [Fact]
    public void ImatrixCalibrator_AllZeroActivations_NormalizesWithoutDivideByZero()
    {
        int rows = 4;
        int cols = 32;
        int tokens = 10;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = MathF.Sin(i * 0.1f);

        float[] zeroActs = new float[tokens * cols];
        var dataset = new CalibrationDataset(tokens, cols, zeroActs);

        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.NotNull(result);
        Assert.NotNull(result.OptimalScales);
        for (int c = 0; c < cols; c++)
        {
            Assert.False(float.IsNaN(result.OptimalScales[c]));
            Assert.False(float.IsInfinity(result.OptimalScales[c]));
        }
        for (int i = 0; i < result.CalibratedWeights.Length; i++)
        {
            Assert.False(float.IsNaN(result.CalibratedWeights[i]));
        }
    }

    [Fact]
    public void Q4_K_AllNaNs_DoesNotCrashOrThrow()
    {
        float[] allNans = new float[BlockQ4_K.BlockSize];
        Array.Fill(allNans, float.NaN);

        var blocks = new BlockQ4_K[1];
        // Must not crash with illegal memory or uncaught exception
        SimdQuantizer.QuantizeRowQ4_K(allNans, blocks);

        float[] dequant = new float[allNans.Length];
        SimdQuantizer.DequantizeRowQ4_K(blocks, dequant);

        Assert.Equal(BlockQ4_K.BlockSize, dequant.Length);
    }

    [Fact]
    public void Q8_0_AllInfinities_DoesNotCrash()
    {
        float[] allInf = new float[BlockQ8_0.BlockSize];
        Array.Fill(allInf, float.PositiveInfinity);

        var blocks = new BlockQ8_0[1];
        SimdQuantizer.QuantizeRowQ8_0(allInf, blocks);

        float[] dequant = new float[allInf.Length];
        SimdQuantizer.DequantizeRowQ8_0(blocks, dequant);

        Assert.Equal(BlockQ8_0.BlockSize, dequant.Length);
    }

    [Fact]
    public void FP8_E4M3_All256ByteValues_ConvertWithoutCrashing()
    {
        for (int b = 0; b <= 255; b++)
        {
            byte val = (byte)b;
            float f = SimdQuantizer.Fp8E4M3ToFloat(val);
            if (!float.IsNaN(f))
            {
                byte reQuant = SimdQuantizer.FloatToFp8E4M3(f);
                Assert.True(reQuant <= 255);
            }
        }
    }

    [Fact]
    public void SafetensorsReader_CorruptedJson_ThrowsJsonException()
    {
        byte[] badJson = "{\"not_a_valid_json"u8.ToArray();
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes((ulong)badJson.Length));
        ms.Write(badJson);
        ms.Position = 0;

        Assert.ThrowsAny<System.Text.Json.JsonException>(() => SafetensorsReader.ReadHeader(ms));
    }

    [Fact]
    public void SafetensorsReader_TruncatedStream_ThrowsEndOfStreamException()
    {
        byte[] truncated = [1, 2, 3]; // less than 8 bytes
        using var ms = new MemoryStream(truncated);
        Assert.Throws<EndOfStreamException>(() => SafetensorsReader.ReadHeader(ms));
    }

    [Fact]
    public void SafetensorsReader_Exceeds100MBLimit_ThrowsInvalidDataException()
    {
        using var ms = new MemoryStream();
        ulong hugeHeader = 101 * 1024 * 1024UL;
        ms.Write(BitConverter.GetBytes(hugeHeader));
        ms.Position = 0;

        Assert.Throws<InvalidDataException>(() => SafetensorsReader.ReadHeader(ms));
    }

    [Fact]
    public void GgufReader_InvalidMagic_ThrowsInvalidDataException()
    {
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(0xDEADBEEFU)); // bad magic
        ms.Write(BitConverter.GetBytes(3U));
        ms.Position = 0;

        Assert.Throws<InvalidDataException>(() => GgufReader.ReadHeader(ms));
    }

    [Fact]
    public void GgufReader_UnsupportedVersion_ThrowsInvalidDataException()
    {
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(GgufWriter.GgufMagic));
        ms.Write(BitConverter.GetBytes(99U)); // unsupported version 99
        ms.Position = 0;

        Assert.Throws<InvalidDataException>(() => GgufReader.ReadHeader(ms));
    }

    [Fact]
    public void ImatrixCalibrator_NonMultipleOf32Cols_ExecutesCleanly()
    {
        int rows = 4;
        int cols = 47; // prime / non-multiple of 32
        int tokens = 5;

        float[] weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (i + 1) * 0.05f;

        float[] acts = new float[tokens * cols];
        for (int i = 0; i < acts.Length; i++) acts[i] = (i % 3) * 0.1f;

        var dataset = new CalibrationDataset(tokens, cols, acts);
        var result = ImatrixCalibrator.Calibrate(weights, rows, cols, dataset);

        Assert.NotNull(result);
        Assert.Equal(weights.Length, result.CalibratedWeights.Length);
        Assert.Equal(cols, result.OptimalScales!.Length);
    }
}

