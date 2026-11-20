namespace Glacier.Quant.Tests.E2E;

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Glacier.Quant.Calibration;
using Glacier.Quant.Evaluation;
using Glacier.Quant.Gpu;
using Glacier.Quant.Kernels;
using Xunit;

[Trait("Suite", "UpgradesE2E")]
public sealed class QuantQualityGateE2ETests
{
    [Fact]
    [Trait("Tier", "Tier3")]
    public void T3_17_QuantKlGateAndSecurityAuditOnIsolatedArtifactDirectory()
    {
        string? repoRoot = TryFindRepoRoot();
        string? scannerPath = repoRoot is not null ? Path.Combine(repoRoot, "scripts", "security_check.py") : null;

        const int numTokens = 6;
        const int vocabSize = 8;
        const int hiddenDim = 256;
        (float[] weights, float[] activations, float[] bias) = BuildSyntheticHead(numTokens, vocabSize, hiddenDim);
        var dataset = new CalibrationDataset(numTokens, hiddenDim, activations);

        KlDivergenceResult gateResult = QuantQualityChecker.EvaluateAwqCalibrated(
            weights,
            vocabSize,
            dataset,
            targetType: QuantType.Q8_0,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 1.0 },
            bias: bias);

        Assert.True(gateResult.Passed, gateResult.FailureReason);

        if (scannerPath is null || !File.Exists(scannerPath))
            return;

        string tempRoot = Path.Combine(Path.GetTempPath(), "glacier_quant_t3_17_" + Guid.NewGuid().ToString("N"));
        string cleanDir = Path.Combine(tempRoot, "clean_artifacts");
        string leakDir = Path.Combine(tempRoot, "negative_probe");
        Directory.CreateDirectory(cleanDir);
        Directory.CreateDirectory(leakDir);

        try
        {
            // Write KL quality report & sanitized template config (env-var only)
            string reportJson = JsonSerializer.Serialize(gateResult, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(cleanDir, "kl_quality_report.json"), reportJson);
            File.WriteAllText(
                Path.Combine(cleanDir, "config_sample.cs"),
                "var apiKey = Environment.GetEnvironmentVariable(\"GEMINI_API_KEY\");\n");

            (int cleanExit, string cleanOut) = RunPythonScript(scannerPath, cleanDir);
            Assert.Equal(0, cleanExit);
            Assert.Contains("[PASSED]", cleanOut, StringComparison.Ordinal);

            // Isolated negative fixture constructed dynamically at runtime outside repo tree
            string syntheticPattern = string.Concat("sk", "-", new string('A', 22), "99");
            string leakyFile = Path.Combine(leakDir, "leaky.json");
            File.WriteAllText(leakyFile, string.Concat("{\"token\": \"", syntheticPattern, "\"}\n"));

            (int leakExit, string leakOut) = RunPythonScript(scannerPath, leakDir);
            Assert.NotEqual(0, leakExit);
            Assert.Contains("leaky.json", leakOut, StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
                Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    [Trait("Tier", "Tier4")]
    public void T4_04_PreReleaseQuantizationQualityGate_Q8BaselineVsFaithfulAndRegressedCandidates()
    {
        const int numTokens = 12;
        const int vocabSize = 16;
        const int hiddenDim = 256;
        (float[] weights, float[] activations, float[] bias) = BuildSyntheticHead(numTokens, vocabSize, hiddenDim);

        // Inject salient outlier activation channels with smaller corresponding weights
        // (classic AWQ regime: small weights multiplied by large activations suffer in RTN 4-bit quantization)
        for (int r = 0; r < vocabSize; r++)
        {
            weights[r * hiddenDim + 0] *= 0.20f;
            weights[r * hiddenDim + 17] *= 0.25f;
        }
        for (int t = 0; t < numTokens; t++)
        {
            activations[t * hiddenDim + 0] *= 20.0f;
            activations[t * hiddenDim + 17] *= 16.0f;
        }

        var dataset = new CalibrationDataset(numTokens, hiddenDim, activations);

        // 1. Q8_0 vs FP32 baseline (non-zero quantization divergence) and Q8_0 vs Q8_0 (exact 0.0)
        KlDivergenceResult q8Result = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            referenceType: QuantType.FP32,
            options: new KlDivergenceOptions { Threshold = 0.01, MaxKlThreshold = 0.02, MinTop1Agreement = 1.0 },
            bias: bias);
        Assert.True(q8Result.Passed, q8Result.FailureReason);
        Assert.True(q8Result.MeanKlDivergence > 0.0);
        Assert.Equal(1.0, q8Result.Top1AgreementRate, precision: 12);

        KlDivergenceResult q8SelfRefResult = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q8_0,
            dataset,
            vocabSize,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.01, MaxKlThreshold = 0.02, MinTop1Agreement = 1.0 },
            bias: bias);
        Assert.True(q8SelfRefResult.Passed, q8SelfRefResult.FailureReason);
        Assert.Equal(0.0, q8SelfRefResult.MeanKlDivergence, precision: 12);

        // 2. Faithful FP8_E4M3 vs Q8_0 reference
        KlDivergenceResult fp8Result = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.FP8_E4M3,
            dataset,
            vocabSize,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.90 },
            bias: bias);
        Assert.True(fp8Result.Passed, fp8Result.FailureReason);

        // 3. Faithful Q4_K_M vs Q8_0 reference (4-bit quantization tolerance)
        KlDivergenceResult q4Result = QuantQualityChecker.EvaluateQuantization(
            weights,
            QuantType.Q4_K_M,
            dataset,
            vocabSize,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.25, MinTop1Agreement = 0.80 },
            bias: bias);
        Assert.True(q4Result.Passed, q4Result.FailureReason);

        // 4. AWQ calibrated Q4_K_M candidate vs Q8_0 reference — must apply OptimalScales
        PtqCalibrationResult awqPtq = AwqCalibrator.Calibrate(weights, vocabSize, hiddenDim, dataset, gridSteps: 20, targetBits: 4);
        Assert.StartsWith("AWQ", awqPtq.Method, StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(awqPtq.OptimalScales);
        Assert.Contains(awqPtq.OptimalScales!, s => MathF.Abs(s - 1.0f) > 0.05f);

        KlDivergenceResult awqResult = QuantQualityChecker.EvaluatePtqResult(
            weights,
            awqPtq,
            dataset,
            vocabSize,
            postCalibrationQuantType: QuantType.Q4_K_M,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.25, MinTop1Agreement = 0.80 },
            bias: bias);
        Assert.True(awqResult.Passed, awqResult.FailureReason);

        // Same CalibratedWeights without OptimalScales exercises the non-AWQ branch;
        // AWQ channel scaling must produce a distinct (and non-equivalent) KL divergence.
        var awqWithoutScales = new PtqCalibrationResult
        {
            Method = awqPtq.Method,
            CalibratedWeights = awqPtq.CalibratedWeights,
            OptimalScales = null,
            BaselineRtnMse = awqPtq.BaselineRtnMse,
            CalibratedMse = awqPtq.CalibratedMse
        };
        KlDivergenceResult awqNoScaleResult = QuantQualityChecker.EvaluatePtqResult(
            weights,
            awqWithoutScales,
            dataset,
            vocabSize,
            postCalibrationQuantType: QuantType.Q4_K_M,
            referenceType: QuantType.Q8_0,
            options: new KlDivergenceOptions { Threshold = 0.50, MinTop1Agreement = 0.50 },
            bias: bias);
        Assert.NotEqual(awqNoScaleResult.MeanKlDivergence, awqResult.MeanKlDivergence, precision: 6);
        Assert.NotEqual(q4Result.MeanKlDivergence, awqResult.MeanKlDivergence, precision: 6);
        Assert.True(
            awqResult.MeanKlDivergence < q4Result.MeanKlDivergence,
            $"Expected AWQ calibrated Q4_K_M KL ({awqResult.MeanKlDivergence}) < uncalibrated Q4_K_M KL ({q4Result.MeanKlDivergence}).");
        Assert.True(
            awqResult.MeanKlDivergence < awqNoScaleResult.MeanKlDivergence,
            $"Expected AWQ scaled KL ({awqResult.MeanKlDivergence}) < unscaled PTQ KL ({awqNoScaleResult.MeanKlDivergence}).");

        // 5. Regressed candidate: permute vocabulary rows (simulates silent tensor layout/transpose bug)
        float[] regressedWeights = new float[weights.Length];
        for (int r = 0; r < vocabSize; r++)
        {
            int srcRow = (r + 1) % vocabSize;
            Array.Copy(weights, srcRow * hiddenDim, regressedWeights, r * hiddenDim, hiddenDim);
        }

        KlDivergenceResult regressedResult = QuantQualityChecker.EvaluateWeights(
            weights,
            regressedWeights,
            dataset,
            vocabSize,
            new KlDivergenceOptions { Threshold = 0.05, MaxKlThreshold = 0.10, MinTop1Agreement = 0.95 },
            bias);

        Assert.False(regressedResult.Passed);
        Assert.True(regressedResult.MeanKlDivergence > 0.5);
        Assert.True(regressedResult.MaxKlDivergence >= regressedResult.MeanKlDivergence);
        Assert.True(regressedResult.Top1AgreementRate < 0.5);
        Assert.NotNull(regressedResult.FailureReason);
    }

    [Fact]
    [Trait("Tier", "Tier4")]
    public void T4_GpuQuantizationDispatcher_GetQuantizedByteCountAndDequantizeTensorRoundTrip()
    {
        const int numTokens = 4;
        const int vocabSize = 4;
        const int hiddenDim = 256;
        (float[] weights, float[] activations, float[] bias) = BuildSyntheticHead(numTokens, vocabSize, hiddenDim);
        int n = weights.Length;

        Assert.Equal(n * sizeof(float), GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.FP32));
        Assert.Equal((n / BlockQ8_0.BlockSize) * Unsafe.SizeOf<BlockQ8_0>(), GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.Q8_0));
        Assert.Equal((n / BlockQ4_K.BlockSize) * Unsafe.SizeOf<BlockQ4_K>(), GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.Q4_K_M));
        Assert.Equal(n, GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.FP8_E4M3));
        Assert.Equal((n / BlockBinary1Bit.BlockSize) * Unsafe.SizeOf<BlockBinary1Bit>(), GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.Binary1Bit));

        Assert.Throws<NotSupportedException>(() => GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.FP16));
        Assert.Throws<NotSupportedException>(() => GpuQuantizationDispatcher.GetQuantizedByteCount(n, QuantType.FP8_E5M2));
        Assert.Throws<ArgumentException>(() => GpuQuantizationDispatcher.GetQuantizedByteCount(100, QuantType.Q4_K_M));

        // FP32 raw byte buffer round-trip via DequantizeTensor and EvaluateQuantizedBuffer
        byte[] fp32Bytes = MemoryMarshal.AsBytes(weights.AsSpan()).ToArray();
        float[] fp32Out = new float[n];
        GpuQuantizationDispatcher.DequantizeTensor(fp32Bytes, QuantType.FP32, fp32Out);
        Assert.Equal(weights, fp32Out);

        var dataset = new CalibrationDataset(numTokens, hiddenDim, activations);
        KlDivergenceResult fp32BufferResult = QuantQualityChecker.EvaluateQuantizedBuffer(
            weights,
            fp32Bytes,
            QuantType.FP32,
            dataset,
            vocabSize,
            bias: bias);
        Assert.True(fp32BufferResult.Passed);
        Assert.Equal(0.0, fp32BufferResult.MeanKlDivergence, precision: 12);

        // Round-trip QuantizeTensor -> DequantizeTensor -> EvaluateQuantizedBuffer for Q8_0, Q4_K_M, FP8_E4M3, Binary1Bit
        QuantType[] types = [QuantType.Q8_0, QuantType.FP8_E4M3, QuantType.Q4_K_M, QuantType.Binary1Bit];
        foreach (QuantType qt in types)
        {
            int byteCount = GpuQuantizationDispatcher.GetQuantizedByteCount(n, qt);
            byte[] quantized = new byte[byteCount];
            GpuQuantizationDispatcher.QuantizeTensor(weights, qt, quantized, out float dynamicScale);

            float[] dequantized = new float[n];
            GpuQuantizationDispatcher.DequantizeTensor(quantized, qt, dequantized, dynamicScale);
            Assert.All(dequantized, v => Assert.True(float.IsFinite(v)));
            Assert.Contains(dequantized, v => MathF.Abs(v) > 1e-3f);

            // Independently verify DequantizeTensor matches SimdQuantizer reference dequantization
            float[] expectedDequantized = new float[n];
            switch (qt)
            {
                case QuantType.Q8_0:
                    SimdQuantizer.DequantizeRowQ8_0(MemoryMarshal.Cast<byte, BlockQ8_0>(quantized), expectedDequantized);
                    break;
                case QuantType.FP8_E4M3:
                    SimdQuantizer.DequantizeRowFP8_E4M3(quantized, dynamicScale, expectedDequantized);
                    break;
                case QuantType.Q4_K_M:
                    SimdQuantizer.DequantizeRowQ4_K(MemoryMarshal.Cast<byte, BlockQ4_K>(quantized), expectedDequantized);
                    break;
                case QuantType.Binary1Bit:
                    SimdQuantizer.DequantizeRow1Bit(MemoryMarshal.Cast<byte, BlockBinary1Bit>(quantized), expectedDequantized);
                    break;
            }
            Assert.Equal(expectedDequantized, dequantized);

            double threshold = qt switch
            {
                QuantType.Q8_0 => 0.01,
                QuantType.FP8_E4M3 => 0.05,
                QuantType.Q4_K_M => 0.25,
                QuantType.Binary1Bit => 1.50,
                _ => 0.25
            };
            double minTop1 = qt == QuantType.Binary1Bit ? 0.50 : 0.90;

            KlDivergenceResult evalResult = QuantQualityChecker.EvaluateQuantizedBuffer(
                weights,
                quantized,
                qt,
                dataset,
                vocabSize,
                dynamicScale: dynamicScale,
                referenceType: QuantType.FP32,
                options: new KlDivergenceOptions { Threshold = threshold, MinTop1Agreement = minTop1 },
                bias: bias);

            Assert.True(double.IsFinite(evalResult.MeanKlDivergence));
            Assert.True(evalResult.Passed, $"{qt}: {evalResult.FailureReason} (MeanKL={evalResult.MeanKlDivergence:G4}, Top1={evalResult.Top1AgreementRate:P1})");
            if (qt is QuantType.Q8_0 or QuantType.FP8_E4M3)
            {
                Assert.Equal(1.0, evalResult.Top1AgreementRate, precision: 12);
            }
            if (qt == QuantType.Q8_0)
            {
                Assert.True(evalResult.MeanKlDivergence > 0.0);
                KlDivergenceResult q8BufferSelfRef = QuantQualityChecker.EvaluateQuantizedBuffer(
                    weights,
                    quantized,
                    QuantType.Q8_0,
                    dataset,
                    vocabSize,
                    dynamicScale: dynamicScale,
                    referenceType: QuantType.Q8_0,
                    options: new KlDivergenceOptions { Threshold = 0.01, MinTop1Agreement = 1.0 },
                    bias: bias);
                Assert.True(q8BufferSelfRef.Passed, q8BufferSelfRef.FailureReason);
                Assert.Equal(0.0, q8BufferSelfRef.MeanKlDivergence, precision: 12);
            }
        }
    }

    private static (float[] Weights, float[] Activations, float[] Bias) BuildSyntheticHead(
        int numTokens,
        int vocabSize,
        int hiddenDim)
    {
        float[] weights = new float[vocabSize * hiddenDim];
        float[] activations = new float[numTokens * hiddenDim];
        float[] bias = new float[vocabSize];

        for (int r = 0; r < vocabSize; r++)
        {
            bias[r] = (r - vocabSize / 2) * 0.02f;
            for (int c = 0; c < hiddenDim; c++)
            {
                weights[r * hiddenDim + c] = MathF.Sin((r + 1) * 0.19f + c * 0.029f) * 0.22f;
            }
        }

        for (int t = 0; t < numTokens; t++)
        {
            for (int c = 0; c < hiddenDim; c++)
            {
                activations[t * hiddenDim + c] = MathF.Cos((t + 1) * 0.31f + c * 0.017f) * 0.45f;
            }

            int preferredRow = (t * 3 + 1) % vocabSize;
            for (int c = 0; c < hiddenDim; c++)
            {
                weights[preferredRow * hiddenDim + c] += 0.035f * MathF.Sign(activations[t * hiddenDim + c]);
            }
        }

        return (weights, activations, bias);
    }

    private static string? TryFindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "scripts", "security_check.py")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    private static (int ExitCode, string Output) RunPythonScript(string scriptPath, string targetDir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = OperatingSystem.IsWindows() ? "python" : "python3",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        psi.ArgumentList.Add(scriptPath);
        psi.ArgumentList.Add(targetDir);

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start python process.");
        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        return (proc.ExitCode, stdout + "\n" + stderr);
    }
}
