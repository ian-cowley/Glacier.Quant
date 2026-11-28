namespace Glacier.Quant.Tests.E2E;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
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

            (int cleanExit, string cleanOut) = RunPureCSharpSecretAudit(cleanDir);
            Assert.Equal(0, cleanExit);
            Assert.Contains("[PASSED]", cleanOut, StringComparison.Ordinal);

            if (TryRunGlacierCliSecurityCheck(repoRoot, cleanDir) is var (cliCleanExit, cliCleanOut))
            {
                Assert.Equal(0, cliCleanExit);
                Assert.Contains("[PASSED]", cliCleanOut, StringComparison.Ordinal);
            }

            // Isolated negative fixture constructed dynamically at runtime outside repo tree
            string syntheticPattern = string.Concat("sk", "-", new string('A', 22), "99");
            string leakyFile = Path.Combine(leakDir, "leaky.json");
            File.WriteAllText(leakyFile, string.Concat("{\"token\": \"", syntheticPattern, "\"}\n"));

            (int leakExit, string leakOut) = RunPureCSharpSecretAudit(leakDir);
            Assert.NotEqual(0, leakExit);
            Assert.Contains("leaky.json", leakOut, StringComparison.Ordinal);

            if (TryRunGlacierCliSecurityCheck(repoRoot, leakDir) is var (cliLeakExit, cliLeakOut))
            {
                Assert.NotEqual(0, cliLeakExit);
                Assert.Contains("leaky.json", cliLeakOut, StringComparison.Ordinal);
            }
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

        // Honest GPU dispatcher reporting & FP32 QuantizeTensor round-trip
        Assert.False(GpuQuantizationDispatcher.IsGpuAccelerationAvailable);
        Assert.Equal("CPU-SIMD", GpuQuantizationDispatcher.ExecutionBackend);

        byte[] fp32Bytes = MemoryMarshal.AsBytes(weights.AsSpan()).ToArray();
        byte[] fp32RoundTripBytes = new byte[n * sizeof(float)];
        GpuQuantizationDispatcher.QuantizeTensor(weights, QuantType.FP32, fp32RoundTripBytes, out float fp32DynScale);
        Assert.Equal(1.0f, fp32DynScale);
        Assert.Equal(fp32Bytes, fp32RoundTripBytes);
        Assert.Throws<ArgumentException>(() => GpuQuantizationDispatcher.QuantizeTensor(weights, QuantType.FP32, new byte[4], out _));

        Assert.Throws<NotSupportedException>(() =>
        {
            byte[] scratch = new byte[64];
            GpuQuantizationDispatcher.QuantizeTensor(weights.AsSpan(0, 16), QuantType.FP32, scratch, out _, requireGpu: true);
        });
        Assert.Throws<NotSupportedException>(() =>
        {
            float[] scratch = new float[16];
            GpuQuantizationDispatcher.DequantizeTensor(fp32Bytes.AsSpan(0, 64), QuantType.FP32, scratch, 1.0f, requireGpu: true);
        });

        // FP32 raw byte buffer round-trip via DequantizeTensor and EvaluateQuantizedBuffer
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
            if (File.Exists(Path.Combine(dir.FullName, "Glacier.slnx")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }

    private static (int ExitCode, string Output) RunPureCSharpSecretAudit(string targetDir)
    {
        var patterns = new (string Name, Regex Pattern, bool IsGeneric)[]
        {
            ("Google AI / Gemini / Maps API Key", new Regex(@"\bAIza[0-9A-Za-z-_]{35}\b", RegexOptions.Compiled), false),
            ("OpenAI API Key", new Regex(@"\bsk-[a-zA-Z0-9]{20,}\b", RegexOptions.Compiled), false),
            ("Anthropic API Key", new Regex(@"\bsk-ant-[a-zA-Z0-9-_]{20,}\b", RegexOptions.Compiled), false),
            ("GitHub Personal Access Token", new Regex(@"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b", RegexOptions.Compiled), false),
            ("AWS Access Key ID", new Regex(@"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b", RegexOptions.Compiled), false),
            ("Private Key Block", new Regex(@"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", RegexOptions.Compiled), false),
            ("HuggingFace Token", new Regex(@"\bhf_[a-zA-Z0-9]{34,}\b", RegexOptions.Compiled), false),
            ("NuGet API Key", new Regex(@"\boy2[a-z0-9]{43}\b", RegexOptions.Compiled), false),
            ("Generic Secret Assignment", new Regex(@"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|password|client[_-]?secret)\s*[:=]\s*[""']([^""'\s]{12,})[""']", RegexOptions.Compiled), true)
        };

        var allowlist = new Regex[]
        {
            new(@"(?i)placeholder", RegexOptions.Compiled),
            new(@"(?i)dummy", RegexOptions.Compiled),
            new(@"(?i)fake", RegexOptions.Compiled),
            new(@"(?i)sample", RegexOptions.Compiled),
            new(@"(?i)example", RegexOptions.Compiled),
            new(@"(?i)environment\.getenvironmentvariable", RegexOptions.Compiled),
            new(@"(?i)process\.env", RegexOptions.Compiled)
        };

        var findings = new List<string>();
        int scanned = 0;

        foreach (string file in Directory.EnumerateFiles(targetDir, "*", SearchOption.AllDirectories))
        {
            scanned++;
            string[] lines = File.ReadAllLines(file);
            for (int idx = 0; idx < lines.Length; idx++)
            {
                string line = lines[idx];
                foreach (var (name, regex, isGeneric) in patterns)
                {
                    foreach (Match m in regex.Matches(line))
                    {
                        string token = m.Groups.Count > 1 && m.Groups[1].Success ? m.Groups[1].Value : m.Value;
                        if (isGeneric && Array.Exists(allowlist, a => a.IsMatch(token)))
                            continue;

                        string rel = Path.GetRelativePath(targetDir, file);
                        findings.Add($"FAILED: {rel}:{idx + 1} -> {name}");
                    }
                }
            }
        }

        if (findings.Count > 0)
            return (1, $"Scanned {scanned} files.\n" + string.Join("\n", findings));

        return (0, $"Scanned {scanned} files.\n[PASSED] Zero secrets or credentials detected.");
    }

    private static (int ExitCode, string Output)? TryRunGlacierCliSecurityCheck(string? repoRoot, string targetDir)
    {
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (!string.IsNullOrEmpty(localAppData))
        {
            string installedExe = Path.Combine(localAppData, "Glacier", "bin", OperatingSystem.IsWindows() ? "glacier.exe" : "glacier");
            if (File.Exists(installedExe))
            {
                var psi = new ProcessStartInfo
                {
                    FileName = installedExe,
                    WorkingDirectory = targetDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                psi.ArgumentList.Add("security-check");
                psi.ArgumentList.Add(targetDir);

                using var proc = Process.Start(psi);
                if (proc is not null)
                {
                    string stdout = proc.StandardOutput.ReadToEnd();
                    string stderr = proc.StandardError.ReadToEnd();
                    proc.WaitForExit();
                    string combined = stdout + "\n" + stderr;
                    if (!combined.Contains("Unknown command", StringComparison.OrdinalIgnoreCase))
                        return (proc.ExitCode, combined);
                }
            }
        }

        if (repoRoot is null)
            return null;

        string preferredCfg = AppContext.BaseDirectory.Contains($"{Path.DirectorySeparatorChar}Release{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase)
            ? "Release"
            : "Debug";
        string? cliDll = null;
        foreach (string cfg in new[] { preferredCfg, preferredCfg == "Release" ? "Debug" : "Release" })
        {
            string candidate = Path.Combine(repoRoot, "Glacier.Cli", "src", "Glacier.Cli", "bin", cfg, "net10.0", "glacier.dll");
            if (File.Exists(candidate) && File.Exists(Path.ChangeExtension(candidate, ".runtimeconfig.json")))
            {
                cliDll = candidate;
                break;
            }
        }
        if (cliDll is null) return null;

        var dllPsi = new ProcessStartInfo
        {
            FileName = "dotnet",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        dllPsi.ArgumentList.Add(cliDll);
        dllPsi.ArgumentList.Add("security-check");
        dllPsi.ArgumentList.Add(targetDir);

        using var dllProc = Process.Start(dllPsi) ?? throw new InvalidOperationException("Failed to start dotnet glacier.dll process.");
        string dllStdout = dllProc.StandardOutput.ReadToEnd();
        string dllStderr = dllProc.StandardError.ReadToEnd();
        dllProc.WaitForExit();
        return (dllProc.ExitCode, dllStdout + "\n" + dllStderr);
    }
}
