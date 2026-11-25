namespace Glacier.Quant.Calibration;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading.Tasks;

/// <summary>
/// Strategy used to rank weight columns when selecting salient high-precision columns in hybrid PTQ.
/// </summary>
public enum SalientSelectionMethod
{
    Auto = 0,
    ActivationWeightedL2 = 1,
    WeightRms = 2,
    WeightMaxAbs = 3
}

/// <summary>
/// Computes per-channel importance multipliers and selects high-salience columns
/// from an external activation importance matrix (<c>imatrix</c>) or weight-channel outlier statistics.
/// </summary>
public sealed class ActivationSalientSelector
{
    private readonly Dictionary<string, float[]> _importanceByTensor;

    private ActivationSalientSelector(Dictionary<string, float[]> importanceByTensor)
    {
        _importanceByTensor = importanceByTensor;
    }

    public int LoadedTensorCount => _importanceByTensor.Count;

    /// <summary>
    /// Creates a selector from an existing tensor-name-to-importance dictionary.
    /// </summary>
    public static ActivationSalientSelector FromDictionary(IReadOnlyDictionary<string, float[]> importanceByTensor)
    {
        ArgumentNullException.ThrowIfNull(importanceByTensor);
        var copy = new Dictionary<string, float[]>(importanceByTensor.Count, StringComparer.Ordinal);
        foreach (var kvp in importanceByTensor)
        {
            ArgumentNullException.ThrowIfNull(kvp.Value);
            copy[kvp.Key] = (float[])kvp.Value.Clone();
        }
        return new ActivationSalientSelector(copy);
    }

    /// <summary>
    /// Loads per-tensor importance vectors from a JSON object mapping tensor names to float arrays,
    /// or returns an empty selector when <paramref name="imatrixJsonPath"/> is null/whitespace.
    /// </summary>
    public static ActivationSalientSelector Load(string? imatrixJsonPath)
    {
        var map = new Dictionary<string, float[]>(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(imatrixJsonPath))
            return new ActivationSalientSelector(map);

        if (!File.Exists(imatrixJsonPath))
            throw new FileNotFoundException($"Calibration imatrix file not found: '{imatrixJsonPath}'.", imatrixJsonPath);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(imatrixJsonPath));
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Imatrix JSON must be a top-level object mapping tensor names to float arrays.");

        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Value.ValueKind != JsonValueKind.Array) continue;
            float[] arr = new float[prop.Value.GetArrayLength()];
            int idx = 0;
            foreach (var item in prop.Value.EnumerateArray())
                arr[idx++] = item.TryGetSingle(out float v) ? v : 1.0f;
            map[prop.Name] = arr;
        }

        return new ActivationSalientSelector(map);
    }

    /// <summary>
    /// Computes normalized column scaling multipliers <c>s[0..cols-1]</c> with mean 1.0,
    /// or returns <c>null</c> when <paramref name="alpha"/> is &lt;= 0.
    /// </summary>
    public float[]? GetColumnScalingFactors(
        string tensorName,
        ReadOnlySpan<float> weightsRowMajor,
        int rows,
        int cols,
        float alpha)
        => GetColumnScalingFactors(tensorName, weightsRowMajor, rows, cols, alpha, ReadOnlySpan<uint>.Empty);

    /// <summary>
    /// Computes normalized column scaling multipliers <c>s[0..cols-1]</c> with mean 1.0 over non-salient columns,
    /// pinning <paramref name="salientCols"/> at <c>1.0f</c> so zeroed salient columns do not skew normalization.
    /// </summary>
    public float[]? GetColumnScalingFactors(
        string tensorName,
        ReadOnlySpan<float> weightsRowMajor,
        int rows,
        int cols,
        float alpha,
        ReadOnlySpan<uint> salientCols)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        long expected = checked((long)rows * cols);
        if (weightsRowMajor.Length != expected)
            throw new ArgumentException($"Weights length ({weightsRowMajor.Length}) does not match {rows}x{cols}.", nameof(weightsRowMajor));

        if (alpha <= 0.0f)
            return null;

        bool[]? isSalient = null;
        if (!salientCols.IsEmpty)
        {
            isSalient = new bool[cols];
            for (int i = 0; i < salientCols.Length; i++)
            {
                uint c = salientCols[i];
                if (c < (uint)cols)
                    isSalient[c] = true;
            }
        }

        float[] rawImportance = new float[cols];
        if (_importanceByTensor.TryGetValue(tensorName, out float[]? imatrix) && imatrix.Length == cols)
        {
            for (int j = 0; j < cols; j++)
                rawImportance[j] = MathF.Sqrt(MathF.Max(imatrix[j], 1e-8f));
        }
        else
        {
            double[] sumSq = ComputeColumnSumSq(weightsRowMajor, rows, cols, expected);
            float invRows = 1.0f / rows;
            for (int j = 0; j < cols; j++)
                rawImportance[j] = MathF.Sqrt(MathF.Max((float)sumSq[j] * invRows, 1e-10f));
        }

        double sum = 0.0;
        int nonSalientCount = 0;
        for (int j = 0; j < cols; j++)
        {
            if (isSalient != null && isSalient[j])
            {
                rawImportance[j] = 1.0f;
                continue;
            }

            double s = Math.Pow(Math.Max(rawImportance[j], 1e-6), alpha);
            rawImportance[j] = (float)s;
            sum += s;
            nonSalientCount++;
        }

        if (nonSalientCount == 0)
            return null;

        double mean = sum / nonSalientCount;
        if (mean <= 1e-12)
            return null;

        float invMean = (float)(1.0 / mean);
        for (int j = 0; j < cols; j++)
        {
            if (isSalient != null && isSalient[j])
                rawImportance[j] = 1.0f;
            else
                rawImportance[j] = Math.Clamp(rawImportance[j] * invMean, 0.1f, 10.0f);
        }

        return rawImportance;
    }

    /// <summary>
    /// Deterministically selects the top-<c>k</c> salient input column indices sorted in strictly ascending order.
    /// </summary>
    public uint[] SelectSalientColumns(
        string tensorName,
        ReadOnlySpan<float> weightsRowMajor,
        int rows,
        int cols,
        double salientFraction,
        SalientSelectionMethod method = SalientSelectionMethod.Auto)
    {
        ArgumentException.ThrowIfNullOrEmpty(tensorName);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        long expected = checked((long)rows * cols);
        if (weightsRowMajor.Length != expected)
            throw new ArgumentException($"Weights length ({weightsRowMajor.Length}) does not match {rows}x{cols}.", nameof(weightsRowMajor));
        if (!double.IsFinite(salientFraction) || salientFraction < 0.0 || salientFraction >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(salientFraction), salientFraction, "SalientFraction must be finite and in [0, 1).");

        if (salientFraction == 0.0 || cols <= 1)
            return [];

        int k = Math.Clamp((int)Math.Round(cols * salientFraction), 1, cols - 1);

        bool hasImatrix = _importanceByTensor.TryGetValue(tensorName, out float[]? imatrix) && imatrix.Length == cols;
        SalientSelectionMethod effective = method == SalientSelectionMethod.Auto
            ? (hasImatrix ? SalientSelectionMethod.ActivationWeightedL2 : SalientSelectionMethod.WeightRms)
            : method;

        double[] scores;
        if (effective == SalientSelectionMethod.WeightMaxAbs)
        {
            scores = ComputeColumnMaxAbs(weightsRowMajor, rows, cols, expected);
        }
        else
        {
            scores = ComputeColumnSumSq(weightsRowMajor, rows, cols, expected);
            double invRows = 1.0 / rows;
            for (int j = 0; j < cols; j++)
            {
                double rms = Math.Sqrt(scores[j] * invRows);
                scores[j] = effective == SalientSelectionMethod.ActivationWeightedL2 && hasImatrix
                    ? Math.Sqrt(Math.Max(imatrix![j], 1e-8f)) * rms
                    : rms;
            }
        }

        int[] ranked = new int[cols];
        for (int j = 0; j < cols; j++)
            ranked[j] = j;

        Array.Sort(ranked, (a, b) =>
        {
            int cmp = scores[b].CompareTo(scores[a]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });

        uint[] selected = new uint[k];
        for (int i = 0; i < k; i++)
            selected[i] = (uint)ranked[i];

        Array.Sort(selected);
        return selected;
    }

    private static double[] ComputeColumnSumSq(ReadOnlySpan<float> weightsRowMajor, int rows, int cols, long expected)
    {
        double[] sumSq = new double[cols];
        if (rows >= 16 && expected >= 65_536L)
        {
            object mergeLock = new();
            unsafe
            {
                fixed (float* pW = weightsRowMajor)
                {
                    nint wAddr = (nint)pW;
                    Parallel.ForEach(
                        Partitioner.Create(0, rows),
                        () =>
                        {
                            double[] buf = ArrayPool<double>.Shared.Rent(cols);
                            Array.Clear(buf, 0, cols);
                            return buf;
                        },
                        (range, _, local) =>
                        {
                            float* wp = (float*)wAddr;
                            for (int i = range.Item1; i < range.Item2; i++)
                                AccumulateRowSumSq(wp + (long)i * cols, local, cols);
                            return local;
                        },
                        local =>
                        {
                            try
                            {
                                lock (mergeLock)
                                {
                                    TensorPrimitives.Add(sumSq, local.AsSpan(0, cols), sumSq);
                                }
                            }
                            finally
                            {
                                ArrayPool<double>.Shared.Return(local);
                            }
                        });
                }
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                int off = i * cols;
                for (int j = 0; j < cols; j++)
                {
                    double w = weightsRowMajor[off + j];
                    sumSq[j] += w * w;
                }
            }
        }
        return sumSq;
    }

    private static double[] ComputeColumnMaxAbs(ReadOnlySpan<float> weightsRowMajor, int rows, int cols, long expected)
    {
        double[] scores = new double[cols];
        if (rows >= 16 && expected >= 65_536L)
        {
            object mergeLock = new();
            unsafe
            {
                fixed (float* pW = weightsRowMajor)
                {
                    nint wAddr = (nint)pW;
                    Parallel.ForEach(
                        Partitioner.Create(0, rows),
                        () =>
                        {
                            double[] buf = ArrayPool<double>.Shared.Rent(cols);
                            Array.Clear(buf, 0, cols);
                            return buf;
                        },
                        (range, _, local) =>
                        {
                            float* wp = (float*)wAddr;
                            for (int i = range.Item1; i < range.Item2; i++)
                                AccumulateRowMaxAbs(wp + (long)i * cols, local, cols);
                            return local;
                        },
                        local =>
                        {
                            try
                            {
                                lock (mergeLock)
                                {
                                    TensorPrimitives.Max(scores, local.AsSpan(0, cols), scores);
                                }
                            }
                            finally
                            {
                                ArrayPool<double>.Shared.Return(local);
                            }
                        });
                }
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                int off = i * cols;
                for (int j = 0; j < cols; j++)
                {
                    double av = Math.Abs(weightsRowMajor[off + j]);
                    if (av > scores[j]) scores[j] = av;
                }
            }
        }
        return scores;
    }

    private static unsafe void AccumulateRowSumSq(float* rowPtr, double[] acc, int cols)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && cols >= Vector<float>.Count)
        {
            int limit = cols - Vector<float>.Count;
            for (; j <= limit; j += Vector<float>.Count)
            {
                var wf = Vector.Load(rowPtr + j);
                Vector.Widen(wf, out Vector<double> lo, out Vector<double> hi);
                var accLo = new Vector<double>(acc, j) + lo * lo;
                var accHi = new Vector<double>(acc, j + Vector<double>.Count) + hi * hi;
                accLo.CopyTo(acc, j);
                accHi.CopyTo(acc, j + Vector<double>.Count);
            }
        }
        for (; j < cols; j++)
        {
            double w = rowPtr[j];
            acc[j] += w * w;
        }
    }

    private static unsafe void AccumulateRowMaxAbs(float* rowPtr, double[] acc, int cols)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && cols >= Vector<float>.Count)
        {
            int limit = cols - Vector<float>.Count;
            for (; j <= limit; j += Vector<float>.Count)
            {
                var wf = Vector.Abs(Vector.Load(rowPtr + j));
                Vector.Widen(wf, out Vector<double> lo, out Vector<double> hi);
                var accLo = Vector.Max(new Vector<double>(acc, j), lo);
                var accHi = Vector.Max(new Vector<double>(acc, j + Vector<double>.Count), hi);
                accLo.CopyTo(acc, j);
                accHi.CopyTo(acc, j + Vector<double>.Count);
            }
        }
        for (; j < cols; j++)
        {
            double av = Math.Abs(rowPtr[j]);
            if (av > acc[j]) acc[j] = av;
        }
    }
}
