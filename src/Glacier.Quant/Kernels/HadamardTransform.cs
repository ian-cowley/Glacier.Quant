namespace Glacier.Quant.Kernels;

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// High-throughput SIMD-accelerated Hadamard transforms for post-training quantization pre-processing
/// (QuaRot, SpinQuant, QuIP#, and NariQ):
/// <list type="bullet">
///   <item>Normalized block Sylvester Fast Walsh-Hadamard Transform (self-inverse orthogonal <c>H * H = I</c>, AVX-512 / AVX2 / scalar).</item>
///   <item>Normalized Paley Type-I Hadamard Transform for orders <c>m in {12, 20, 68}</c> (<c>H * H^T = I</c>) plus Kronecker Paley-Sylvester slices.</item>
///   <item>Randomized Hadamard Rotation (<c>H * diag(signs) * x</c>) and its exact inverse (<c>diag(signs) * H * x</c>).</item>
/// </list>
/// </summary>
public static unsafe class HadamardTransform
{
    private const int Paley12Stride = 16;
    private const int Paley20Stride = 32;
    private const int Paley68Stride = 80;

    private static readonly sbyte[] s_paleyS12 = BuildPaleySByte(12);
    private static readonly sbyte[] s_paleyS20 = BuildPaleySByte(20);
    private static readonly sbyte[] s_paleyS68 = BuildPaleySByte(68);

    private static readonly float[] s_paleyFwd12 = BuildPaleyFloatPadded(s_paleyS12, 12, Paley12Stride, transpose: false);
    private static readonly float[] s_paleyInv12 = BuildPaleyFloatPadded(s_paleyS12, 12, Paley12Stride, transpose: true);
    private static readonly float[] s_paleyFwd20 = BuildPaleyFloatPadded(s_paleyS20, 20, Paley20Stride, transpose: false);
    private static readonly float[] s_paleyInv20 = BuildPaleyFloatPadded(s_paleyS20, 20, Paley20Stride, transpose: true);
    private static readonly float[] s_paleyFwd68 = BuildPaleyFloatPadded(s_paleyS68, 68, Paley68Stride, transpose: false);
    private static readonly float[] s_paleyInv68 = BuildPaleyFloatPadded(s_paleyS68, 68, Paley68Stride, transpose: true);

    /// <summary>
    /// Returns <c>true</c> if <paramref name="m"/> is a supported Paley Type-I Hadamard order (12, 20, or 68).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsSupportedPaleyOrder(int m) => m is 12 or 20 or 68;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPowerOfTwo(int v) => v > 0 && (v & (v - 1)) == 0;

    // =========================================================================
    // 1. SYLVESTER FAST WALSH-HADAMARD TRANSFORM (H * H = I)
    // =========================================================================

    /// <summary>
    /// Applies the normalized block Sylvester Fast Walsh-Hadamard Transform in place with
    /// <c>1 / sqrt(blockLen)</c> scaling so <c>H * H = I</c> (self-inverse orthogonal).
    /// Supports any power-of-2 <paramref name="blockLen"/> &gt;= 2 (SIMD fast paths for <paramref name="blockLen"/> &gt;= 16).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplySylvesterInPlace(Span<float> data, int blockLen)
    {
        ValidateSylvesterArgs(data.Length, blockLen);
        if (data.IsEmpty) return;

        float scale = (float)(1.0 / Math.Sqrt(blockLen));

        fixed (float* pData = data)
        {
            if (Avx512F.IsSupported && blockLen >= 16)
            {
                for (int offset = 0; offset < data.Length; offset += blockLen)
                {
                    FwhtBlockAvx512(pData + offset, blockLen, scale);
                }
            }
            else if (Avx2.IsSupported && blockLen >= 16)
            {
                for (int offset = 0; offset < data.Length; offset += blockLen)
                {
                    FwhtBlockAvx2(pData + offset, blockLen, scale);
                }
            }
            else
            {
                for (int offset = 0; offset < data.Length; offset += blockLen)
                {
                    FwhtBlockScalar(pData + offset, blockLen, scale);
                }
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="ApplySylvesterInPlace"/> used for bit-exact parity testing.
    /// </summary>
    internal static void ApplySylvesterScalarInPlace(Span<float> data, int blockLen)
    {
        ValidateSylvesterArgs(data.Length, blockLen);
        if (data.IsEmpty) return;

        float scale = (float)(1.0 / Math.Sqrt(blockLen));
        fixed (float* pData = data)
        {
            for (int offset = 0; offset < data.Length; offset += blockLen)
            {
                FwhtBlockScalar(pData + offset, blockLen, scale);
            }
        }
    }

    private static void ValidateSylvesterArgs(int dataLength, int blockLen)
    {
        if (blockLen < 2 || !IsPowerOfTwo(blockLen))
            throw new ArgumentOutOfRangeException(nameof(blockLen), blockLen, "Block length must be a power of 2 >= 2.");
        if ((dataLength % blockLen) != 0)
            throw new ArgumentException($"Data length ({dataLength}) must be a multiple of blockLen ({blockLen}).", nameof(dataLength));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtBlockScalar(float* u, int blockLen, float scale)
    {
        for (int h = 1; h < blockLen; h <<= 1)
        {
            int step = h << 1;
            for (int i = 0; i < blockLen; i += step)
            {
                for (int j = i; j < i + h; j++)
                {
                    float a = u[j];
                    float b = u[j + h];
                    u[j] = a + b;
                    u[j + h] = a - b;
                }
            }
        }

        for (int i = 0; i < blockLen; i++)
        {
            u[i] *= scale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtBlockAvx2(float* u, int blockLen, float scale)
    {
        var sign1 = Vector256.Create(0f, -0.0f, 0f, -0.0f, 0f, -0.0f, 0f, -0.0f);
        var sign2 = Vector256.Create(0f, 0f, -0.0f, -0.0f, 0f, 0f, -0.0f, -0.0f);
        var sign4 = Vector256.Create(0f, 0f, 0f, 0f, -0.0f, -0.0f, -0.0f, -0.0f);
        var vScale = Vector256.Create(scale);

        // Fused in-register stages h = 1, 2, 4, 8 across 16-float chunks
        bool scaleInFirstPass = blockLen == 16;
        for (int i = 0; i < blockLen; i += 16)
        {
            var v0 = Vector256.Load(u + i);
            var v1 = Vector256.Load(u + i + 8);

            // h = 1
            v0 = Avx.Add(Avx.Xor(v0, sign1), Avx.Permute(v0, 0xB1));
            v1 = Avx.Add(Avx.Xor(v1, sign1), Avx.Permute(v1, 0xB1));

            // h = 2
            v0 = Avx.Add(Avx.Xor(v0, sign2), Avx.Permute(v0, 0x4E));
            v1 = Avx.Add(Avx.Xor(v1, sign2), Avx.Permute(v1, 0x4E));

            // h = 4
            v0 = Avx.Add(Avx.Xor(v0, sign4), Avx2.Permute2x128(v0, v0, 0x01));
            v1 = Avx.Add(Avx.Xor(v1, sign4), Avx2.Permute2x128(v1, v1, 0x01));

            // h = 8
            var t0 = Avx.Add(v0, v1);
            var t1 = Avx.Subtract(v0, v1);

            if (scaleInFirstPass)
            {
                Vector256.Store(Avx.Multiply(t0, vScale), u + i);
                Vector256.Store(Avx.Multiply(t1, vScale), u + i + 8);
            }
            else
            {
                Vector256.Store(t0, u + i);
                Vector256.Store(t1, u + i + 8);
            }
        }

        if (scaleInFirstPass) return;

        // Outer stages h = 16, 32, ..., blockLen / 2
        for (int h = 16; h < blockLen; h <<= 1)
        {
            int step = h << 1;
            if (step == blockLen)
            {
                // Final stage: fuse normalization multiply to avoid an extra memory pass
                for (int j = 0; j < h; j += 16)
                {
                    var a0 = Vector256.Load(u + j);
                    var b0 = Vector256.Load(u + j + h);
                    var a1 = Vector256.Load(u + j + 8);
                    var b1 = Vector256.Load(u + j + h + 8);

                    Vector256.Store(Avx.Multiply(Avx.Add(a0, b0), vScale), u + j);
                    Vector256.Store(Avx.Multiply(Avx.Subtract(a0, b0), vScale), u + j + h);
                    Vector256.Store(Avx.Multiply(Avx.Add(a1, b1), vScale), u + j + 8);
                    Vector256.Store(Avx.Multiply(Avx.Subtract(a1, b1), vScale), u + j + h + 8);
                }
            }
            else
            {
                for (int i = 0; i < blockLen; i += step)
                {
                    for (int j = i; j < i + h; j += 16)
                    {
                        var a0 = Vector256.Load(u + j);
                        var b0 = Vector256.Load(u + j + h);
                        var a1 = Vector256.Load(u + j + 8);
                        var b1 = Vector256.Load(u + j + h + 8);

                        Vector256.Store(Avx.Add(a0, b0), u + j);
                        Vector256.Store(Avx.Subtract(a0, b0), u + j + h);
                        Vector256.Store(Avx.Add(a1, b1), u + j + 8);
                        Vector256.Store(Avx.Subtract(a1, b1), u + j + h + 8);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtBlockAvx512(float* u, int blockLen, float scale)
    {
        var sign1 = Vector512.Create(
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u);
        var sign2 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u);
        var sign4 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
        var sign8 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);

        var perm4 = Vector512.Create(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
        var perm8 = Vector512.Create(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        var vScale = Vector512.Create(scale);

        bool scaleInFirstPass = blockLen == 16;
        for (int i = 0; i < blockLen; i += 16)
        {
            var v = Vector512.Load(u + i);

            // h = 1
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign1).AsSingle(), Avx512F.Shuffle(v, v, 0xB1));
            // h = 2
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign2).AsSingle(), Avx512F.Shuffle(v, v, 0x4E));
            // h = 4
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign4).AsSingle(), Vector512.Shuffle(v, perm4));
            // h = 8
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign8).AsSingle(), Vector512.Shuffle(v, perm8));

            if (scaleInFirstPass)
            {
                Vector512.Store(Avx512F.Multiply(v, vScale), u + i);
            }
            else
            {
                Vector512.Store(v, u + i);
            }
        }

        if (scaleInFirstPass) return;

        for (int h = 16; h < blockLen; h <<= 1)
        {
            int step = h << 1;
            if (step == blockLen)
            {
                for (int j = 0; j < h; j += 16)
                {
                    var a = Vector512.Load(u + j);
                    var b = Vector512.Load(u + j + h);
                    Vector512.Store(Avx512F.Multiply(Avx512F.Add(a, b), vScale), u + j);
                    Vector512.Store(Avx512F.Multiply(Avx512F.Subtract(a, b), vScale), u + j + h);
                }
            }
            else
            {
                for (int i = 0; i < blockLen; i += step)
                {
                    for (int j = i; j < i + h; j += 16)
                    {
                        var a = Vector512.Load(u + j);
                        var b = Vector512.Load(u + j + h);
                        Vector512.Store(Avx512F.Add(a, b), u + j);
                        Vector512.Store(Avx512F.Subtract(a, b), u + j + h);
                    }
                }
            }
        }
    }

    // =========================================================================
    // 2. PALEY TYPE-I HADAMARD TRANSFORM (m in {12, 20, 68}, H * H^T = I)
    // =========================================================================

    /// <summary>
    /// Applies the normalized block Paley Type-I Hadamard transform <c>(1 / sqrt(m)) * P_m</c>
    /// in place for <paramref name="m"/> in {12, 20, 68}. Satisfies <c>H * H^T = I</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyPaleyInPlace(Span<float> data, int m)
    {
        ApplyPaleyCore(data, m, transpose: false);
    }

    /// <summary>
    /// Applies the inverse (transpose) normalized block Paley Type-I Hadamard transform
    /// <c>(1 / sqrt(m)) * P_m^T</c> in place for <paramref name="m"/> in {12, 20, 68}.
    /// Satisfies <c>ApplyPaleyInverseInPlace(ApplyPaleyInPlace(x, m), m) == x</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyPaleyInverseInPlace(Span<float> data, int m)
    {
        ApplyPaleyCore(data, m, transpose: true);
    }

    /// <summary>
    /// Applies a full-slice NariQ Kronecker Paley-Sylvester orthogonal transform (<c>P_m ⊗ H_run</c>)
    /// in place where <c>data.Length == m * 2^p</c> and <paramref name="m"/> is in {12, 20, 68}.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyKroneckerPaleyInPlace(Span<float> data, int m)
    {
        ApplyKroneckerPaleyCore(data, m, inverse: false);
    }

    /// <summary>
    /// Applies the exact inverse of <see cref="ApplyKroneckerPaleyInPlace"/> (<c>(P_m ⊗ H_run)^T = P_m^T ⊗ H_run</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyKroneckerPaleyInverseInPlace(Span<float> data, int m)
    {
        ApplyKroneckerPaleyCore(data, m, inverse: true);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ApplyPaleyCore(Span<float> data, int m, bool transpose)
    {
        (float[] mat, int stride) = GetPaleyPaddedMatrix(m, transpose);
        if ((data.Length % m) != 0)
            throw new ArgumentException($"Data length ({data.Length}) must be a multiple of Paley order m ({m}).", nameof(data));
        if (data.IsEmpty) return;

        float scale = (float)(1.0 / Math.Sqrt(m));
        float* tmp = stackalloc float[Paley68Stride];
        new Span<float>(tmp, stride).Clear();

        fixed (float* pData = data)
        fixed (float* pMat = mat)
        {
            for (int offset = 0; offset < data.Length; offset += m)
            {
                float* block = pData + offset;
                Buffer.MemoryCopy(block, tmp, stride * sizeof(float), m * sizeof(float));

                for (int r = 0; r < m; r++)
                {
                    block[r] = DotPaddedSimd(pMat + r * stride, tmp, stride) * scale;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ApplyKroneckerPaleyCore(Span<float> data, int m, bool inverse)
    {
        sbyte[] h = GetPaleySByte(m);
        if (data.Length < m || (data.Length % m) != 0)
            throw new ArgumentException($"Data length ({data.Length}) must be a positive multiple of Paley order m ({m}).", nameof(data));

        int run = data.Length / m;
        if (!IsPowerOfTwo(run))
            throw new ArgumentException($"For Kronecker Paley transform, data.Length / m ({run}) must be a power of 2.", nameof(data));

        if (run == 1)
        {
            ApplyPaleyCore(data, m, transpose: inverse);
            return;
        }

        float scaleM = (float)(1.0 / Math.Sqrt(m));
        float[]? rented = data.Length > 1024 ? ArrayPool<float>.Shared.Rent(data.Length) : null;
        try
        {
            Span<float> scratch = rented != null ? rented.AsSpan(0, data.Length) : stackalloc float[data.Length];

            if (!inverse)
            {
                ApplySylvesterInPlace(data, run);
                CrossPaleySimd(data, scratch, h, m, run, transpose: false, scaleM);
                scratch.CopyTo(data);
            }
            else
            {
                CrossPaleySimd(data, scratch, h, m, run, transpose: true, scaleM);
                ApplySylvesterInPlace(scratch, run);
                scratch.CopyTo(data);
            }
        }
        finally
        {
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void CrossPaleySimd(
        ReadOnlySpan<float> input, Span<float> output, sbyte[] h, int m, int run, bool transpose, float scale)
    {
        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            for (int a = 0; a < m; a++)
            {
                float* o = pOut + a * run;
                new Span<float>(o, run).Clear();

                for (int b = 0; b < m; b++)
                {
                    int s = transpose ? h[b * m + a] : h[a * m + b];
                    float* x = pIn + b * run;
                    int j = 0;

                    if (s > 0)
                    {
                        if (Avx512F.IsSupported && run >= 16)
                        {
                            for (; j <= run - 16; j += 16)
                                Vector512.Store(Avx512F.Add(Vector512.Load(o + j), Vector512.Load(x + j)), o + j);
                        }
                        else if (Avx2.IsSupported && run >= 8)
                        {
                            for (; j <= run - 8; j += 8)
                                Vector256.Store(Avx.Add(Vector256.Load(o + j), Vector256.Load(x + j)), o + j);
                        }
                        for (; j < run; j++) o[j] += x[j];
                    }
                    else
                    {
                        if (Avx512F.IsSupported && run >= 16)
                        {
                            for (; j <= run - 16; j += 16)
                                Vector512.Store(Avx512F.Subtract(Vector512.Load(o + j), Vector512.Load(x + j)), o + j);
                        }
                        else if (Avx2.IsSupported && run >= 8)
                        {
                            for (; j <= run - 8; j += 8)
                                Vector256.Store(Avx.Subtract(Vector256.Load(o + j), Vector256.Load(x + j)), o + j);
                        }
                        for (; j < run; j++) o[j] -= x[j];
                    }
                }

                int k = 0;
                if (Avx512F.IsSupported && run >= 16)
                {
                    var vScale = Vector512.Create(scale);
                    for (; k <= run - 16; k += 16)
                        Vector512.Store(Avx512F.Multiply(Vector512.Load(o + k), vScale), o + k);
                }
                else if (Avx2.IsSupported && run >= 8)
                {
                    var vScale = Vector256.Create(scale);
                    for (; k <= run - 8; k += 8)
                        Vector256.Store(Avx.Multiply(Vector256.Load(o + k), vScale), o + k);
                }
                for (; k < run; k++) o[k] *= scale;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotPaddedSimd(float* a, float* b, int stride)
    {
        int i = 0;
        if (Avx512F.IsSupported)
        {
            var acc = Vector512<float>.Zero;
            for (; i < stride; i += 16)
            {
                acc = Avx512F.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc);
            }
            return Vector512.Sum(acc);
        }
        if (Avx2.IsSupported)
        {
            var acc = Vector256<float>.Zero;
            if (Fma.IsSupported)
            {
                for (; i < stride; i += 8)
                    acc = Fma.MultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), acc);
            }
            else
            {
                for (; i < stride; i += 8)
                    acc = Avx.Add(acc, Avx.Multiply(Vector256.Load(a + i), Vector256.Load(b + i)));
            }
            return Vector256.Sum(acc);
        }

        float sum = 0f;
        for (; i < stride; i++) sum += a[i] * b[i];
        return sum;
    }

    // =========================================================================
    // 3. RANDOMIZED HADAMARD ROTATION (QuaRot / SpinQuant / QuIP# / NariQ)
    // =========================================================================

    /// <summary>
    /// Deterministically fills <paramref name="signs"/> with Rademacher <c>+1.0f</c> / <c>-1.0f</c>
    /// values using SplitMix64, suitable for reproducible QuaRot / SpinQuant / NariQ rotations.
    /// </summary>
    public static void GenerateRandomSigns(Span<float> signs, ulong seed)
    {
        ulong state = seed;
        int i = 0;
        while (i < signs.Length)
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;

            int bits = Math.Min(64, signs.Length - i);
            for (int b = 0; b < bits; b++)
            {
                signs[i++] = ((z >> b) & 1UL) == 0 ? 1.0f : -1.0f;
            }
        }
    }

    /// <summary>
    /// Applies a Randomized Hadamard Rotation in place: <c>y = H * (signs ⊙ x)</c>.
    /// <paramref name="signs"/> may match either <c>data.Length</c> (per-element signs) or
    /// <paramref name="blockLen"/> (shared block signs).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyRandomizedRotation(Span<float> data, ReadOnlySpan<float> signs, int blockLen)
    {
        ValidateSylvesterArgs(data.Length, blockLen);
        ValidateSignsLength(data.Length, signs.Length, blockLen);
        if (data.IsEmpty) return;

        MultiplySignsInPlace(data, signs, blockLen);
        ApplySylvesterInPlace(data, blockLen);
    }

    /// <summary>
    /// Applies the exact inverse Randomized Hadamard Rotation in place: <c>x = signs ⊙ (H * y)</c>.
    /// When <paramref name="signs"/> contains <c>±1.0f</c>, <c>Inverse(Forward(x)) == x</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyRandomizedRotationInverse(Span<float> data, ReadOnlySpan<float> signs, int blockLen)
    {
        ValidateSylvesterArgs(data.Length, blockLen);
        ValidateSignsLength(data.Length, signs.Length, blockLen);
        if (data.IsEmpty) return;

        ApplySylvesterInPlace(data, blockLen);
        MultiplySignsInPlace(data, signs, blockLen);
    }

    private static void ValidateSignsLength(int dataLength, int signsLength, int blockLen)
    {
        if (signsLength != dataLength && signsLength != blockLen)
            throw new ArgumentException($"Signs length ({signsLength}) must equal either data.Length ({dataLength}) or blockLen ({blockLen}).", nameof(signsLength));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MultiplySignsInPlace(Span<float> data, ReadOnlySpan<float> signs, int blockLen)
    {
        fixed (float* pData = data)
        fixed (float* pSigns = signs)
        {
            if (signs.Length == data.Length)
            {
                MultiplyVecInPlace(pData, pSigns, data.Length);
            }
            else
            {
                for (int offset = 0; offset < data.Length; offset += blockLen)
                {
                    MultiplyVecInPlace(pData + offset, pSigns, blockLen);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyVecInPlace(float* dst, float* src, int length)
    {
        int i = 0;
        if (Avx512F.IsSupported && length >= 16)
        {
            for (; i <= length - 16; i += 16)
                Vector512.Store(Avx512F.Multiply(Vector512.Load(dst + i), Vector512.Load(src + i)), dst + i);
        }
        else if (Avx2.IsSupported && length >= 8)
        {
            for (; i <= length - 8; i += 8)
                Vector256.Store(Avx.Multiply(Vector256.Load(dst + i), Vector256.Load(src + i)), dst + i);
        }
        for (; i < length; i++)
            dst[i] *= src[i];
    }

    // =========================================================================
    // 4. PALEY MATRIX CONSTRUCTION HELPERS
    // =========================================================================

    private static int LegendreChi(int d, int q)
    {
        d = ((d % q) + q) % q;
        if (d == 0) return 0;
        for (int i = 1; i < q; i++)
        {
            if ((i * i) % q == d) return 1;
        }
        return -1;
    }

    private static sbyte[] BuildPaleySByte(int m)
    {
        var h = new sbyte[m * m];
        int q = m - 1;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : LegendreChi(j - i, q));
                h[i * m + j] = (sbyte)(s + (i == j ? 1 : 0));
            }
        }
        return h;
    }

    private static float[] BuildPaleyFloatPadded(sbyte[] src, int m, int stride, bool transpose)
    {
        var padded = new float[m * stride];
        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < m; c++)
            {
                padded[r * stride + c] = transpose ? src[c * m + r] : src[r * m + c];
            }
        }
        return padded;
    }

    private static sbyte[] GetPaleySByte(int m) => m switch
    {
        12 => s_paleyS12,
        20 => s_paleyS20,
        68 => s_paleyS68,
        _ => throw new ArgumentOutOfRangeException(nameof(m), m, "Supported Paley Hadamard orders are 12, 20, and 68.")
    };

    private static (float[] Matrix, int Stride) GetPaleyPaddedMatrix(int m, bool transpose) => m switch
    {
        12 => (transpose ? s_paleyInv12 : s_paleyFwd12, Paley12Stride),
        20 => (transpose ? s_paleyInv20 : s_paleyFwd20, Paley20Stride),
        68 => (transpose ? s_paleyInv68 : s_paleyFwd68, Paley68Stride),
        _ => throw new ArgumentOutOfRangeException(nameof(m), m, "Supported Paley Hadamard orders are 12, 20, and 68.")
    };
}
