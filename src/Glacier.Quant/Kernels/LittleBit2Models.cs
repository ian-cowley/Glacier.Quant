namespace Glacier.Quant.Kernels;

using System;
using System.Buffers.Binary;
using System.IO;
using System.Runtime.InteropServices;

/// <summary>
/// Orthogonal rotation strategy applied to the low-rank factors <c>A</c> and <c>B</c>
/// prior to Dual-SVID magnitude extraction and <c>{+1, -1}</c> binarization.
/// </summary>
public enum LittleBit2RotationMode : byte
{
    JointItq = 0,
    RandomOrthogonal = 1,
    None = 2
}

/// <summary>
/// Configuration parameters for clean-room LittleBit-2 SVD + Joint-ITQ + Dual-SVID quantization.
/// </summary>
public sealed record LittleBit2Options
{
    /// <summary>Target bits per weight used to derive rank when <see cref="Rank"/> is <c>null</c>.</summary>
    public double BitsPerWeight { get; init; } = 1.0;

    /// <summary>Optional explicit low-rank factorization rank <c>r</c> for the primary path.</summary>
    public int? Rank { get; init; }

    /// <summary>Optional explicit low-rank factorization rank for the residual compensation path.</summary>
    public int? ResidualRank { get; init; }

    /// <summary>Whether to fit a second Dual-SVID residual compensation path on <c>W - W_hat_pri</c>.</summary>
    public bool UseResidualCompensation { get; init; } = true;

    /// <summary>Rotation alignment mode applied to symmetric factors <c>(A, B)</c>.</summary>
    public LittleBit2RotationMode RotationMode { get; init; } = LittleBit2RotationMode.JointItq;

    /// <summary>Maximum number of Joint-ITQ alternating Procrustes iterations.</summary>
    public int JointItqIterations { get; init; } = 50;

    /// <summary>Deterministic PRNG seed for randomized SVD and orthogonal rotation initialization.</summary>
    public int Seed { get; init; } = 42;

    /// <summary>Oversampling dimension <c>p</c> for randomized range finding.</summary>
    public int SvdOversampling { get; init; } = 16;

    /// <summary>Number of subspace power iterations <c>q</c> for randomized SVD.</summary>
    public int PowerIterations { get; init; } = 2;

    /// <summary>Maximum power-iteration steps for Dual-SVID rank-1 non-negative magnitude factorization.</summary>
    public int Rank1SvdIterations { get; init; } = 50;

    /// <summary>Whether inference and dequantization kernels may use multi-threaded row parallelism.</summary>
    public bool UseParallel { get; init; } = true;

    internal void Validate(int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);

        int maxRank = Math.Min(rows, cols);

        if (Rank is null)
        {
            if (!double.IsFinite(BitsPerWeight) || BitsPerWeight <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(BitsPerWeight), BitsPerWeight, "BitsPerWeight must be finite and strictly positive.");
        }
        else if (Rank.Value < 1 || Rank.Value > maxRank)
        {
            throw new ArgumentOutOfRangeException(nameof(Rank), Rank.Value, $"Rank must be in [1, {maxRank}].");
        }

        if (ResidualRank is int resRank && (resRank < 1 || resRank > maxRank))
            throw new ArgumentOutOfRangeException(nameof(ResidualRank), resRank, $"ResidualRank must be in [1, {maxRank}].");

        if (!Enum.IsDefined(RotationMode))
            throw new ArgumentOutOfRangeException(nameof(RotationMode), RotationMode, "Unsupported LittleBit2RotationMode value.");

        ArgumentOutOfRangeException.ThrowIfNegative(JointItqIterations);
        ArgumentOutOfRangeException.ThrowIfNegative(SvdOversampling);
        ArgumentOutOfRangeException.ThrowIfNegative(PowerIterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Rank1SvdIterations);
    }
}

/// <summary>
/// Single Dual-SVID bit-packed factorized path:
/// <c>W_path = diag(h) * U_sign * diag(ell) * V_sign^T * diag(g)</c>.
/// </summary>
public sealed class LittleBit2Path
{
    public int Rows { get; }
    public int Cols { get; }
    public int Rank { get; }
    public int UStrideWords { get; }
    public int VStrideWords { get; }
    public ulong[] UBits { get; }
    public ulong[] VBits { get; }
    public Half[] RowScales { get; }
    public Half[] ColScales { get; }
    public Half[] LatentScales { get; }

    public long TotalModelBits =>
        (long)Rank * ((long)Rows + Cols + 16L) + 16L * ((long)Rows + Cols);

    public long PackedByteCount =>
        ((long)UBits.Length + VBits.Length) * sizeof(ulong) +
        ((long)RowScales.Length + ColScales.Length + LatentScales.Length) * 2L;

    public LittleBit2Path(
        int rows,
        int cols,
        int rank,
        ulong[] uBits,
        ulong[] vBits,
        Half[] rowScales,
        Half[] colScales,
        Half[] latentScales)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

        int maxRank = Math.Min(rows, cols);
        if (rank > maxRank)
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"Rank ({rank}) cannot exceed min(rows, cols) ({maxRank}).");

        ArgumentNullException.ThrowIfNull(uBits);
        ArgumentNullException.ThrowIfNull(vBits);
        ArgumentNullException.ThrowIfNull(rowScales);
        ArgumentNullException.ThrowIfNull(colScales);
        ArgumentNullException.ThrowIfNull(latentScales);

        int uStride = (rank + 63) >> 6;
        int vStride = (cols + 63) >> 6;

        long expectedUBits = (long)rows * uStride;
        long expectedVBits = (long)rank * vStride;

        if (uBits.Length != expectedUBits)
            throw new ArgumentException($"UBits length ({uBits.Length}) must equal {expectedUBits}.", nameof(uBits));
        if (vBits.Length != expectedVBits)
            throw new ArgumentException($"VBits length ({vBits.Length}) must equal {expectedVBits}.", nameof(vBits));
        if (rowScales.Length != rows)
            throw new ArgumentException($"RowScales length ({rowScales.Length}) must equal {rows}.", nameof(rowScales));
        if (colScales.Length != cols)
            throw new ArgumentException($"ColScales length ({colScales.Length}) must equal {cols}.", nameof(colScales));
        if (latentScales.Length != rank)
            throw new ArgumentException($"LatentScales length ({latentScales.Length}) must equal {rank}.", nameof(latentScales));

        EnsureAllFinite(rowScales, nameof(rowScales));
        EnsureAllFinite(colScales, nameof(colScales));
        EnsureAllFinite(latentScales, nameof(latentScales));

        Rows = rows;
        Cols = cols;
        Rank = rank;
        UStrideWords = uStride;
        VStrideWords = vStride;
        UBits = uBits;
        VBits = vBits;
        RowScales = rowScales;
        ColScales = colScales;
        LatentScales = latentScales;

        ClearUnusedTailBits(UBits, rows, uStride, rank);
        ClearUnusedTailBits(VBits, rank, vStride, cols);
    }

    internal static void ClearUnusedTailBits(ulong[] bits, int rowCount, int strideWords, int logicalWidth)
    {
        int tail = logicalWidth & 63;
        if (tail == 0 || strideWords == 0)
            return;

        ulong mask = (1UL << tail) - 1UL;
        int lastWordOffset = strideWords - 1;
        for (int r = 0; r < rowCount; r++)
        {
            bits[r * strideWords + lastWordOffset] &= mask;
        }
    }

    private static void EnsureAllFinite(Half[] values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!Half.IsFinite(values[i]))
                throw new ArgumentException($"Scale array '{paramName}' contains non-finite value at index {i}.", paramName);
        }
    }
}

/// <summary>
/// LittleBit-2 quantized 2D weight matrix container holding a primary Dual-SVID path
/// and an optional residual compensation path.
/// </summary>
public sealed class LittleBit2Tensor
{
    private const uint BinaryMagic = 0x3242544Cu; // "LTB2"
    private const uint BinaryFormatVersion = 1u;
    private const int HeaderByteSize = 20;

    public int Rows { get; }
    public int Cols { get; }
    public QuantType Type => QuantType.LittleBit2;
    public LittleBit2Path Primary { get; }
    public LittleBit2Path? Residual { get; }
    public int Rank => Primary.Rank;
    public int PathCount => Residual is null ? 1 : 2;
    public bool HasResidual => Residual is not null;

    public double EffectiveBitsPerWeight =>
        LittleBit2Quantizer.ComputeEffectiveBpw(Rows, Cols, Primary.Rank, Residual?.Rank ?? 0);

    public long CompressedByteSize =>
        Primary.PackedByteCount + (Residual?.PackedByteCount ?? 0L);

    public LittleBit2Tensor(int rows, int cols, LittleBit2Path primary, LittleBit2Path? residual = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        ArgumentNullException.ThrowIfNull(primary);

        if (primary.Rows != rows || primary.Cols != cols)
            throw new ArgumentException($"Primary path dimensions ({primary.Rows}x{primary.Cols}) must match tensor dimensions ({rows}x{cols}).", nameof(primary));

        if (residual is not null && (residual.Rows != rows || residual.Cols != cols))
            throw new ArgumentException($"Residual path dimensions ({residual.Rows}x{residual.Cols}) must match tensor dimensions ({rows}x{cols}).", nameof(residual));

        Rows = rows;
        Cols = cols;
        Primary = primary;
        Residual = residual;
    }

    public byte[] ToBytes()
    {
        long totalLengthLong = checked(HeaderByteSize + 4L + Primary.PackedByteCount + (Residual is not null ? 4L + Residual.PackedByteCount : 0L));
        if (totalLengthLong > int.MaxValue)
            throw new InvalidOperationException($"Serialized LittleBit2Tensor size ({totalLengthLong} bytes) exceeds maximum single-array length.");

        byte[] output = new byte[(int)totalLengthLong];
        Span<byte> span = output;

        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(0, 4), BinaryMagic);
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(4, 4), BinaryFormatVersion);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(8, 4), Rows);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(12, 4), Cols);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(16, 4), PathCount);

        int offset = HeaderByteSize;
        WritePath(span, ref offset, Primary);
        if (Residual is not null)
            WritePath(span, ref offset, Residual);

        return output;
    }

    public static LittleBit2Tensor FromBytes(ReadOnlySpan<byte> data)
    {
        if (data.Length < HeaderByteSize)
            throw new InvalidDataException($"LittleBit2 payload too short ({data.Length} bytes); expected at least {HeaderByteSize} bytes.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(0, 4));
        if (magic != BinaryMagic)
            throw new InvalidDataException($"Invalid LittleBit2 magic header: 0x{magic:X8}.");

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(4, 4));
        if (version != BinaryFormatVersion)
            throw new InvalidDataException($"Unsupported LittleBit2 format version: {version}.");

        int rows = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(8, 4));
        int cols = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(12, 4));
        int pathCount = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(16, 4));

        if (rows <= 0 || cols <= 0)
            throw new InvalidDataException($"Invalid LittleBit2 dimensions ({rows}x{cols}).");
        if (pathCount != 1 && pathCount != 2)
            throw new InvalidDataException($"Invalid LittleBit2 path count: {pathCount}.");

        int offset = HeaderByteSize;
        LittleBit2Path primary = ReadPath(data, ref offset, rows, cols);
        LittleBit2Path? residual = pathCount == 2 ? ReadPath(data, ref offset, rows, cols) : null;

        if (offset != data.Length)
            throw new InvalidDataException($"Unexpected trailing bytes in LittleBit2 payload: consumed {offset} of {data.Length} bytes.");

        return new LittleBit2Tensor(rows, cols, primary, residual);
    }

    public void WriteTo(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        Span<byte> header = stackalloc byte[HeaderByteSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(0, 4), BinaryMagic);
        BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(4, 4), BinaryFormatVersion);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(8, 4), Rows);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(12, 4), Cols);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(16, 4), PathCount);
        stream.Write(header);

        WritePathToStream(stream, Primary);
        if (Residual is not null)
            WritePathToStream(stream, Residual);
    }

    public static LittleBit2Tensor ReadFrom(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        Span<byte> header = stackalloc byte[HeaderByteSize];
        ReadExactOrInvalidData(stream, header, "LittleBit2 header");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(0, 4));
        if (magic != BinaryMagic)
            throw new InvalidDataException($"Invalid LittleBit2 magic header: 0x{magic:X8}.");

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(4, 4));
        if (version != BinaryFormatVersion)
            throw new InvalidDataException($"Unsupported LittleBit2 format version: {version}.");

        int rows = BinaryPrimitives.ReadInt32LittleEndian(header.Slice(8, 4));
        int cols = BinaryPrimitives.ReadInt32LittleEndian(header.Slice(12, 4));
        int pathCount = BinaryPrimitives.ReadInt32LittleEndian(header.Slice(16, 4));

        if (rows <= 0 || cols <= 0)
            throw new InvalidDataException($"Invalid LittleBit2 dimensions ({rows}x{cols}).");
        if (pathCount != 1 && pathCount != 2)
            throw new InvalidDataException($"Invalid LittleBit2 path count: {pathCount}.");

        LittleBit2Path primary = ReadPathFromStream(stream, rows, cols);
        LittleBit2Path? residual = pathCount == 2 ? ReadPathFromStream(stream, rows, cols) : null;

        return new LittleBit2Tensor(rows, cols, primary, residual);
    }

    private static void WritePath(Span<byte> dst, ref int offset, LittleBit2Path path)
    {
        BinaryPrimitives.WriteInt32LittleEndian(dst.Slice(offset, 4), path.Rank);
        offset += 4;

        WriteUInt64Array(dst, ref offset, path.UBits);
        WriteUInt64Array(dst, ref offset, path.VBits);
        WriteHalfArray(dst, ref offset, path.RowScales);
        WriteHalfArray(dst, ref offset, path.ColScales);
        WriteHalfArray(dst, ref offset, path.LatentScales);
    }

    private static LittleBit2Path ReadPath(ReadOnlySpan<byte> src, ref int offset, int rows, int cols)
    {
        if (offset + 4 > src.Length)
            throw new InvalidDataException("Unexpected end of LittleBit2 payload reading path rank.");

        int rank = BinaryPrimitives.ReadInt32LittleEndian(src.Slice(offset, 4));
        offset += 4;

        ComputeAndValidatePathCounts(rows, cols, rank, out int uCount, out int vCount, out long neededBytes);

        long remaining = (long)src.Length - offset;
        if (neededBytes > remaining)
            throw new InvalidDataException($"LittleBit2 payload truncated: needed {neededBytes} path bytes at offset {offset}, remaining {remaining}.");

        ulong[] uBits = ReadUInt64Array(src, ref offset, uCount);
        ulong[] vBits = ReadUInt64Array(src, ref offset, vCount);
        Half[] rowScales = ReadHalfArray(src, ref offset, rows);
        Half[] colScales = ReadHalfArray(src, ref offset, cols);
        Half[] latentScales = ReadHalfArray(src, ref offset, rank);

        return new LittleBit2Path(rows, cols, rank, uBits, vBits, rowScales, colScales, latentScales);
    }

    private static void WritePathToStream(Stream stream, LittleBit2Path path)
    {
        Span<byte> rankBuf = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(rankBuf, path.Rank);
        stream.Write(rankBuf);

        WriteUInt64ArrayToStream(stream, path.UBits);
        WriteUInt64ArrayToStream(stream, path.VBits);
        WriteHalfArrayToStream(stream, path.RowScales);
        WriteHalfArrayToStream(stream, path.ColScales);
        WriteHalfArrayToStream(stream, path.LatentScales);
    }

    private static LittleBit2Path ReadPathFromStream(Stream stream, int rows, int cols)
    {
        Span<byte> rankBuf = stackalloc byte[4];
        ReadExactOrInvalidData(stream, rankBuf, "LittleBit2 path rank");
        int rank = BinaryPrimitives.ReadInt32LittleEndian(rankBuf);

        ComputeAndValidatePathCounts(rows, cols, rank, out int uCount, out int vCount, out long neededBytes);

        if (stream.CanSeek)
        {
            long remaining = stream.Length - stream.Position;
            if (neededBytes > remaining)
                throw new InvalidDataException($"LittleBit2 payload truncated: needed {neededBytes} path bytes, remaining {remaining}.");
        }

        ulong[] uBits = ReadUInt64ArrayFromStream(stream, uCount);
        ulong[] vBits = ReadUInt64ArrayFromStream(stream, vCount);
        Half[] rowScales = ReadHalfArrayFromStream(stream, rows);
        Half[] colScales = ReadHalfArrayFromStream(stream, cols);
        Half[] latentScales = ReadHalfArrayFromStream(stream, rank);

        return new LittleBit2Path(rows, cols, rank, uBits, vBits, rowScales, colScales, latentScales);
    }

    private static void ComputeAndValidatePathCounts(
        int rows,
        int cols,
        int rank,
        out int uCount,
        out int vCount,
        out long neededBytes)
    {
        ValidateRankForDeserialize(rows, cols, rank, out int uStride, out int vStride);

        long uCountLong = checked((long)rows * uStride);
        long vCountLong = checked((long)rank * vStride);

        int maxUInt64Count = Math.Min(Array.MaxLength, int.MaxValue / sizeof(ulong));
        int maxHalfCount = Math.Min(Array.MaxLength, int.MaxValue / 2);

        if (uCountLong > maxUInt64Count || vCountLong > maxUInt64Count)
            throw new InvalidDataException($"LittleBit2 path bit-array dimensions exceed maximum supported array length ({rows}x{cols}, rank={rank}).");

        if (rows > maxHalfCount || cols > maxHalfCount || rank > maxHalfCount)
            throw new InvalidDataException($"LittleBit2 path scale-array dimensions exceed maximum supported array length ({rows}x{cols}, rank={rank}).");

        neededBytes = checked((uCountLong + vCountLong) * sizeof(ulong) + ((long)rows + cols + rank) * 2L);
        uCount = (int)uCountLong;
        vCount = (int)vCountLong;
    }

    private static void ReadExactOrInvalidData(Stream stream, Span<byte> destination, string context)
    {
        int read = stream.ReadAtLeast(destination, destination.Length, throwOnEndOfStream: false);
        if (read < destination.Length)
            throw new InvalidDataException($"Unexpected end of stream reading {context}: expected {destination.Length} bytes, got {read}.");
    }

    private static void ValidateRankForDeserialize(int rows, int cols, int rank, out int uStride, out int vStride)
    {
        int maxRank = Math.Min(rows, cols);
        if (rank <= 0 || rank > maxRank)
            throw new InvalidDataException($"Invalid LittleBit2 path rank {rank} for dimensions ({rows}x{cols}).");

        uStride = (rank + 63) >> 6;
        vStride = (cols + 63) >> 6;
    }

    private static void WriteUInt64Array(Span<byte> dst, ref int offset, ulong[] values)
    {
        int byteCount = checked(values.Length * sizeof(ulong));
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(dst.Slice(offset, byteCount));
            offset += byteCount;
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(dst.Slice(offset, 8), values[i]);
            offset += 8;
        }
    }

    private static ulong[] ReadUInt64Array(ReadOnlySpan<byte> src, ref int offset, int count)
    {
        int byteCount = checked(count * sizeof(ulong));
        ulong[] result = new ulong[count];
        if (BitConverter.IsLittleEndian)
        {
            src.Slice(offset, byteCount).CopyTo(MemoryMarshal.AsBytes(result.AsSpan()));
            offset += byteCount;
            return result;
        }

        for (int i = 0; i < count; i++)
        {
            result[i] = BinaryPrimitives.ReadUInt64LittleEndian(src.Slice(offset, 8));
            offset += 8;
        }
        return result;
    }

    private static void WriteHalfArray(Span<byte> dst, ref int offset, Half[] values)
    {
        int byteCount = checked(values.Length * 2);
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(dst.Slice(offset, byteCount));
            offset += byteCount;
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(dst.Slice(offset, 2), BitConverter.HalfToUInt16Bits(values[i]));
            offset += 2;
        }
    }

    private static Half[] ReadHalfArray(ReadOnlySpan<byte> src, ref int offset, int count)
    {
        int byteCount = checked(count * 2);
        Half[] result = new Half[count];
        if (BitConverter.IsLittleEndian)
        {
            src.Slice(offset, byteCount).CopyTo(MemoryMarshal.AsBytes(result.AsSpan()));
            offset += byteCount;
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(offset, 2));
                result[i] = BitConverter.UInt16BitsToHalf(bits);
                offset += 2;
            }
        }

        for (int i = 0; i < result.Length; i++)
        {
            if (!Half.IsFinite(result[i]))
                throw new InvalidDataException($"Deserialized LittleBit2 scale contains non-finite Half value at index {i}.");
        }
        return result;
    }

    private static void WriteUInt64ArrayToStream(Stream stream, ulong[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }

        Span<byte> buf = stackalloc byte[8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(buf, values[i]);
            stream.Write(buf);
        }
    }

    private static ulong[] ReadUInt64ArrayFromStream(Stream stream, int count)
    {
        if (count == 0)
            return [];

        const int MaxChunkElements = 65_536; // 512 KB (<= 1 MB bounded chunk)
        if (count <= MaxChunkElements)
        {
            ulong[] smallResult = new ulong[count];
            if (BitConverter.IsLittleEndian)
            {
                ReadExactOrInvalidData(stream, MemoryMarshal.AsBytes(smallResult.AsSpan()), "LittleBit2 UInt64 array");
                return smallResult;
            }

            byte[] smallBuf = new byte[count * sizeof(ulong)];
            ReadExactOrInvalidData(stream, smallBuf, "LittleBit2 UInt64 array");
            for (int i = 0; i < count; i++)
                smallResult[i] = BinaryPrimitives.ReadUInt64LittleEndian(smallBuf.AsSpan(i * 8, 8));
            return smallResult;
        }

        byte[] chunkBuf = new byte[MaxChunkElements * sizeof(ulong)];
        var chunks = new List<ulong[]>();
        int remaining = count;
        while (remaining > 0)
        {
            int elems = Math.Min(remaining, MaxChunkElements);
            int bytes = elems * sizeof(ulong);
            ReadExactOrInvalidData(stream, chunkBuf.AsSpan(0, bytes), "LittleBit2 UInt64 chunk");

            ulong[] chunk = new ulong[elems];
            if (BitConverter.IsLittleEndian)
            {
                chunkBuf.AsSpan(0, bytes).CopyTo(MemoryMarshal.AsBytes(chunk.AsSpan()));
            }
            else
            {
                for (int i = 0; i < elems; i++)
                    chunk[i] = BinaryPrimitives.ReadUInt64LittleEndian(chunkBuf.AsSpan(i * 8, 8));
            }
            chunks.Add(chunk);
            remaining -= elems;
        }

        ulong[] result = new ulong[count];
        int offset = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            ulong[] c = chunks[i];
            c.CopyTo(result, offset);
            offset += c.Length;
        }
        return result;
    }

    private static void WriteHalfArrayToStream(Stream stream, Half[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }

        Span<byte> buf = stackalloc byte[2];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(buf, BitConverter.HalfToUInt16Bits(values[i]));
            stream.Write(buf);
        }
    }

    private static Half[] ReadHalfArrayFromStream(Stream stream, int count)
    {
        if (count == 0)
            return [];

        const int MaxChunkElements = 262_144; // 512 KB (<= 1 MB bounded chunk)
        Half[] result;
        if (count <= MaxChunkElements)
        {
            result = new Half[count];
            if (BitConverter.IsLittleEndian)
            {
                ReadExactOrInvalidData(stream, MemoryMarshal.AsBytes(result.AsSpan()), "LittleBit2 Half array");
            }
            else
            {
                byte[] smallBuf = new byte[count * 2];
                ReadExactOrInvalidData(stream, smallBuf, "LittleBit2 Half array");
                for (int i = 0; i < count; i++)
                    result[i] = BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(smallBuf.AsSpan(i * 2, 2)));
            }
        }
        else
        {
            byte[] chunkBuf = new byte[MaxChunkElements * 2];
            var chunks = new List<Half[]>();
            int remaining = count;
            while (remaining > 0)
            {
                int elems = Math.Min(remaining, MaxChunkElements);
                int bytes = elems * 2;
                ReadExactOrInvalidData(stream, chunkBuf.AsSpan(0, bytes), "LittleBit2 Half chunk");

                Half[] chunk = new Half[elems];
                if (BitConverter.IsLittleEndian)
                {
                    chunkBuf.AsSpan(0, bytes).CopyTo(MemoryMarshal.AsBytes(chunk.AsSpan()));
                }
                else
                {
                    for (int i = 0; i < elems; i++)
                        chunk[i] = BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(chunkBuf.AsSpan(i * 2, 2)));
                }
                for (int i = 0; i < chunk.Length; i++)
                {
                    if (!Half.IsFinite(chunk[i]))
                        throw new InvalidDataException($"Deserialized LittleBit2 scale contains non-finite Half value.");
                }
                chunks.Add(chunk);
                remaining -= elems;
            }

            result = new Half[count];
            int offset = 0;
            for (int i = 0; i < chunks.Count; i++)
            {
                Half[] c = chunks[i];
                c.CopyTo(result, offset);
                offset += c.Length;
            }
            return result;
        }

        for (int i = 0; i < result.Length; i++)
        {
            if (!Half.IsFinite(result[i]))
                throw new InvalidDataException($"Deserialized LittleBit2 scale contains non-finite Half value at index {i}.");
        }
        return result;
    }
}
