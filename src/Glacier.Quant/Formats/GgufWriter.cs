namespace Glacier.Quant.Formats;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Glacier.Quant.Kernels;

/// <summary>
/// Standard GGUF v2/v3 metadata value types (gguf_metadata_value_type).
/// </summary>
public enum GgufMetadataValueType : uint
{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12
}

/// <summary>
/// Streaming binary GGUF v3 writer adhering to power-of-two alignment and typed metadata contracts.
/// </summary>
public static class GgufWriter
{
    public const uint GgufMagic = 0x46554747; // "GGUF" in LE
    public const uint GgufVersion = 3;
    public const int DefaultAlignment = 32;

    public static void WriteToFile(
        string filePath,
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        int alignment = DefaultAlignment)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024);
        WriteToStream(stream, metadata, tensors, alignment);
    }

    public static void WriteToFile(
        string filePath,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        int alignment = DefaultAlignment)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024);
        WriteToStream(stream, metadata, tensors, alignment);
    }

    public static void WriteToStream(
        Stream stream,
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        int alignment = DefaultAlignment)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);

        var typed = new Dictionary<string, object>(metadata.Count);
        foreach (var (k, v) in metadata)
            typed[k] = v;

        WriteToStream(stream, typed, tensors, alignment);
    }

    public static void WriteToStream(
        Stream stream,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        int alignment = DefaultAlignment)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);

        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(alignment), alignment, "GGUF alignment must be a positive power of two.");

        long startPos = stream.Position;
        using var bw = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Header
        bw.Write(GgufMagic);
        bw.Write(GgufVersion);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        // 2. Metadata Key-Values (all 13 GGUF value types)
        foreach (var (k, v) in metadata)
        {
            ArgumentNullException.ThrowIfNull(k);
            ArgumentNullException.ThrowIfNull(v);
            WriteString(bw, k);
            GgufMetadataValueType valType = InferValueType(v);
            bw.Write((uint)valType);
            WriteValue(bw, valType, v);
        }

        // 3. Tensor Info Table (with relative offsets from data start)
        ulong currentDataOffset = 0;
        var calculatedOffsets = new List<ulong>(tensors.Count);
        ulong alignMask = (ulong)(alignment - 1);

        foreach (var (_, data) in tensors)
        {
            ArgumentNullException.ThrowIfNull(data);
            ulong alignedOffset = (currentDataOffset + alignMask) & ~alignMask;
            calculatedOffsets.Add(alignedOffset);
            currentDataOffset = checked(alignedOffset + (ulong)data.Length);
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (info, _) = tensors[i];
            ArgumentNullException.ThrowIfNull(info);
            WriteString(bw, info.Name);
            bw.Write((uint)info.Shape.Length);
            for (int d = 0; d < info.Shape.Length; d++)
            {
                if (info.Shape[d] < 0)
                    throw new ArgumentOutOfRangeException(nameof(tensors), $"Tensor '{info.Name}' contains negative dimension {info.Shape[d]}.");
                bw.Write((ulong)info.Shape[d]);
            }
            bw.Write(ToGgmlType(info.Type));
            bw.Write(calculatedOffsets[i]);
        }

        bw.Flush();

        // 4. Pad header section to alignment boundary before tensor data
        long headerEndRel = stream.Position - startPos;
        long alignLongMask = alignment - 1L;
        long dataStartRel = (headerEndRel + alignLongMask) & ~alignLongMask;
        long padBytes = dataStartRel - headerEndRel;
        if (padBytes > 0)
        {
            WriteZeroPadding(stream, padBytes);
        }

        // 5. Write Tensor Payloads
        for (int i = 0; i < tensors.Count; i++)
        {
            var (_, data) = tensors[i];
            long currentRel = stream.Position - startPos;
            long expectedRel = checked(dataStartRel + (long)calculatedOffsets[i]);
            if (expectedRel > currentRel)
            {
                WriteZeroPadding(stream, expectedRel - currentRel);
            }
            stream.Write(data);
        }

        stream.Flush();
    }

    internal static uint ToGgmlType(QuantType type) => type switch
    {
        QuantType.FP32 => 0u,
        QuantType.FP16 => 1u,
        QuantType.Q8_0 => 8u,
        QuantType.Q4_K_M => 12u,
        QuantType.FP8_E4M3 => 104u,
        QuantType.FP8_E5M2 => 105u,
        QuantType.Binary1Bit => 106u,
        QuantType.LittleBit2 => 107u,
        _ => throw new NotSupportedException($"Unsupported QuantType for GGUF serialization: {type}.")
    };

    internal static QuantType FromGgmlType(uint ggmlType) => ggmlType switch
    {
        0u => QuantType.FP32,
        1u => QuantType.FP16,
        8u => QuantType.Q8_0,
        12u => QuantType.Q4_K_M,
        104u => QuantType.FP8_E4M3,
        105u => QuantType.FP8_E5M2,
        106u => QuantType.Binary1Bit,
        107u => QuantType.LittleBit2,
        _ => throw new NotSupportedException($"Unsupported GGUF ggml_type ID: {ggmlType}.")
    };

    private static GgufMetadataValueType InferValueType(object value) => value switch
    {
        byte => GgufMetadataValueType.Uint8,
        sbyte => GgufMetadataValueType.Int8,
        ushort => GgufMetadataValueType.Uint16,
        short => GgufMetadataValueType.Int16,
        uint => GgufMetadataValueType.Uint32,
        int => GgufMetadataValueType.Int32,
        float => GgufMetadataValueType.Float32,
        bool => GgufMetadataValueType.Bool,
        string => GgufMetadataValueType.String,
        ulong => GgufMetadataValueType.Uint64,
        long => GgufMetadataValueType.Int64,
        double => GgufMetadataValueType.Float64,
        Array or IList => GgufMetadataValueType.Array,
        _ => throw new NotSupportedException($"Unsupported GGUF metadata value CLR type: {value.GetType().FullName}.")
    };

    private static void WriteValue(BinaryWriter bw, GgufMetadataValueType valType, object value)
    {
        switch (valType)
        {
            case GgufMetadataValueType.Uint8:
                bw.Write((byte)value);
                break;
            case GgufMetadataValueType.Int8:
                bw.Write((sbyte)value);
                break;
            case GgufMetadataValueType.Uint16:
                bw.Write((ushort)value);
                break;
            case GgufMetadataValueType.Int16:
                bw.Write((short)value);
                break;
            case GgufMetadataValueType.Uint32:
                bw.Write((uint)value);
                break;
            case GgufMetadataValueType.Int32:
                bw.Write((int)value);
                break;
            case GgufMetadataValueType.Float32:
                bw.Write((float)value);
                break;
            case GgufMetadataValueType.Bool:
                bw.Write((byte)((bool)value ? 1 : 0));
                break;
            case GgufMetadataValueType.String:
                WriteString(bw, (string)value);
                break;
            case GgufMetadataValueType.Uint64:
                bw.Write((ulong)value);
                break;
            case GgufMetadataValueType.Int64:
                bw.Write((long)value);
                break;
            case GgufMetadataValueType.Float64:
                bw.Write((double)value);
                break;
            case GgufMetadataValueType.Array:
                WriteArray(bw, value);
                break;
            default:
                throw new NotSupportedException($"Unsupported GGUF metadata type: {valType}.");
        }
    }

    private static void WriteArray(BinaryWriter bw, object value)
    {
        if (value is Array arr)
        {
            Type? elemClrType = arr.GetType().GetElementType();
            if (elemClrType == typeof(byte))
            {
                var typed = (byte[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Uint8, typed.Length);
                bw.Write(typed);
                return;
            }
            if (elemClrType == typeof(sbyte))
            {
                var typed = (sbyte[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Int8, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(ushort))
            {
                var typed = (ushort[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Uint16, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(short))
            {
                var typed = (short[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Int16, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(uint))
            {
                var typed = (uint[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Uint32, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(int))
            {
                var typed = (int[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Int32, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(float))
            {
                var typed = (float[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Float32, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(bool))
            {
                var typed = (bool[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Bool, typed.Length);
                foreach (var item in typed) bw.Write((byte)(item ? 1 : 0));
                return;
            }
            if (elemClrType == typeof(string))
            {
                var typed = (string[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.String, typed.Length);
                foreach (var item in typed) WriteString(bw, item ?? "");
                return;
            }
            if (elemClrType == typeof(ulong))
            {
                var typed = (ulong[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Uint64, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(long))
            {
                var typed = (long[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Int64, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
            if (elemClrType == typeof(double))
            {
                var typed = (double[])arr;
                WriteTypedArrayHeader(bw, GgufMetadataValueType.Float64, typed.Length);
                foreach (var item in typed) bw.Write(item);
                return;
            }
        }

        if (value is IList list)
        {
            if (list.Count == 0)
                throw new ArgumentException("Untyped empty list metadata values cannot determine GGUF array element type; pass a typed array (e.g., string[] or int[]).");

            object? first = list[0];
            ArgumentNullException.ThrowIfNull(first);
            GgufMetadataValueType elemType = InferValueType(first);
            WriteTypedArrayHeader(bw, elemType, list.Count);
            for (int i = 0; i < list.Count; i++)
            {
                object? elem = list[i];
                ArgumentNullException.ThrowIfNull(elem);
                if (InferValueType(elem) != elemType)
                    throw new ArgumentException("GGUF array elements must be homogeneous.");
                WriteValue(bw, elemType, elem);
            }
            return;
        }

        throw new NotSupportedException($"Unsupported GGUF array container type: {value.GetType().FullName}.");
    }

    private static void WriteTypedArrayHeader(BinaryWriter bw, GgufMetadataValueType elemType, int count)
    {
        bw.Write((uint)elemType);
        bw.Write((ulong)count);
    }

    private static void WriteString(BinaryWriter bw, string str)
    {
        int maxByteCount = Encoding.UTF8.GetByteCount(str);
        byte[]? rented = maxByteCount > 256 ? ArrayPool<byte>.Shared.Rent(maxByteCount) : null;
        Span<byte> buf = rented is not null ? rented.AsSpan(0, maxByteCount) : stackalloc byte[maxByteCount];
        try
        {
            int written = Encoding.UTF8.GetBytes(str, buf);
            bw.Write((ulong)written);
            bw.Write(buf.Slice(0, written));
        }
        finally
        {
            if (rented is not null)
                ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static void WriteZeroPadding(Stream stream, long padBytes)
    {
        Span<byte> zeros = stackalloc byte[4096];
        zeros.Clear();

        long remaining = padBytes;
        while (remaining > 0)
        {
            int chunk = (int)Math.Min(remaining, zeros.Length);
            stream.Write(zeros.Slice(0, chunk));
            remaining -= chunk;
        }
    }
}

/// <summary>
/// Streaming binary GGUF v2/v3 reader validating magic, version, all 13 metadata types,
/// bounds/OOM safety limits, and tensor alignment offsets.
/// </summary>
public static class GgufReader
{
    private const ulong MaxMetadataEntries = 1_000_000UL;
    private const ulong MaxTensorCount = 1_000_000UL;
    private const uint MaxDimensionsCount = 8u;
    private const ulong MaxStringLength = 4UL * 1024 * 1024;
    private const ulong MaxArrayElements = 2_000_000UL;
    private const int MaxNestingDepth = 4;
    private const ulong MaxInitialCollectionCapacity = 65_536UL;

    public static (Dictionary<string, string> Metadata, List<QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeader(Stream stream)
    {
        var (typedMeta, tensors, dataStart) = ReadHeaderTyped(stream);
        var stringMeta = new Dictionary<string, string>(typedMeta.Count);
        foreach (var (k, v) in typedMeta)
        {
            stringMeta[k] = FormatMetadataValue(v);
        }
        return (stringMeta, tensors, dataStart);
    }

    public static (Dictionary<string, object> Metadata, List<QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeaderTyped(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        bool canSeek = stream.CanSeek;
        long startPos = canSeek ? stream.Position : 0L;
        CountingReadStream? counting = canSeek ? null : new CountingReadStream(stream);
        Stream readStream = counting ?? stream;

        uint magic = ReadUInt32(readStream);
        if (magic != GgufWriter.GgufMagic)
            throw new InvalidDataException($"Invalid GGUF magic constant: 0x{magic:X8}");

        uint version = ReadUInt32(readStream);
        if (version != 2 && version != 3)
            throw new InvalidDataException($"Unsupported GGUF version: {version}");

        ulong tensorCount = ReadUInt64(readStream);
        ulong metadataCount = ReadUInt64(readStream);

        if (metadataCount > MaxMetadataEntries)
            throw new InvalidDataException($"GGUF metadata count ({metadataCount}) exceeds safety limit ({MaxMetadataEntries}).");
        if (tensorCount > MaxTensorCount)
            throw new InvalidDataException($"GGUF tensor count ({tensorCount}) exceeds safety limit ({MaxTensorCount}).");

        if (canSeek)
        {
            long remaining = Math.Max(0L, stream.Length - stream.Position);
            if (metadataCount > (ulong)remaining / 8UL)
                throw new InvalidDataException($"GGUF metadata count ({metadataCount}) exceeds remaining stream bytes ({remaining}).");
            if (tensorCount > (ulong)remaining / 16UL)
                throw new InvalidDataException($"GGUF tensor count ({tensorCount}) exceeds remaining stream bytes ({remaining}).");
        }

        int initMetaCap = checked((int)Math.Min(metadataCount, MaxInitialCollectionCapacity));
        var metadata = new Dictionary<string, object>(initMetaCap);

        for (ulong i = 0; i < metadataCount; i++)
        {
            string key = ReadString(readStream);
            uint rawType = ReadUInt32(readStream);
            if (rawType > (uint)GgufMetadataValueType.Float64)
                throw new InvalidDataException($"Unknown GGUF metadata value type ID: {rawType} for key '{key}'.");

            object val = ReadValue(readStream, (GgufMetadataValueType)rawType, depth: 0);
            metadata[key] = val;
        }

        int initTensorCap = checked((int)Math.Min(tensorCount, MaxInitialCollectionCapacity));
        var tensors = new List<QuantTensorMetadata>(initTensorCap);

        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(readStream);
            uint nDims = ReadUInt32(readStream);
            if (nDims > MaxDimensionsCount)
                throw new InvalidDataException($"GGUF tensor '{name}' rank ({nDims}) exceeds safety limit ({MaxDimensionsCount}).");

            long[] shape = new long[nDims];
            for (int d = 0; d < nDims; d++)
            {
                ulong rawDim = ReadUInt64(readStream);
                if (rawDim > long.MaxValue)
                    throw new InvalidDataException($"GGUF tensor '{name}' dimension ({rawDim}) exceeds Int64.MaxValue.");
                shape[d] = (long)rawDim;
            }

            uint type = ReadUInt32(readStream);
            ulong offset = ReadUInt64(readStream);

            tensors.Add(new QuantTensorMetadata
            {
                Name = name,
                Shape = shape,
                Type = GgufWriter.FromGgmlType(type),
                DataOffset = offset
            });
        }

        int alignment = ResolveAlignment(metadata);
        long alignMask = alignment - 1L;
        long relHeaderPos = canSeek ? (stream.Position - startPos) : counting!.BytesRead;
        long relDataStart = (relHeaderPos + alignMask) & ~alignMask;
        long dataStart = canSeek ? (startPos + relDataStart) : relDataStart;

        return (metadata, tensors, dataStart);
    }

    private sealed class CountingReadStream(Stream inner) : Stream
    {
        public long BytesRead { get; private set; }
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = inner.Read(buffer, offset, count);
            BytesRead += n;
            return n;
        }
        public override int Read(Span<byte> buffer)
        {
            int n = inner.Read(buffer);
            BytesRead += n;
            return n;
        }
        public override int ReadByte()
        {
            int b = inner.ReadByte();
            if (b >= 0) BytesRead++;
            return b;
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static int ResolveAlignment(Dictionary<string, object> metadata)
    {
        if (!metadata.TryGetValue("general.alignment", out object? alignObj))
            return GgufWriter.DefaultAlignment;

        long parsed = alignObj switch
        {
            uint u => u,
            int i => i,
            ulong ul when ul <= int.MaxValue => (long)ul,
            long l => l,
            ushort us => us,
            short s => s,
            byte b => b,
            sbyte sb => sb,
            _ => 0L
        };

        if (parsed > 0 && parsed <= int.MaxValue && (parsed & (parsed - 1)) == 0)
            return (int)parsed;

        return GgufWriter.DefaultAlignment;
    }

    private static object ReadValue(Stream stream, GgufMetadataValueType valType, int depth)
    {
        switch (valType)
        {
            case GgufMetadataValueType.Uint8:
                return ReadByte(stream);
            case GgufMetadataValueType.Int8:
                return unchecked((sbyte)ReadByte(stream));
            case GgufMetadataValueType.Uint16:
                return ReadUInt16(stream);
            case GgufMetadataValueType.Int16:
                return ReadInt16(stream);
            case GgufMetadataValueType.Uint32:
                return ReadUInt32(stream);
            case GgufMetadataValueType.Int32:
                return ReadInt32(stream);
            case GgufMetadataValueType.Float32:
                return ReadSingle(stream);
            case GgufMetadataValueType.Bool:
                return ReadByte(stream) != 0;
            case GgufMetadataValueType.String:
                return ReadString(stream);
            case GgufMetadataValueType.Uint64:
                return ReadUInt64(stream);
            case GgufMetadataValueType.Int64:
                return ReadInt64(stream);
            case GgufMetadataValueType.Float64:
                return ReadDouble(stream);
            case GgufMetadataValueType.Array:
                return ReadArray(stream, depth);
            default:
                throw new InvalidDataException($"Unknown GGUF metadata value type: {valType}.");
        }
    }

    private static object ReadArray(Stream stream, int depth)
    {
        if (depth >= MaxNestingDepth)
            throw new InvalidDataException($"GGUF nested array depth exceeded safety limit ({MaxNestingDepth}).");

        uint rawElemType = ReadUInt32(stream);
        if (rawElemType > (uint)GgufMetadataValueType.Float64)
            throw new InvalidDataException($"Unknown GGUF array element type ID: {rawElemType}.");

        var elemType = (GgufMetadataValueType)rawElemType;
        ulong count = ReadUInt64(stream);
        if (count > MaxArrayElements)
            throw new InvalidDataException($"GGUF array element count ({count}) exceeds safety limit ({MaxArrayElements}).");

        ulong minElemBytes = elemType switch
        {
            GgufMetadataValueType.Uint8 or GgufMetadataValueType.Int8 or GgufMetadataValueType.Bool => 1UL,
            GgufMetadataValueType.Uint16 or GgufMetadataValueType.Int16 => 2UL,
            GgufMetadataValueType.Uint32 or GgufMetadataValueType.Int32 or GgufMetadataValueType.Float32 => 4UL,
            GgufMetadataValueType.Uint64 or GgufMetadataValueType.Int64 or GgufMetadataValueType.Float64 or GgufMetadataValueType.String => 8UL,
            GgufMetadataValueType.Array => 12UL,
            _ => 1UL
        };

        if (stream.CanSeek)
        {
            long remaining = Math.Max(0L, stream.Length - stream.Position);
            if (count > (ulong)remaining / minElemBytes)
                throw new InvalidDataException($"GGUF array element count ({count}) exceeds remaining stream bytes ({remaining}).");
        }

        int n = checked((int)count);
        switch (elemType)
        {
            case GgufMetadataValueType.Uint8:
            {
                var arr = new byte[n];
                if (n > 0) stream.ReadExactly(arr);
                return arr;
            }
            case GgufMetadataValueType.Int8:
            {
                var arr = new sbyte[n];
                for (int i = 0; i < n; i++) arr[i] = unchecked((sbyte)ReadByte(stream));
                return arr;
            }
            case GgufMetadataValueType.Uint16:
            {
                var arr = new ushort[n];
                for (int i = 0; i < n; i++) arr[i] = ReadUInt16(stream);
                return arr;
            }
            case GgufMetadataValueType.Int16:
            {
                var arr = new short[n];
                for (int i = 0; i < n; i++) arr[i] = ReadInt16(stream);
                return arr;
            }
            case GgufMetadataValueType.Uint32:
            {
                var arr = new uint[n];
                for (int i = 0; i < n; i++) arr[i] = ReadUInt32(stream);
                return arr;
            }
            case GgufMetadataValueType.Int32:
            {
                var arr = new int[n];
                for (int i = 0; i < n; i++) arr[i] = ReadInt32(stream);
                return arr;
            }
            case GgufMetadataValueType.Float32:
            {
                var arr = new float[n];
                for (int i = 0; i < n; i++) arr[i] = ReadSingle(stream);
                return arr;
            }
            case GgufMetadataValueType.Bool:
            {
                var arr = new bool[n];
                for (int i = 0; i < n; i++) arr[i] = ReadByte(stream) != 0;
                return arr;
            }
            case GgufMetadataValueType.String:
            {
                var arr = new string[n];
                for (int i = 0; i < n; i++) arr[i] = ReadString(stream);
                return arr;
            }
            case GgufMetadataValueType.Uint64:
            {
                var arr = new ulong[n];
                for (int i = 0; i < n; i++) arr[i] = ReadUInt64(stream);
                return arr;
            }
            case GgufMetadataValueType.Int64:
            {
                var arr = new long[n];
                for (int i = 0; i < n; i++) arr[i] = ReadInt64(stream);
                return arr;
            }
            case GgufMetadataValueType.Float64:
            {
                var arr = new double[n];
                for (int i = 0; i < n; i++) arr[i] = ReadDouble(stream);
                return arr;
            }
            case GgufMetadataValueType.Array:
            {
                var arr = new object[n];
                for (int i = 0; i < n; i++) arr[i] = ReadArray(stream, depth + 1);
                return arr;
            }
            default:
                throw new InvalidDataException($"Unknown GGUF array element type: {elemType}.");
        }
    }

    private static string ReadString(Stream stream)
    {
        ulong len = ReadUInt64(stream);
        if (len > MaxStringLength)
            throw new InvalidDataException($"GGUF string length ({len} bytes) exceeds safety limit ({MaxStringLength}).");

        if (stream.CanSeek)
        {
            long remaining = Math.Max(0L, stream.Length - stream.Position);
            if (len > (ulong)remaining)
                throw new EndOfStreamException($"GGUF string length ({len} bytes) exceeds remaining stream bytes ({remaining}).");
        }

        int byteLen = checked((int)len);
        if (byteLen == 0)
            return string.Empty;

        byte[]? rented = byteLen > 256 ? ArrayPool<byte>.Shared.Rent(byteLen) : null;
        Span<byte> buf = rented is not null ? rented.AsSpan(0, byteLen) : stackalloc byte[byteLen];
        try
        {
            stream.ReadExactly(buf);
            return Encoding.UTF8.GetString(buf);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static byte ReadByte(Stream stream)
    {
        Span<byte> buf = stackalloc byte[1];
        stream.ReadExactly(buf);
        return buf[0];
    }

    private static ushort ReadUInt16(Stream stream)
    {
        Span<byte> buf = stackalloc byte[2];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadUInt16LittleEndian(buf);
    }

    private static short ReadInt16(Stream stream)
    {
        Span<byte> buf = stackalloc byte[2];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadInt16LittleEndian(buf);
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> buf = stackalloc byte[4];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadUInt32LittleEndian(buf);
    }

    private static int ReadInt32(Stream stream)
    {
        Span<byte> buf = stackalloc byte[4];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadInt32LittleEndian(buf);
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> buf = stackalloc byte[4];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadSingleLittleEndian(buf);
    }

    private static ulong ReadUInt64(Stream stream)
    {
        Span<byte> buf = stackalloc byte[8];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadUInt64LittleEndian(buf);
    }

    private static long ReadInt64(Stream stream)
    {
        Span<byte> buf = stackalloc byte[8];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadInt64LittleEndian(buf);
    }

    private static double ReadDouble(Stream stream)
    {
        Span<byte> buf = stackalloc byte[8];
        stream.ReadExactly(buf);
        return BinaryPrimitives.ReadDoubleLittleEndian(buf);
    }

    private static string FormatMetadataValue(object value) => value switch
    {
        string s => s,
        bool b => b ? "true" : "false",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        Array arr => FormatArray(arr),
        _ => value.ToString() ?? string.Empty
    };

    private static string FormatArray(Array arr)
    {
        var sb = new StringBuilder();
        sb.Append('[');
        for (int i = 0; i < arr.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            object? item = arr.GetValue(i);
            sb.Append(item is null ? string.Empty : FormatMetadataValue(item));
        }
        sb.Append(']');
        return sb.ToString();
    }
}
