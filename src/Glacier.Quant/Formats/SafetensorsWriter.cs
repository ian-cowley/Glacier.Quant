namespace Glacier.Quant.Formats;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Glacier.Quant.Kernels;

/// <summary>
/// High-throughput streaming Safetensors format writer with UTF-8 JSON header prefix
/// and HuggingFace-compliant <c>__metadata__</c> preservation of Glacier quantization types and scales.
/// </summary>
public static class SafetensorsWriter
{
    public static void WriteToFile(
        string filePath,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024);
        WriteToStream(stream, tensors);
    }

    public static void WriteToFile(
        string filePath,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        IReadOnlyDictionary<string, string>? metadata)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024);
        WriteToStream(stream, tensors, metadata);
    }

    public static void WriteToStream(
        Stream stream,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors)
        => WriteToStream(stream, tensors, metadata: null);

    public static void WriteToStream(
        Stream stream,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        IReadOnlyDictionary<string, string>? metadata)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(tensors);

        // 1. Build JSON metadata object using AOT-safe Utf8JsonWriter
        byte[] headerJsonBytes;
        using (var ms = new MemoryStream())
        {
            using (var writer = new Utf8JsonWriter(ms))
            {
                writer.WriteStartObject();

                // Write HF-compliant string-valued __metadata__ section first
                writer.WriteStartObject("__metadata__");
                bool hasFormat = false;
                if (metadata is not null)
                {
                    foreach (var (k, v) in metadata)
                    {
                        ArgumentNullException.ThrowIfNull(k);
                        if (k.StartsWith("glacier.quant.", StringComparison.Ordinal))
                            continue;
                        if (k == "format")
                            hasFormat = true;
                        writer.WriteString(k, v ?? string.Empty);
                    }
                }
                if (!hasFormat)
                    writer.WriteString("format", "pt");
                writer.WriteString("glacier.quant.version", "1");

                foreach (var (info, _) in tensors)
                {
                    ArgumentNullException.ThrowIfNull(info);
                    writer.WriteString($"glacier.quant.type.{info.Name}", info.Type.ToString());
                    if (info.DynamicScale.HasValue)
                    {
                        writer.WriteString(
                            $"glacier.quant.scale.{info.Name}",
                            info.DynamicScale.Value.ToString("R", CultureInfo.InvariantCulture));
                    }
                    if (info.ScaleTensorName is not null)
                    {
                        writer.WriteString($"glacier.quant.scales.{info.Name}", info.ScaleTensorName);
                    }
                }
                writer.WriteEndObject();

                ulong currentOffset = 0;
                foreach (var (info, data) in tensors)
                {
                    ArgumentNullException.ThrowIfNull(data);
                    writer.WriteStartObject(info.Name);
                    writer.WriteString("dtype", MapQuantTypeToSafetensorsDtype(info.Type));

                    writer.WriteStartArray("shape");
                    foreach (var dim in info.Shape)
                    {
                        if (dim < 0)
                            throw new ArgumentOutOfRangeException(nameof(tensors), $"Tensor '{info.Name}' contains negative dimension {dim}.");
                        writer.WriteNumberValue(dim);
                    }
                    writer.WriteEndArray();

                    writer.WriteStartArray("data_offsets");
                    writer.WriteNumberValue(currentOffset);
                    currentOffset = checked(currentOffset + (ulong)data.Length);
                    writer.WriteNumberValue(currentOffset);
                    writer.WriteEndArray();

                    writer.WriteEndObject();
                }

                writer.WriteEndObject();
            }
            headerJsonBytes = ms.ToArray();
        }
        ulong headerLen = (ulong)headerJsonBytes.Length;

        // 2. Write 8-byte LE header length prefix
        Span<byte> lenBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lenBytes, headerLen);
        stream.Write(lenBytes);

        // 3. Write UTF-8 JSON header bytes
        stream.Write(headerJsonBytes);

        // 4. Write contiguous raw tensor data buffers
        foreach (var (_, data) in tensors)
        {
            stream.Write(data);
        }

        stream.Flush();
    }

    /// <summary>
    /// Writes a Q8_0 quantized tensor as separated HF-compatible <c>I8</c> weights (<paramref name="name"/>)
    /// and <c>F16</c> block scales (<c>{name}.scale</c>) linked via <c>glacier.quant.scales.{name}</c>.
    /// </summary>
    public static unsafe void WriteQuantizedQ8_0(
        Stream stream,
        string name,
        long[] shape,
        ReadOnlySpan<BlockQ8_0> blocks,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(shape);

        int numBlocks = blocks.Length;
        byte[] quantBytes = new byte[checked(numBlocks * BlockQ8_0.BlockSize)];
        byte[] scaleBytes = new byte[checked(numBlocks * sizeof(ushort))];

        for (int b = 0; b < numBlocks; b++)
        {
            BlockQ8_0 blk = blocks[b];
            ushort scaleBits = BitConverter.HalfToUInt16Bits(blk.Scale);
            BinaryPrimitives.WriteUInt16LittleEndian(scaleBytes.AsSpan(b * 2, 2), scaleBits);

            int qOff = b * BlockQ8_0.BlockSize;
            for (int i = 0; i < BlockQ8_0.BlockSize; i++)
            {
                quantBytes[qOff + i] = unchecked((byte)blk.Quants[i]);
            }
        }

        string scaleName = $"{name}.scale";
        var weightInfo = new QuantTensorMetadata
        {
            Name = name,
            Shape = (long[])shape.Clone(),
            Type = QuantType.Q8_0,
            ScaleTensorName = scaleName
        };
        var scaleInfo = new QuantTensorMetadata
        {
            Name = scaleName,
            Shape = [numBlocks],
            Type = QuantType.FP16
        };

        WriteToStream(stream, [(weightInfo, quantBytes), (scaleInfo, scaleBytes)], metadata);
    }

    public static string MapQuantTypeToSafetensorsDtype(QuantType type) => type switch
    {
        QuantType.FP32 => "F32",
        QuantType.FP16 => "F16",
        QuantType.Q8_0 => "I8",
        QuantType.FP8_E4M3 => "F8_E4M3",
        QuantType.FP8_E5M2 => "F8_E5M2",
        QuantType.Binary1Bit => "U64",
        QuantType.LittleBit2 => "U8",
        _ => "U8"
    };
}

/// <summary>
/// Streaming Safetensors reader extracting header length, <c>__metadata__</c>, and tensor buffers.
/// </summary>
public static class SafetensorsReader
{
    private const ulong MaxHeaderBytes = 100UL * 1024 * 1024;

    public static (Dictionary<string, QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeader(Stream stream)
    {
        var (_, tensors, dataStartPos) = ReadHeaderWithMetadata(stream);
        return (tensors, dataStartPos);
    }

    public static (Dictionary<string, string> Metadata, Dictionary<string, QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeaderWithMetadata(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        Span<byte> lenBytes = stackalloc byte[8];
        stream.ReadExactly(lenBytes);

        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBytes);
        if (headerLen == 0 || headerLen > MaxHeaderBytes)
            throw new InvalidDataException($"Safetensors header size ({headerLen} bytes) is invalid or exceeds safety limit.");

        if (stream.CanSeek)
        {
            long remaining = Math.Max(0L, stream.Length - stream.Position);
            if (headerLen > (ulong)remaining)
                throw new EndOfStreamException($"Safetensors header size ({headerLen} bytes) exceeds remaining stream bytes ({remaining}).");
        }

        byte[] headerBytes = new byte[checked((int)headerLen)];
        stream.ReadExactly(headerBytes);

        string json = Encoding.UTF8.GetString(headerBytes);
        using var doc = JsonDocument.Parse(json);

        if (doc.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Safetensors JSON header root must be an object.");

        long dataStartPos = stream.CanSeek ? stream.Position : checked(8L + (long)headerLen);
        long? remainingAfterHeader = stream.CanSeek ? Math.Max(0L, stream.Length - dataStartPos) : null;

        // Pass 1: parse __metadata__ regardless of property ordering
        var metadata = new Dictionary<string, string>();
        if (doc.RootElement.TryGetProperty("__metadata__", out JsonElement metaEl))
        {
            if (metaEl.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Safetensors '__metadata__' entry must be a JSON object.");

            foreach (var kv in metaEl.EnumerateObject())
            {
                if (kv.Value.ValueKind == JsonValueKind.String)
                    metadata[kv.Name] = kv.Value.GetString() ?? string.Empty;
                else
                    metadata[kv.Name] = kv.Value.ToString();
            }
        }

        // Pass 2: parse tensor entries and restore Glacier quantization type & scale metadata
        var tensors = new Dictionary<string, QuantTensorMetadata>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
                continue;

            var val = prop.Value;
            if (val.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"Safetensors tensor entry '{prop.Name}' must be a JSON object.");

            string dtype = val.GetProperty("dtype").GetString() ?? "F32";

            var shapeEl = val.GetProperty("shape");
            if (shapeEl.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException($"Safetensors tensor '{prop.Name}' shape must be a JSON array.");

            long[] shape = new long[shapeEl.GetArrayLength()];
            int idx = 0;
            foreach (var d in shapeEl.EnumerateArray())
            {
                long dim = d.GetInt64();
                if (dim < 0)
                    throw new InvalidDataException($"Safetensors tensor '{prop.Name}' contains negative dimension {dim}.");
                shape[idx++] = dim;
            }

            var offsetsEl = val.GetProperty("data_offsets");
            if (offsetsEl.ValueKind != JsonValueKind.Array || offsetsEl.GetArrayLength() != 2)
                throw new InvalidDataException($"Safetensors tensor '{prop.Name}' data_offsets must be a 2-element array.");

            long rawStart = offsetsEl[0].GetInt64();
            long rawEnd = offsetsEl[1].GetInt64();
            if (rawStart < 0 || rawEnd < rawStart)
                throw new InvalidDataException($"Safetensors tensor '{prop.Name}' has invalid data_offsets [{rawStart}, {rawEnd}].");
            if (remainingAfterHeader.HasValue && rawEnd > remainingAfterHeader.Value)
                throw new InvalidDataException($"Safetensors tensor '{prop.Name}' end offset ({rawEnd}) exceeds payload size ({remainingAfterHeader.Value}).");

            ulong startOffset = (ulong)rawStart;
            ulong endOffset = (ulong)rawEnd;

            QuantType quantType = metadata.TryGetValue($"glacier.quant.type.{prop.Name}", out string? qTypeStr)
                && Enum.TryParse(qTypeStr, ignoreCase: false, out QuantType parsedType)
                ? parsedType
                : MapDtypeToQuantType(dtype);

            float? dynamicScale = metadata.TryGetValue($"glacier.quant.scale.{prop.Name}", out string? scaleStr)
                && float.TryParse(scaleStr, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float parsedScale)
                ? parsedScale
                : null;

            string? scaleTensorName = metadata.TryGetValue($"glacier.quant.scales.{prop.Name}", out string? scalesName)
                ? scalesName
                : null;

            tensors[prop.Name] = new QuantTensorMetadata
            {
                Name = prop.Name,
                Shape = shape,
                Type = quantType,
                DataOffset = startOffset,
                DataLength = endOffset - startOffset,
                DynamicScale = dynamicScale,
                ScaleTensorName = scaleTensorName
            };
        }

        return (metadata, tensors, dataStartPos);
    }

    /// <summary>
    /// Reads a Q8_0 tensor written with <see cref="SafetensorsWriter.WriteQuantizedQ8_0"/> by combining
    /// its <c>I8</c> quants and companion <c>F16</c> scale tensor into <see cref="BlockQ8_0"/> blocks.
    /// </summary>
    public static unsafe BlockQ8_0[] ReadQuantizedQ8_0(Stream stream, string name)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (!stream.CanSeek)
            throw new NotSupportedException("ReadQuantizedQ8_0 requires a seekable stream.");

        var (_, tensors, dataStartPos) = ReadHeaderWithMetadata(stream);
        if (!tensors.TryGetValue(name, out QuantTensorMetadata? weightInfo))
            throw new KeyNotFoundException($"Tensor '{name}' not found in Safetensors stream.");

        string scaleName = weightInfo.ScaleTensorName ?? $"{name}.scale";
        if (!tensors.TryGetValue(scaleName, out QuantTensorMetadata? scaleInfo))
            throw new KeyNotFoundException($"Companion scale tensor '{scaleName}' not found in Safetensors stream.");

        if (weightInfo.DataLength % BlockQ8_0.BlockSize != 0)
            throw new InvalidDataException($"Tensor '{name}' byte length ({weightInfo.DataLength}) is not a multiple of Q8_0 block size ({BlockQ8_0.BlockSize}).");

        int numBlocks = checked((int)(weightInfo.DataLength / BlockQ8_0.BlockSize));
        ulong expectedScaleBytes = checked((ulong)numBlocks * sizeof(ushort));
        if (scaleInfo.DataLength != expectedScaleBytes)
            throw new InvalidDataException($"Scale tensor '{scaleName}' length ({scaleInfo.DataLength}) does not match expected ({expectedScaleBytes}).");

        byte[] quantBytes = new byte[checked((int)weightInfo.DataLength)];
        byte[] scaleBytes = new byte[checked((int)scaleInfo.DataLength)];

        stream.Position = checked(dataStartPos + (long)weightInfo.DataOffset);
        stream.ReadExactly(quantBytes);

        stream.Position = checked(dataStartPos + (long)scaleInfo.DataOffset);
        stream.ReadExactly(scaleBytes);

        var blocks = new BlockQ8_0[numBlocks];
        for (int b = 0; b < numBlocks; b++)
        {
            ushort scaleBits = BinaryPrimitives.ReadUInt16LittleEndian(scaleBytes.AsSpan(b * 2, 2));
            BlockQ8_0 blk = default;
            blk.Scale = BitConverter.UInt16BitsToHalf(scaleBits);

            int qOff = b * BlockQ8_0.BlockSize;
            for (int i = 0; i < BlockQ8_0.BlockSize; i++)
            {
                blk.Quants[i] = unchecked((sbyte)quantBytes[qOff + i]);
            }
            blocks[b] = blk;
        }

        return blocks;
    }

    private static QuantType MapDtypeToQuantType(string dtype) => dtype.ToUpperInvariant() switch
    {
        "F32" => QuantType.FP32,
        "F16" => QuantType.FP16,
        "I8" => QuantType.Q8_0,
        "F8_E4M3" => QuantType.FP8_E4M3,
        "F8_E5M2" => QuantType.FP8_E5M2,
        "U64" => QuantType.Binary1Bit,
        _ => QuantType.Q4_K_M
    };
}
