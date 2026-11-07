namespace Glacier.Quant.Formats;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

/// <summary>
/// High-throughput streaming Safetensors format writer with UTF-8 JSON header prefix.
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

    public static void WriteToStream(
        Stream stream,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors)
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
                ulong currentOffset = 0;

                foreach (var (info, data) in tensors)
                {
                    writer.WriteStartObject(info.Name);
                    writer.WriteString("dtype", MapQuantTypeToSafetensorsDtype(info.Type));

                    writer.WriteStartArray("shape");
                    foreach (var dim in info.Shape) writer.WriteNumberValue(dim);
                    writer.WriteEndArray();

                    writer.WriteStartArray("data_offsets");
                    writer.WriteNumberValue((long)currentOffset);
                    currentOffset += (ulong)data.Length;
                    writer.WriteNumberValue((long)currentOffset);
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

    public static string MapQuantTypeToSafetensorsDtype(Kernels.QuantType type) => type switch
    {
        Kernels.QuantType.FP32 => "F32",
        Kernels.QuantType.FP16 => "F16",
        Kernels.QuantType.Q8_0 => "I8",
        Kernels.QuantType.FP8_E4M3 => "F8_E4M3",
        Kernels.QuantType.FP8_E5M2 => "F8_E5M2",
        Kernels.QuantType.Binary1Bit => "U64",
        _ => "U8"
    };
}

/// <summary>
/// Streaming Safetensors reader extracting header length, JSON metadata, and tensor buffers.
/// </summary>
public static class SafetensorsReader
{
    public static (Dictionary<string, QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeader(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        Span<byte> lenBytes = stackalloc byte[8];
        int read = stream.Read(lenBytes);
        if (read < 8) throw new EndOfStreamException("Unable to read Safetensors 8-byte header length.");

        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBytes);
        if (headerLen > 100 * 1024 * 1024)
            throw new InvalidDataException($"Safetensors header size ({headerLen} bytes) exceeds safety limit.");

        byte[] headerBytes = new byte[headerLen];
        int totalRead = 0;
        while (totalRead < (int)headerLen)
        {
            int r = stream.Read(headerBytes, totalRead, (int)headerLen - totalRead);
            if (r == 0) throw new EndOfStreamException("Unexpected EOF while reading Safetensors JSON header.");
            totalRead += r;
        }

        string json = Encoding.UTF8.GetString(headerBytes);
        using var doc = JsonDocument.Parse(json);

        var result = new Dictionary<string, QuantTensorMetadata>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;

            var val = prop.Value;
            string dtype = val.GetProperty("dtype").GetString() ?? "F32";

            var shapeEl = val.GetProperty("shape");
            long[] shape = new long[shapeEl.GetArrayLength()];
            int idx = 0;
            foreach (var d in shapeEl.EnumerateArray()) shape[idx++] = d.GetInt64();

            var offsetsEl = val.GetProperty("data_offsets");
            ulong startOffset = (ulong)offsetsEl[0].GetInt64();
            ulong endOffset = (ulong)offsetsEl[1].GetInt64();

            result[prop.Name] = new QuantTensorMetadata
            {
                Name = prop.Name,
                Shape = shape,
                Type = MapDtypeToQuantType(dtype),
                DataOffset = startOffset,
                DataLength = endOffset - startOffset
            };
        }

        long dataStartPos = stream.Position;
        return (result, dataStartPos);
    }

    private static Kernels.QuantType MapDtypeToQuantType(string dtype) => dtype.ToUpperInvariant() switch
    {
        "F32" => Kernels.QuantType.FP32,
        "F16" => Kernels.QuantType.FP16,
        "I8" => Kernels.QuantType.Q8_0,
        "F8_E4M3" => Kernels.QuantType.FP8_E4M3,
        "F8_E5M2" => Kernels.QuantType.FP8_E5M2,
        "U64" => Kernels.QuantType.Binary1Bit,
        _ => Kernels.QuantType.Q4_K_M
    };
}
