namespace Glacier.Quant.Formats;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Quant.Kernels;

/// <summary>
/// Streaming binary GGUF v3 writer adhering to standard 32-byte alignment and metadata contracts.
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

    public static void WriteToStream(
        Stream stream,
        IReadOnlyDictionary<string, string> metadata,
        IReadOnlyList<(QuantTensorMetadata Info, byte[] Data)> tensors,
        int alignment = DefaultAlignment)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);

        using var bw = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Header
        bw.Write(GgufMagic);
        bw.Write(GgufVersion);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        // 2. Metadata Key-Values (string type = 8)
        foreach (var (k, v) in metadata)
        {
            WriteString(bw, k);
            bw.Write((uint)8); // GGUF_METADATA_VALUE_TYPE_STRING
            WriteString(bw, v);
        }

        // 3. Tensor Info Table (with relative offsets from data start)
        ulong currentDataOffset = 0;
        var calculatedOffsets = new List<ulong>(tensors.Count);

        foreach (var (info, data) in tensors)
        {
            // Align current offset to boundary
            ulong alignedOffset = (currentDataOffset + (ulong)(alignment - 1)) & ~((ulong)(alignment - 1));
            calculatedOffsets.Add(alignedOffset);
            currentDataOffset = alignedOffset + (ulong)data.Length;
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (info, _) = tensors[i];
            WriteString(bw, info.Name);
            bw.Write((uint)info.Shape.Length);
            for (int d = 0; d < info.Shape.Length; d++)
            {
                bw.Write((ulong)info.Shape[d]);
            }
            bw.Write((uint)info.Type);
            bw.Write(calculatedOffsets[i]);
        }

        // 4. Pad header section to alignment boundary before tensor data
        long headerEndPos = stream.Position;
        long dataStartPos = (headerEndPos + (alignment - 1)) & ~(alignment - 1);
        int padBytes = (int)(dataStartPos - headerEndPos);
        if (padBytes > 0)
        {
            bw.Write(new byte[padBytes]);
        }

        // 5. Write Tensor Payloads
        for (int i = 0; i < tensors.Count; i++)
        {
            var (_, data) = tensors[i];
            long currentPos = stream.Position;
            long expectedPos = dataStartPos + (long)calculatedOffsets[i];
            if (expectedPos > currentPos)
            {
                bw.Write(new byte[(int)(expectedPos - currentPos)]);
            }
            bw.Write(data);
        }

        bw.Flush();
    }

    private static void WriteString(BinaryWriter bw, string str)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(str);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }
}

/// <summary>
/// Streaming binary GGUF v3 reader validating magic, version, metadata, and tensor offsets.
/// </summary>
public static class GgufReader
{
    public static (Dictionary<string, string> Metadata, List<QuantTensorMetadata> Tensors, long DataStartOffset) ReadHeader(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var br = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        uint magic = br.ReadUInt32();
        if (magic != GgufWriter.GgufMagic)
            throw new InvalidDataException($"Invalid GGUF magic constant: 0x{magic:X8}");

        uint version = br.ReadUInt32();
        if (version != 2 && version != 3)
            throw new InvalidDataException($"Unsupported GGUF version: {version}");

        ulong tensorCount = br.ReadUInt64();
        ulong metadataCount = br.ReadUInt64();

        var metadata = new Dictionary<string, string>((int)metadataCount);
        for (ulong i = 0; i < metadataCount; i++)
        {
            string key = ReadString(br);
            uint valType = br.ReadUInt32();
            if (valType == 8) // string
            {
                string val = ReadString(br);
                metadata[key] = val;
            }
            else
            {
                // Skip unsupported types
                throw new NotSupportedException($"Metadata type {valType} reader not implemented.");
            }
        }

        var tensors = new List<QuantTensorMetadata>((int)tensorCount);
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(br);
            uint nDims = br.ReadUInt32();
            long[] shape = new long[nDims];
            for (int d = 0; d < nDims; d++) shape[d] = (long)br.ReadUInt64();
            uint type = br.ReadUInt32();
            ulong offset = br.ReadUInt64();

            tensors.Add(new QuantTensorMetadata
            {
                Name = name,
                Shape = shape,
                Type = (QuantType)type,
                DataOffset = offset
            });
        }

        long headerPos = stream.Position;
        long dataStart = (headerPos + 31) & ~31;

        return (metadata, tensors, dataStart);
    }

    private static string ReadString(BinaryReader br)
    {
        ulong len = br.ReadUInt64();
        byte[] bytes = br.ReadBytes((int)len);
        return Encoding.UTF8.GetString(bytes);
    }
}
