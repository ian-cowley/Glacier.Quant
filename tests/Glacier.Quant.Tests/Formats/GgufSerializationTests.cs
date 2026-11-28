namespace Glacier.Quant.Tests.Formats;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using Glacier.Quant.Formats;
using Glacier.Quant.Kernels;
using Xunit;

public class GgufSerializationTests
{
    [Fact]
    public void GgufWriter_And_GgufReader_RoundtripTensorsAndMetadata_Correctly()
    {
        var metadata = new Dictionary<string, string>
        {
            ["general.architecture"] = "llama",
            ["llama.block_count"] = "32",
            ["general.name"] = "Glacier-Quant-Test"
        };

        byte[] tensor1Data = new byte[64];
        for (int i = 0; i < tensor1Data.Length; i++) tensor1Data[i] = (byte)(i + 1);

        byte[] tensor2Data = new byte[128];
        for (int i = 0; i < tensor2Data.Length; i++) tensor2Data[i] = (byte)(255 - i);

        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata
            {
                Name = "blk.0.attn_k.weight",
                Shape = [16, 4],
                Type = QuantType.Q8_0
            }, tensor1Data),
            (new QuantTensorMetadata
            {
                Name = "blk.0.attn_v.weight",
                Shape = [32, 4],
                Type = QuantType.FP16
            }, tensor2Data)
        };

        using var ms = new MemoryStream();
        GgufWriter.WriteToStream(ms, metadata, tensors, alignment: 32);

        ms.Position = 0;
        var (readMeta, readTensors, dataStart) = GgufReader.ReadHeader(ms);

        // Verify metadata
        Assert.Equal("llama", readMeta["general.architecture"]);
        Assert.Equal("32", readMeta["llama.block_count"]);
        Assert.Equal("Glacier-Quant-Test", readMeta["general.name"]);

        // Verify tensor table
        Assert.Equal(2, readTensors.Count);
        Assert.Equal("blk.0.attn_k.weight", readTensors[0].Name);
        Assert.Equal([16, 4], readTensors[0].Shape);
        Assert.Equal(QuantType.Q8_0, readTensors[0].Type);
        Assert.Equal(0UL, readTensors[0].DataOffset);

        Assert.Equal("blk.0.attn_v.weight", readTensors[1].Name);
        Assert.Equal([32, 4], readTensors[1].Shape);
        Assert.Equal(QuantType.FP16, readTensors[1].Type);
        Assert.True(readTensors[1].DataOffset >= 64);
        Assert.Equal(0UL, readTensors[1].DataOffset % 32); // 32-byte aligned

        // Verify raw bytes
        ms.Position = dataStart + (long)readTensors[0].DataOffset;
        byte[] read1 = new byte[tensor1Data.Length];
        ms.ReadExactly(read1);
        Assert.Equal(tensor1Data, read1);

        ms.Position = dataStart + (long)readTensors[1].DataOffset;
        byte[] read2 = new byte[tensor2Data.Length];
        ms.ReadExactly(read2);
        Assert.Equal(tensor2Data, read2);

        // Verify standard ggml_type IDs are written (Q8_0=8, Q4_K=12, FP16=1)
        var q4kTensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "w", Shape = [256], Type = QuantType.Q4_K_M }, new byte[144])
        };
        using var ms2 = new MemoryStream();
        GgufWriter.WriteToStream(ms2, new Dictionary<string, string>(), q4kTensors);
        ms2.Position = 0;
        var (_, rt2, _) = GgufReader.ReadHeader(ms2);
        Assert.Equal(QuantType.Q4_K_M, rt2[0].Type);
        Assert.Equal(8u, GgufWriter.ToGgmlType(QuantType.Q8_0));
        Assert.Equal(12u, GgufWriter.ToGgmlType(QuantType.Q4_K_M));
    }

    [Fact]
    public void GgufWriter_And_GgufReader_All13MetadataTypesAndAlignment_Roundtrip()
    {
        var metadata = new Dictionary<string, object>
        {
            ["t.uint8"] = (byte)200,
            ["t.int8"] = (sbyte)-42,
            ["t.uint16"] = (ushort)50000,
            ["t.int16"] = (short)-12345,
            ["t.uint32"] = 3_000_000_000u,
            ["t.int32"] = -987654321,
            ["t.float32"] = 3.25f,
            ["t.bool"] = true,
            ["t.string"] = "glacier-quant-gguf",
            ["t.uint64"] = 9_000_000_000_000UL,
            ["t.int64"] = -9_000_000_000_000L,
            ["t.float64"] = 6.5d,
            ["t.arr_sbyte"] = new sbyte[] { -1, 0, 127 },
            ["t.arr_short"] = new short[] { -100, 200 },
            ["t.arr_int"] = new[] { 10, 20, 30 },
            ["t.arr_uint"] = new uint[] { 10u, 20u, 30u },
            ["t.arr_long"] = new long[] { -500L, 500L },
            ["t.arr_float"] = new float[] { 1.5f, -2.5f },
            ["t.arr_str"] = new[] { "alpha", "beta" },
            ["general.alignment"] = 64u
        };

        byte[] payload = new byte[64];
        for (int i = 0; i < payload.Length; i++) payload[i] = (byte)i;
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "w.weight", Shape = [8, 8], Type = QuantType.LittleBit2 }, payload)
        };

        using var ms = new MemoryStream();
        GgufWriter.WriteToStream(ms, metadata, tensors, alignment: 64);

        ms.Position = 0;
        var (typedMeta, readTensors, dataStart) = GgufReader.ReadHeaderTyped(ms);

        Assert.Equal((byte)200, typedMeta["t.uint8"]);
        Assert.Equal((sbyte)-42, typedMeta["t.int8"]);
        Assert.Equal((ushort)50000, typedMeta["t.uint16"]);
        Assert.Equal((short)-12345, typedMeta["t.int16"]);
        Assert.Equal(3_000_000_000u, typedMeta["t.uint32"]);
        Assert.Equal(-987654321, typedMeta["t.int32"]);
        Assert.Equal(3.25f, typedMeta["t.float32"]);
        Assert.Equal(true, typedMeta["t.bool"]);
        Assert.Equal("glacier-quant-gguf", typedMeta["t.string"]);
        Assert.Equal(9_000_000_000_000UL, typedMeta["t.uint64"]);
        Assert.Equal(-9_000_000_000_000L, typedMeta["t.int64"]);
        Assert.Equal(6.5d, typedMeta["t.float64"]);
        Assert.Equal(new sbyte[] { -1, 0, 127 }, Assert.IsType<sbyte[]>(typedMeta["t.arr_sbyte"]));
        Assert.Equal(new short[] { -100, 200 }, Assert.IsType<short[]>(typedMeta["t.arr_short"]));
        Assert.Equal(new[] { 10, 20, 30 }, Assert.IsType<int[]>(typedMeta["t.arr_int"]));
        Assert.Equal(new uint[] { 10u, 20u, 30u }, Assert.IsType<uint[]>(typedMeta["t.arr_uint"]));
        Assert.Equal(new long[] { -500L, 500L }, Assert.IsType<long[]>(typedMeta["t.arr_long"]));
        Assert.Equal(new float[] { 1.5f, -2.5f }, Assert.IsType<float[]>(typedMeta["t.arr_float"]));
        Assert.Equal(new[] { "alpha", "beta" }, Assert.IsType<string[]>(typedMeta["t.arr_str"]));
        Assert.Equal(0L, dataStart % 64);
        Assert.Single(readTensors);
        Assert.Equal(QuantType.LittleBit2, readTensors[0].Type);

        ms.Position = dataStart;
        byte[] readPayload = new byte[payload.Length];
        ms.ReadExactly(readPayload);
        Assert.Equal(payload, readPayload);

        // String ReadHeader compatibility also handles all 13 metadata types cleanly
        ms.Position = 0;
        var (strMeta, _, strDataStart) = GgufReader.ReadHeader(ms);
        Assert.Equal(dataStart, strDataStart);
        Assert.Equal("true", strMeta["t.bool"]);
        Assert.Equal("[10, 20, 30]", strMeta["t.arr_int"]);
    }

    [Fact]
    public void GgufReader_OomAndTruncatedGuards_ThrowExpectedExceptions()
    {
        // 1. Metadata count exceeds remaining stream bytes / limit
        using (var ms = new MemoryStream())
        {
            Span<byte> hdr = stackalloc byte[24];
            BinaryPrimitives.WriteUInt32LittleEndian(hdr[0..4], GgufWriter.GgufMagic);
            BinaryPrimitives.WriteUInt32LittleEndian(hdr[4..8], GgufWriter.GgufVersion);
            BinaryPrimitives.WriteUInt64LittleEndian(hdr[8..16], 0UL);
            BinaryPrimitives.WriteUInt64LittleEndian(hdr[16..24], 500_000UL);
            ms.Write(hdr);
            ms.Position = 0;
            Assert.Throws<InvalidDataException>(() => GgufReader.ReadHeaderTyped(ms));
        }

        // 2. Non-power-of-two alignment rejected by GgufWriter
        using (var ms = new MemoryStream())
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GgufWriter.WriteToStream(ms, new Dictionary<string, string>(), Array.Empty<(QuantTensorMetadata, byte[])>(), alignment: 24));
        }

        // 3. Truncated string payload throws EndOfStreamException
        using (var ms = new MemoryStream())
        {
            Span<byte> buf = stackalloc byte[32];
            BinaryPrimitives.WriteUInt32LittleEndian(buf[0..4], GgufWriter.GgufMagic);
            BinaryPrimitives.WriteUInt32LittleEndian(buf[4..8], GgufWriter.GgufVersion);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[8..16], 0UL);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[16..24], 1UL);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[24..32], 128UL); // claims 128-byte key, 0 bytes follow
            ms.Write(buf);
            ms.Position = 0;
            Assert.Throws<EndOfStreamException>(() => GgufReader.ReadHeaderTyped(ms));
        }
    }

    [Fact]
    public void GgufReader_NonSeekableStream_ReadsHeaderAndDataStartOffsetCorrectly()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "llama",
            ["general.alignment"] = 64u,
            ["llama.block_count"] = 12
        };
        byte[] payload = new byte[64];
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "w.weight", Shape = [8, 8], Type = QuantType.Q8_0 }, payload)
        };

        using var ms = new MemoryStream();
        GgufWriter.WriteToStream(ms, metadata, tensors, alignment: 64);
        byte[] bytes = ms.ToArray();

        ms.Position = 0;
        var (_, _, seekableDataStart) = GgufReader.ReadHeaderTyped(ms);

        using var nonSeekable = new NonSeekableReadStream(bytes);
        var (readMeta, readTensors, nonSeekableDataStart) = GgufReader.ReadHeaderTyped(nonSeekable);

        Assert.Equal(seekableDataStart, nonSeekableDataStart);
        Assert.Equal("llama", readMeta["general.architecture"]);
        Assert.Equal(12, readMeta["llama.block_count"]);
        Assert.Single(readTensors);
        Assert.Equal(QuantType.Q8_0, readTensors[0].Type);
    }

    private sealed class NonSeekableReadStream(byte[] data) : Stream
    {
        private int _pos;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(count, data.Length - _pos);
            Array.Copy(data, _pos, buffer, offset, n);
            _pos += n;
            return n;
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
