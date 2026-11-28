namespace Glacier.Quant.Tests.Formats;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Quant.Formats;
using Glacier.Quant.Kernels;
using Xunit;

public class SafetensorsSerializationTests
{
    [Fact]
    public void SafetensorsWriter_And_SafetensorsReader_RoundtripTensors_Correctly()
    {
        byte[] tensorAData = new byte[80];
        for (int i = 0; i < tensorAData.Length; i++) tensorAData[i] = (byte)(i % 50);

        byte[] tensorBData = new byte[160];
        for (int i = 0; i < tensorBData.Length; i++) tensorBData[i] = (byte)((i * 3) % 256);

        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata
            {
                Name = "model.embed_tokens.weight",
                Shape = [20, 1],
                Type = QuantType.FP32
            }, tensorAData),
            (new QuantTensorMetadata
            {
                Name = "model.layers.0.mlp.gate_proj.weight",
                Shape = [40, 1],
                Type = QuantType.FP8_E4M3
            }, tensorBData)
        };

        using var ms = new MemoryStream();
        SafetensorsWriter.WriteToStream(ms, tensors);

        ms.Position = 0;
        var (readDict, dataStart) = SafetensorsReader.ReadHeader(ms);

        Assert.Equal(2, readDict.Count);
        Assert.True(readDict.ContainsKey("model.embed_tokens.weight"));
        Assert.True(readDict.ContainsKey("model.layers.0.mlp.gate_proj.weight"));

        var tA = readDict["model.embed_tokens.weight"];
        Assert.Equal([20, 1], tA.Shape);
        Assert.Equal(QuantType.FP32, tA.Type);
        Assert.Equal(0UL, tA.DataOffset);
        Assert.Equal((ulong)tensorAData.Length, tA.DataLength);

        var tB = readDict["model.layers.0.mlp.gate_proj.weight"];
        Assert.Equal([40, 1], tB.Shape);
        Assert.Equal(QuantType.FP8_E4M3, tB.Type);
        Assert.Equal((ulong)tensorAData.Length, tB.DataOffset);
        Assert.Equal((ulong)tensorBData.Length, tB.DataLength);

        // Read tensor payloads
        ms.Position = dataStart + (long)tA.DataOffset;
        byte[] readA = new byte[tensorAData.Length];
        ms.ReadExactly(readA);
        Assert.Equal(tensorAData, readA);

        ms.Position = dataStart + (long)tB.DataOffset;
        byte[] readB = new byte[tensorBData.Length];
        ms.ReadExactly(readB);
        Assert.Equal(tensorBData, readB);
    }

    [Fact]
    public void SafetensorsWriter_And_SafetensorsReader_PreservesLittleBit2Q8AndDynamicScaleMetadata()
    {
        byte[] lb2Data = new byte[64];
        byte[] q4kData = new byte[144];
        byte[] fp8Data = new byte[32];
        for (int i = 0; i < lb2Data.Length; i++) lb2Data[i] = (byte)(i + 1);
        for (int i = 0; i < q4kData.Length; i++) q4kData[i] = (byte)(255 - i);
        for (int i = 0; i < fp8Data.Length; i++) fp8Data[i] = (byte)(i * 2);

        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "lb2.weight", Shape = [8, 8], Type = QuantType.LittleBit2 }, lb2Data),
            (new QuantTensorMetadata { Name = "q4k.weight", Shape = [256], Type = QuantType.Q4_K_M }, q4kData),
            (new QuantTensorMetadata { Name = "fp8.weight", Shape = [32], Type = QuantType.FP8_E4M3, DynamicScale = 0.125f }, fp8Data)
        };

        var customMeta = new Dictionary<string, string> { ["model.name"] = "Glacier-Test" };

        using var ms = new MemoryStream();
        SafetensorsWriter.WriteToStream(ms, tensors, customMeta);

        ms.Position = 0;
        var (meta, readTensors, _) = SafetensorsReader.ReadHeaderWithMetadata(ms);

        Assert.Equal("Glacier-Test", meta["model.name"]);
        Assert.Equal(QuantType.LittleBit2, readTensors["lb2.weight"].Type);
        Assert.Equal(QuantType.Q4_K_M, readTensors["q4k.weight"].Type);
        Assert.Equal(QuantType.FP8_E4M3, readTensors["fp8.weight"].Type);
        Assert.Equal(0.125f, readTensors["fp8.weight"].DynamicScale);
    }

    [Fact]
    public void SafetensorsWriter_And_SafetensorsReader_WriteAndReadQuantizedQ8_0_RoundtripsBlocksLosslessly()
    {
        float[] weights = new float[64];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = MathF.Sin(i * 0.25f) * 3.5f;

        var blocks = new BlockQ8_0[2];
        SimdQuantizer.QuantizeRowQ8_0(weights, blocks);

        using var ms = new MemoryStream();
        SafetensorsWriter.WriteQuantizedQ8_0(ms, "layer0.proj.weight", [2, 32], blocks);

        ms.Position = 0;
        var (meta, tensors, _) = SafetensorsReader.ReadHeaderWithMetadata(ms);
        Assert.Equal("layer0.proj.weight.scale", tensors["layer0.proj.weight"].ScaleTensorName);
        Assert.Equal(QuantType.Q8_0, tensors["layer0.proj.weight"].Type);
        Assert.Equal(QuantType.FP16, tensors["layer0.proj.weight.scale"].Type);
        Assert.Equal("layer0.proj.weight.scale", meta["glacier.quant.scales.layer0.proj.weight"]);

        ms.Position = 0;
        BlockQ8_0[] roundtripped = SafetensorsReader.ReadQuantizedQ8_0(ms, "layer0.proj.weight");
        Assert.Equal(blocks.Length, roundtripped.Length);
        unsafe
        {
            for (int b = 0; b < blocks.Length; b++)
            {
                Assert.Equal(blocks[b].Scale, roundtripped[b].Scale);
                for (int i = 0; i < BlockQ8_0.BlockSize; i++)
                    Assert.Equal(blocks[b].Quants[i], roundtripped[b].Quants[i]);
            }
        }
    }

    [Fact]
    public void SafetensorsReader_ZeroHeaderLengthAndInvalidOffsets_ThrowInvalidDataException()
    {
        using (var ms = new MemoryStream())
        {
            ms.Write(BitConverter.GetBytes(0UL));
            ms.Position = 0;
            Assert.Throws<InvalidDataException>(() => SafetensorsReader.ReadHeader(ms));
        }

        byte[] badOffsetsJson = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,9999]}}"u8.ToArray();
        using (var ms = new MemoryStream())
        {
            ms.Write(BitConverter.GetBytes((ulong)badOffsetsJson.Length));
            ms.Write(badOffsetsJson);
            ms.Position = 0;
            Assert.Throws<InvalidDataException>(() => SafetensorsReader.ReadHeader(ms));
        }
    }

    [Fact]
    public void SafetensorsReader_NonSeekableStream_ReadsHeaderAndDataStartOffsetCorrectly()
    {
        byte[] fp8Data = new byte[32];
        for (int i = 0; i < fp8Data.Length; i++) fp8Data[i] = (byte)i;
        var tensors = new List<(QuantTensorMetadata Info, byte[] Data)>
        {
            (new QuantTensorMetadata { Name = "fp8.weight", Shape = [32], Type = QuantType.FP8_E4M3, DynamicScale = 0.25f }, fp8Data)
        };

        using var ms = new MemoryStream();
        SafetensorsWriter.WriteToStream(ms, tensors, new Dictionary<string, string> { ["model"] = "non-seekable" });
        byte[] bytes = ms.ToArray();

        ms.Position = 0;
        var (_, _, seekableDataStart) = SafetensorsReader.ReadHeaderWithMetadata(ms);

        using var nonSeekable = new NonSeekableReadStream(bytes);
        var (meta, readTensors, nonSeekableDataStart) = SafetensorsReader.ReadHeaderWithMetadata(nonSeekable);

        Assert.Equal(seekableDataStart, nonSeekableDataStart);
        Assert.Equal("non-seekable", meta["model"]);
        Assert.Equal(QuantType.FP8_E4M3, readTensors["fp8.weight"].Type);
        Assert.Equal(0.25f, readTensors["fp8.weight"].DynamicScale);
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
