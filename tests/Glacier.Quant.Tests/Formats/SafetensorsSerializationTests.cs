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
}
