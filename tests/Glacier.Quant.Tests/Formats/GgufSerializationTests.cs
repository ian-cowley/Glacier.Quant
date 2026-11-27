namespace Glacier.Quant.Tests.Formats;

using System;
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
}
