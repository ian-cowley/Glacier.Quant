namespace Glacier.Quant.Formats;

using System;
using Glacier.Quant.Kernels;

public sealed record QuantTensorMetadata
{
    public string Name { get; init; } = "";
    public long[] Shape { get; init; } = [];
    public QuantType Type { get; init; } = QuantType.FP32;
    public ulong DataOffset { get; set; }
    public ulong DataLength { get; set; }

    public long TotalElements
    {
        get
        {
            if (Shape.Length == 0) return 0;
            long count = 1;
            foreach (var dim in Shape) count *= dim;
            return count;
        }
    }
}
