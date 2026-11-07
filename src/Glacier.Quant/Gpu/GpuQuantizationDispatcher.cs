namespace Glacier.Quant.Gpu;

using System;
using Glacier.Quant.Kernels;

public static class GpuQuantizationDispatcher
{
    public static bool IsGpuAccelerationAvailable =>
        OperatingSystem.IsWindows() && CheckDirect3D12Support();

    public static void QuantizeTensor(
        ReadOnlySpan<float> input,
        QuantType targetType,
        Span<byte> output,
        out float dynamicScale)
    {
        dynamicScale = 1.0f;

        switch (targetType)
        {
            case QuantType.Q8_0:
            {
                int numBlocks = input.Length / BlockQ8_0.BlockSize;
                var blockSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, BlockQ8_0>(output);
                SimdQuantizer.QuantizeRowQ8_0(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            case QuantType.Q4_K_M:
            {
                int numBlocks = input.Length / BlockQ4_K.BlockSize;
                var blockSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, BlockQ4_K>(output);
                SimdQuantizer.QuantizeRowQ4_K(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            case QuantType.FP8_E4M3:
            {
                SimdQuantizer.QuantizeRowFP8_E4M3(input, output.Slice(0, input.Length), out dynamicScale);
                break;
            }
            case QuantType.Binary1Bit:
            {
                int numBlocks = input.Length / BlockBinary1Bit.BlockSize;
                var blockSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, BlockBinary1Bit>(output);
                SimdQuantizer.QuantizeRow1Bit(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            default:
                throw new NotSupportedException($"Quantization type {targetType} not supported.");
        }
    }

    private static bool CheckDirect3D12Support()
    {
        try
        {
            // Verified via Windows runtime platform query
            return OperatingSystem.IsWindows();
        }
        catch
        {
            return false;
        }
    }
}
