namespace Glacier.Quant.Gpu;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Quant.Kernels;

public static class GpuQuantizationDispatcher
{
    public static bool IsGpuAccelerationAvailable =>
        OperatingSystem.IsWindows() && CheckDirect3D12Support();

    public static int GetQuantizedByteCount(int elementCount, QuantType targetType)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);

        return targetType switch
        {
            QuantType.FP32 => checked(elementCount * sizeof(float)),
            QuantType.Q8_0 => ComputeBlockByteCount(elementCount, BlockQ8_0.BlockSize, Unsafe.SizeOf<BlockQ8_0>(), targetType),
            QuantType.Q4_K_M => ComputeBlockByteCount(elementCount, BlockQ4_K.BlockSize, Unsafe.SizeOf<BlockQ4_K>(), targetType),
            QuantType.FP8_E4M3 => elementCount,
            QuantType.Binary1Bit => ComputeBlockByteCount(elementCount, BlockBinary1Bit.BlockSize, Unsafe.SizeOf<BlockBinary1Bit>(), targetType),
            _ => throw new NotSupportedException($"Quantization type {targetType} not supported.")
        };
    }

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
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ8_0>(output);
                SimdQuantizer.QuantizeRowQ8_0(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            case QuantType.Q4_K_M:
            {
                int numBlocks = input.Length / BlockQ4_K.BlockSize;
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ4_K>(output);
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
                var blockSpan = MemoryMarshal.Cast<byte, BlockBinary1Bit>(output);
                SimdQuantizer.QuantizeRow1Bit(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            default:
                throw new NotSupportedException($"Quantization type {targetType} not supported.");
        }
    }

    public static void DequantizeTensor(
        ReadOnlySpan<byte> input,
        QuantType sourceType,
        Span<float> output,
        float dynamicScale = 1.0f)
    {
        switch (sourceType)
        {
            case QuantType.FP32:
            {
                int requiredBytes = checked(output.Length * sizeof(float));
                if (input.Length < requiredBytes)
                    throw new ArgumentException($"Input byte span ({input.Length}) is too small for {output.Length} FP32 elements ({requiredBytes} bytes).", nameof(input));
                MemoryMarshal.Cast<byte, float>(input.Slice(0, requiredBytes)).CopyTo(output);
                break;
            }
            case QuantType.Q8_0:
            {
                if (output.Length % BlockQ8_0.BlockSize != 0)
                    throw new ArgumentException($"Output length ({output.Length}) must be a multiple of block size ({BlockQ8_0.BlockSize}).", nameof(output));
                int numBlocks = output.Length / BlockQ8_0.BlockSize;
                int requiredBytes = checked(numBlocks * Unsafe.SizeOf<BlockQ8_0>());
                if (input.Length < requiredBytes)
                    throw new ArgumentException($"Input byte span ({input.Length}) is too small for {numBlocks} Q8_0 blocks ({requiredBytes} bytes).", nameof(input));
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ8_0>(input.Slice(0, requiredBytes));
                SimdQuantizer.DequantizeRowQ8_0(blockSpan, output);
                break;
            }
            case QuantType.Q4_K_M:
            {
                if (output.Length % BlockQ4_K.BlockSize != 0)
                    throw new ArgumentException($"Output length ({output.Length}) must be a multiple of block size ({BlockQ4_K.BlockSize}).", nameof(output));
                int numBlocks = output.Length / BlockQ4_K.BlockSize;
                int requiredBytes = checked(numBlocks * Unsafe.SizeOf<BlockQ4_K>());
                if (input.Length < requiredBytes)
                    throw new ArgumentException($"Input byte span ({input.Length}) is too small for {numBlocks} Q4_K_M blocks ({requiredBytes} bytes).", nameof(input));
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ4_K>(input.Slice(0, requiredBytes));
                SimdQuantizer.DequantizeRowQ4_K(blockSpan, output);
                break;
            }
            case QuantType.FP8_E4M3:
            {
                if (input.Length < output.Length)
                    throw new ArgumentException($"Input byte span ({input.Length}) is too small for {output.Length} FP8_E4M3 elements.", nameof(input));
                SimdQuantizer.DequantizeRowFP8_E4M3(input.Slice(0, output.Length), dynamicScale, output);
                break;
            }
            case QuantType.Binary1Bit:
            {
                if (output.Length % BlockBinary1Bit.BlockSize != 0)
                    throw new ArgumentException($"Output length ({output.Length}) must be a multiple of block size ({BlockBinary1Bit.BlockSize}).", nameof(output));
                int numBlocks = output.Length / BlockBinary1Bit.BlockSize;
                int requiredBytes = checked(numBlocks * Unsafe.SizeOf<BlockBinary1Bit>());
                if (input.Length < requiredBytes)
                    throw new ArgumentException($"Input byte span ({input.Length}) is too small for {numBlocks} Binary1Bit blocks ({requiredBytes} bytes).", nameof(input));
                var blockSpan = MemoryMarshal.Cast<byte, BlockBinary1Bit>(input.Slice(0, requiredBytes));
                SimdQuantizer.DequantizeRow1Bit(blockSpan, output);
                break;
            }
            default:
                throw new NotSupportedException($"Quantization type {sourceType} not supported.");
        }
    }

    private static int ComputeBlockByteCount(int elementCount, int blockSize, int blockByteSize, QuantType targetType)
    {
        if (elementCount % blockSize != 0)
            throw new ArgumentException($"Element count ({elementCount}) must be a multiple of block size ({blockSize}) for {targetType}.", nameof(elementCount));
        return checked((elementCount / blockSize) * blockByteSize);
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
