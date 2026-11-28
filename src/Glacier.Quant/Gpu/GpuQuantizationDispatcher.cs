namespace Glacier.Quant.Gpu;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Quant.Kernels;

/// <summary>
/// Quantization dispatcher for tensor-level quantization and dequantization.
/// <para>
/// <c>Glacier.Quant</c> is a pure CPU-SIMD library (<see cref="ExecutionBackend"/> = <c>"CPU-SIMD"</c>)
/// and does not ship Direct3D 12 compute shaders; <see cref="IsGpuAccelerationAvailable"/> therefore
/// always returns <see langword="false"/>. Use <c>Glacier.LittleBit</c> for D3D12 GPU dispatch.
/// Callers invoking the 4-argument overloads receive CPU-SIMD execution via <see cref="SimdQuantizer"/>,
/// while the 5-argument overloads with <c>requireGpu: true</c> throw <see cref="NotSupportedException"/>.
/// </para>
/// </summary>
public static class GpuQuantizationDispatcher
{
    /// <summary>
    /// Gets whether native GPU quantization acceleration is available in <c>Glacier.Quant</c>.
    /// Always returns <see langword="false"/> because this package executes exclusively on CPU-SIMD.
    /// </summary>
    public static bool IsGpuAccelerationAvailable => false;

    /// <summary>
    /// Gets the active execution backend identifier (<c>"CPU-SIMD"</c>).
    /// </summary>
    public static string ExecutionBackend => "CPU-SIMD";

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
            QuantType.LittleBit2 => throw new NotSupportedException("QuantType.LittleBit2 requires 2D matrix dimensions (rows, cols); use LittleBit2Quantizer directly."),
            _ => throw new NotSupportedException($"Quantization type {targetType} not supported.")
        };
    }

    public static void QuantizeTensor(
        ReadOnlySpan<float> input,
        QuantType targetType,
        Span<byte> output,
        out float dynamicScale,
        bool requireGpu)
    {
        if (requireGpu && !IsGpuAccelerationAvailable)
            throw new NotSupportedException($"GPU acceleration is not available in Glacier.Quant (ExecutionBackend: {ExecutionBackend}). Use Glacier.LittleBit for D3D12 GPU dispatch or pass requireGpu: false for CPU-SIMD fallback.");

        QuantizeTensor(input, targetType, output, out dynamicScale);
    }

    public static void QuantizeTensor(
        ReadOnlySpan<float> input,
        QuantType targetType,
        Span<byte> output,
        out float dynamicScale)
    {
        dynamicScale = 1.0f;
        int requiredBytes = GetQuantizedByteCount(input.Length, targetType);
        if (output.Length < requiredBytes)
            throw new ArgumentException($"Output byte span ({output.Length}) is too small for {input.Length} {targetType} elements ({requiredBytes} bytes).", nameof(output));

        switch (targetType)
        {
            case QuantType.FP32:
            {
                MemoryMarshal.AsBytes(input).CopyTo(output);
                break;
            }
            case QuantType.Q8_0:
            {
                int numBlocks = input.Length / BlockQ8_0.BlockSize;
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ8_0>(output.Slice(0, requiredBytes));
                SimdQuantizer.QuantizeRowQ8_0(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            case QuantType.Q4_K_M:
            {
                int numBlocks = input.Length / BlockQ4_K.BlockSize;
                var blockSpan = MemoryMarshal.Cast<byte, BlockQ4_K>(output.Slice(0, requiredBytes));
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
                var blockSpan = MemoryMarshal.Cast<byte, BlockBinary1Bit>(output.Slice(0, requiredBytes));
                SimdQuantizer.QuantizeRow1Bit(input, blockSpan.Slice(0, numBlocks));
                break;
            }
            case QuantType.LittleBit2:
                throw new NotSupportedException("QuantType.LittleBit2 requires 2D matrix dimensions (rows, cols); use LittleBit2Quantizer directly.");
            default:
                throw new NotSupportedException($"Quantization type {targetType} not supported.");
        }
    }

    public static void DequantizeTensor(
        ReadOnlySpan<byte> input,
        QuantType sourceType,
        Span<float> output,
        float dynamicScale,
        bool requireGpu)
    {
        if (requireGpu && !IsGpuAccelerationAvailable)
            throw new NotSupportedException($"GPU acceleration is not available in Glacier.Quant (ExecutionBackend: {ExecutionBackend}). Use Glacier.LittleBit for D3D12 GPU dispatch or pass requireGpu: false for CPU-SIMD fallback.");

        DequantizeTensor(input, sourceType, output, dynamicScale);
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
            case QuantType.LittleBit2:
                throw new NotSupportedException("QuantType.LittleBit2 requires 2D matrix dimensions (rows, cols); use LittleBit2Quantizer directly.");
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
}
