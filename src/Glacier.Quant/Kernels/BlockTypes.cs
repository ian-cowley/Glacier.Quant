namespace Glacier.Quant.Kernels;

using System;
using System.Runtime.InteropServices;

public enum QuantType : byte
{
    FP32 = 0,
    FP16 = 1,
    Q8_0 = 2,
    Q4_K_M = 3,
    FP8_E4M3 = 4,
    FP8_E5M2 = 5,
    Binary1Bit = 6
}

/// <summary>
/// Q8_0 block format: 32 elements quantized into 32 signed bytes with one 16-bit float scale factor.
/// Total block size = 2 (Half scale) + 32 (bytes) = 34 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 34)]
public unsafe struct BlockQ8_0
{
    public const int BlockSize = 32;
    public Half Scale;
    public fixed sbyte Quants[BlockSize];
}

/// <summary>
/// Q4_K super-block format: 256 elements in 8 sub-blocks of 32 elements.
/// Total super-block size = 2 (Half d) + 2 (Half dmin) + 12 (6-bit scales/mins) + 128 (4-bit nibbles) = 144 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 144)]
public unsafe struct BlockQ4_K
{
    public const int BlockSize = 256;
    public Half D;
    public Half DMin;
    public fixed byte Scales[12];
    public fixed byte Quants[128];
}

/// <summary>
/// Packed 1-bit binary representation: 64 weights represented as sign bits in a single 64-bit integer,
/// with an FP32 scale factor and optional mean bias offset.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 16)]
public struct BlockBinary1Bit
{
    public const int BlockSize = 64;
    public ulong Bits;
    public float Scale;
    public float Bias;
}
