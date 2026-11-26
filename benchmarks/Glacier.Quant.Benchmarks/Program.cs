using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Glacier.Quant.Kernels;

Console.WriteLine("================================================================================");
Console.WriteLine("          GLACIER.QUANT EMPIRICAL HADAMARD & NARIQ TRELLIS BENCHMARKS          ");
Console.WriteLine("================================================================================");
Console.WriteLine($"SIMD ISA: AVX-512F={Avx512F.IsSupported} | AVX2={Avx2.IsSupported} | FMA={Fma.IsSupported} | Threads={Environment.ProcessorCount}");

BenchmarkSylvesterFwht();
BenchmarkPaleyAndKronecker();
BenchmarkSwarStateDecode();
BenchmarkTrellisTileDecodeAndMatVec();

static void BenchmarkSylvesterFwht()
{
    Console.WriteLine("\n[1] Sylvester Block FWHT (16,384 floats, Normalized H * H = I):");
    const int n = 16_384;
    float[] data = new float[n];
    for (int i = 0; i < n; i++) data[i] = ((i % 29) - 14) * 0.1f;

    foreach (int blockLen in new[] { 128, 512, 4096 })
    {
        for (int w = 0; w < 200; w++) HadamardTransform.ApplySylvesterInPlace(data, blockLen);

        const int iters = 5_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) HadamardTransform.ApplySylvesterInPlace(data, blockLen);
        sw.Stop();

        double us = sw.Elapsed.TotalMicroseconds / iters;
        double gbs = (n * sizeof(float) / 1e9) / (us / 1e6);
        Console.WriteLine($"  blockLen={blockLen,4}: {us,6:F2} us | {gbs,5:F2} GB/s");
    }
}

static void BenchmarkPaleyAndKronecker()
{
    Console.WriteLine("\n[2] Paley Type-I & Kronecker Paley-Sylvester NariQ Transforms:");
    foreach (int m in new[] { 12, 20, 68 })
    {
        float[] vec = new float[m];
        for (int i = 0; i < m; i++) vec[i] = (i + 1) * 0.25f;
        for (int w = 0; w < 10_000; w++) HadamardTransform.ApplyPaleyInPlace(vec, m);

        const int iters = 500_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) HadamardTransform.ApplyPaleyInPlace(vec, m);
        sw.Stop();
        Console.WriteLine($"  Paley m={m,2}:       {sw.Elapsed.TotalNanoseconds / iters,6:F1} ns/call");
    }

    foreach (var (n, m, label) in new[] { (5120, 20, "5120 (20*256)"), (17408, 68, "17408 (68*256)") })
    {
        float[] vec = new float[n];
        for (int i = 0; i < n; i++) vec[i] = ((i % 17) - 8) * 0.125f;
        for (int w = 0; w < 200; w++) HadamardTransform.ApplyKroneckerPaleyInPlace(vec, m);

        const int iters = 5_000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) HadamardTransform.ApplyKroneckerPaleyInPlace(vec, m);
        sw.Stop();

        double us = sw.Elapsed.TotalMicroseconds / iters;
        double gbs = (n * sizeof(float) / 1e9) / (us / 1e6);
        Console.WriteLine($"  Kronecker {label}: {us,6:F2} us | {gbs,5:F2} GB/s");
    }
}

static void BenchmarkSwarStateDecode()
{
    Console.WriteLine("\n[3] 65,536-State Codebook Decode (SWAR vs Scalar 0x83DCD12Du):");
    int sink = 0;
    for (int w = 0; w < 5; w++)
    {
        for (uint s = 0; s < 65536; s++) sink += NariQTrellisCodec.DecodeStateSwar(s);
    }

    const int sweeps = 1_000;
    var swSwar = Stopwatch.StartNew();
    for (int r = 0; r < sweeps; r++)
    {
        int acc = 0;
        for (uint s = 0; s < 65536; s++) acc += NariQTrellisCodec.DecodeStateSwar(s);
        sink ^= acc;
    }
    swSwar.Stop();

    var swScalar = Stopwatch.StartNew();
    for (int r = 0; r < sweeps; r++)
    {
        int acc = 0;
        for (uint s = 0; s < 65536; s++) acc += NariQTrellisCodec.DecodeStateScalar(s);
        sink ^= acc;
    }
    swScalar.Stop();

    double swarNs = swSwar.Elapsed.TotalNanoseconds / (sweeps * 65536.0);
    double scalarNs = swScalar.Elapsed.TotalNanoseconds / (sweeps * 65536.0);
    Console.WriteLine($"  DecodeStateSwar:   {swarNs:F3} ns/state | DecodeStateScalar: {scalarNs:F3} ns/state (guard={sink})");
}

static void BenchmarkTrellisTileDecodeAndMatVec()
{
    Console.WriteLine("\n[4] NariQ TRL2 / TRL3 / TRL4 16x16 Tile Matrix (5120 x 256, 5,120 tiles = 1.31M weights):");
    const int k = 256;
    const int rows = 5120;
    const int tilesK = k / 16;
    const int tilesN = rows / 16;
    const int totalTiles = tilesK * tilesN;

    float[] x = new float[k];
    float[] y = new float[rows];
    sbyte[] dstTile = new sbyte[256];
    for (int i = 0; i < k; i++) x[i] = ((i % 13) - 6) * 0.2f;

    foreach (int bits in new[] { 2, 3, 4 })
    {
        int nw = NariQTrellisCodec.GetTileWordCount(bits);
        uint[] words = new uint[totalTiles * nw];
        for (int i = 0; i < words.Length; i++) words[i] = unchecked((uint)(i * 2654435761u + 17u));

        // Warmup
        for (int t = 0; t < totalTiles; t++)
        {
            NariQTrellisCodec.DecodeTileInt8(words.AsSpan(t * nw, nw), bits, dstTile);
            NariQTrellisCodec.TileMatVecAdd(words.AsSpan(t * nw, nw), bits, 0.05f, x.AsSpan((t % tilesK) * 16, 16), y.AsSpan((t / tilesK) * 16, 16));
        }

        const int runs = 100;
        var swDec = Stopwatch.StartNew();
        for (int r = 0; r < runs; r++)
        {
            for (int t = 0; t < totalTiles; t++)
            {
                NariQTrellisCodec.DecodeTileInt8(words.AsSpan(t * nw, nw), bits, dstTile);
            }
        }
        swDec.Stop();

        var swMv = Stopwatch.StartNew();
        for (int r = 0; r < runs; r++)
        {
            for (int t = 0; t < totalTiles; t++)
            {
                int jt = t / tilesK;
                int it = t % tilesK;
                NariQTrellisCodec.TileMatVecAdd(words.AsSpan(t * nw, nw), bits, 0.05f, x.AsSpan(it * 16, 16), y.AsSpan(jt * 16, 16));
            }
        }
        swMv.Stop();

        double decMs = swDec.Elapsed.TotalMilliseconds / runs;
        double mvMs = swMv.Elapsed.TotalMilliseconds / runs;
        double gWeightsDec = ((double)rows * k / 1e9) / (decMs / 1000.0);
        double gWeightsMv = ((double)rows * k / 1e9) / (mvMs / 1000.0);
        Console.WriteLine($"  TRL{bits}: DecodeTileInt8={decMs:F2} ms ({gWeightsDec:F2} Gw/s) | TileMatVecAdd={mvMs:F2} ms ({gWeightsMv:F2} Gw/s)");
    }
}
