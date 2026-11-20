namespace Glacier.Quant.Evaluation;

using System;
using System.Collections.Generic;

/// <summary>
/// Numerically stable, allocation-free KL-divergence evaluator D_KL(P_ref || Q_quant)
/// over single-token or batched logits and probability distributions.
/// </summary>
public static class KlDivergenceEvaluator
{
    public const double DefaultThreshold = 0.05;
    public const double DefaultEpsilon = 1e-12;

    /// <summary>
    /// Computes log-softmax(logits) in double precision using a 3-pass max-shifted log-sum-exp formulation.
    /// </summary>
    public static void ComputeLogSoftmax(ReadOnlySpan<float> logits, Span<double> logProbs)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits span cannot be empty.", nameof(logits));
        if (logProbs.Length < logits.Length)
            throw new ArgumentException($"Destination span length ({logProbs.Length}) is smaller than logits length ({logits.Length}).", nameof(logProbs));

        double maxLogit = FindMaxAndValidateLogits(logits, nameof(logits));
        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (!float.IsNegativeInfinity(v))
                sumExp += Math.Exp((double)v - maxLogit);
        }

        double logSumExpShift = Math.Log(sumExp);
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (float.IsNegativeInfinity(v))
            {
                logProbs[i] = double.NegativeInfinity;
            }
            else
            {
                double lp = ((double)v - maxLogit) - logSumExpShift;
                logProbs[i] = lp > 0.0 ? 0.0 : lp;
            }
        }
    }

    /// <summary>
    /// Computes softmax(logits) in double precision and writes float probabilities to <paramref name="probs"/>.
    /// </summary>
    public static void ComputeSoftmax(ReadOnlySpan<float> logits, Span<float> probs)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits span cannot be empty.", nameof(logits));
        if (probs.Length < logits.Length)
            throw new ArgumentException($"Destination span length ({probs.Length}) is smaller than logits length ({logits.Length}).", nameof(probs));

        double maxLogit = FindMaxAndValidateLogits(logits, nameof(logits));
        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (!float.IsNegativeInfinity(v))
                sumExp += Math.Exp((double)v - maxLogit);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            probs[i] = float.IsNegativeInfinity(v)
                ? 0.0f
                : (float)(Math.Exp((double)v - maxLogit) / sumExp);
        }
    }

    /// <summary>
    /// Returns the index of the maximum element using lowest-index tie-breaking.
    /// </summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
            throw new ArgumentException("Values span cannot be empty.", nameof(values));

        float first = values[0];
        if (float.IsNaN(first))
            throw new ArgumentException("Input values cannot contain NaN.", nameof(values));

        int bestIdx = 0;
        float bestVal = first;
        for (int i = 1; i < values.Length; i++)
        {
            float v = values[i];
            if (float.IsNaN(v))
                throw new ArgumentException("Input values cannot contain NaN.", nameof(values));
            if (v > bestVal)
            {
                bestVal = v;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Computes D_KL(P_ref || Q_quant) in nats for a single pair of logit vectors without heap allocations.
    /// </summary>
    public static double ComputeTokenKlDivergence(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double epsilon = DefaultEpsilon)
    {
        if (referenceLogits.IsEmpty)
            throw new ArgumentException("Reference logits span cannot be empty.", nameof(referenceLogits));
        if (referenceLogits.Length != quantizedLogits.Length)
            throw new ArgumentException($"Logit span lengths must match (reference: {referenceLogits.Length}, quantized: {quantizedLogits.Length}).", nameof(quantizedLogits));
        if (!double.IsFinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be finite and in (0.0, 1.0).");

        return ComputeTokenKlCore(referenceLogits, quantizedLogits, epsilon, out _);
    }

    public static KlDivergenceResult EvaluateSingleTokenLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        if (referenceLogits.IsEmpty)
            throw new ArgumentException("Reference logits span cannot be empty.", nameof(referenceLogits));

        return EvaluateLogits(
            referenceLogits,
            quantizedLogits,
            referenceLogits.Length,
            threshold,
            maxKlThreshold,
            minTop1Agreement,
            epsilon);
    }

    public static KlDivergenceResult EvaluateSingleTokenLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        return EvaluateSingleTokenLogits(
            referenceLogits,
            quantizedLogits,
            opts.Threshold,
            opts.MaxKlThreshold,
            opts.MinTop1Agreement,
            opts.Epsilon);
    }

    public static KlDivergenceResult EvaluateLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        int vocabSize,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement,
            Epsilon = epsilon
        };
        return EvaluateLogits(referenceLogits, quantizedLogits, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        int vocabSize,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateBatchShapes(referenceLogits.Length, quantizedLogits.Length, vocabSize);
        int tokenCount = referenceLogits.Length / vocabSize;

        double sumKl = 0.0;
        double maxKl = 0.0;
        double minKl = double.PositiveInfinity;
        int top1Matches = 0;

        for (int t = 0; t < tokenCount; t++)
        {
            int offset = t * vocabSize;
            var refRow = referenceLogits.Slice(offset, vocabSize);
            var quantRow = quantizedLogits.Slice(offset, vocabSize);

            double kl = ComputeTokenKlCore(refRow, quantRow, opts.Epsilon, out bool top1Match);
            sumKl += kl;
            if (kl > maxKl) maxKl = kl;
            if (kl < minKl) minKl = kl;
            if (top1Match) top1Matches++;
        }

        return BuildResult(sumKl, maxKl, minKl, top1Matches, tokenCount, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateProbabilities(
        ReadOnlySpan<float> referenceProbs,
        ReadOnlySpan<float> quantizedProbs,
        int vocabSize,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement,
            Epsilon = epsilon
        };
        return EvaluateProbabilities(referenceProbs, quantizedProbs, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateProbabilities(
        ReadOnlySpan<float> referenceProbs,
        ReadOnlySpan<float> quantizedProbs,
        int vocabSize,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateBatchShapes(referenceProbs.Length, quantizedProbs.Length, vocabSize);
        int tokenCount = referenceProbs.Length / vocabSize;

        double sumKl = 0.0;
        double maxKl = 0.0;
        double minKl = double.PositiveInfinity;
        int top1Matches = 0;

        for (int t = 0; t < tokenCount; t++)
        {
            int offset = t * vocabSize;
            var refRow = referenceProbs.Slice(offset, vocabSize);
            var quantRow = quantizedProbs.Slice(offset, vocabSize);

            double sumP = 0.0;
            double sumQ = 0.0;
            int argMaxP = 0;
            int argMaxQ = 0;
            float maxP = float.NegativeInfinity;
            float maxQ = float.NegativeInfinity;

            for (int i = 0; i < vocabSize; i++)
            {
                float rp = refRow[i];
                float qp = quantRow[i];

                if (!float.IsFinite(rp) || rp < 0.0f)
                    throw new ArgumentException($"Reference probability at token {t}, index {i} must be finite and non-negative.", nameof(referenceProbs));
                if (!float.IsFinite(qp) || qp < 0.0f)
                    throw new ArgumentException($"Quantized probability at token {t}, index {i} must be finite and non-negative.", nameof(quantizedProbs));

                sumP += rp;
                sumQ += qp;

                if (i == 0 || rp > maxP)
                {
                    maxP = rp;
                    argMaxP = i;
                }
                if (i == 0 || qp > maxQ)
                {
                    maxQ = qp;
                    argMaxQ = i;
                }
            }

            if (sumP <= 0.0)
                throw new ArgumentException($"Reference probability distribution at token {t} sums to zero.", nameof(referenceProbs));
            if (sumQ <= 0.0)
                throw new ArgumentException($"Quantized probability distribution at token {t} sums to zero.", nameof(quantizedProbs));

            if (argMaxP == argMaxQ)
                top1Matches++;

            double kl = 0.0;
            for (int i = 0; i < vocabSize; i++)
            {
                double p = (double)refRow[i] / sumP;
                if (p <= 0.0)
                    continue;

                double q = Math.Max((double)quantRow[i] / sumQ, opts.Epsilon);
                kl += p * (Math.Log(p) - Math.Log(q));
            }

            if (kl < 0.0)
                kl = 0.0;

            sumKl += kl;
            if (kl > maxKl) maxKl = kl;
            if (kl < minKl) minKl = kl;
        }

        return BuildResult(sumKl, maxKl, minKl, top1Matches, tokenCount, vocabSize, opts);
    }

    private static double ComputeTokenKlCore(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double epsilon,
        out bool top1Match)
    {
        int n = referenceLogits.Length;

        // Pass 1: validate finite/non-NaN/+Inf, find maxes and argmaxes (lowest-index tie-breaker)
        double maxRef = double.NegativeInfinity;
        double maxQuant = double.NegativeInfinity;
        int argMaxRef = 0;
        int argMaxQuant = 0;
        bool hasFiniteRef = false;
        bool hasFiniteQuant = false;

        for (int i = 0; i < n; i++)
        {
            float r = referenceLogits[i];
            float q = quantizedLogits[i];

            if (float.IsNaN(r) || float.IsPositiveInfinity(r))
                throw new ArgumentException($"Reference logits contain invalid value ({r}) at index {i}.", nameof(referenceLogits));
            if (float.IsNaN(q) || float.IsPositiveInfinity(q))
                throw new ArgumentException($"Quantized logits contain invalid value ({q}) at index {i}.", nameof(quantizedLogits));

            if (!float.IsNegativeInfinity(r))
                hasFiniteRef = true;
            if (!float.IsNegativeInfinity(q))
                hasFiniteQuant = true;

            if (i == 0 || r > maxRef)
            {
                maxRef = r;
                argMaxRef = i;
            }
            if (i == 0 || q > maxQuant)
            {
                maxQuant = q;
                argMaxQuant = i;
            }
        }

        if (!hasFiniteRef)
            throw new ArgumentException("Reference logits cannot consist exclusively of -Infinity.", nameof(referenceLogits));
        if (!hasFiniteQuant)
            throw new ArgumentException("Quantized logits cannot consist exclusively of -Infinity.", nameof(quantizedLogits));

        top1Match = argMaxRef == argMaxQuant;

        // Pass 2: shifted sum of exponentials in double precision
        double sumExpRef = 0.0;
        double sumExpQuant = 0.0;
        for (int i = 0; i < n; i++)
        {
            float r = referenceLogits[i];
            if (!float.IsNegativeInfinity(r))
                sumExpRef += Math.Exp((double)r - maxRef);

            float q = quantizedLogits[i];
            if (!float.IsNegativeInfinity(q))
                sumExpQuant += Math.Exp((double)q - maxQuant);
        }

        double logSumRef = Math.Log(sumExpRef);
        double logSumQuant = Math.Log(sumExpQuant);
        double logEpsilon = Math.Log(epsilon);

        // Pass 3: accumulate KL divergence D_KL(P || Q) = sum_i P_i * (log P_i - log Q_i)
        double kl = 0.0;
        for (int i = 0; i < n; i++)
        {
            float r = referenceLogits[i];
            if (float.IsNegativeInfinity(r))
                continue;

            double logP = ((double)r - maxRef) - logSumRef;
            if (logP > 0.0) logP = 0.0;
            double p = Math.Exp(logP);
            if (p <= 0.0)
                continue;

            float q = quantizedLogits[i];
            double logQ;
            if (float.IsNegativeInfinity(q))
            {
                logQ = logEpsilon;
            }
            else
            {
                logQ = ((double)q - maxQuant) - logSumQuant;
                if (logQ > 0.0) logQ = 0.0;
                if (logQ < logEpsilon) logQ = logEpsilon;
            }

            kl += p * (logP - logQ);
        }

        return kl < 0.0 ? 0.0 : kl;
    }

    private static double FindMaxAndValidateLogits(ReadOnlySpan<float> logits, string paramName)
    {
        double maxLogit = double.NegativeInfinity;
        bool hasFinite = false;

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (float.IsNaN(v) || float.IsPositiveInfinity(v))
                throw new ArgumentException($"Logits contain invalid value ({v}) at index {i}.", paramName);

            if (!float.IsNegativeInfinity(v))
            {
                hasFinite = true;
                if (v > maxLogit)
                    maxLogit = v;
            }
        }

        if (!hasFinite)
            throw new ArgumentException("Logits cannot consist exclusively of -Infinity.", paramName);

        return maxLogit;
    }

    private static void ValidateBatchShapes(int refLength, int quantLength, int vocabSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        if (refLength == 0)
            throw new ArgumentException("Reference distribution span cannot be empty.");
        if (refLength != quantLength)
            throw new ArgumentException($"Reference length ({refLength}) must equal quantized length ({quantLength}).");
        if (refLength % vocabSize != 0)
            throw new ArgumentException($"Total element count ({refLength}) must be a multiple of vocabSize ({vocabSize}).");
    }

    private static KlDivergenceResult BuildResult(
        double sumKl,
        double maxKl,
        double minKl,
        int top1Matches,
        int tokenCount,
        int vocabSize,
        KlDivergenceOptions opts)
    {
        double meanKl = sumKl / tokenCount;
        double top1Rate = (double)top1Matches / tokenCount;

        List<string>? failures = null;
        if (meanKl > opts.Threshold)
        {
            failures ??= new List<string>();
            failures.Add($"Mean KL divergence {meanKl:G6} exceeded threshold {opts.Threshold:G6}.");
        }
        if (opts.MaxKlThreshold.HasValue && maxKl > opts.MaxKlThreshold.Value)
        {
            failures ??= new List<string>();
            failures.Add($"Max KL divergence {maxKl:G6} exceeded MaxKlThreshold {opts.MaxKlThreshold.Value:G6}.");
        }
        if (top1Rate < opts.MinTop1Agreement)
        {
            failures ??= new List<string>();
            failures.Add($"Top-1 agreement rate {top1Rate:P2} was below MinTop1Agreement {opts.MinTop1Agreement:P2}.");
        }

        bool passed = failures is null;
        return new KlDivergenceResult
        {
            MeanKlDivergence = meanKl,
            MaxKlDivergence = maxKl,
            MinKlDivergence = minKl,
            Top1AgreementRate = top1Rate,
            Top1Matches = top1Matches,
            TokenCount = tokenCount,
            VocabSize = vocabSize,
            Threshold = opts.Threshold,
            MaxKlThreshold = opts.MaxKlThreshold,
            MinTop1Agreement = opts.MinTop1Agreement,
            HasNonFiniteQuantizedValues = false,
            Passed = passed,
            FailureReason = passed ? null : string.Join(" ", failures!)
        };
    }
}
