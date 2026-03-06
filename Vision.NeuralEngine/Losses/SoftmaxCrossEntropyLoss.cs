namespace Vision.NeuralEngine.Losses;

/// <summary>
/// Численно стабильная связка softmax + cross-entropy.
/// </summary>
public sealed class SoftmaxCrossEntropyLoss : ILoss
{
    private float[]? _probabilities;
    private int _target;

    public float Forward(float[] logits, int targetClass)
    {
        if (logits.Length == 0) throw new ArgumentException("Logits are empty.", nameof(logits));
        if (targetClass < 0 || targetClass >= logits.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(targetClass));
        }

        _target = targetClass;
        var probs = _probabilities;
        if (probs is null || probs.Length != logits.Length)
        {
            probs = new float[logits.Length];
            _probabilities = probs;
        }

        var maxLogit = logits[0];
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        var sumExp = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            var e = System.MathF.Exp(logits[i] - maxLogit);
            probs[i] = e;
            sumExp += e;
        }

        var invSum = 1f / sumExp;
        for (var i = 0; i < probs.Length; i++)
        {
            probs[i] *= invSum;
        }

        var p = probs[targetClass];
        // защита от log(0)
        const float eps = 1e-12f;
        return -System.MathF.Log(p + eps);
    }

    public float[] Backward()
    {
        if (_probabilities is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        var grad = new float[_probabilities.Length];
        Array.Copy(_probabilities, grad, _probabilities.Length);
        grad[_target] -= 1f;
        return grad;
    }
}
