namespace Vision.NeuralEngine.Math;

internal static class VectorOps
{
    public static int ArgMax(ReadOnlySpan<float> v)
    {
        if (v.Length == 0)
        {
            throw new ArgumentException("Vector is empty.", nameof(v));
        }

        var bestIndex = 0;
        var bestValue = v[0];

        for (var i = 1; i < v.Length; i++)
        {
            var value = v[i];
            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
