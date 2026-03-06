using Vision.NeuralEngine.Core;

namespace Vision.NeuralEngine.Random;

public static class WeightInitializers
{
    public static void XavierUniform(Parameter weights, int fanIn, int fanOut, IRandomSource rng)
    {
        var limit = System.MathF.Sqrt(6f / (fanIn + fanOut));
        var w = weights.Value;

        for (var i = 0; i < w.Length; i++)
        {
            // [-limit, +limit)
            w[i] = (rng.NextSingle() * 2f - 1f) * limit;
        }
    }
}
