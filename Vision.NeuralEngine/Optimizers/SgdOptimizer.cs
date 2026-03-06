using Vision.NeuralEngine.Core;

namespace Vision.NeuralEngine.Optimizers;

public sealed class SgdOptimizer : IOptimizer
{
    public SgdOptimizer(float learningRate)
    {
        if (learningRate <= 0f) throw new ArgumentOutOfRangeException(nameof(learningRate));
        LearningRate = learningRate;
    }

    public float LearningRate { get; }

    public void Step(IReadOnlyList<Parameter> parameters)
    {
        foreach (var p in parameters)
        {
            var w = p.Value;
            var g = p.Grad;

            for (var i = 0; i < w.Length; i++)
            {
                w[i] -= LearningRate * g[i];
            }
        }
    }
}
