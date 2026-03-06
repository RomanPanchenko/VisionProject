using Vision.NeuralEngine.Core;

namespace Vision.NeuralEngine.Optimizers;

public interface IOptimizer
{
    void Step(IReadOnlyList<Parameter> parameters);
}
