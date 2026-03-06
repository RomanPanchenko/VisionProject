namespace Vision.NeuralEngine.Core;

public interface ILayer
{
    float[] Forward(float[] input, bool training);

    float[] Backward(float[] dOutput);

    IReadOnlyList<Parameter> Parameters { get; }
}
