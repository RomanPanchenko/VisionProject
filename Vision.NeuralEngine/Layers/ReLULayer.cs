using Vision.NeuralEngine.Core;

namespace Vision.NeuralEngine.Layers;

public sealed class ReLULayer : ILayer
{
    private float[]? _lastInput;

    public IReadOnlyList<Parameter> Parameters => Array.Empty<Parameter>();

    public float[] Forward(float[] input, bool training)
    {
        _lastInput = input;
        var output = new float[input.Length];

        for (var i = 0; i < input.Length; i++)
        {
            var v = input[i];
            output[i] = v > 0f ? v : 0f;
        }

        return output;
    }

    public float[] Backward(float[] dOutput)
    {
        if (_lastInput is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        if (dOutput.Length != _lastInput.Length)
        {
            throw new ArgumentException("dOutput size must match layer input size.", nameof(dOutput));
        }

        var dInput = new float[dOutput.Length];
        var x = _lastInput;

        for (var i = 0; i < dOutput.Length; i++)
        {
            dInput[i] = x[i] > 0f ? dOutput[i] : 0f;
        }

        return dInput;
    }
}
