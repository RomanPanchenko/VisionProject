using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Random;

namespace Vision.NeuralEngine.Layers;

public sealed class DenseLayer : ILayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;

    private float[]? _lastInput;

    public DenseLayer(int inputSize, int outputSize, IRandomSource? rng = null)
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));

        _inputSize = inputSize;
        _outputSize = outputSize;

        Weights = new Parameter(new float[outputSize * inputSize]);
        Bias = new Parameter(new float[outputSize]);

        if (rng != null)
        {
            WeightInitializers.XavierUniform(Weights, fanIn: inputSize, fanOut: outputSize, rng);
        }
    }

    public Parameter Weights { get; }
    public Parameter Bias { get; }

    public IReadOnlyList<Parameter> Parameters => new[] { Weights, Bias };

    public float[] Forward(float[] input, bool training)
    {
        if (input.Length != _inputSize)
        {
            throw new ArgumentException($"Expected input size {_inputSize}, got {input.Length}.", nameof(input));
        }

        _lastInput = input;

        var output = new float[_outputSize];
        var w = Weights.Value;
        var b = Bias.Value;

        for (var o = 0; o < _outputSize; o++)
        {
            var sum = b[o];
            var rowOffset = o * _inputSize;

            for (var i = 0; i < _inputSize; i++)
            {
                sum += w[rowOffset + i] * input[i];
            }

            output[o] = sum;
        }

        return output;
    }

    public float[] Backward(float[] dOutput)
    {
        if (dOutput.Length != _outputSize)
        {
            throw new ArgumentException($"Expected dOutput size {_outputSize}, got {dOutput.Length}.", nameof(dOutput));
        }

        if (_lastInput is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        var x = _lastInput;
        var dInput = new float[_inputSize];

        var w = Weights.Value;
        var dw = Weights.Grad;
        var db = Bias.Grad;

        for (var o = 0; o < _outputSize; o++)
        {
            var grad = dOutput[o];
            db[o] += grad;

            var rowOffset = o * _inputSize;
            for (var i = 0; i < _inputSize; i++)
            {
                dw[rowOffset + i] += grad * x[i];
                dInput[i] += w[rowOffset + i] * grad;
            }
        }

        return dInput;
    }
}
