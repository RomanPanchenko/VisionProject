using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Random;

namespace Vision.App.WinForms;

internal static class MnistModelVariants
{
    public static SequentialModel BuildConv32_64_Dense128(int classCount)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

        var rng = new SplitMix64Random(123);

        var conv1 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 1,
            outputChannels: 32,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var conv2 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 32,
            outputChannels: 64,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var dense1 = new DenseLayer(inputSize: conv2.OutputSize, outputSize: 128, rng: rng);
        var denseOut = new DenseLayer(inputSize: 128, outputSize: classCount, rng: rng);

        return new SequentialModel(new ILayer[]
        {
            conv1,
            new ReLULayer(),
            conv2,
            new ReLULayer(),
            dense1,
            new ReLULayer(),
            denseOut
        });
    }

    public static SequentialModel BuildConv32_64_128_Dense128(int classCount)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

        var rng = new SplitMix64Random(123);

        var conv1 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 1,
            outputChannels: 32,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var conv2 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 32,
            outputChannels: 64,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var conv3 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 64,
            outputChannels: 128,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var dense1 = new DenseLayer(inputSize: conv3.OutputSize, outputSize: 128, rng: rng);
        var denseOut = new DenseLayer(inputSize: 128, outputSize: classCount, rng: rng);

        return new SequentialModel(new ILayer[]
        {
            conv1,
            new ReLULayer(),
            conv2,
            new ReLULayer(),
            conv3,
            new ReLULayer(),
            dense1,
            new ReLULayer(),
            denseOut
        });
    }

    public static SequentialModel BuildConv32_64_128_Dense128_Dense128(int classCount)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

        var rng = new SplitMix64Random(123);

        var conv1 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 1,
            outputChannels: 32,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var conv2 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 32,
            outputChannels: 64,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var conv3 = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 64,
            outputChannels: 128,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var dense1 = new DenseLayer(inputSize: conv3.OutputSize, outputSize: 128, rng: rng);
        var dense2 = new DenseLayer(inputSize: 128, outputSize: 128, rng: rng);
        var denseOut = new DenseLayer(inputSize: 128, outputSize: classCount, rng: rng);

        return new SequentialModel(new ILayer[]
        {
            conv1,
            new ReLULayer(),
            conv2,
            new ReLULayer(),
            conv3,
            new ReLULayer(),
            dense1,
            new ReLULayer(),
            dense2,
            new ReLULayer(),
            denseOut
        });
    }

    public static SequentialModel BuildLegacyConv8(int classCount)
    {
        if (classCount <= 0) throw new ArgumentOutOfRangeException(nameof(classCount));

        var rng = new SplitMix64Random(123);
        var conv = new Conv2DLayer(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 1,
            outputChannels: 8,
            kernelHeight: 3,
            kernelWidth: 3,
            stride: 1,
            padding: 1,
            rng: rng);

        var denseOut = new DenseLayer(inputSize: conv.OutputSize, outputSize: classCount, rng: rng);

        return new SequentialModel(new ILayer[]
        {
            conv,
            new ReLULayer(),
            denseOut
        });
    }
}
