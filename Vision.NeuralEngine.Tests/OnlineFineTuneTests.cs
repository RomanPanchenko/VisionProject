using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Losses;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Optimizers;
using Vision.NeuralEngine.Training;

namespace Vision.NeuralEngine.Tests;

public class OnlineFineTuneTests
{
    [Fact]
    public void Trainer_FineTuneOnSingleSample_IncreasesProbabilityOfCorrectClass()
    {
        // Делаем максимально простой и детерминированный сценарий:
        // один Dense слой без RNG => веса инициализируются нулями.
        var model = new SequentialModel(new ILayer[]
        {
            new DenseLayer(inputSize: 28 * 28, outputSize: 10)
        });

        var input = new float[28 * 28];
        input[10] = 1f;
        var label = 7;

        var before = Softmax(model.Forward(input, training: false))[label];

        var trainer = new Trainer();
        trainer.Train(
            model,
            samples: RepeatSample(input, label, count: 30),
            loss: new SoftmaxCrossEntropyLoss(),
            optimizer: new SgdOptimizer(learningRate: 0.1f),
            options: new TrainingOptions { Epochs = 1, Shuffle = false });

        var after = Softmax(model.Forward(input, training: false))[label];

        Assert.True(after > before, $"Expected P(correct) to increase, before={before}, after={after}.");
    }

    private static TrainingSample[] RepeatSample(float[] input, int label, int count)
    {
        var samples = new TrainingSample[count];
        for (var i = 0; i < count; i++)
            samples[i] = new TrainingSample(input, label);
        return samples;
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits[0];
        for (var i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        var exps = new float[logits.Length];
        var sum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            var e = MathF.Exp(logits[i] - max);
            exps[i] = e;
            sum += e;
        }

        for (var i = 0; i < exps.Length; i++)
            exps[i] /= sum;

        return exps;
    }
}
