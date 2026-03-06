using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Losses;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Optimizers;
using Vision.NeuralEngine.Training;

namespace Vision.NeuralEngine.Tests;

public class NeuralEngineTrainingTests
{
    [Fact]
    public void SoftmaxCrossEntropy_Backward_ForZeroLogits_Target0()
    {
        var loss = new SoftmaxCrossEntropyLoss();

        var value = loss.Forward(new[] { 0f, 0f }, targetClass: 0);
        var grad = loss.Backward();

        Assert.True(value > 0f);
        Assert.Equal(2, grad.Length);
        Assert.InRange(grad[0], -0.500001f, -0.499999f);
        Assert.InRange(grad[1], 0.499999f, 0.500001f);
        Assert.InRange(grad[0] + grad[1], -1e-6f, 1e-6f);
    }

    [Fact]
    public void Trainer_CanLearn_ORGate_WithSingleDense()
    {
        // Линейно разделимая задача: логистическая регрессия (Dense -> SoftmaxCE) должна справиться.
        // DenseLayer без rng инициализируется нулями => полностью детерминированный старт.

        ILayer dense = new DenseLayer(inputSize: 2, outputSize: 2);
        var model = new SequentialModel(new[] { dense });

        var samples = new TrainingSample[]
        {
            new(new[] { 0f, 0f }, Label: 0),
            new(new[] { 0f, 1f }, Label: 1),
            new(new[] { 1f, 0f }, Label: 1),
            new(new[] { 1f, 1f }, Label: 1),
        };

        var trainer = new Trainer();
        var metrics = trainer.Train(
            model,
            samples,
            loss: new SoftmaxCrossEntropyLoss(),
            optimizer: new SgdOptimizer(learningRate: 0.1f),
            options: new TrainingOptions { Epochs = 2000, Shuffle = false });

        Assert.NotEmpty(metrics);
        Assert.True(metrics[0].Loss > metrics[^1].Loss);
        Assert.InRange(metrics[^1].Accuracy, 0.99f, 1.0f);
    }
}
