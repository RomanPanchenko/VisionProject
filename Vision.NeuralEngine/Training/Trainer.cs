using Vision.NeuralEngine.Losses;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Optimizers;
using System.Diagnostics;

namespace Vision.NeuralEngine.Training;

public sealed class Trainer
{
    public IReadOnlyList<TrainingMetrics> Train(
        SequentialModel model,
        IReadOnlyList<TrainingSample> samples,
        ILoss loss,
        IOptimizer optimizer,
        TrainingOptions options,
        System.Random? shuffleRng = null)
    {
        if (options.Epochs <= 0) throw new ArgumentOutOfRangeException(nameof(options.Epochs));
        if (samples.Count == 0) throw new ArgumentException("Samples are empty.", nameof(samples));

        var rng = shuffleRng ?? new System.Random(123);
        var indices = Enumerable.Range(0, samples.Count).ToArray();
        var metrics = new List<TrainingMetrics>(capacity: options.Epochs);

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            if (options.ReportProgressToConsole)
            {
                Console.WriteLine($"Epoch {epoch}/{options.Epochs}");
            }

            if (options.Shuffle)
            {
                Shuffle(indices, rng);
            }

            var totalLoss = 0f;
            var correct = 0;

            var progressStep = options.ProgressPercentStep;
            if (progressStep < 1) progressStep = 1;
            if (progressStep > 100) progressStep = 100;
            var lastPrintedPercentInt = -1;
            var minReportInterval = options.ProgressMinReportInterval;
            if (minReportInterval < TimeSpan.Zero) minReportInterval = TimeSpan.Zero;
            var lastPrintedAt = TimeSpan.Zero;
            var stopwatch = options.ReportProgressToConsole ? Stopwatch.StartNew() : null;

            for (var s = 0; s < indices.Length; s++)
            {
                var sample = samples[indices[s]];

                model.ZeroGrad();

                var logits = model.Forward(sample.Input, training: true);
                var l = loss.Forward(logits, sample.Label);
                totalLoss += l;

                var predicted = ArgMax(logits);
                if (predicted == sample.Label) correct++;

                var dLogits = loss.Backward();
                model.Backward(dLogits);
                optimizer.Step(model.Parameters);

                if (options.ReportProgressToConsole)
                {
                    var percentInt = (int)(((s + 1L) * 100L) / indices.Length);
                    var percentPrecise = ((s + 1d) * 100d) / indices.Length;

                    var timeOk = stopwatch is not null && (stopwatch.Elapsed - lastPrintedAt) >= minReportInterval;
                    var stepOk = percentInt == 100 || (percentInt % progressStep == 0 && percentInt != lastPrintedPercentInt);
                    var shouldPrint = timeOk || stepOk;
                    if (shouldPrint)
                    {
                        Console.Write($"\r  {percentPrecise,6:F2}%");
                        lastPrintedPercentInt = percentInt;
                        if (stopwatch is not null) lastPrintedAt = stopwatch.Elapsed;
                        if (percentInt == 100) Console.WriteLine();
                    }
                }
            }

            var avgLoss = totalLoss / samples.Count;
            var accuracy = correct / (float)samples.Count;
            metrics.Add(new TrainingMetrics(epoch, avgLoss, accuracy));

            if (options.ReportProgressToConsole)
            {
                Console.WriteLine($"  loss={avgLoss:F4} acc={accuracy:P2}");
            }
        }

        return metrics;
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
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

    private static void Shuffle(int[] a, System.Random rng)
    {
        for (var i = a.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
