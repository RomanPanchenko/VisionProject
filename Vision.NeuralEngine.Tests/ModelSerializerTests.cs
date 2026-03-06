using Vision.NeuralEngine.IO;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Models;

namespace Vision.NeuralEngine.Tests;

public class ModelSerializerTests
{
    [Fact]
    public void SaveLoadParameters_RoundTrip_RestoresWeights()
    {
        // DenseLayer без rng инициализируется нулями => сначала зададим явные веса.
        var dense = new DenseLayer(inputSize: 3, outputSize: 2);
        var model = new SequentialModel(new[] { (Vision.NeuralEngine.Core.ILayer)dense });

        FillWithTestPattern(model);

        var tmp = Path.Combine(Path.GetTempPath(), $"vision_model_{Guid.NewGuid():N}.vnd");
        try
        {
            var expected = Snapshot(model);
            ModelSerializer.SaveParameters(model, tmp);

            // Искажаем параметры.
            ZeroOut(model);

            ModelSerializer.LoadParameters(model, tmp);
            var actual = Snapshot(model);

            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    private static float[] Snapshot(SequentialModel model)
    {
        var list = new List<float>();
        foreach (var p in model.Parameters)
        {
            list.AddRange(p.Value);
        }

        return list.ToArray();
    }

    private static void FillWithTestPattern(SequentialModel model)
    {
        var v = 0.1f;
        foreach (var p in model.Parameters)
        {
            for (var i = 0; i < p.Value.Length; i++)
            {
                // Дет. шаблон, чтобы отличать элементы.
                p.Value[i] = v;
                v += 0.1f;
            }
        }
    }

    private static void ZeroOut(SequentialModel model)
    {
        foreach (var p in model.Parameters)
        {
            Array.Clear(p.Value);
        }
    }
}
