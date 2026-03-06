using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Vision.NeuralEngine.Training;

namespace Vision.Trainer;

public static class MnistPngDataset
{
    public static List<TrainingSample> Load(string rootDir, int? maxSamples = null)
    {
        if (string.IsNullOrWhiteSpace(rootDir)) throw new ArgumentException("Root directory is empty.", nameof(rootDir));
        if (!Directory.Exists(rootDir)) throw new DirectoryNotFoundException($"Dataset directory not found: '{rootDir}'");
        if (maxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(maxSamples));

        // Важно: нельзя просто «взять первые N файлов», потому что тогда при лимите
        // легко получить только один класс (например, все файлы из папки '0').
        // Поэтому сначала собираем список всех файлов с метками, затем (если нужно)
        // случайно подвыбираем maxSamples и только потом загружаем пиксели.

        var labeledFiles = new List<(string Path, int Label)>(capacity: 16_384);

        // Ожидаем структуру: root\0\*.png ... root\9\*.png
        for (var label = 0; label <= 9; label++)
        {
            var classDir = Path.Combine(rootDir, label.ToString());
            if (!Directory.Exists(classDir)) continue;

            foreach (var file in Directory.EnumerateFiles(classDir, "*.png", SearchOption.TopDirectoryOnly))
            {
                labeledFiles.Add((file, label));
            }
        }

        if (labeledFiles.Count == 0)
        {
            throw new InvalidOperationException($"No png samples found in '{rootDir}'.");
        }

        if (maxSamples.HasValue && labeledFiles.Count > maxSamples.Value)
        {
            ShuffleInPlace(labeledFiles, seed: 123);
            labeledFiles.RemoveRange(maxSamples.Value, labeledFiles.Count - maxSamples.Value);
        }

        var samples = new List<TrainingSample>(capacity: labeledFiles.Count);
        foreach (var (path, label) in labeledFiles)
        {
            var input = LoadPngAs28x28Hwc1(path);
            samples.Add(new TrainingSample(input, label));
        }

        return samples;
    }

    private static void ShuffleInPlace<T>(IList<T> list, int seed)
    {
        var rnd = new System.Random(seed);
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rnd.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    public static float[] LoadPngAs28x28Hwc1(string path)
    {
        using var image = Image.Load<L8>(path);
        if (image.Width != 28 || image.Height != 28)
        {
            throw new InvalidOperationException($"Expected 28x28 image, got {image.Width}x{image.Height}: '{path}'");
        }

        var input = new float[28 * 28];
        for (var y = 0; y < 28; y++)
        {
            for (var x = 0; x < 28; x++)
            {
                // L8: один байт яркости 0..255
                var v = image[x, y].PackedValue / 255f;
                input[y * 28 + x] = v;
            }
        }

        return input;
    }
}
