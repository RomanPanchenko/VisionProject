using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Vision.NeuralEngine.Training;

namespace Vision.Trainer;

public sealed record DatasetLoadResult(
    List<TrainingSample> Samples,
    IReadOnlyList<string> Labels);

public static class PngClassificationDataset
{
    public static string[] BuildLabels(IEnumerable<string> datasetRoots)
    {
        if (datasetRoots is null) throw new ArgumentNullException(nameof(datasetRoots));

        var roots = datasetRoots
            .Where(r => !string.IsNullOrWhiteSpace(r))
            .Select(r => r.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (roots.Length == 0)
            throw new ArgumentException("No dataset roots provided.", nameof(datasetRoots));

        // Собираем все доступные классы по всем корням и split'ам (train/test),
        // чтобы индексация меток была одинаковой в обучении и валидации.
        var classNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var root in roots)
        {
            if (!Directory.Exists(root)) continue;

            foreach (var split in new[] { "train", "test" })
            {
                var splitDir = Path.Combine(root, split);
                if (!Directory.Exists(splitDir)) continue;

                foreach (var classDir in Directory.EnumerateDirectories(splitDir))
                {
                    var name = Path.GetFileName(classDir);
                    if (!string.IsNullOrWhiteSpace(name))
                        classNames.Add(name);
                }
            }
        }

        if (classNames.Count == 0)
            throw new InvalidOperationException($"No class directories found in: {string.Join(", ", roots)}");

        // 2) Фиксируем порядок меток: цифры 0..9, затем латиница A..Z, затем остальное по имени.
        var digits = classNames
            .Select(n => int.TryParse(n, out var d) ? (Name: n, Digit: (int?)d) : (Name: n, Digit: null))
            .Where(t => t.Digit is >= 0 and <= 9)
            .OrderBy(t => t.Digit!.Value)
            .Select(t => t.Name)
            .ToList();

        var latin = classNames
            .Where(n => n.Length == 1 && n[0] is >= 'A' and <= 'Z')
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var rest = classNames
            .Except(digits, StringComparer.OrdinalIgnoreCase)
            .Except(latin, StringComparer.OrdinalIgnoreCase)
            .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return digits
            .Concat(latin)
            .Concat(rest)
            .ToArray();
    }

    public static DatasetLoadResult LoadMany(
        IEnumerable<string> datasetRoots,
        string split,
        IReadOnlyList<string> labels,
        int? maxSamples = null)
    {
        if (datasetRoots is null) throw new ArgumentNullException(nameof(datasetRoots));
        if (string.IsNullOrWhiteSpace(split)) throw new ArgumentException("Split is empty.", nameof(split));
        if (!string.Equals(split, "train", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(split, "test", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("Split must be 'train' or 'test'.", nameof(split));
        }
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (labels.Count == 0) throw new ArgumentException("Labels are empty.", nameof(labels));
        if (maxSamples is <= 0) throw new ArgumentOutOfRangeException(nameof(maxSamples));

        var roots = datasetRoots
            .Where(r => !string.IsNullOrWhiteSpace(r))
            .Select(r => r.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (roots.Length == 0)
            throw new ArgumentException("No dataset roots provided.", nameof(datasetRoots));

        var labelToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < labels.Count; i++)
            labelToIndex[labels[i]] = i;

        // Собираем список файлов (путь + labelIndex) и затем (опционально) делаем случайную подвыборку.
        var labeledFiles = new List<(string Path, int LabelIndex)>(capacity: 16_384);
        foreach (var root in roots)
        {
            var splitDir = Path.Combine(root, split);
            if (!Directory.Exists(splitDir)) continue;

            foreach (var classDir in Directory.EnumerateDirectories(splitDir))
            {
                var className = Path.GetFileName(classDir);
                if (string.IsNullOrWhiteSpace(className)) continue;
                if (!labelToIndex.TryGetValue(className, out var labelIndex)) continue;

                foreach (var file in Directory.EnumerateFiles(classDir, "*.png", SearchOption.TopDirectoryOnly))
                {
                    labeledFiles.Add((file, labelIndex));
                }
            }
        }

        if (labeledFiles.Count == 0)
        {
            throw new InvalidOperationException(
                $"No png samples found for split '{split}' in: {string.Join(", ", roots)}");
        }

        if (maxSamples.HasValue && labeledFiles.Count > maxSamples.Value)
        {
            ShuffleInPlace(labeledFiles, seed: 123);
            labeledFiles.RemoveRange(maxSamples.Value, labeledFiles.Count - maxSamples.Value);
        }

        var samples = new List<TrainingSample>(capacity: labeledFiles.Count);
        foreach (var (path, labelIndex) in labeledFiles)
        {
            var input = LoadPngAs28x28Hwc1(path);
            samples.Add(new TrainingSample(input, labelIndex));
        }

        return new DatasetLoadResult(samples, labels);
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

    public static string GetLabelsSidecarPath(string modelPath) => modelPath + ".labels.json";

    public static void SaveLabels(string modelPath, IReadOnlyList<string> labels)
    {
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path is empty.", nameof(modelPath));
        if (labels is null) throw new ArgumentNullException(nameof(labels));

        var sidecar = GetLabelsSidecarPath(modelPath);
        var dir = Path.GetDirectoryName(sidecar);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var json = JsonSerializer.Serialize(labels, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(sidecar, json);
    }

    public static string[]? TryLoadLabelsForModel(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath)) return null;

        var sidecar = GetLabelsSidecarPath(modelPath);
        if (!File.Exists(sidecar)) return null;

        try
        {
            var json = File.ReadAllText(sidecar);
            var labels = JsonSerializer.Deserialize<string[]>(json);
            return labels is { Length: > 0 } ? labels : null;
        }
        catch
        {
            return null;
        }
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
}
