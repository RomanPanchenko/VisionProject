using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Losses;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Optimizers;
using Vision.NeuralEngine.Random;
using Vision.NeuralEngine.Training;
using Vision.NeuralEngine.IO;
using Vision.Trainer;

var dataRoot = GetArgValue(args, "--dataRoot");
var extraDataRoots = GetArgValues(args, "--extraDataRoot");
var classCountArg = GetArgValueInt(args, "--classCount");

// По умолчанию обучаемся на цифрах + латинице (без кириллицы, чтобы не смешивать похожие символы).
var datasetRoots = BuildDefaultDatasetRoots(dataRoot, extraDataRoots);

// Для совместимости со старым выводом (train/test dirs).
var primaryRoot = !string.IsNullOrWhiteSpace(dataRoot) ? dataRoot : datasetRoots[0];
var trainDir = Path.Combine(primaryRoot, "train");
var testDir = Path.Combine(primaryRoot, "test");

var trainLimit = GetArgValueInt(args, "--trainLimit") ?? 5_000;
var testLimit = GetArgValueInt(args, "--testLimit") ?? 1_000;
var epochs = GetArgValueInt(args, "--epochs") ?? 3;
var learningRate = GetArgValueFloat(args, "--lr") ?? 0.01f;
var loadModelPath = GetArgValue(args, "--loadModel");
var saveModelPath = GetArgValue(args, "--saveModel");
var printProbs = GetArgValueInt(args, "--printProbs") ?? 0;
var topK = GetArgValueInt(args, "--topK") ?? 3;
var predictPngPath = GetArgValue(args, "--predictPng");
var predictDirPath = GetArgValue(args, "--predictDir");
var diagnosePredict = HasFlag(args, "--diagnosePredict");
var predictRecursive = HasFlag(args, "--predictRecursive");
var predictLimit = GetArgValueInt(args, "--predictLimit");

Console.WriteLine($"Data root: {Path.GetFullPath(primaryRoot)}");
Console.WriteLine($"Train dir: {Path.GetFullPath(trainDir)} (limit={trainLimit})");
Console.WriteLine($"Test dir:  {Path.GetFullPath(testDir)} (limit={testLimit})");

Console.WriteLine("Datasets:");
foreach (var r in datasetRoots)
    Console.WriteLine($"  - {Path.GetFullPath(r)}");

List<TrainingSample>? trainSamples = null;
List<TrainingSample>? testSamples = null;
string[]? labels = null;
if (string.IsNullOrWhiteSpace(predictPngPath) && string.IsNullOrWhiteSpace(predictDirPath))
{
    labels = PngClassificationDataset.BuildLabels(datasetRoots);

    trainSamples = PngClassificationDataset.LoadMany(datasetRoots, split: "train", labels, maxSamples: trainLimit).Samples;
    testSamples = PngClassificationDataset.LoadMany(datasetRoots, split: "test", labels, maxSamples: testLimit).Samples;
}
else
{
    // Для предикта нам нужны имена классов, чтобы красиво печатать результаты.
    // Если sidecar нет — просто работаем с индексацией 0..N-1.
    if (!string.IsNullOrWhiteSpace(loadModelPath))
        labels = PngClassificationDataset.TryLoadLabelsForModel(loadModelPath);
}

var classCount = classCountArg ?? labels?.Length;
if (classCount is null || classCount <= 0)
{
    throw new InvalidOperationException(
        "Не удалось определить число классов. " +
        "Для обучения оно берётся из датасетов, для предикта нужно либо наличие sidecar '*.labels.json', " +
        "либо указать --classCount <N>.");
}

// Модель:
//   Conv(32,3x3,pad=1) -> ReLU -> Conv(64,3x3,pad=1) -> ReLU
//   -> Dense(128) -> ReLU -> Dense(N)
// Вход: 28x28x1, выход: N логитов классов.
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
var denseOut = new DenseLayer(inputSize: 128, outputSize: classCount.Value, rng: rng);

var model = new SequentialModel(new ILayer[]
{
    conv1,
    new ReLULayer(),
    conv2,
    new ReLULayer(),
    dense1,
    new ReLULayer(),
    denseOut
});

if (!string.IsNullOrWhiteSpace(loadModelPath))
{
    Console.WriteLine($"Loading model parameters from: {Path.GetFullPath(loadModelPath)}");
    ModelSerializer.LoadParameters(model, loadModelPath);
}

if (!string.IsNullOrWhiteSpace(predictPngPath))
{
    if (string.IsNullOrWhiteSpace(loadModelPath))
    {
        throw new ArgumentException("For --predictPng you must also specify --loadModel <path-to-vnd>.");
    }

    var fullPngPath = Path.GetFullPath(predictPngPath);
    Console.WriteLine($"\nPredicting PNG: {fullPngPath}");

    var expectedLabel = TryInferExpectedLabelFromPath(fullPngPath, labels);
    if (expectedLabel.HasValue)
        Console.WriteLine($"Expected label (from parent dir): {FormatLabel(expectedLabel.Value, labels)}");

    var input = PngClassificationDataset.LoadPngAs28x28Hwc1(fullPngPath);

    if (diagnosePredict)
    {
        PrintInputStats(input);
    }

    var logits = model.Forward(input, training: false);
    var probs = Softmax(logits);

    var pred = 0;
    var bestP = probs[0];
    for (var c = 1; c < probs.Length; c++)
    {
        if (probs[c] > bestP)
        {
            bestP = probs[c];
            pred = c;
        }
    }

    Console.WriteLine($"Prediction: {FormatLabel(pred, labels)} (p={bestP:F4})");

    var probsClassCount = probs.Length;
    var k = Math.Clamp(topK, 1, probsClassCount);
    var top = Enumerable.Range(0, probs.Length)
        .Select(c => (Class: c, P: probs[c]))
        .OrderByDescending(t => t.P)
        .Take(k);
    Console.WriteLine("Top: " + string.Join(", ", top.Select(t => $"{FormatLabel(t.Class, labels)}:{t.P:F4}")));

    if (expectedLabel.HasValue)
    {
        Console.WriteLine($"Result: {(pred == expectedLabel.Value ? "CORRECT" : "WRONG")}");
    }

    return;
}

if (!string.IsNullOrWhiteSpace(predictDirPath))
{
    if (string.IsNullOrWhiteSpace(loadModelPath))
    {
        throw new ArgumentException("For --predictDir you must also specify --loadModel <path-to-vnd>.");
    }

    var fullDirPath = Path.GetFullPath(predictDirPath);
    Console.WriteLine($"\nPredicting directory: {fullDirPath}");
    Console.WriteLine($"Recursive: {predictRecursive}, Limit: {(predictLimit.HasValue ? predictLimit.Value.ToString() : "none")}");

    if (!Directory.Exists(fullDirPath))
        throw new DirectoryNotFoundException($"Directory not found: '{fullDirPath}'");

    var option = predictRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
    var files = Directory.EnumerateFiles(fullDirPath, "*.png", option);

    var total = 0;
    var correct = 0;
    var cc = classCount.Value;
    var perLabelTotal = new int[cc];
    var perLabelCorrect = new int[cc];

    foreach (var file in files)
    {
        var expected = TryInferExpectedLabelFromPath(file, labels);
        if (!expected.HasValue) continue;

        if ((uint)expected.Value >= (uint)cc) continue;

        var input = PngClassificationDataset.LoadPngAs28x28Hwc1(file);
        var pred = model.PredictClass(input);

        total++;
        perLabelTotal[expected.Value]++;
        if (pred == expected.Value)
        {
            correct++;
            perLabelCorrect[expected.Value]++;
        }

        if (predictLimit.HasValue && total >= predictLimit.Value)
            break;
    }

    if (total == 0)
    {
        Console.WriteLine("No labeled PNGs found (expected parent dir name to match known labels).");
        return;
    }

    Console.WriteLine($"Accuracy: {correct}/{total} = {(correct / (float)total):P2}");
    Console.WriteLine("Per-label accuracy:");
    for (var label = 0; label < cc; label++)
    {
        if (perLabelTotal[label] == 0) continue;
        var acc = perLabelCorrect[label] / (float)perLabelTotal[label];
        Console.WriteLine($"  {FormatLabel(label, labels)}: {perLabelCorrect[label]}/{perLabelTotal[label]} = {acc:P2}");
    }

    return;
}

var loss = new SoftmaxCrossEntropyLoss();
var optimizer = new SgdOptimizer(learningRate);
var trainer = new Trainer();
var options = new TrainingOptions { Epochs = epochs, Shuffle = true, ReportProgressToConsole = true, ProgressPercentStep = 5 };

if (epochs > 0)
{
    Console.WriteLine($"Start training: epochs={epochs}, lr={learningRate}");
    var metrics = trainer.Train(model, trainSamples!, loss, optimizer, options);
    foreach (var m in metrics)
    {
        Console.WriteLine($"epoch={m.Epoch} loss={m.Loss:F4} acc={m.Accuracy:P2}");
    }
}
else
{
    Console.WriteLine("Skip training (epochs <= 0). Evaluating only...");
}

var testAcc = EvaluateAccuracy(model, testSamples!);
Console.WriteLine($"Test accuracy: {testAcc:P2}");

if (printProbs > 0)
{
    Console.WriteLine($"\nPredictions (first {printProbs} samples, topK={topK}):");
    PrintPredictionsWithProbabilities(model, testSamples!, labels, limit: printProbs, topK: topK);
}

if (!string.IsNullOrWhiteSpace(saveModelPath))
{
    Console.WriteLine($"Saving model parameters to: {Path.GetFullPath(saveModelPath)}");
    ModelSerializer.SaveParameters(model, saveModelPath);

    if (labels is { Length: > 0 })
    {
        PngClassificationDataset.SaveLabels(saveModelPath, labels);
        Console.WriteLine($"Saved labels to: {Path.GetFullPath(PngClassificationDataset.GetLabelsSidecarPath(saveModelPath))}");
    }
}

static float EvaluateAccuracy(SequentialModel model, IReadOnlyList<TrainingSample> samples)
{
    var correct = 0;
    for (var i = 0; i < samples.Count; i++)
    {
        var s = samples[i];
        var pred = model.PredictClass(s.Input);
        if (pred == s.Label) correct++;
    }

    return correct / (float)samples.Count;
}

static void PrintPredictionsWithProbabilities(
    SequentialModel model,
    IReadOnlyList<TrainingSample> samples,
    IReadOnlyList<string>? labels,
    int limit,
    int topK)
{
    if (limit <= 0) return;
    if (topK <= 0) topK = 1;

    limit = Math.Min(limit, samples.Count);
    topK = Math.Min(topK, 10);

    for (var i = 0; i < limit; i++)
    {
        var s = samples[i];
        var logits = model.Forward(s.Input, training: false);
        var probs = Softmax(logits);

        var pred = 0;
        var bestP = probs[0];
        for (var c = 1; c < probs.Length; c++)
        {
            if (probs[c] > bestP)
            {
                bestP = probs[c];
                pred = c;
            }
        }

        Console.WriteLine($"[{i}] true={s.Label} pred={pred} p(pred)={bestP:F4}");

        var top = Enumerable.Range(0, probs.Length)
            .Select(c => (Class: c, P: probs[c]))
            .OrderByDescending(t => t.P)
            .Take(topK);
        Console.WriteLine("    top: " + string.Join(", ", top.Select(t => $"{FormatLabel(t.Class, labels)}:{t.P:F4}")));
    }
}

static float[] Softmax(float[] logits)
{
    // Численно устойчивый softmax.
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

    // Защита от NaN/Inf на всякий случай.
    if (!(sum > 0f) || float.IsNaN(sum) || float.IsInfinity(sum))
    {
        var uniform = 1f / logits.Length;
        for (var i = 0; i < exps.Length; i++) exps[i] = uniform;
        return exps;
    }

    for (var i = 0; i < exps.Length; i++)
        exps[i] /= sum;

    return exps;
}

static string? GetArgValue(string[] args, string key)
{
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
        {
            return args[i + 1];
        }
    }

    return null;
}

static string[] GetArgValues(string[] args, string key)
{
    var values = new List<string>();
    for (var i = 0; i < args.Length - 1; i++)
    {
        if (string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
        {
            var v = args[i + 1];
            if (!string.IsNullOrWhiteSpace(v))
                values.Add(v);
        }
    }

    return values.ToArray();
}

static int? GetArgValueInt(string[] args, string key)
{
    var v = GetArgValue(args, key);
    return v != null && int.TryParse(v, out var n) ? n : null;
}

static float? GetArgValueFloat(string[] args, string key)
{
    var v = GetArgValue(args, key);
    return v != null && float.TryParse(v, System.Globalization.CultureInfo.InvariantCulture, out var n) ? n : null;
}

static bool HasFlag(string[] args, string key)
{
    for (var i = 0; i < args.Length; i++)
    {
        if (string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
            return true;
    }

    return false;
}

static int? TryInferExpectedLabelFromPath(string path, IReadOnlyList<string>? labels)
{
    try
    {
        var dir = Path.GetDirectoryName(path);
        if (string.IsNullOrWhiteSpace(dir)) return null;

        var parentName = new DirectoryInfo(dir).Name;

        if (labels is null || labels.Count == 0)
        {
            // Фоллбэк для старого MNIST формата.
            return int.TryParse(parentName, out var d) && d is >= 0 and <= 9 ? d : null;
        }

        for (var i = 0; i < labels.Count; i++)
        {
            if (string.Equals(labels[i], parentName, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return null;
    }
    catch
    {
        return null;
    }
}

static string FormatLabel(int index, IReadOnlyList<string>? labels)
{
    if (labels is null || index < 0 || index >= labels.Count)
        return index.ToString();
    return $"{index}('{labels[index]}')";
}

static string[] BuildDefaultDatasetRoots(string? dataRootArg, string[] extraDataRoots)
{
    if (!string.IsNullOrWhiteSpace(dataRootArg))
    {
        // Совместимость: если явно указан --dataRoot, используем его как единственный датасет.
        // Доп. датасеты всё равно можно подключить через --extraDataRoot.
        return new[] { dataRootArg }
            .Concat(extraDataRoots)
            .ToArray();
    }

    return new[]
        {
            Path.Combine("datasets", "mnist-digits"),
            Path.Combine("datasets", "mnist-latin")
        }
        .Concat(extraDataRoots)
        .ToArray();
}

static void PrintInputStats(float[] input)
{
    if (input.Length == 0)
    {
        Console.WriteLine("Input stats: empty");
        return;
    }

    var min = input[0];
    var max = input[0];
    var sum = 0f;
    var nonZero = 0;

    for (var i = 0; i < input.Length; i++)
    {
        var v = input[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v > 0f) nonZero++;
    }

    var mean = sum / input.Length;
    Console.WriteLine($"Input stats: len={input.Length}, min={min:F4}, max={max:F4}, mean={mean:F4}, nonZero={nonZero}");
}
