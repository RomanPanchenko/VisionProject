using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.IO;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Losses;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Optimizers;
using Vision.NeuralEngine.Random;
using Vision.NeuralEngine.Training;

namespace Vision.App.WinForms;

public partial class Form1 : Form
{
    private readonly System.Windows.Forms.Timer _predictDebounceTimer;
    private SequentialModel? _model;
    private string? _modelPath;

    private string[]? _labels;
    private int _classCount = 10;

    private readonly Button[] _feedbackButtons;

    public Form1()
    {
        InitializeComponent();

        _feedbackButtons = CreateFeedbackButtons();
        foreach (var b in _feedbackButtons)
            flowFeedbackButtons.Controls.Add(b);
        SetFeedbackEnabled(enabled: false);

        btnClear.Click += (_, _) =>
        {
            drawingCanvas1.Clear();
            UpdatePredictionUi(null);
        };

        btnLoadModel.Click += (_, _) => LoadModelViaDialog();

        drawingCanvas1.CanvasChanged += (_, _) => SchedulePredict();

        _predictDebounceTimer = new System.Windows.Forms.Timer
        {
            Interval = 120
        };
        _predictDebounceTimer.Tick += (_, _) =>
        {
            _predictDebounceTimer.Stop();
            PredictNow();
        };

        TryLoadDefaultModel();
        drawingCanvas1.Clear();
        UpdatePredictionUi(null);
    }

    private Button[] CreateFeedbackButtons()
    {
        var buttons = new List<Button>(capacity: 40);

        // 0..9
        for (var d = 0; d <= 9; d++)
            buttons.Add(CreateFeedbackButton(d.ToString(), labelIndex: d));

        // A..Z (по умолчанию индексация как в trainer: 0..9 затем A..Z)
        for (var i = 0; i < 26; i++)
        {
            var ch = (char)('A' + i);
            buttons.Add(CreateFeedbackButton(ch.ToString(), labelIndex: 10 + i));
        }

        return buttons.ToArray();
    }

    private Button CreateFeedbackButton(string text, int labelIndex)
    {
        var btn = new Button
        {
            Text = text,
            // В правой панели `flowFeedbackButtons` имеет ширину ~306px.
            // При Width=46 и Margin=3 получается 5 кнопок в ряд и последняя (обычно `Z`) уезжает
            // в отдельную строку вниз, из-за чего кажется что "кнопки Z нет".
            // Width=45 позволяет уместить 6 кнопок в ряд (45 + 2*3 = 51; 6*51 = 306).
            Width = 45,
            Height = 32,
            Margin = new Padding(3),
            Tag = labelIndex
        };
        btn.Click += async (_, _) => await FineTuneOnCurrentDrawingAsync(labelIndex);
        return btn;
    }

    private void TryLoadDefaultModel()
    {
        var repoRoot = RepoRootLocator.FindRepoRootOrNull(AppContext.BaseDirectory);
        var defaultPath = repoRoot is null
            ? null
            : Path.Combine(repoRoot, "models", "mnist_conv_relu_dense.vnd");

        if (string.IsNullOrWhiteSpace(defaultPath) || !File.Exists(defaultPath))
        {
            lblModel.Text = "Модель: (не найдена, нажмите 'Загрузить модель...')";
            return;
        }

        // При старте приложения лучше не показывать ошибку модального окна из-за старой/несовместимой модели.
        // Если дефолтная модель не подходит по размерностям — просто не автозагружаем её.
        if (!IsModelCompatibleWithAnyKnownFormat(defaultPath))
        {
            lblModel.Text = "Модель: (дефолтная модель несовместима — нажмите 'Загрузить модель...')";
            return;
        }

        try
        {
            LoadModel(defaultPath);
        }
        catch
        {
            lblModel.Text = "Модель: (не удалось загрузить — нажмите 'Загрузить модель...')";
        }
    }

    private static bool IsModelCompatibleWithAnyKnownFormat(string modelPath)
    {
        if (!ModelSerializer.TryReadSignature(modelPath, out var sig, out _))
            return false;

        // 1) Если рядом лежит labels.json — проверяем под него.
        var labels = TryLoadLabelsForModel(modelPath);
        if (labels is { Length: > 0 })
        {
            if (IsSignatureCompatibleWithModel(sig, BuildMnistModel(classCount: labels.Length)))
                return true;
        }

        // 2) Иначе пробуем наиболее частые варианты.
        if (IsSignatureCompatibleWithModel(sig, BuildMnistModel(classCount: 36)))
            return true;
        if (IsSignatureCompatibleWithModel(sig, BuildMnistModel(classCount: 10)))
            return true;

        return false;
    }

    private static bool IsSignatureCompatibleWithModel(ModelSerializer.ModelFileSignature sig, SequentialModel model)
    {
        var parameters = model.Parameters;
        if (sig.ParamCount != parameters.Count) return false;
        if (sig.Lengths.Length != parameters.Count) return false;

        for (var i = 0; i < parameters.Count; i++)
        {
            if (sig.Lengths[i] != parameters[i].Value.Length)
                return false;
        }

        return true;
    }

    private void LoadModelViaDialog()
    {
        using var dialog = new OpenFileDialog
        {
            Title = "Выберите файл модели (*.vnd)",
            Filter = "Vision Neural Data (*.vnd)|*.vnd|All files (*.*)|*.*",
            CheckFileExists = true,
            CheckPathExists = true
        };

        if (dialog.ShowDialog(this) != DialogResult.OK) return;

        try
        {
            LoadModel(dialog.FileName);
            SchedulePredict();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка загрузки модели", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void LoadModel(string path)
    {
        _labels = TryLoadLabelsForModel(path);

        if (_labels is { Length: > 0 })
        {
            _classCount = _labels.Length;

            if (!VndModelLoader.TryLoadCompatible(path, classCount: _classCount, out var model, out _, out var error))
                throw BuildModelLoadException(path, error);

            _model = model;
        }
        else
        {
            // Если labels рядом с моделью отсутствуют, пробуем сначала формат "digits+latin" (36 классов),
            // затем откатываемся к классическому MNIST (10 классов).
            if (!VndModelLoader.TryLoadCompatible(path, classCount: 36, out var model, out _, out _))
            {
                if (!VndModelLoader.TryLoadCompatible(path, classCount: 10, out model, out _, out var error))
                    throw BuildModelLoadException(path, error);
            }

            _model = model;
            _classCount = model.Layers.Count > 0 && model.Layers[^1] is DenseLayer dense
                ? dense.Bias.Value.Length
                : 10;
            _labels = _classCount == 36 ? BuildDefaultDigitsLatinLabels() : null;
        }

        _modelPath = path;
        lblModel.Text = $"Модель: {Path.GetFileName(path)}";

        SetFeedbackEnabled(enabled: true);
        lblFeedbackStatus.Text = "";
    }

    private static InvalidOperationException BuildModelLoadException(string path, Exception? error)
    {
        var details = error is null ? "" : $"\n\nДетали: {error.Message}";
        return new InvalidOperationException(
            "Не удалось загрузить модель: несовпадение архитектуры/размерностей. " +
            "Выберите другой файл модели (*.vnd) или переобучите и сохраните модель заново." +
            $"\nФайл: {Path.GetFileName(path)}" +
            details);
    }

    private static string[] BuildDefaultDigitsLatinLabels()
    {
        var labels = new List<string>(capacity: 36);
        for (var d = 0; d <= 9; d++) labels.Add(d.ToString());
        for (var i = 0; i < 26; i++) labels.Add(((char)('A' + i)).ToString());
        return labels.ToArray();
    }

    private static SequentialModel BuildMnistModel(int classCount)
    {
        // Оставлено для обратной совместимости (старый код может вызывать этот метод).
        return MnistModelVariants.BuildConv32_64_Dense128(classCount);
    }

    private void SchedulePredict()
    {
        if (_model is null) return;
        _predictDebounceTimer.Stop();
        _predictDebounceTimer.Start();
    }

    private void PredictNow()
    {
        if (_model is null)
        {
            UpdatePredictionUi(null);
            return;
        }

        var input = drawingCanvas1.CaptureAs28x28Hwc1();
        if (IsBlankInput(input))
        {
            picturePreview.Image?.Dispose();
            picturePreview.Image = null;
            UpdatePredictionUi(null);
            return;
        }
        var logits = _model.Forward(input, training: false);
        var probs = Softmax(logits);

        var pred = 0;
        var bestP = probs[0];
        for (var i = 1; i < probs.Length; i++)
        {
            if (probs[i] > bestP)
            {
                bestP = probs[i];
                pred = i;
            }
        }

        using var preview = drawingCanvas1.CapturePreview28x28();
        picturePreview.Image?.Dispose();
        picturePreview.Image = (Bitmap)preview.Clone();

        UpdatePredictionUi((pred, probs));
    }

    private void SetFeedbackEnabled(bool enabled)
    {
        foreach (var b in _feedbackButtons)
        {
            var idx = b.Tag is int i ? i : -1;
            b.Enabled = enabled && idx >= 0 && idx < _classCount;
        }
    }

    private async Task FineTuneOnCurrentDrawingAsync(int correctLabel)
    {
        if (_model is null)
        {
            MessageBox.Show(this, "Сначала загрузите модель.", "Нет модели", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        var input = drawingCanvas1.CaptureAs28x28Hwc1();
        if (IsBlankInput(input))
        {
            lblFeedbackStatus.Text = "Нечего дообучать: холст пуст.";
            return;
        }

        TrySaveCurrentDrawingToDataset(correctLabel);

        SetFeedbackEnabled(enabled: false);
        btnClear.Enabled = false;
        btnLoadModel.Enabled = false;
        lblFeedbackStatus.Text = $"Дообучение на классе {FormatLabel(correctLabel, _labels)}...";

        try
        {
            await Task.Run(() => FineTuneBlocking(_model, input, correctLabel));

            var savePath = _modelPath;
            if (string.IsNullOrWhiteSpace(savePath))
            {
                using var dialog = new SaveFileDialog
                {
                    Title = "Куда сохранить обновлённую модель (*.vnd)",
                    Filter = "Vision Neural Data (*.vnd)|*.vnd|All files (*.*)|*.*",
                    AddExtension = true,
                    DefaultExt = "vnd"
                };
                if (dialog.ShowDialog(this) != DialogResult.OK)
                {
                    lblFeedbackStatus.Text = "Дообучено, но не сохранено (отменено пользователем).";
                    return;
                }

                savePath = dialog.FileName;
                _modelPath = savePath;
                lblModel.Text = $"Модель: {Path.GetFileName(savePath)}";
            }

            ModelSerializer.SaveParameters(_model, savePath);

            // Сохраняем labels рядом с моделью, чтобы можно было корректно интерпретировать индексы.
            if (_labels is { Length: > 0 })
                TrySaveLabelsForModel(savePath, _labels);

            lblFeedbackStatus.Text = $"Готово: дообучено как {FormatLabel(correctLabel, _labels)} и сохранено.";

            // Обновляем предсказание после изменения весов.
            PredictNow();
        }
        catch (Exception ex)
        {
            lblFeedbackStatus.Text = "Ошибка дообучения.";
            MessageBox.Show(this, ex.Message, "Ошибка дообучения", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            btnClear.Enabled = true;
            btnLoadModel.Enabled = true;
            SetFeedbackEnabled(enabled: true);
        }
    }

    private void TrySaveCurrentDrawingToDataset(int correctLabel)
    {
        try
        {
            var repoRoot = RepoRootLocator.FindRepoRootOrNull(AppContext.BaseDirectory);
            if (string.IsNullOrWhiteSpace(repoRoot)) return;

            var (datasetRoot, className) = MapLabelToDataset(repoRoot, correctLabel, _labels);
            using var preview = drawingCanvas1.CapturePreview28x28();
            LabeledSampleWriter.Save28x28Png(preview, className, datasetRoot);
        }
        catch
        {
            // Сохранение примера — побочный эффект. Не мешаем дообучению, если не получилось записать файл.
        }
    }

    private static bool IsBlankInput(float[] input)
    {
        // На чёрном фоне все значения близки к 0. Если чернил нет — не обучаем.
        var sum = 0f;
        for (var i = 0; i < input.Length; i++)
            sum += input[i];

        return sum < 1e-3f;
    }

    private static void FineTuneBlocking(SequentialModel model, float[] input, int correctLabel)
    {
        var trainer = new Trainer();
        var loss = new SoftmaxCrossEntropyLoss();
        var optimizer = new SgdOptimizer(learningRate: 0.01f);

        // Повторяем один и тот же пример много раз: получаем несколько SGD-шагов.
        const int steps = 40;
        var samples = new TrainingSample[steps];
        for (var i = 0; i < steps; i++)
        {
            // Важно: Trainer не модифицирует sample.Input, можно шарить один и тот же массив.
            samples[i] = new TrainingSample(input, correctLabel);
        }

        trainer.Train(
            model,
            samples,
            loss,
            optimizer,
            new TrainingOptions { Epochs = 1, Shuffle = false },
            shuffleRng: null);
    }

    private void UpdatePredictionUi((int Pred, float[] Probs)? result)
    {
        if (result is null)
        {
            lblPred.Text = "Предсказание: -";
            lblTop.Text = "top-3: -";
            return;
        }

        var (pred, probs) = result.Value;
        lblPred.Text = $"Предсказание: {FormatLabel(pred, _labels)}";

        var top = Enumerable.Range(0, probs.Length)
            .Select(i => (Class: i, P: probs[i]))
            .OrderByDescending(t => t.P)
            .Take(3)
            .ToArray();

        lblTop.Text = "top-3:\r\n" + string.Join("\r\n", top.Select(t => $"{FormatLabel(t.Class, _labels)}: {t.P:P2}"));
    }

    private static string FormatLabel(int index, IReadOnlyList<string>? labels)
    {
        if (labels is null || index < 0 || index >= labels.Count)
            return index.ToString();
        return $"{index}('{labels[index]}')";
    }

    private static string[]? TryLoadLabelsForModel(string modelPath)
    {
        try
        {
            var sidecar = modelPath + ".labels.json";
            if (!File.Exists(sidecar)) return null;

            var json = File.ReadAllText(sidecar);
            var labels = System.Text.Json.JsonSerializer.Deserialize<string[]>(json);
            return labels is { Length: > 0 } ? labels : null;
        }
        catch
        {
            return null;
        }
    }

    private static void TrySaveLabelsForModel(string modelPath, IReadOnlyList<string> labels)
    {
        try
        {
            var sidecar = modelPath + ".labels.json";
            var dir = Path.GetDirectoryName(sidecar);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            var json = System.Text.Json.JsonSerializer.Serialize(labels, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(sidecar, json);
        }
        catch
        {
            // Sidecar — удобство. Ошибка записи не должна ломать дообучение/сохранение модели.
        }
    }

    private static (string DatasetRoot, string ClassName) MapLabelToDataset(string repoRoot, int labelIndex, IReadOnlyList<string>? labels)
    {
        // Если labels известны — используем их для выбора папки класса.
        if (labels is { Count: > 0 } && labelIndex >= 0 && labelIndex < labels.Count)
        {
            var className = labels[labelIndex];
            var isDigit = int.TryParse(className, out var d) && d is >= 0 and <= 9;
            var dataset = isDigit ? "mnist-digits" : "mnist-latin";
            return (Path.Combine(repoRoot, "datasets", dataset, "train"), className);
        }

        // Фоллбэк: старая логика (только цифры).
        return (Path.Combine(repoRoot, "datasets", "mnist-digits", "train"), labelIndex.ToString());
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
}
