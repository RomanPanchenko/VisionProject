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
        var buttons = new Button[10];
        for (var d = 0; d <= 9; d++)
        {
            var digit = d;
            var btn = new Button
            {
                Text = digit.ToString(),
                Width = 46,
                Height = 32,
                Margin = new Padding(3)
            };
            btn.Click += async (_, _) => await FineTuneOnCurrentDrawingAsync(digit);
            buttons[digit] = btn;
        }

        return buttons;
    }

    private void TryLoadDefaultModel()
    {
        var defaultPath = Path.Combine(@"C:\Users\roman\RiderProjects\VisionProject", "models", "mnist_conv_relu_dense.vnd");
        if (!File.Exists(defaultPath))
        {
            lblModel.Text = "Модель: (не найдена, нажмите 'Загрузить модель...')";
            return;
        }

        try
        {
            LoadModel(defaultPath);
        }
        catch (Exception ex)
        {
            lblModel.Text = "Модель: ошибка загрузки";
            MessageBox.Show(this, ex.Message, "Ошибка загрузки модели", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
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
        var model = BuildMnistModel();
        ModelSerializer.LoadParameters(model, path);

        _model = model;
        _modelPath = path;
        lblModel.Text = $"Модель: {Path.GetFileName(path)}";

        SetFeedbackEnabled(enabled: true);
        lblFeedbackStatus.Text = "";
    }

    private static SequentialModel BuildMnistModel()
    {
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

        return new SequentialModel(new ILayer[]
        {
            conv,
            new ReLULayer(),
            new DenseLayer(inputSize: conv.OutputSize, outputSize: 10, rng: rng)
        });
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
        flowFeedbackButtons.Enabled = enabled;
        foreach (var b in _feedbackButtons)
            b.Enabled = enabled;
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
        lblFeedbackStatus.Text = $"Дообучение на классе {correctLabel}...";

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
            lblFeedbackStatus.Text = $"Готово: дообучено как {correctLabel} и сохранено.";

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

            var datasetRoot = Path.Combine(repoRoot, "datasets", "mnist-pngs", "train");
            using var preview = drawingCanvas1.CapturePreview28x28();
            LabeledSampleWriter.Save28x28Png(preview, correctLabel, datasetRoot);
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
        lblPred.Text = $"Предсказание: {pred}";

        var top = Enumerable.Range(0, probs.Length)
            .Select(i => (Class: i, P: probs[i]))
            .OrderByDescending(t => t.P)
            .Take(3)
            .ToArray();

        lblTop.Text = "top-3:\r\n" + string.Join("\r\n", top.Select(t => $"{t.Class}: {t.P:P2}"));
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
