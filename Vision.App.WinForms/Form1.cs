using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.IO;
using Vision.NeuralEngine.Layers;
using Vision.NeuralEngine.Models;
using Vision.NeuralEngine.Random;

namespace Vision.App.WinForms;

public partial class Form1 : Form
{
    private readonly System.Windows.Forms.Timer _predictDebounceTimer;
    private SequentialModel? _model;
    private string? _modelPath;

    public Form1()
    {
        InitializeComponent();

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
