using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Random;

namespace Vision.NeuralEngine.Layers;

/// <summary>
/// 2D свёртка (convolution) для входа в формате HWC, упакованного в <see cref="float"/>[].
///
/// Формат входа (HWC):
/// - H = height (высота)
/// - W = width (ширина)
/// - C = channels (каналы)
///
/// Упаковка HWC в одномерный массив:
/// - <c>inputIndex = ((y * W + x) * C + c)</c>
///
/// Формат выхода (тоже HWC):
/// - OH = output height
/// - OW = output width
/// - OC = output channels
/// - <c>outputIndex = ((oy * OW + ox) * OC + oc)</c>
///
/// Что именно вычисляет слой:
/// - Это «корреляция» (как в большинстве фреймворков), то есть ядро не разворачивается.
/// - Для каждой позиции выхода (oy, ox) берётся окно входа размером (KH, KW)
///   со сдвигом <see cref="_stride"/> и виртуальным дополнением нулями по краям (<see cref="_padding"/>).
/// - Для каждого выходного канала <c>oc</c> суммируется по всем входным каналам <c>ic</c>:
///   <c>sum = bias[oc] + Σ_{ic,ky,kx} W[oc,ic,ky,kx] * X[inY,inX,ic]</c>
/// </summary>
public sealed class Conv2DLayer : ILayer
{
    // Параметры входа.
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;

    // Кол-во фильтров (выходных каналов). Каждый фильтр даёт один канал на выходе.
    private readonly int _outputChannels;

    // Размер ядра (kernel) по Y и X.
    private readonly int _kernelHeight;
    private readonly int _kernelWidth;

    // Шаг свёртки (stride): на сколько пикселей смещаем окно по входу при переходе к следующей позиции выхода.
    private readonly int _stride;

    // Padding (дополнение нулями по краям входа).
    // Важно: физически мы НЕ создаём «паддинговый» массив. Мы просто пропускаем индексы,
    // выходящие за границы входа, как если бы там были нули.
    private readonly int _padding;

    // Размеры выхода вычисляются один раз в конструкторе.
    private readonly int _outputHeight;
    private readonly int _outputWidth;

    // Кэш последнего входа — нужен для Backward, чтобы посчитать градиенты по весам и по входу.
    private float[]? _lastInput;

    public Conv2DLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputChannels,
        int kernelHeight,
        int kernelWidth,
        int stride = 1,
        int padding = 0,
        IRandomSource? rng = null)
    {
        if (inputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(inputHeight));
        if (inputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(inputWidth));
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelHeight <= 0) throw new ArgumentOutOfRangeException(nameof(kernelHeight));
        if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;
        _stride = stride;
        _padding = padding;

        // Сначала считаем «виртуальные» размеры входа с паддингом.
        // Например: H=28, padding=1 => paddedH=30.
        var paddedHeight = _inputHeight + 2 * _padding;
        var paddedWidth = _inputWidth + 2 * _padding;

        // Формула для размера выхода:
        // OH = (paddedH - KH) / stride + 1
        // OW = (paddedW - KW) / stride + 1
        // (Если деление не целое — значит конфигурация stride/padding некорректна для данного входа/ядра).
        var hNumerator = paddedHeight - _kernelHeight;
        var wNumerator = paddedWidth - _kernelWidth;
        if (hNumerator < 0 || wNumerator < 0)
        {
            throw new ArgumentException("Kernel is larger than padded input.");
        }
        if (hNumerator % _stride != 0 || wNumerator % _stride != 0)
        {
            throw new ArgumentException("Invalid stride/padding: output size is not an integer.");
        }

        _outputHeight = hNumerator / _stride + 1;
        _outputWidth = wNumerator / _stride + 1;

        // Развёртка весов в 1D массив.
        // Логическая форма: W[OC, IC, KH, KW]
        // Индексация в массиве:
        //   base = ((oc * IC + ic) * KH + ky) * KW
        //   index = base + kx
        // То есть блок весов (KH*KW) для пары (oc,ic) лежит непрерывно.
        Weights = new Parameter(new float[_outputChannels * _inputChannels * _kernelHeight * _kernelWidth]);

        // Bias по одному значению на каждый выходной канал.
        Bias = new Parameter(new float[_outputChannels]);

        if (rng != null)
        {
            // XavierUniform: хорошо подходит для начальной инициализации линейных слоёв.
            // Здесь fanIn/fanOut считаем как кол-во входов/выходов на один фильтр,
            // умноженное на размер ядра.
            var fanIn = _inputChannels * _kernelHeight * _kernelWidth;
            var fanOut = _outputChannels * _kernelHeight * _kernelWidth;
            WeightInitializers.XavierUniform(Weights, fanIn, fanOut, rng);
        }
    }

    public int InputSize => _inputHeight * _inputWidth * _inputChannels;

    public int OutputSize => _outputHeight * _outputWidth * _outputChannels;

    public int OutputHeight => _outputHeight;
    public int OutputWidth => _outputWidth;

    public Parameter Weights { get; }
    public Parameter Bias { get; }

    public IReadOnlyList<Parameter> Parameters => new[] { Weights, Bias };

    public float[] Forward(float[] input, bool training)
    {
        // training сейчас не влияет на вычисление (нет dropout/batchnorm и т.п.),
        // но параметр нужен, чтобы соответствовать общему контракту ILayer.

        if (input.Length != InputSize)
        {
            throw new ArgumentException($"Expected input size {InputSize}, got {input.Length}.", nameof(input));
        }

        // Запоминаем вход для Backward.
        _lastInput = input;

        // Выход в HWC (OH, OW, OC).
        var output = new float[OutputSize];

        // Короткие локальные ссылки на массивы (чуть быстрее и читабельнее в циклах).
        var w = Weights.Value;
        var b = Bias.Value;

        // Перебираем все координаты выходной карты признаков.
        // oy, ox — координаты в выходном пространстве.
        for (var oy = 0; oy < _outputHeight; oy++)
        {
            // inY0/inX0 — координата верхнего левого угла окна на входе,
            // соответствующего текущему (oy, ox) на выходе.
            // Важно: здесь мы учитываем padding, поэтому координата может стать отрицательной.
            var inY0 = oy * _stride - _padding;
            for (var ox = 0; ox < _outputWidth; ox++)
            {
                var inX0 = ox * _stride - _padding;

                // Базовый индекс для (oy, ox, 0) в выходном HWC массиве.
                // Дальше просто прибавляем oc.
                var outBase = (oy * _outputWidth + ox) * _outputChannels;

                // Для каждого выходного канала считаем свёртку своим фильтром.
                for (var oc = 0; oc < _outputChannels; oc++)
                {
                    // Начинаем с bias.
                    var sum = b[oc];

                    // Суммируем вклад всех входных каналов.
                    for (var ic = 0; ic < _inputChannels; ic++)
                    {
                        // Начало блока весов для фиксированных (oc, ic).
                        // Далее по ky/kx пробегаем по ядру.
                        var wBase = ((oc * _inputChannels + ic) * _kernelHeight) * _kernelWidth;

                        // Пробегаем окно ядра по Y.
                        for (var ky = 0; ky < _kernelHeight; ky++)
                        {
                            var inY = inY0 + ky;

                            // Если inY вне входа, это часть паддинга (виртуальные нули) — вклад = 0,
                            // значит можно пропустить.
                            // Трюк с (uint) позволяет одной проверкой покрыть < 0 и >= H.
                            if ((uint)inY >= (uint)_inputHeight) continue;

                            // Пробегаем окно ядра по X.
                            for (var kx = 0; kx < _kernelWidth; kx++)
                            {
                                var inX = inX0 + kx;
                                // Аналогично: если inX вне входа, это паддинг => вклад 0 => пропускаем.
                                if ((uint)inX >= (uint)_inputWidth) continue;

                                // Индекс входа в HWC:
                                // ((inY * W + inX) * C + ic)
                                var inIndex = ((inY * _inputWidth + inX) * _inputChannels) + ic;

                                // Индекс веса для (oc, ic, ky, kx).
                                var wIndex = wBase + ky * _kernelWidth + kx;

                                // Основная формула свёртки: sum += W * X
                                sum += w[wIndex] * input[inIndex];
                            }
                        }
                    }

                    // Сохраняем значение в выходной тензор.
                    output[outBase + oc] = sum;
                }
            }
        }

        return output;
    }

    public float[] Backward(float[] dOutput)
    {
        // Backward получает dL/dY (градиент лосса по выходу слоя)
        // и должен вернуть dL/dX (градиент по входу слоя), а также накопить градиенты
        // по параметрам: dL/dW и dL/dB.
        //
        // Обозначения:
        // - X — вход (HWC)
        // - Y — выход (HWC)
        // - W — веса (OC, IC, KH, KW)
        // - B — bias (OC)
        // - dY = dOutput
        // - dX = dInput
        //
        // Производные:
        // - dB[oc] += dY[oy,ox,oc]
        // - dW[oc,ic,ky,kx] += dY[oy,ox,oc] * X[inY,inX,ic]
        // - dX[inY,inX,ic] += W[oc,ic,ky,kx] * dY[oy,ox,oc]

        if (dOutput.Length != OutputSize)
        {
            throw new ArgumentException($"Expected dOutput size {OutputSize}, got {dOutput.Length}.", nameof(dOutput));
        }

        if (_lastInput is null)
        {
            throw new InvalidOperationException("Forward must be called before Backward.");
        }

        // Вход, который был на Forward.
        var x = _lastInput;

        // dInput имеет ту же форму, что и вход: (H, W, C) в виде 1D массива.
        var dInput = new float[InputSize];

        // Ссылки на значения/градиенты параметров.
        var w = Weights.Value;
        var dw = Weights.Grad;
        var db = Bias.Grad;

        // Логика циклов совпадает с Forward: идём по каждой позиции выхода и распределяем её градиент
        // по соответствующим параметрам и входным элементам.
        for (var oy = 0; oy < _outputHeight; oy++)
        {
            var inY0 = oy * _stride - _padding;
            for (var ox = 0; ox < _outputWidth; ox++)
            {
                var inX0 = ox * _stride - _padding;
                var outBase = (oy * _outputWidth + ox) * _outputChannels;

                for (var oc = 0; oc < _outputChannels; oc++)
                {
                    // grad = dL/dY для конкретного элемента выхода (oy, ox, oc).
                    var grad = dOutput[outBase + oc];

                    // dL/dB[oc] — это сумма всех grad по пространственным позициям.
                    db[oc] += grad;

                    for (var ic = 0; ic < _inputChannels; ic++)
                    {
                        var wBase = ((oc * _inputChannels + ic) * _kernelHeight) * _kernelWidth;

                        for (var ky = 0; ky < _kernelHeight; ky++)
                        {
                            var inY = inY0 + ky;
                            if ((uint)inY >= (uint)_inputHeight) continue;

                            for (var kx = 0; kx < _kernelWidth; kx++)
                            {
                                var inX = inX0 + kx;
                                if ((uint)inX >= (uint)_inputWidth) continue;

                                var inIndex = ((inY * _inputWidth + inX) * _inputChannels) + ic;
                                var wIndex = wBase + ky * _kernelWidth + kx;

                                // dW += dY * X
                                dw[wIndex] += grad * x[inIndex];

                                // dX += W * dY
                                dInput[inIndex] += w[wIndex] * grad;
                            }
                        }
                    }
                }
            }
        }

        return dInput;
    }
}
