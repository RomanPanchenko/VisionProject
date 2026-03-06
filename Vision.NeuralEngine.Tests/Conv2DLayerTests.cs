using Vision.NeuralEngine.Layers;

namespace Vision.NeuralEngine.Tests;

public class Conv2DLayerTests
{
    [Fact]
    public void Forward_ReturnsExpectedSize_AndBackwardReturnsInputSize()
    {
        var layer = new Conv2DLayer(
            inputHeight: 4,
            inputWidth: 4,
            inputChannels: 1,
            outputChannels: 3,
            kernelHeight: 2,
            kernelWidth: 2,
            stride: 1,
            padding: 0);

        var input = new float[4 * 4 * 1];
        for (var i = 0; i < input.Length; i++) input[i] = i;

        var output = layer.Forward(input, training: true);
        Assert.Equal(3 * 3 * 3, output.Length);
        Assert.Equal(3, layer.OutputHeight);
        Assert.Equal(3, layer.OutputWidth);

        var dOutput = new float[output.Length];
        for (var i = 0; i < dOutput.Length; i++) dOutput[i] = 1f;

        var dInput = layer.Backward(dOutput);
        Assert.Equal(input.Length, dInput.Length);
    }

    [Fact]
    public void Backward_MatchesNumericalGradient_ForSmallCase()
    {
        // Маленький кейс, чтобы численный градиент был дешёвым:
        // Input: 3x3x1, Kernel: 2x2, Stride 1, Padding 0, OC=1 => Output 2x2x1 (4 значения)
        var layer = new Conv2DLayer(
            inputHeight: 3,
            inputWidth: 3,
            inputChannels: 1,
            outputChannels: 1,
            kernelHeight: 2,
            kernelWidth: 2,
            stride: 1,
            padding: 0);

        // Зафиксируем веса/биасы вручную, чтобы избежать случайности.
        var w = layer.Weights.Value;
        w[0] = 0.10f;
        w[1] = -0.20f;
        w[2] = 0.05f;
        w[3] = 0.30f;
        layer.Bias.Value[0] = 0.01f;

        var x = new float[3 * 3 * 1]
        {
            0.2f, -0.1f, 0.0f,
            0.3f, 0.4f, -0.2f,
            0.1f, -0.3f, 0.5f
        };

        layer.Weights.ZeroGrad();
        layer.Bias.ZeroGrad();

        var y = layer.Forward(x, training: true);
        var loss = Sum(y);

        var dOutput = new float[y.Length];
        for (var i = 0; i < dOutput.Length; i++) dOutput[i] = 1f; // d/dy Sum(y) = 1

        var dInput = layer.Backward(dOutput);

        const float eps = 1e-3f;
        const float tol = 5e-2f;

        // Проверим несколько производных по весам.
        for (var wi = 0; wi < w.Length; wi++)
        {
            var numeric = NumericalGradWeight(layer, x, wi, eps);
            var analytic = layer.Weights.Grad[wi];
            Assert.InRange(analytic - numeric, -tol, tol);
        }

        // Проверим несколько производных по входу.
        for (var xi = 0; xi < x.Length; xi++)
        {
            var numeric = NumericalGradInput(layer, x, xi, eps);
            var analytic = dInput[xi];
            Assert.InRange(analytic - numeric, -tol, tol);
        }

        // sanity-check: loss должен совпасть с пересчитанным
        Assert.InRange(loss - Sum(layer.Forward(x, training: false)), -1e-6f, 1e-6f);
    }

    private static float NumericalGradWeight(Conv2DLayer layer, float[] x, int wIndex, float eps)
    {
        var w = layer.Weights.Value;
        var old = w[wIndex];

        w[wIndex] = old + eps;
        var lp = Sum(layer.Forward(x, training: false));

        w[wIndex] = old - eps;
        var lm = Sum(layer.Forward(x, training: false));

        w[wIndex] = old;
        return (lp - lm) / (2f * eps);
    }

    private static float NumericalGradInput(Conv2DLayer layer, float[] x, int xIndex, float eps)
    {
        var old = x[xIndex];

        x[xIndex] = old + eps;
        var lp = Sum(layer.Forward(x, training: false));

        x[xIndex] = old - eps;
        var lm = Sum(layer.Forward(x, training: false));

        x[xIndex] = old;
        return (lp - lm) / (2f * eps);
    }

    private static float Sum(float[] v)
    {
        var s = 0f;
        for (var i = 0; i < v.Length; i++) s += v[i];
        return s;
    }
}
