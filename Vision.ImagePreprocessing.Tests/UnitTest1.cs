using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Vision.ImagePreprocessing.Tests;

public class Letter28x28PreprocessorTests
{
    [Fact]
    public void Preprocess_Blank_ReturnsBlack28x28()
    {
        using var src = new Image<L8>(120, 90, new L8(0));

        using var out28 = Letter28x28Preprocessor.PreprocessTo28x28(src, new Letter28x28PreprocessOptions(InkThreshold: 24));

        Assert.Equal(28, out28.Width);
        Assert.Equal(28, out28.Height);
        Assert.All(Flatten(out28), v => Assert.Equal(0, v));
    }

    [Fact]
    public void Preprocess_OffsetInk_IsCenteredToCanvas()
    {
        using var src = new Image<L8>(200, 150, new L8(0));
        PutRect(src, x0: 140, y0: 10, w: 30, h: 60, value: 255);

        using var out28 = Letter28x28Preprocessor.PreprocessTo28x28(src, new Letter28x28PreprocessOptions(InkThreshold: 24));

        Assert.Equal(28, out28.Width);
        Assert.Equal(28, out28.Height);

        var (cx, cy) = CenterOfMass(out28, threshold: 24);
        Assert.InRange(cx, 12f, 16f);
        Assert.InRange(cy, 12f, 16f);
    }

    [Fact]
    public void Preprocess_WhiteBackgroundBlackInk_AutoInvertsToWhiteInkOnBlack()
    {
        using var src = new Image<L8>(180, 180, new L8(255));
        PutRect(src, x0: 40, y0: 60, w: 35, h: 50, value: 0);

        using var out28 = Letter28x28Preprocessor.PreprocessTo28x28(src, new Letter28x28PreprocessOptions(InkThreshold: 24));

        // На выходе должен быть чёрный фон и яркие чернила.
        var max = Flatten(out28).Max();
        Assert.True(max >= 200, $"Expected bright ink after invert, got max={max}");

        // Углы должны оставаться близкими к чёрному.
        Assert.True(out28[0, 0].PackedValue <= 10);
        Assert.True(out28[27, 0].PackedValue <= 10);
        Assert.True(out28[0, 27].PackedValue <= 10);
        Assert.True(out28[27, 27].PackedValue <= 10);
    }

    [Fact]
    public void LoadAndPreprocess_AlphaAlmostOpaque_DoesNotTreatAlphaAsInk()
    {
        using var rgba = new Image<Rgba32>(64, 64, new Rgba32(255, 255, 255, 255));

        // Чёрные «чернила» в RGB.
        for (var y = 20; y < 44; y++)
        for (var x = 22; x < 42; x++)
            rgba[x, y] = new Rgba32(0, 0, 0, 255);

        // Небольшая вариативность альфы (антиалиасинг/ресемплинг) — не должна переключать режим на A-канал.
        rgba[0, 0] = new Rgba32(255, 255, 255, 254);

        var tmpPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png");
        try
        {
            rgba.SaveAsPng(tmpPath);

            using var out28 = Letter28x28Preprocessor.LoadAndPreprocessTo28x28(
                tmpPath,
                new Letter28x28PreprocessOptions(InkThreshold: 24));

            var max = Flatten(out28).Max();
            Assert.True(max >= 200, $"Expected non-empty output, got max={max}");
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }

    private static void PutRect(Image<L8> img, int x0, int y0, int w, int h, byte value)
    {
        for (var y = y0; y < y0 + h; y++)
        for (var x = x0; x < x0 + w; x++)
        {
            if ((uint)x >= (uint)img.Width || (uint)y >= (uint)img.Height) continue;
            img[x, y] = new L8(value);
        }
    }

    private static IEnumerable<byte> Flatten(Image<L8> img)
    {
        for (var y = 0; y < img.Height; y++)
        for (var x = 0; x < img.Width; x++)
            yield return img[x, y].PackedValue;
    }

    private static (float Cx, float Cy) CenterOfMass(Image<L8> img, byte threshold)
    {
        double sum = 0;
        double sumX = 0;
        double sumY = 0;

        for (var y = 0; y < img.Height; y++)
        for (var x = 0; x < img.Width; x++)
        {
            var v = img[x, y].PackedValue;
            if (v <= threshold) continue;

            sum += v;
            sumX += x * v;
            sumY += y * v;
        }

        if (sum <= 0) return (img.Width / 2f, img.Height / 2f);
        return ((float)(sumX / sum), (float)(sumY / sum));
    }
}
