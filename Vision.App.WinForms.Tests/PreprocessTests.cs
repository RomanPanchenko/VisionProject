using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Vision.App.WinForms;

namespace Vision.App.WinForms.Tests;

public class PreprocessTests
{
    [Fact]
    public void PreprocessTo28x28_SmallInkIsUpscaledAndCentered()
    {
        using var bmp = new Bitmap(280, 280, PixelFormat.Format24bppRgb);
        using (var g = Graphics.FromImage(bmp))
        {
            g.Clear(Color.Black);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            using var pen = new Pen(Color.White, 18f)
            {
                StartCap = LineCap.Round,
                EndCap = LineCap.Round,
                LineJoin = LineJoin.Round
            };

            // Маленькая «1» в правом нижнем углу.
            g.DrawLine(pen, 250, 190, 250, 260);
        }

        using var processed = DrawingCanvas.PreprocessTo28x28(bmp);
        var (minX, minY, maxX, maxY, count) = GetInkBounds(processed, threshold: 20);

        Assert.True(count > 30, $"Expected enough ink pixels, got {count}.");

        // Должно оказаться примерно в центре, а не в углу.
        var cx = (minX + maxX) / 2f;
        var cy = (minY + maxY) / 2f;
        Assert.InRange(cx, 10f, 18f);
        Assert.InRange(cy, 10f, 18f);
    }

    private static (int MinX, int MinY, int MaxX, int MaxY, int Count) GetInkBounds(Bitmap bmp, byte threshold)
    {
        var minX = int.MaxValue;
        var minY = int.MaxValue;
        var maxX = int.MinValue;
        var maxY = int.MinValue;
        var count = 0;

        for (var y = 0; y < bmp.Height; y++)
        {
            for (var x = 0; x < bmp.Width; x++)
            {
                var r = bmp.GetPixel(x, y).R;
                if (r <= threshold) continue;
                count++;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }

        if (count == 0) return (0, 0, 0, 0, 0);
        return (minX, minY, maxX, maxY, count);
    }
}
