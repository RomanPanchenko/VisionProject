using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Vision.ImagePreprocessing;

public sealed record Letter28x28PreprocessOptions(
    byte InkThreshold = 24,
    int TargetSize = 28,
    int InnerSize = 20,
    int BboxPadding = 2,
    bool AutoInvertByMean = true);

public static class Letter28x28Preprocessor
{
    public static Image<L8> LoadAndPreprocessTo28x28(string inputPath, Letter28x28PreprocessOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(inputPath))
            throw new ArgumentException("Input path is empty.", nameof(inputPath));

        options ??= new Letter28x28PreprocessOptions();

        using var rgba = Image.Load<Rgba32>(inputPath);
        using var gray = ToL8BestEffort(rgba);
        return PreprocessTo28x28(gray, options);
    }

    private static Image<L8> ToL8BestEffort(Image<Rgba32> rgba)
    {
        // В некоторых датасетах «чернила» закодированы в альфа-канале:
        // RGB может быть весь чёрный, а контур/буква видны только по A.
        // Поэтому сначала проверяем вариативность альфы и используем её как интенсивность,
        // иначе — стандартная яркость из RGB.

        var stats = SampleRanges(rgba);
        var alphaRange = stats.MaxA - stats.MinA;

        // Важно: альфа почти всегда «немного» варьируется из-за антиалиасинга/ресемплинга,
        // но это не значит, что в ней закодированы чернила. Поэтому используем A только если:
        // 1) у альфы заметный динамический диапазон
        // 2) при этом RGB почти не содержит сигнала (картинка «почти чёрная»)
        var useAlpha = alphaRange >= 32 && stats.MaxLuma <= 8;

        var gray = new Image<L8>(rgba.Width, rgba.Height);
        rgba.ProcessPixelRows(srcAccessor =>
        {
            for (var y = 0; y < rgba.Height; y++)
            {
                var srcRow = srcAccessor.GetRowSpan(y);
                for (var x = 0; x < rgba.Width; x++)
                {
                    var p = srcRow[x];
                    byte v;
                    if (useAlpha)
                    {
                        v = p.A;
                    }
                    else
                    {
                        // Rec.601 luma approximation.
                        v = (byte)((p.R * 77 + p.G * 150 + p.B * 29 + 128) >> 8);
                    }

                    gray[x, y] = new L8(v);
                }
            }
        });

        return gray;
    }

    private readonly record struct RangeStats(byte MinA, byte MaxA, byte MinLuma, byte MaxLuma);

    private static RangeStats SampleRanges(Image<Rgba32> rgba)
    {
        var stepX = Math.Max(1, rgba.Width / 128);
        var stepY = Math.Max(1, rgba.Height / 128);

        var minA = (byte)255;
        var maxA = (byte)0;
        var minL = (byte)255;
        var maxL = (byte)0;

        rgba.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < rgba.Height; y += stepY)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < rgba.Width; x += stepX)
                {
                    var p = row[x];
                    var a = p.A;
                    if (a < minA) minA = a;
                    if (a > maxA) maxA = a;

                    // Rec.601 luma approximation.
                    var l = (byte)((p.R * 77 + p.G * 150 + p.B * 29 + 128) >> 8);
                    if (l < minL) minL = l;
                    if (l > maxL) maxL = l;
                }
            }
        });

        return new RangeStats(minA, maxA, minL, maxL);
    }

    public static Image<L8> PreprocessTo28x28(Image<L8> gray, Letter28x28PreprocessOptions options)
    {
        if (gray is null) throw new ArgumentNullException(nameof(gray));
        if (options.TargetSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.TargetSize));
        if (options.InnerSize <= 0 || options.InnerSize > options.TargetSize)
            throw new ArgumentOutOfRangeException(nameof(options.InnerSize));

        // Стандартизируем в формат "белые чернила на чёрном фоне".
        // Для большинства входов достаточно эвристики по среднему: если картинка в среднем светлая — это белый фон.
        var inverted = options.AutoInvertByMean && IsLikelyWhiteBackground(gray);

        var bbox = FindInkBoundingBox(gray, options.InkThreshold, inverted);
        if (bbox is null)
        {
            // Пустое изображение (или слишком высокий порог) — возвращаем чёрный 28x28.
            return new Image<L8>(options.TargetSize, options.TargetSize, new L8(0));
        }

        var padded = ExpandRect(bbox.Value, options.BboxPadding, gray.Width, gray.Height);

        using var cropped = gray.Clone(ctx => ctx.Crop(padded));
        if (inverted)
            cropped.Mutate(ctx => ctx.Invert());

        // Масштабируем так, чтобы максимум сторон стал InnerSize (как в MNIST: 20x20 внутри 28x28).
        var scale = options.InnerSize / (float)Math.Max(cropped.Width, cropped.Height);
        var newW = Math.Max(1, (int)MathF.Round(cropped.Width * scale));
        var newH = Math.Max(1, (int)MathF.Round(cropped.Height * scale));

        cropped.Mutate(ctx => ctx.Resize(new ResizeOptions
        {
            Size = new Size(newW, newH),
            Mode = ResizeMode.Stretch,
            Sampler = KnownResamplers.Bicubic
        }));

        var canvas = new Image<L8>(options.TargetSize, options.TargetSize, new L8(0));
        var offsetX = (options.TargetSize - newW) / 2;
        var offsetY = (options.TargetSize - newH) / 2;

        canvas.Mutate(ctx => ctx.DrawImage(cropped, new Point(offsetX, offsetY), 1f));
        return canvas;
    }

    private static bool IsLikelyWhiteBackground(Image<L8> gray)
    {
        // Быстрая оценка среднего (sub-sample), чтобы не гонять полный проход на больших изображениях.
        var stepX = Math.Max(1, gray.Width / 128);
        var stepY = Math.Max(1, gray.Height / 128);

        long sum = 0;
        long count = 0;
        for (var y = 0; y < gray.Height; y += stepY)
        {
            for (var x = 0; x < gray.Width; x += stepX)
            {
                sum += gray[x, y].PackedValue;
                count++;
            }
        }

        var mean = (float)sum / Math.Max(1, count);
        return mean > 127f;
    }

    private static Rectangle? FindInkBoundingBox(Image<L8> gray, byte threshold, bool inverted)
    {
        var minX = int.MaxValue;
        var minY = int.MaxValue;
        var maxX = int.MinValue;
        var maxY = int.MinValue;

        for (var y = 0; y < gray.Height; y++)
        {
            for (var x = 0; x < gray.Width; x++)
            {
                var v = gray[x, y].PackedValue;
                if (inverted) v = (byte)(255 - v);

                if (v <= threshold) continue;

                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }

        if (minX == int.MaxValue) return null;
        return Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
    }

    private static Rectangle ExpandRect(Rectangle rect, int padding, int limitW, int limitH)
    {
        if (padding <= 0) return rect;

        var x0 = Math.Max(0, rect.X - padding);
        var y0 = Math.Max(0, rect.Y - padding);
        var x1 = Math.Min(limitW, rect.Right + padding);
        var y1 = Math.Min(limitH, rect.Bottom + padding);
        return Rectangle.FromLTRB(x0, y0, x1, y1);
    }
}
