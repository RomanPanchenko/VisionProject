using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace Vision.App.WinForms;

public sealed class DrawingCanvas : Control
{
    private const int BaseCanvasSize = 280;
    private const float BasePenWidth = 18f;

    private const int ModelSize = 28;
    private const int ContentSize = 20;
    private const byte InkThreshold = 12; // 0..255, учитываем антиалиас

    private Bitmap? _buffer;
    private Point _lastPoint;
    private bool _isDrawing;

    public DrawingCanvas()
    {
        DoubleBuffered = true;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);

        BackColor = Color.Black;
        ForeColor = Color.White;

        ResizeRedraw = true;
        MinimumSize = new Size(56, 56);
    }

    public event EventHandler? CanvasChanged;

    public void Clear()
    {
        EnsureBuffer();
        using var g = Graphics.FromImage(_buffer!);
        g.Clear(BackColor);
        Invalidate();
        CanvasChanged?.Invoke(this, EventArgs.Empty);
    }

    public float[] CaptureAs28x28Hwc1()
    {
        using var processed = CaptureProcessed28x28();

        var input = new float[28 * 28];
        for (var y = 0; y < 28; y++)
        {
            for (var x = 0; x < 28; x++)
            {
                var c = processed.GetPixel(x, y);
                // Канвас: фон чёрный, цифра белая. Берём яркость 0..1.
                input[y * 28 + x] = c.R / 255f;
            }
        }

        return input;
    }

    public Bitmap CapturePreview28x28()
    {
        return CaptureProcessed28x28();
    }

    private Bitmap CaptureProcessed28x28()
    {
        EnsureBuffer();

        // Важно: если пользователь рисует маленькую цифру, простое масштабирование всего холста
        // приводит к тому, что цифра становится «крошечной» в 28x28 и сеть ошибается.
        // Поэтому делаем MNIST-подобную нормализацию: bbox -> масштаб до 20x20 -> центрирование -> паддинг до 28x28.
        var outBmp = PreprocessTo28x28(_buffer!);

        try
        {
            var previewPath = Path.Combine(AppContext.BaseDirectory, "preview.png");
            outBmp.Save(previewPath, ImageFormat.Png);
        }
        catch
        {
            // ignore
        }

        return outBmp;
    }

    internal static Bitmap PreprocessTo28x28(Bitmap source)
    {
        if (source.Width <= 0 || source.Height <= 0)
            return new Bitmap(ModelSize, ModelSize, PixelFormat.Format24bppRgb);

        var bounds = FindInkBounds(source, InkThreshold);
        if (bounds is null)
        {
            var blank = new Bitmap(ModelSize, ModelSize, PixelFormat.Format24bppRgb);
            using var gBlank = Graphics.FromImage(blank);
            gBlank.Clear(Color.Black);
            return blank;
        }

        var b = bounds.Value;

        // Небольшой паддинг вокруг содержимого, чтобы штрихи у границы bbox не обрезались.
        const int pad = 2;
        var x = Math.Max(0, b.X - pad);
        var y = Math.Max(0, b.Y - pad);
        var r = Math.Min(source.Width, b.Right + pad);
        var bt = Math.Min(source.Height, b.Bottom + pad);
        var w = Math.Max(1, r - x);
        var h = Math.Max(1, bt - y);

        using var cropped = new Bitmap(w, h, PixelFormat.Format24bppRgb);
        using (var gCrop = Graphics.FromImage(cropped))
        {
            gCrop.Clear(Color.Black);
            gCrop.InterpolationMode = InterpolationMode.NearestNeighbor;
            gCrop.PixelOffsetMode = PixelOffsetMode.HighQuality;
            gCrop.DrawImage(source, new Rectangle(0, 0, w, h), new Rectangle(x, y, w, h), GraphicsUnit.Pixel);
        }

        var scale = ContentSize / (float)Math.Max(w, h);
        var dstW = Math.Max(1, (int)MathF.Round(w * scale));
        var dstH = Math.Max(1, (int)MathF.Round(h * scale));

        using var scaled = new Bitmap(dstW, dstH, PixelFormat.Format24bppRgb);
        using (var gScale = Graphics.FromImage(scaled))
        {
            gScale.Clear(Color.Black);
            gScale.InterpolationMode = InterpolationMode.HighQualityBilinear;
            gScale.PixelOffsetMode = PixelOffsetMode.HighQuality;
            gScale.SmoothingMode = SmoothingMode.None;
            gScale.DrawImage(cropped, 0, 0, dstW, dstH);
        }

        var outBmp = new Bitmap(ModelSize, ModelSize, PixelFormat.Format24bppRgb);
        using (var g = Graphics.FromImage(outBmp))
        {
            g.Clear(Color.Black);
            g.InterpolationMode = InterpolationMode.HighQualityBilinear;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.SmoothingMode = SmoothingMode.None;

            var ox = (ModelSize - dstW) / 2;
            var oy = (ModelSize - dstH) / 2;
            g.DrawImage(scaled, ox, oy, dstW, dstH);
        }

        return outBmp;
    }

    private static Rectangle? FindInkBounds(Bitmap bmp, byte threshold)
    {
        // Ожидаем чёрный фон и белые/серые штрихи. Берём яркость по каналу R (формат 24bpp).
        var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
        var data = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            var minX = int.MaxValue;
            var minY = int.MaxValue;
            var maxX = int.MinValue;
            var maxY = int.MinValue;
            var stride = data.Stride;

            // Без unsafe: копируем буфер в managed-массив и сканируем.
            var bytes = new byte[Math.Abs(stride) * bmp.Height];
            System.Runtime.InteropServices.Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);

            for (var y = 0; y < bmp.Height; y++)
            {
                var rowStart = y * stride;
                for (var x = 0; x < bmp.Width; x++)
                {
                    // BGR
                    var r = bytes[rowStart + x * 3 + 2];
                    if (r <= threshold) continue;

                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }

            if (minX == int.MaxValue) return null;
            return Rectangle.FromLTRB(minX, minY, maxX + 1, maxY + 1);
        }
        finally
        {
            bmp.UnlockBits(data);
        }
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        EnsureBuffer();

        e.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        e.Graphics.PixelOffsetMode = PixelOffsetMode.Half;
        e.Graphics.DrawImage(_buffer!, 0, 0, ClientSize.Width, ClientSize.Height);
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        ResizeBufferIfNeeded();
        Invalidate();
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button == MouseButtons.Right)
        {
            Clear();
            return;
        }

        if (e.Button != MouseButtons.Left) return;

        EnsureBuffer();
        _isDrawing = true;
        _lastPoint = e.Location;
        DrawDot(e.Location);
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (!_isDrawing) return;

        DrawLine(_lastPoint, e.Location);
        _lastPoint = e.Location;
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        if (e.Button != MouseButtons.Left) return;
        _isDrawing = false;
    }

    private void DrawLine(Point a, Point b)
    {
        EnsureBuffer();
        using var g = Graphics.FromImage(_buffer!);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.PixelOffsetMode = PixelOffsetMode.HighQuality;

        using var pen = new Pen(ForeColor, GetScaledPenWidth())
        {
            StartCap = LineCap.Round,
            EndCap = LineCap.Round,
            LineJoin = LineJoin.Round
        };

        var sa = ClientToBuffer(a);
        var sb = ClientToBuffer(b);
        g.DrawLine(pen, sa, sb);

        Invalidate();
        CanvasChanged?.Invoke(this, EventArgs.Empty);
    }

    private void DrawDot(Point p)
    {
        EnsureBuffer();
        using var g = Graphics.FromImage(_buffer!);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.PixelOffsetMode = PixelOffsetMode.HighQuality;

        var bw = GetScaledPenWidth();
        var s = ClientToBuffer(p);
        using var brush = new SolidBrush(ForeColor);
        g.FillEllipse(brush, s.X - bw / 2f, s.Y - bw / 2f, bw, bw);

        Invalidate();
        CanvasChanged?.Invoke(this, EventArgs.Empty);
    }

    private float GetScaledPenWidth()
    {
        var w = Math.Max(1, ClientSize.Width);
        var h = Math.Max(1, ClientSize.Height);
        var scale = Math.Min(w, h) / (float)BaseCanvasSize;
        return Math.Max(2f, BasePenWidth * scale);
    }

    private PointF ClientToBuffer(Point p)
    {
        EnsureBuffer();

        var sx = _buffer!.Width / (float)Math.Max(1, ClientSize.Width);
        var sy = _buffer!.Height / (float)Math.Max(1, ClientSize.Height);
        return new PointF(p.X * sx, p.Y * sy);
    }

    private void EnsureBuffer()
    {
        if (_buffer is not null && _buffer.Width > 0 && _buffer.Height > 0) return;
        ResizeBufferIfNeeded(force: true);
    }

    private void ResizeBufferIfNeeded(bool force = false)
    {
        var w = Math.Max(1, ClientSize.Width);
        var h = Math.Max(1, ClientSize.Height);

        if (!force && _buffer is not null && _buffer.Width == w && _buffer.Height == h) return;

        var newBuffer = new Bitmap(w, h, PixelFormat.Format24bppRgb);
        using (var g = Graphics.FromImage(newBuffer))
        {
            g.Clear(BackColor);
            if (_buffer is not null)
            {
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(_buffer, 0, 0, w, h);
            }
        }

        _buffer?.Dispose();
        _buffer = newBuffer;
    }
}
