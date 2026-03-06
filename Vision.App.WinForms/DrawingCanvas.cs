using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace Vision.App.WinForms;

public sealed class DrawingCanvas : Control
{
    private const int BaseCanvasSize = 280;
    private const float BasePenWidth = 18f;

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

        // Препроцессинг как в Trainer:
        // - нужен 28x28 grayscale (яркость 0..255), дальше в CaptureAs28x28Hwc1() делим на 255 => 0..1
        // - без пороговой бинаризации
        // Здесь делаем только масштабирование всего буфера до 28x28.
        var outBmp = new Bitmap(28, 28, PixelFormat.Format24bppRgb);
        using (var g = Graphics.FromImage(outBmp))
        {
            g.Clear(Color.Black);
            g.InterpolationMode = InterpolationMode.HighQualityBilinear;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.SmoothingMode = SmoothingMode.None;
            g.DrawImage(_buffer!, 0, 0, 28, 28);
        }

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
