using System.Drawing.Imaging;
using Vision.App.WinForms;

namespace Vision.App.WinForms.Tests;

public class LabeledSampleWriterTests
{
    [Fact]
    public void Save28x28Png_CreatesFileInCorrectLabelFolder_WithUniqueNames()
    {
        var root = Path.Combine(Path.GetTempPath(), "VisionProject_Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);

        try
        {
            using var bmp = new Bitmap(28, 28, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.Black);
                g.FillRectangle(Brushes.White, 10, 10, 8, 8);
            }

            var p1 = LabeledSampleWriter.Save28x28Png(bmp, label: 7, datasetRootDir: root);
            var p2 = LabeledSampleWriter.Save28x28Png(bmp, label: 7, datasetRootDir: root);

            Assert.True(File.Exists(p1), $"Expected file to exist: {p1}");
            Assert.True(File.Exists(p2), $"Expected file to exist: {p2}");
            Assert.NotEqual(p1, p2);

            var expectedDir = Path.Combine(root, "7");
            Assert.Equal(expectedDir, Path.GetDirectoryName(p1));
            Assert.Equal(expectedDir, Path.GetDirectoryName(p2));
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}
