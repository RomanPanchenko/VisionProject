using System.Drawing.Imaging;

namespace Vision.App.WinForms;

public static class LabeledSampleWriter
{
    public static string Save28x28Png(Bitmap image28x28, int label, string datasetRootDir)
    {
        if (label is < 0 or > 9) throw new ArgumentOutOfRangeException(nameof(label));
        if (string.IsNullOrWhiteSpace(datasetRootDir))
            throw new ArgumentException("Dataset root directory is empty.", nameof(datasetRootDir));

        var classDir = Path.Combine(datasetRootDir, label.ToString());
        Directory.CreateDirectory(classDir);

        var fileName = $"{Guid.NewGuid():N}.png";
        var path = Path.Combine(classDir, fileName);

        image28x28.Save(path, ImageFormat.Png);
        return path;
    }
}
