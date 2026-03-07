namespace Vision.App.WinForms;

public static class RepoRootLocator
{
    public static string? FindRepoRootOrNull(string startDir)
    {
        if (string.IsNullOrWhiteSpace(startDir)) return null;

        var current = new DirectoryInfo(startDir);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "VisionProject.sln")))
                return current.FullName;

            if (Directory.Exists(Path.Combine(current.FullName, "datasets")) &&
                Directory.Exists(Path.Combine(current.FullName, "models")))
                return current.FullName;

            current = current.Parent;
        }

        return null;
    }
}
