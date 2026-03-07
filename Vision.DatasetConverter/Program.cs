using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using Vision.ImagePreprocessing;

static int Run(string[] args)
{
    try
    {
        var options = ConverterOptions.Parse(args);
        if (options.ShowHelp)
        {
            ConverterOptions.PrintHelp();
            return 0;
        }

        var repoRoot = options.RepoRoot ?? RepoRootLocator.FindRepoRootOrNull(AppContext.BaseDirectory);
        if (string.IsNullOrWhiteSpace(repoRoot))
        {
            Console.Error.WriteLine("Не удалось определить корень репозитория. Укажите --repo-root.");
            return 2;
        }

        var datasetsDir = Path.Combine(repoRoot, "datasets");
        var jobs = BuildJobs(datasetsDir, options);

        var pre = new Letter28x28PreprocessOptions(
            InkThreshold: options.InkThreshold,
            TargetSize: 28,
            InnerSize: options.InnerSize,
            BboxPadding: options.BboxPadding,
            AutoInvertByMean: !options.DisableAutoInvert);

        var pngEncoder = new PngEncoder
        {
            // Для L8 палитра не нужна; оставляем дефолт.
            CompressionLevel = PngCompressionLevel.Level6
        };

        foreach (var job in jobs)
        {
            ConvertFolder(job.SourceRoot, job.TargetRoot, pre, pngEncoder, options);
        }

        return 0;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine(ex.ToString());
        return 1;
    }
}

return Run(args);

static List<(string SourceRoot, string TargetRoot)> BuildJobs(string datasetsDir, ConverterOptions options)
{
    var jobs = new List<(string, string)>(capacity: 2);

    if (options.Mode is ConvertMode.Latin or ConvertMode.Both)
        jobs.Add((Path.Combine(datasetsDir, "Latin"), Path.Combine(datasetsDir, "mnist-latin")));

    if (options.Mode is ConvertMode.Cyrillic or ConvertMode.Both)
        jobs.Add((Path.Combine(datasetsDir, "Cyrillic"), Path.Combine(datasetsDir, "mnist-cyrillic")));

    return jobs;
}

static void ConvertFolder(
    string sourceRoot,
    string targetRoot,
    Letter28x28PreprocessOptions preprocessOptions,
    PngEncoder encoder,
    ConverterOptions options)
{
    if (!Directory.Exists(sourceRoot))
        throw new DirectoryNotFoundException($"Исходная папка не найдена: '{sourceRoot}'");

    Directory.CreateDirectory(targetRoot);

    Console.WriteLine($"Источник: {sourceRoot}");
    Console.WriteLine($"Назначение: {targetRoot}");

    var total = 0;
    var written = 0;
    var skipped = 0;
    var failed = 0;

    foreach (var classDir in Directory.EnumerateDirectories(sourceRoot))
    {
        var className = Path.GetFileName(classDir);
        if (string.IsNullOrWhiteSpace(className)) continue;

        if (options.Classes.Length > 0 && !options.Classes.Contains(className, StringComparer.OrdinalIgnoreCase))
            continue;

        var outClassDir = Path.Combine(targetRoot, className);
        if (!options.DryRun)
            Directory.CreateDirectory(outClassDir);

        var classWritten = 0;

        foreach (var file in Directory.EnumerateFiles(classDir, "*.*", SearchOption.TopDirectoryOnly))
        {
            var ext = Path.GetExtension(file);
            if (!ext.Equals(".png", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".jpg", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".jpeg", StringComparison.OrdinalIgnoreCase) &&
                !ext.Equals(".bmp", StringComparison.OrdinalIgnoreCase))
                continue;

            total++;
            var outPath = Path.Combine(outClassDir, Path.GetFileName(file));

            if (File.Exists(outPath) && !options.Overwrite)
            {
                skipped++;
                continue;
            }

            if (options.DryRun)
            {
                written++;
                classWritten++;
                if (options.MaxPerClass.HasValue && classWritten >= options.MaxPerClass.Value)
                    break;
                continue;
            }

            try
            {
                using var img28 = Letter28x28Preprocessor.LoadAndPreprocessTo28x28(file, preprocessOptions);
                img28.Save(outPath, encoder);
                written++;
                classWritten++;
                if (options.MaxPerClass.HasValue && classWritten >= options.MaxPerClass.Value)
                    break;
            }
            catch
            {
                failed++;
            }
        }
    }

    Console.WriteLine(options.DryRun
        ? $"DRY-RUN: обработано={total}, было бы записано={written}, пропущено={skipped}, ошибок={failed}"
        : $"Готово: обработано={total}, записано={written}, пропущено={skipped}, ошибок={failed}");
}

enum ConvertMode
{
    Both,
    Latin,
    Cyrillic
}

sealed record ConverterOptions
{
    public bool ShowHelp { get; init; }
    public string? RepoRoot { get; init; }
    public ConvertMode Mode { get; init; } = ConvertMode.Both;
    public bool DryRun { get; init; }
    public bool Overwrite { get; init; }
    public string[] Classes { get; init; } = [];
    public int? MaxPerClass { get; init; }
    public byte InkThreshold { get; init; } = 24;
    public int InnerSize { get; init; } = 20;
    public int BboxPadding { get; init; } = 2;
    public bool DisableAutoInvert { get; init; }

    public static ConverterOptions Parse(string[] args)
    {
        var opt = new ConverterOptions();
        for (var i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
                case "-h" or "--help":
                    opt = opt with { ShowHelp = true };
                    break;

                case "--repo-root":
                    opt = opt with { RepoRoot = ReadValue(args, ref i, "--repo-root") };
                    break;

                case "--mode":
                {
                    var v = ReadValue(args, ref i, "--mode");
                    opt = opt with
                    {
                        Mode = v.ToLowerInvariant() switch
                        {
                            "both" => ConvertMode.Both,
                            "latin" => ConvertMode.Latin,
                            "cyrillic" => ConvertMode.Cyrillic,
                            _ => throw new ArgumentException($"Неизвестный режим '{v}'. Ожидалось: both|latin|cyrillic")
                        }
                    };
                    break;
                }

                case "--dry-run":
                    opt = opt with { DryRun = true };
                    break;

                case "--overwrite":
                    opt = opt with { Overwrite = true };
                    break;

                case "--class":
                {
                    var v = ReadValue(args, ref i, "--class");
                    opt = opt with { Classes = opt.Classes.Concat([v]).ToArray() };
                    break;
                }

                case "--max-per-class":
                    opt = opt with { MaxPerClass = int.Parse(ReadValue(args, ref i, "--max-per-class")) };
                    break;

                case "--threshold":
                    opt = opt with { InkThreshold = byte.Parse(ReadValue(args, ref i, "--threshold")) };
                    break;

                case "--inner":
                    opt = opt with { InnerSize = int.Parse(ReadValue(args, ref i, "--inner")) };
                    break;

                case "--padding":
                    opt = opt with { BboxPadding = int.Parse(ReadValue(args, ref i, "--padding")) };
                    break;

                case "--no-auto-invert":
                    opt = opt with { DisableAutoInvert = true };
                    break;

                default:
                    throw new ArgumentException($"Неизвестный аргумент: '{a}'. Используйте --help.");
            }
        }

        return opt;
    }

    private static string ReadValue(string[] args, ref int i, string key)
    {
        if (i + 1 >= args.Length)
            throw new ArgumentException($"Ожидалось значение после {key}.");
        i++;
        return args[i];
    }

    public static void PrintHelp()
    {
        Console.WriteLine("Конвертация датасета букв в 28x28 (MNIST-стиль).\n");
        Console.WriteLine("Использование:");
        Console.WriteLine("  dotnet run --project Vision.DatasetConverter -- [опции]\n");
        Console.WriteLine("Опции:");
        Console.WriteLine("  --repo-root <PATH>        Корень репозитория (если авто-детект не сработал)");
        Console.WriteLine("  --mode <both|latin|cyrillic>  Что конвертировать (по умолчанию both)");
        Console.WriteLine("  --dry-run                 Ничего не записывать, только посчитать");
        Console.WriteLine("  --overwrite               Перезаписывать существующие файлы");
        Console.WriteLine("  --class <NAME>            Конвертировать только указанный класс (можно повторять)");
        Console.WriteLine("  --max-per-class <N>       Ограничить число файлов на класс (первые N в папке)");
        Console.WriteLine("  --threshold <0..255>      Порог 'чернил' (по умолчанию 24)");
        Console.WriteLine("  --inner <N>               Максимальный размер символа внутри 28x28 (по умолчанию 20)");
        Console.WriteLine("  --padding <N>             Паддинг bbox перед ресайзом (по умолчанию 2)\n");
        Console.WriteLine("  --no-auto-invert          Отключить авто-инверсию (если фон не распознаётся корректно)");
    }
}

static class RepoRootLocator
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
