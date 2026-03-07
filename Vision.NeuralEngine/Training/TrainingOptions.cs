namespace Vision.NeuralEngine.Training;

public sealed class TrainingOptions
{
    public int Epochs { get; init; } = 10;
    public bool Shuffle { get; init; } = true;

    /// <summary>
    /// Whether to print training progress to console.
    /// </summary>
    public bool ReportProgressToConsole { get; init; } = false;

    /// <summary>
    /// How often to update progress within an epoch, in percent points (1..100).
    /// </summary>
    public int ProgressPercentStep { get; init; } = 5;

    /// <summary>
    /// Minimum time between progress updates within an epoch.
    /// Ensures progress is printed regularly even when <see cref="ProgressPercentStep"/> changes rarely.
    /// </summary>
    public TimeSpan ProgressMinReportInterval { get; init; } = TimeSpan.FromSeconds(10);
}
