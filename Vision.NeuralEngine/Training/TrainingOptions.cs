namespace Vision.NeuralEngine.Training;

public sealed class TrainingOptions
{
    public int Epochs { get; init; } = 10;
    public bool Shuffle { get; init; } = true;
}
