namespace Vision.NeuralEngine.Training;

public readonly record struct TrainingMetrics(int Epoch, float Loss, float Accuracy);
