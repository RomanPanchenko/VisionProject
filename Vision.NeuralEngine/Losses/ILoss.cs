namespace Vision.NeuralEngine.Losses;

public interface ILoss
{
    float Forward(float[] logits, int targetClass);

    float[] Backward();
}
