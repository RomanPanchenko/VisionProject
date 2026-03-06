namespace Vision.NeuralEngine.Random;

public interface IRandomSource
{
    ulong NextUInt64();

    float NextSingle();
}
