namespace Vision.NeuralEngine.Random;

/// <summary>
/// Простой детерминированный генератор для воспроизводимых инициализаций.
/// </summary>
public sealed class SplitMix64Random : IRandomSource
{
    private ulong _state;

    public SplitMix64Random(ulong seed)
    {
        _state = seed;
    }

    public ulong NextUInt64()
    {
        var z = (_state += 0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    public float NextSingle()
    {
        // 24 бита мантиссы для float
        var v = (uint)(NextUInt64() >> 40);
        return v / (float)(1 << 24);
    }
}
