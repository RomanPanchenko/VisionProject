namespace Vision.NeuralEngine.Core;

public sealed class Parameter
{
    public Parameter(float[] value)
    {
        Value = value;
        Grad = new float[value.Length];
    }

    public float[] Value { get; }

    public float[] Grad { get; }

    public void ZeroGrad()
    {
        Array.Clear(Grad);
    }
}
