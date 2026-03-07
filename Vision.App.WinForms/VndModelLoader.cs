using Vision.NeuralEngine.IO;
using Vision.NeuralEngine.Models;

namespace Vision.App.WinForms;

internal static class VndModelLoader
{
    public static bool TryLoadCompatible(
        string path,
        int classCount,
        out SequentialModel model,
        out string? variant,
        out Exception? error)
    {
        model = null!;
        variant = null;
        error = null;

        if (!ModelSerializer.TryReadSignature(path, out var sig, out var sigError))
        {
            error = new InvalidDataException(sigError);
            return false;
        }

        // Порядок важен: сначала текущая архитектура, затем предыдущие варианты,
        // затем совсем старый legacy (Conv8->ReLU->Dense).
        var candidates = new (string Name, Func<int, SequentialModel> Build)[]
        {
            ("conv32_64_dense128", MnistModelVariants.BuildConv32_64_Dense128),
            ("conv32_64_128_dense128", MnistModelVariants.BuildConv32_64_128_Dense128),
            ("conv32_64_128_dense128_dense128", MnistModelVariants.BuildConv32_64_128_Dense128_Dense128),
            ("legacy_conv8", MnistModelVariants.BuildLegacyConv8)
        };

        foreach (var c in candidates)
        {
            SequentialModel m;
            try
            {
                m = c.Build(classCount);
            }
            catch (Exception ex)
            {
                error = ex;
                continue;
            }

            if (!IsSignatureCompatible(sig, m))
                continue;

            try
            {
                ModelSerializer.LoadParameters(m, path);
                model = m;
                variant = c.Name;
                return true;
            }
            catch (Exception ex)
            {
                error = ex;
            }
        }

        return false;
    }

    private static bool IsSignatureCompatible(ModelSerializer.ModelFileSignature sig, SequentialModel model)
    {
        var parameters = model.Parameters;
        if (sig.ParamCount != parameters.Count) return false;
        if (sig.Lengths.Length != parameters.Count) return false;

        for (var i = 0; i < parameters.Count; i++)
        {
            if (sig.Lengths[i] != parameters[i].Value.Length)
                return false;
        }

        return true;
    }
}
