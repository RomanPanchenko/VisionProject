using System.Buffers.Binary;
using Vision.NeuralEngine.Core;
using Vision.NeuralEngine.Models;

namespace Vision.NeuralEngine.IO;

public static class ModelSerializer
{
    // Формат: [magic:4][version:int32][paramCount:int32][len0:int32..][floats...]
    // Числа пишем в little-endian (стандарт для .NET на Windows).
    private const uint Magic = 0x31444E56; // 'V''N''D''1' (Vision Neural Data v1)
    private const int Version = 1;

    public sealed record ModelFileSignature(int Version, int ParamCount, int[] Lengths);

    public static bool TryReadSignature(string path, out ModelFileSignature signature, out string error)
    {
        signature = null!;
        error = string.Empty;

        if (string.IsNullOrWhiteSpace(path))
        {
            error = "Path is empty.";
            return false;
        }
        if (!File.Exists(path))
        {
            error = $"Model file not found: '{path}'";
            return false;
        }

        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var br = new BinaryReader(fs);

            var magic = br.ReadUInt32();
            if (magic != Magic)
            {
                error = $"Invalid model file magic: 0x{magic:X8}.";
                return false;
            }

            var version = br.ReadInt32();
            if (version != Version)
            {
                error = $"Unsupported model file version: {version} (expected {Version}).";
                return false;
            }

            var paramCount = br.ReadInt32();
            if (paramCount < 0)
            {
                error = "Invalid parameter count in file.";
                return false;
            }

            var lengths = new int[paramCount];
            for (var i = 0; i < paramCount; i++)
            {
                var len = br.ReadInt32();
                if (len < 0)
                {
                    error = "Invalid parameter length in file.";
                    return false;
                }
                lengths[i] = len;
            }

            signature = new ModelFileSignature(version, paramCount, lengths);
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static void SaveParameters(SequentialModel model, string path)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Path is empty.", nameof(path));

        var parameters = model.Parameters;

        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        bw.Write(Magic);
        bw.Write(Version);
        bw.Write(parameters.Count);

        for (var i = 0; i < parameters.Count; i++)
        {
            bw.Write(parameters[i].Value.Length);
        }

        for (var p = 0; p < parameters.Count; p++)
        {
            WriteFloatArray(bw, parameters[p].Value);
        }
    }

    public static void LoadParameters(SequentialModel model, string path)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Path is empty.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Model file not found: '{path}'", path);

        var parameters = model.Parameters;

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs);

        var magic = br.ReadUInt32();
        if (magic != Magic)
        {
            throw new InvalidDataException($"Invalid model file magic: 0x{magic:X8}.");
        }

        var version = br.ReadInt32();
        if (version != Version)
        {
            throw new InvalidDataException($"Unsupported model file version: {version} (expected {Version}).");
        }

        var paramCount = br.ReadInt32();
        if (paramCount != parameters.Count)
        {
            throw new InvalidDataException($"Model parameter count mismatch: file={paramCount}, model={parameters.Count}.");
        }

        var lengths = new int[paramCount];
        for (var i = 0; i < paramCount; i++)
        {
            var len = br.ReadInt32();
            if (len < 0) throw new InvalidDataException("Invalid parameter length in file.");
            lengths[i] = len;
        }

        for (var p = 0; p < paramCount; p++)
        {
            var expectedLen = parameters[p].Value.Length;
            if (lengths[p] != expectedLen)
            {
                throw new InvalidDataException($"Parameter[{p}] length mismatch: file={lengths[p]}, model={expectedLen}.");
            }

            ReadFloatArray(br, parameters[p].Value);
        }
    }

    private static void WriteFloatArray(BinaryWriter bw, float[] data)
    {
        // Пишем float как 4 байта без лишних аллокаций.
        Span<byte> buf = stackalloc byte[4];
        for (var i = 0; i < data.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(buf, data[i]);
            bw.Write(buf);
        }
    }

    private static void ReadFloatArray(BinaryReader br, float[] target)
    {
        Span<byte> buf = stackalloc byte[4];
        for (var i = 0; i < target.Length; i++)
        {
            var read = br.Read(buf);
            if (read != 4) throw new EndOfStreamException("Unexpected end of model file.");
            target[i] = BinaryPrimitives.ReadSingleLittleEndian(buf);
        }
    }
}
