using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace MiniTransformer;

/// <summary>Model ağırlıklarını kendi formatımızda saklar. Uzantı .txt ise okunabilir metin, değilse ikili yazar.</summary>
public static class Checkpoint
{
    private const string Magic = "MTRF";
    private const string TextHeader = "minitransformer-text";
    private const int Version = 1;

    public static void Save(string path, GptModel model, Tokenizer tokenizer)
    {
        if (IsText(path))
        {
            SaveText(path, model, tokenizer);
        }
        else
        {
            SaveBinary(path, model, tokenizer);
        }
    }

    public static (GptModel Model, Tokenizer Tokenizer) Load(string path) =>
        IsText(path) ? LoadText(path) : LoadBinary(path);

    private static bool IsText(string path) =>
        Path.GetExtension(path).Equals(".txt", StringComparison.OrdinalIgnoreCase);

    private static void SaveBinary(string path, GptModel model, Tokenizer tokenizer)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);

        writer.Write(Magic);
        writer.Write(Version);

        var config = model.Config;
        writer.Write(config.VocabSize);
        writer.Write(config.BlockSize);
        writer.Write(config.EmbedSize);
        writer.Write(config.Heads);
        writer.Write(config.Layers);
        writer.Write(tokenizer.Vocabulary);

        var parameters = model.Parameters().ToArray();
        writer.Write(parameters.Length);

        foreach (var p in parameters)
        {
            writer.Write(p.Rows);
            writer.Write(p.Cols);
            writer.Write(MemoryMarshal.AsBytes<float>(p.Data));
        }
    }

    private static (GptModel Model, Tokenizer Tokenizer) LoadBinary(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8);

        if (reader.ReadString() != Magic)
        {
            throw new InvalidDataException($"{path} bir MiniTransformer checkpoint dosyası değil.");
        }

        int version = reader.ReadInt32();
        if (version != Version)
        {
            throw new InvalidDataException($"Desteklenmeyen checkpoint sürümü: {version}.");
        }

        var config = new GptConfig(
            reader.ReadInt32(),
            reader.ReadInt32(),
            reader.ReadInt32(),
            reader.ReadInt32(),
            reader.ReadInt32());

        var tokenizer = new Tokenizer(reader.ReadString());
        var (model, parameters) = BuildTarget(config, reader.ReadInt32());

        foreach (var p in parameters)
        {
            ExpectShape(p, reader.ReadInt32(), reader.ReadInt32());
            stream.ReadExactly(MemoryMarshal.AsBytes(p.Data.AsSpan()));
        }

        return (model, tokenizer);
    }

    private static void SaveText(string path, GptModel model, Tokenizer tokenizer)
    {
        var config = model.Config;
        var parameters = model.Parameters().ToArray();

        using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
        writer.WriteLine($"{TextHeader} {Version}");
        writer.WriteLine($"config {config.VocabSize} {config.BlockSize} {config.EmbedSize} {config.Heads} {config.Layers}");

        // Sözlükte satır sonu gibi karakterler olabildiği için kod noktası olarak yazılır.
        writer.WriteLine("vocab " + string.Join(' ', tokenizer.Vocabulary.Select(c => (int)c)));
        writer.WriteLine($"params {parameters.Length}");

        var line = new StringBuilder();
        foreach (var p in parameters)
        {
            writer.WriteLine($"tensor {p.Rows} {p.Cols}");
            for (int r = 0; r < p.Rows; r++)
            {
                line.Clear();
                for (int c = 0; c < p.Cols; c++)
                {
                    if (c > 0)
                    {
                        line.Append(' ');
                    }

                    line.Append(p.Data[(r * p.Cols) + c].ToString("G9", CultureInfo.InvariantCulture));
                }

                writer.WriteLine(line);
            }
        }
    }

    private static (GptModel Model, Tokenizer Tokenizer) LoadText(string path)
    {
        using var reader = new StreamReader(path, Encoding.UTF8);

        var header = NextTokens(reader);
        if (header[0] != TextHeader || header[1] != Version.ToString(CultureInfo.InvariantCulture))
        {
            throw new InvalidDataException($"{path} tanınmayan bir metin checkpoint başlığına sahip.");
        }

        var configLine = NextTokens(reader);
        var config = new GptConfig(
            int.Parse(configLine[1], CultureInfo.InvariantCulture),
            int.Parse(configLine[2], CultureInfo.InvariantCulture),
            int.Parse(configLine[3], CultureInfo.InvariantCulture),
            int.Parse(configLine[4], CultureInfo.InvariantCulture),
            int.Parse(configLine[5], CultureInfo.InvariantCulture));

        var vocabLine = NextTokens(reader);
        var vocab = new string(vocabLine.Skip(1).Select(t => (char)int.Parse(t, CultureInfo.InvariantCulture)).ToArray());
        var tokenizer = new Tokenizer(vocab);

        var paramLine = NextTokens(reader);
        var (model, parameters) = BuildTarget(config, int.Parse(paramLine[1], CultureInfo.InvariantCulture));

        foreach (var p in parameters)
        {
            var shape = NextTokens(reader);
            ExpectShape(p, int.Parse(shape[1], CultureInfo.InvariantCulture), int.Parse(shape[2], CultureInfo.InvariantCulture));

            for (int r = 0; r < p.Rows; r++)
            {
                var values = NextTokens(reader);
                for (int c = 0; c < p.Cols; c++)
                {
                    p.Data[(r * p.Cols) + c] = float.Parse(values[c], CultureInfo.InvariantCulture);
                }
            }
        }

        return (model, tokenizer);
    }

    private static string[] NextTokens(StreamReader reader)
    {
        string? line = reader.ReadLine() ?? throw new InvalidDataException("Checkpoint dosyası beklenenden erken bitti.");
        return line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static (GptModel Model, Tensor[] Parameters) BuildTarget(GptConfig config, int expectedCount)
    {
        // Ağırlıklar dosyadan gelecek, bu yüzden başlangıç değerleri önemsiz.
        var model = new GptModel(config, new Random(0));
        var parameters = model.Parameters().ToArray();

        if (expectedCount != parameters.Length)
        {
            throw new InvalidDataException($"Parametre sayısı uyuşmuyor: dosyada {expectedCount}, modelde {parameters.Length}.");
        }

        return (model, parameters);
    }

    private static void ExpectShape(Tensor p, int rows, int cols)
    {
        if (rows != p.Rows || cols != p.Cols)
        {
            throw new InvalidDataException($"Tensör boyutu uyuşmuyor: dosyada [{rows}x{cols}], modelde [{p.Rows}x{p.Cols}].");
        }
    }
}
