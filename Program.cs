using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace MiniTransformer;

public static class Program
{
    private const int BlockSize = 64;
    private const int EmbedSize = 96;
    private const int Heads = 4;
    private const int Layers = 3;
    private const int BatchSize = 8;
    private const int TrainingSteps = 1500;
    private const double LearningRate = 3e-3;
    private const string DefaultPrompt = "toyota corolla 2021 ";

    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        string command = args.Length > 0 ? args[0].ToLowerInvariant() : "auto";
        string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "model.mtf");
        string[] rest = args.Length > 1 ? args[1..] : [];

        switch (command)
        {
            case "train":
                if (rest.Length > 0)
                {
                    checkpointPath = Path.GetFullPath(rest[0]);
                }

                Train(checkpointPath);
                return 0;

            case "generate":
                if (rest.Length > 0 && File.Exists(rest[0]))
                {
                    checkpointPath = Path.GetFullPath(rest[0]);
                    rest = rest[1..];
                }

                if (!File.Exists(checkpointPath))
                {
                    Console.Error.WriteLine($"Kayıtlı model yok: {checkpointPath}");
                    Console.Error.WriteLine("Önce `dotnet run -c Release -- train` çalıştırın.");
                    return 1;
                }

                GenerateFrom(checkpointPath, rest.Length > 0 ? string.Join(' ', rest) : DefaultPrompt);
                return 0;

            case "loss":
                string lossPath = rest.Length > 0
                    ? Path.GetFullPath(rest[0])
                    : Path.Combine(Directory.GetCurrentDirectory(), "loss.csv");

                if (!File.Exists(lossPath))
                {
                    Console.Error.WriteLine($"Loss kaydı yok: {lossPath}");
                    Console.Error.WriteLine("Önce `dotnet run -c Release -- train` çalıştırın.");
                    return 1;
                }

                ShowLoss(lossPath);
                return 0;

            default:
                if (File.Exists(checkpointPath))
                {
                    GenerateFrom(checkpointPath, args.Length > 0 ? string.Join(' ', args) : DefaultPrompt);
                }
                else
                {
                    Train(checkpointPath);
                }

                return 0;
        }
    }

    private static void GenerateFrom(string checkpointPath, string prompt)
    {
        var (model, tokenizer) = Checkpoint.Load(checkpointPath);
        Console.WriteLine($"model     : {checkpointPath} ({new FileInfo(checkpointPath).Length / 1024.0:F0} KB, vocab {tokenizer.VocabSize})");
        Console.WriteLine();
        Console.WriteLine(model.Generate(tokenizer, prompt, maxNewTokens: 400, temperature: 0.8, topK: 10, new Random()));
    }

    private static void ShowLoss(string csvPath)
    {
        var losses = new List<double>();
        foreach (string line in File.ReadLines(csvPath).Skip(1))
        {
            var fields = line.Split(',');
            if (fields.Length >= 2 && double.TryParse(fields[1], CultureInfo.InvariantCulture, out double value))
            {
                losses.Add(value);
            }
        }

        if (losses.Count == 0)
        {
            Console.Error.WriteLine($"{csvPath} içinde okunabilir loss değeri bulunamadı.");
            return;
        }

        Console.WriteLine($"kaynak: {csvPath} ({losses.Count} adım)");
        Console.WriteLine();
        Console.Write(LossChart.Render(losses));
        PrintLossSummary(losses);
    }

    private static void PrintLossSummary(IReadOnlyList<double> losses)
    {
        int tail = Math.Min(50, losses.Count);
        Console.WriteLine($"ilk: {losses[0]:F4}   en iyi: {losses.Min():F4}   son {tail} ortalama: {losses.TakeLast(tail).Average():F4}");
    }

    private static void Train(string checkpointPath)
    {
        var rng = new Random(1337);
        string corpus = LoadCorpus();
        var tokenizer = new Tokenizer(corpus);
        int[] data = tokenizer.Encode(corpus);

        var config = new GptConfig(tokenizer.VocabSize, BlockSize, EmbedSize, Heads, Layers);
        var model = new GptModel(config, rng);
        var optimizer = new AdamOptimizer(model.Parameters(), LearningRate);

        int parameterCount = model.Parameters().Sum(p => p.Length);
        Console.WriteLine($"corpus     : {data.Length} tokens, vocab {tokenizer.VocabSize}");
        Console.WriteLine($"model      : {Layers} layers, {Heads} heads, d_model {EmbedSize}, {parameterCount:N0} parameters");
        Console.WriteLine($"training   : {TrainingSteps} steps, batch {BatchSize}, block {BlockSize}");
        Console.WriteLine();

        var stopwatch = Stopwatch.StartNew();
        var lossHistory = new List<double>(TrainingSteps);
        var lrHistory = new List<double>(TrainingSteps);

        for (int step = 1; step <= TrainingSteps; step++)
        {
            optimizer.LearningRate = CosineSchedule(step);
            optimizer.ZeroGrad();

            var (inputs, targets) = SampleBatch(data, rng);
            var loss = model.Loss(inputs, targets);
            loss.Backward();

            optimizer.ClipGradients(1.0);
            optimizer.Step();

            lossHistory.Add(loss.Data[0]);
            lrHistory.Add(optimizer.LearningRate);

            if (step % 50 == 0 || step == 1)
            {
                Console.WriteLine($"step {step,5}/{TrainingSteps}  loss {loss.Data[0]:F4}  lr {optimizer.LearningRate:E2}  {stopwatch.Elapsed.TotalSeconds:F1}s");
            }
        }

        Console.WriteLine();
        Console.WriteLine("--- loss eğrisi ---");
        Console.Write(LossChart.Render(lossHistory));
        PrintLossSummary(lossHistory);
        Console.WriteLine($"csv : {SaveLossCsv(lossHistory, lrHistory)}");

        Checkpoint.Save(checkpointPath, model, tokenizer);
        Console.WriteLine($"model: {checkpointPath} ({new FileInfo(checkpointPath).Length / 1024.0:F0} KB)");

        Console.WriteLine();
        Console.WriteLine("--- örnek üretim ---");
        Console.WriteLine(model.Generate(tokenizer, DefaultPrompt, maxNewTokens: 400, temperature: 0.8, topK: 10, rng));
    }

    private static string SaveLossCsv(List<double> losses, List<double> learningRates)
    {
        string path = Path.Combine(Directory.GetCurrentDirectory(), "loss.csv");
        var sb = new StringBuilder("step,loss,lr\n");
        for (int i = 0; i < losses.Count; i++)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{i + 1},{losses[i]:F6},{learningRates[i]:G6}\n");
        }

        File.WriteAllText(path, sb.ToString());
        return path;
    }

    private static double CosineSchedule(int step)
    {
        const int Warmup = 100;
        if (step < Warmup)
        {
            return LearningRate * step / Warmup;
        }

        double progress = (double)(step - Warmup) / Math.Max(1, TrainingSteps - Warmup);
        return 0.1 * LearningRate + (0.9 * LearningRate * 0.5 * (1.0 + Math.Cos(Math.PI * progress)));
    }

    private static (int[][] Inputs, int[][] Targets) SampleBatch(int[] data, Random rng)
    {
        var inputs = new int[BatchSize][];
        var targets = new int[BatchSize][];

        for (int b = 0; b < BatchSize; b++)
        {
            int start = rng.Next(data.Length - BlockSize - 1);
            inputs[b] = new int[BlockSize];
            targets[b] = new int[BlockSize];
            Array.Copy(data, start, inputs[b], 0, BlockSize);
            Array.Copy(data, start + 1, targets[b], 0, BlockSize);
        }

        return (inputs, targets);
    }

    private static string LoadCorpus()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "data", "input.txt");
        if (File.Exists(path))
        {
            return File.ReadAllText(path);
        }

        throw new FileNotFoundException($"Training corpus not found at {path}.");
    }
}
