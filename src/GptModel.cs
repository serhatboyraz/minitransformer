namespace MiniTransformer;

public sealed record GptConfig(int VocabSize, int BlockSize, int EmbedSize, int Heads, int Layers);

/// <summary>A decoder-only (GPT-style) transformer operating on one sequence at a time.</summary>
public sealed class GptModel : IModule
{
    private readonly GptConfig _config;
    private readonly Tensor _tokenEmbedding;
    private readonly Tensor _positionEmbedding;
    private readonly TransformerBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _head;

    public GptModel(GptConfig config, Random rng)
    {
        _config = config;
        _tokenEmbedding = new Tensor(config.VocabSize, config.EmbedSize);
        _positionEmbedding = new Tensor(config.BlockSize, config.EmbedSize);

        for (int i = 0; i < _tokenEmbedding.Length; i++)
        {
            _tokenEmbedding.Data[i] = (float)(Linear.NextGaussian(rng) * 0.02);
        }

        for (int i = 0; i < _positionEmbedding.Length; i++)
        {
            _positionEmbedding.Data[i] = (float)(Linear.NextGaussian(rng) * 0.02);
        }

        _blocks = new TransformerBlock[config.Layers];
        for (int i = 0; i < config.Layers; i++)
        {
            _blocks[i] = new TransformerBlock(config.EmbedSize, config.Heads, rng);
        }

        _finalNorm = new LayerNorm(config.EmbedSize);
        _head = new Linear(config.EmbedSize, config.VocabSize, rng);
    }

    public GptConfig Config => _config;

    /// <summary>Runs a whole batch as a single [batch*seq x embed] matrix. Returns logits of shape [batch*seq x VocabSize].</summary>
    public Tensor Forward(int[][] sequences)
    {
        int batch = sequences.Length;
        int seqLen = sequences[0].Length;

        if (seqLen > _config.BlockSize)
        {
            throw new ArgumentException($"Sequence of {seqLen} exceeds block size {_config.BlockSize}.");
        }

        var tokens = new int[batch * seqLen];
        var positions = new int[batch * seqLen];

        for (int b = 0; b < batch; b++)
        {
            if (sequences[b].Length != seqLen)
            {
                throw new ArgumentException("Batch içindeki tüm diziler aynı uzunlukta olmalı.");
            }

            Array.Copy(sequences[b], 0, tokens, b * seqLen, seqLen);
            for (int t = 0; t < seqLen; t++)
            {
                positions[(b * seqLen) + t] = t;
            }
        }

        var x = Ops.Add(Ops.Gather(_tokenEmbedding, tokens), Ops.Gather(_positionEmbedding, positions));

        foreach (var block in _blocks)
        {
            x = block.Forward(x, batch, seqLen);
        }

        return _head.Forward(_finalNorm.Forward(x));
    }

    public Tensor Loss(int[][] inputs, int[][] targets)
    {
        var flatTargets = new int[targets.Length * targets[0].Length];
        for (int b = 0; b < targets.Length; b++)
        {
            Array.Copy(targets[b], 0, flatTargets, b * targets[0].Length, targets[0].Length);
        }

        return Ops.CrossEntropy(Forward(inputs), flatTargets);
    }

    public IEnumerable<Tensor> Parameters()
    {
        yield return _tokenEmbedding;
        yield return _positionEmbedding;

        foreach (var p in _blocks.SelectMany(b => b.Parameters()).Concat(_finalNorm.Parameters()).Concat(_head.Parameters()))
        {
            yield return p;
        }
    }

    public string Generate(Tokenizer tokenizer, string prompt, int maxNewTokens, double temperature, int topK, Random rng)
    {
        var context = new List<int>(tokenizer.Encode(prompt));
        if (context.Count == 0)
        {
            context.Add(0);
        }

        var generated = new List<int>();

        for (int step = 0; step < maxNewTokens; step++)
        {
            var window = context.Skip(Math.Max(0, context.Count - _config.BlockSize)).ToArray();
            var logits = Forward([window]);
            int next = SampleLastRow(logits, temperature, topK, rng);
            context.Add(next);
            generated.Add(next);
        }

        return prompt + tokenizer.Decode(generated);
    }

    private static int SampleLastRow(Tensor logits, double temperature, int topK, Random rng)
    {
        int vocab = logits.Cols;
        int offset = (logits.Rows - 1) * vocab;

        var scores = new double[vocab];
        for (int i = 0; i < vocab; i++)
        {
            scores[i] = logits.Data[offset + i] / Math.Max(temperature, 1e-6);
        }

        if (topK > 0 && topK < vocab)
        {
            double threshold = scores.OrderByDescending(s => s).ElementAt(topK - 1);
            for (int i = 0; i < vocab; i++)
            {
                if (scores[i] < threshold)
                {
                    scores[i] = double.NegativeInfinity;
                }
            }
        }

        double max = scores.Max();
        double sum = 0.0;
        for (int i = 0; i < vocab; i++)
        {
            scores[i] = double.IsNegativeInfinity(scores[i]) ? 0.0 : Math.Exp(scores[i] - max);
            sum += scores[i];
        }

        double pick = rng.NextDouble() * sum;
        for (int i = 0; i < vocab; i++)
        {
            pick -= scores[i];
            if (pick <= 0.0)
            {
                return i;
            }
        }

        return vocab - 1;
    }
}
