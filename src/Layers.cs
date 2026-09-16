namespace MiniTransformer;

public interface IModule
{
    IEnumerable<Tensor> Parameters();
}

public sealed class Linear : IModule
{
    public readonly Tensor Weight;
    public readonly Tensor Bias;

    public Linear(int inFeatures, int outFeatures, Random rng, double std = 0.02)
    {
        Weight = new Tensor(inFeatures, outFeatures);
        Bias = new Tensor(1, outFeatures);
        for (int i = 0; i < Weight.Length; i++)
        {
            Weight.Data[i] = (float)(NextGaussian(rng) * std);
        }
    }

    public Tensor Forward(Tensor x) => Ops.AddRow(Ops.MatMul(x, Weight), Bias);

    public IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
        yield return Bias;
    }

    public static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

public sealed class LayerNorm : IModule
{
    public readonly Tensor Gain;
    public readonly Tensor Bias;

    public LayerNorm(int features)
    {
        Gain = new Tensor(1, features);
        Bias = new Tensor(1, features);
        Array.Fill(Gain.Data, 1.0f);
    }

    public Tensor Forward(Tensor x) => Ops.LayerNorm(x, Gain, Bias);

    public IEnumerable<Tensor> Parameters()
    {
        yield return Gain;
        yield return Bias;
    }
}

public sealed class MultiHeadSelfAttention : IModule
{
    private readonly int _heads;
    private readonly int _headSize;
    private readonly Linear _query;
    private readonly Linear _key;
    private readonly Linear _value;
    private readonly Linear _projection;

    public MultiHeadSelfAttention(int embedSize, int heads, Random rng)
    {
        _heads = heads;
        _headSize = embedSize / heads;
        _query = new Linear(embedSize, embedSize, rng);
        _key = new Linear(embedSize, embedSize, rng);
        _value = new Linear(embedSize, embedSize, rng);
        _projection = new Linear(embedSize, embedSize, rng);
    }

    /// <summary>x is [batchCount*seqLen, embed]; attention stays inside each sequence.</summary>
    public Tensor Forward(Tensor x, int batchCount, int seqLen)
    {
        var q = _query.Forward(x);
        var k = _key.Forward(x);
        var v = _value.Forward(x);

        var sequences = new Tensor[batchCount];
        float scale = (float)(1.0 / Math.Sqrt(_headSize));

        for (int b = 0; b < batchCount; b++)
        {
            int rowStart = b * seqLen;
            var headOutputs = new Tensor[_heads];

            for (int h = 0; h < _heads; h++)
            {
                int offset = h * _headSize;
                var qh = Ops.Slice(q, rowStart, seqLen, offset, _headSize);
                var kh = Ops.Slice(k, rowStart, seqLen, offset, _headSize);
                var vh = Ops.Slice(v, rowStart, seqLen, offset, _headSize);

                var scores = Ops.Scale(Ops.MatMul(qh, Ops.Transpose(kh)), scale);
                var weights = Ops.SoftmaxRows(Ops.CausalMask(scores));
                headOutputs[h] = Ops.MatMul(weights, vh);
            }

            sequences[b] = Ops.ConcatCols(headOutputs);
        }

        return _projection.Forward(Ops.ConcatRows(sequences));
    }

    public IEnumerable<Tensor> Parameters() =>
        _query.Parameters()
            .Concat(_key.Parameters())
            .Concat(_value.Parameters())
            .Concat(_projection.Parameters());
}

public sealed class FeedForward : IModule
{
    private readonly Linear _up;
    private readonly Linear _down;

    public FeedForward(int embedSize, Random rng)
    {
        _up = new Linear(embedSize, 4 * embedSize, rng);
        _down = new Linear(4 * embedSize, embedSize, rng);
    }

    public Tensor Forward(Tensor x) => _down.Forward(Ops.Gelu(_up.Forward(x)));

    public IEnumerable<Tensor> Parameters() => _up.Parameters().Concat(_down.Parameters());
}

/// <summary>Pre-norm transformer block: x + attn(ln(x)), then x + ffn(ln(x)).</summary>
public sealed class TransformerBlock : IModule
{
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly MultiHeadSelfAttention _attention;
    private readonly FeedForward _feedForward;

    public TransformerBlock(int embedSize, int heads, Random rng)
    {
        _norm1 = new LayerNorm(embedSize);
        _norm2 = new LayerNorm(embedSize);
        _attention = new MultiHeadSelfAttention(embedSize, heads, rng);
        _feedForward = new FeedForward(embedSize, rng);
    }

    public Tensor Forward(Tensor x, int batchCount, int seqLen)
    {
        x = Ops.Add(x, _attention.Forward(_norm1.Forward(x), batchCount, seqLen));
        return Ops.Add(x, _feedForward.Forward(_norm2.Forward(x)));
    }

    public IEnumerable<Tensor> Parameters() =>
        _norm1.Parameters()
            .Concat(_norm2.Parameters())
            .Concat(_attention.Parameters())
            .Concat(_feedForward.Parameters());
}
