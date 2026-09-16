namespace MiniTransformer;

/// <summary>Adam optimizer with decoupled gradient clipping.</summary>
public sealed class AdamOptimizer
{
    private readonly Tensor[] _parameters;
    private readonly double[][] _m;
    private readonly double[][] _v;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _eps;
    private int _step;

    public AdamOptimizer(IEnumerable<Tensor> parameters, double learningRate, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
    {
        _parameters = parameters.ToArray();
        _m = _parameters.Select(p => new double[p.Length]).ToArray();
        _v = _parameters.Select(p => new double[p.Length]).ToArray();
        LearningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
    }

    public double LearningRate { get; set; }

    public void ZeroGrad()
    {
        foreach (var p in _parameters)
        {
            p.ZeroGrad();
        }
    }

    public void ClipGradients(double maxNorm)
    {
        double total = 0.0;
        foreach (var p in _parameters)
        {
            foreach (float g in p.Grad)
            {
                total += (double)g * g;
            }
        }

        double norm = Math.Sqrt(total);
        if (norm <= maxNorm || norm == 0.0)
        {
            return;
        }

        float scale = (float)(maxNorm / norm);
        foreach (var p in _parameters)
        {
            for (int i = 0; i < p.Grad.Length; i++)
            {
                p.Grad[i] *= scale;
            }
        }
    }

    public void Step()
    {
        _step++;
        double bc1 = 1.0 - Math.Pow(_beta1, _step);
        double bc2 = 1.0 - Math.Pow(_beta2, _step);

        for (int pi = 0; pi < _parameters.Length; pi++)
        {
            var p = _parameters[pi];
            var m = _m[pi];
            var v = _v[pi];

            for (int i = 0; i < p.Length; i++)
            {
                double g = p.Grad[i];
                m[i] = (_beta1 * m[i]) + ((1.0 - _beta1) * g);
                v[i] = (_beta2 * v[i]) + ((1.0 - _beta2) * g * g);
                p.Data[i] -= (float)(LearningRate * (m[i] / bc1) / (Math.Sqrt(v[i] / bc2) + _eps));
            }
        }
    }
}
