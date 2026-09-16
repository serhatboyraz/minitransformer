using System.Numerics;

namespace MiniTransformer;

/// <summary>Differentiable tensor operations. Every op records a closure that accumulates gradients into its inputs.</summary>
public static class Ops
{
    private const float NegativeInfinityMask = -1e9f;

    /// <summary>dst += src * scalar, vectorized.</summary>
    private static void Axpy(float[] dst, int dstOffset, float[] src, int srcOffset, float scalar, int count)
    {
        int width = Vector<float>.Count;
        var vScalar = new Vector<float>(scalar);
        int j = 0;

        for (; j <= count - width; j += width)
        {
            var acc = new Vector<float>(dst, dstOffset + j) + (new Vector<float>(src, srcOffset + j) * vScalar);
            acc.CopyTo(dst, dstOffset + j);
        }

        for (; j < count; j++)
        {
            dst[dstOffset + j] += src[srcOffset + j] * scalar;
        }
    }

    private static float Dot(float[] x, int xOffset, float[] y, int yOffset, int count)
    {
        int width = Vector<float>.Count;
        var acc = Vector<float>.Zero;
        int j = 0;

        for (; j <= count - width; j += width)
        {
            acc += new Vector<float>(x, xOffset + j) * new Vector<float>(y, yOffset + j);
        }

        float sum = Vector.Sum(acc);
        for (; j < count; j++)
        {
            sum += x[xOffset + j] * y[yOffset + j];
        }

        return sum;
    }

    public static Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"Shape mismatch: [{a.Rows}x{a.Cols}] * [{b.Rows}x{b.Cols}]");
        }

        var outTensor = new Tensor(a.Rows, b.Cols) { Parents = [a, b] };
        int m = a.Rows, k = a.Cols, n = b.Cols;
        float[] ad = a.Data, bd = b.Data, od = outTensor.Data;

        // Her iş parçacığı ayrı bir çıktı/gradyan satırına yazar, bu yüzden kilit gerekmez ve sonuç deterministiktir.
        bool parallel = (long)m * k * n >= 64_000;

        void ForwardRow(int i)
        {
            int oRow = i * n;
            int aRow = i * k;
            for (int p = 0; p < k; p++)
            {
                float av = ad[aRow + p];
                if (av != 0f)
                {
                    Axpy(od, oRow, bd, p * n, av, n);
                }
            }
        }

        if (parallel)
        {
            Parallel.For(0, m, ForwardRow);
        }
        else
        {
            for (int i = 0; i < m; i++)
            {
                ForwardRow(i);
            }
        }

        outTensor.BackwardFn = () =>
        {
            float[] og = outTensor.Grad;

            // dA = dC * B^T : i satırları birbirinden bağımsız
            void GradARow(int i)
            {
                int oRow = i * n;
                int aRow = i * k;
                for (int p = 0; p < k; p++)
                {
                    a.Grad[aRow + p] += Dot(og, oRow, bd, p * n, n);
                }
            }

            // dB = A^T * dC : p satırları birbirinden bağımsız
            void GradBRow(int p)
            {
                int bRow = p * n;
                for (int i = 0; i < m; i++)
                {
                    float av = ad[i * k + p];
                    if (av != 0f)
                    {
                        Axpy(b.Grad, bRow, og, i * n, av, n);
                    }
                }
            }

            if (parallel)
            {
                Parallel.For(0, m, GradARow);
                Parallel.For(0, k, GradBRow);
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    GradARow(i);
                }

                for (int p = 0; p < k; p++)
                {
                    GradBRow(p);
                }
            }
        };

        return outTensor;
    }

    public static Tensor Add(Tensor a, Tensor b)
    {
        var outTensor = new Tensor(a.Rows, a.Cols) { Parents = [a, b] };
        for (int i = 0; i < outTensor.Length; i++)
        {
            outTensor.Data[i] = a.Data[i] + b.Data[i];
        }

        outTensor.BackwardFn = () =>
        {
            for (int i = 0; i < outTensor.Length; i++)
            {
                a.Grad[i] += outTensor.Grad[i];
                b.Grad[i] += outTensor.Grad[i];
            }
        };

        return outTensor;
    }

    /// <summary>Adds a 1 x Cols bias row to every row of <paramref name="x"/>.</summary>
    public static Tensor AddRow(Tensor x, Tensor row)
    {
        var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x, row] };
        for (int r = 0; r < x.Rows; r++)
        {
            for (int c = 0; c < x.Cols; c++)
            {
                outTensor.Data[r * x.Cols + c] = x.Data[r * x.Cols + c] + row.Data[c];
            }
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < x.Rows; r++)
            {
                for (int c = 0; c < x.Cols; c++)
                {
                    float g = outTensor.Grad[r * x.Cols + c];
                    x.Grad[r * x.Cols + c] += g;
                    row.Grad[c] += g;
                }
            }
        };

        return outTensor;
    }

    public static Tensor Scale(Tensor x, float factor)
    {
        var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        for (int i = 0; i < x.Length; i++)
        {
            outTensor.Data[i] = x.Data[i] * factor;
        }

        outTensor.BackwardFn = () =>
        {
            for (int i = 0; i < x.Length; i++)
            {
                x.Grad[i] += outTensor.Grad[i] * factor;
            }
        };

        return outTensor;
    }

    public static Tensor Transpose(Tensor x)
    {
        var outTensor = new Tensor(x.Cols, x.Rows) { Parents = [x] };
        for (int r = 0; r < x.Rows; r++)
        {
            for (int c = 0; c < x.Cols; c++)
            {
                outTensor.Data[c * x.Rows + r] = x.Data[r * x.Cols + c];
            }
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < x.Rows; r++)
            {
                for (int c = 0; c < x.Cols; c++)
                {
                    x.Grad[r * x.Cols + c] += outTensor.Grad[c * x.Rows + r];
                }
            }
        };

        return outTensor;
    }

    /// <summary>Extracts the [rowStart, rowStart+rowCount) x [colStart, colStart+colCount) block.</summary>
    public static Tensor Slice(Tensor x, int rowStart, int rowCount, int colStart, int colCount)
    {
        var outTensor = new Tensor(rowCount, colCount) { Parents = [x] };
        for (int r = 0; r < rowCount; r++)
        {
            Array.Copy(x.Data, ((rowStart + r) * x.Cols) + colStart, outTensor.Data, r * colCount, colCount);
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < rowCount; r++)
            {
                int src = r * colCount;
                int dst = ((rowStart + r) * x.Cols) + colStart;
                for (int c = 0; c < colCount; c++)
                {
                    x.Grad[dst + c] += outTensor.Grad[src + c];
                }
            }
        };

        return outTensor;
    }

    public static Tensor ConcatRows(Tensor[] parts)
    {
        if (parts.Length == 1)
        {
            return parts[0];
        }

        int cols = parts[0].Cols;
        int rows = parts.Sum(p => p.Rows);
        var outTensor = new Tensor(rows, cols) { Parents = parts };

        int offset = 0;
        foreach (var part in parts)
        {
            Array.Copy(part.Data, 0, outTensor.Data, offset, part.Length);
            offset += part.Length;
        }

        outTensor.BackwardFn = () =>
        {
            int off = 0;
            foreach (var part in parts)
            {
                for (int i = 0; i < part.Length; i++)
                {
                    part.Grad[i] += outTensor.Grad[off + i];
                }

                off += part.Length;
            }
        };

        return outTensor;
    }

    public static Tensor ConcatCols(Tensor[] parts)
    {
        int rows = parts[0].Rows;
        int cols = parts.Sum(p => p.Cols);
        var outTensor = new Tensor(rows, cols) { Parents = parts };

        int offset = 0;
        foreach (var part in parts)
        {
            for (int r = 0; r < rows; r++)
            {
                Array.Copy(part.Data, r * part.Cols, outTensor.Data, r * cols + offset, part.Cols);
            }

            offset += part.Cols;
        }

        outTensor.BackwardFn = () =>
        {
            int off = 0;
            foreach (var part in parts)
            {
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < part.Cols; c++)
                    {
                        part.Grad[r * part.Cols + c] += outTensor.Grad[r * cols + off + c];
                    }
                }

                off += part.Cols;
            }
        };

        return outTensor;
    }

    /// <summary>Blocks attention to future positions by pushing the upper triangle to -inf.</summary>
    public static Tensor CausalMask(Tensor scores)
    {
        var outTensor = new Tensor(scores.Rows, scores.Cols) { Parents = [scores] };
        for (int r = 0; r < scores.Rows; r++)
        {
            for (int c = 0; c < scores.Cols; c++)
            {
                outTensor.Data[r * scores.Cols + c] = c <= r ? scores.Data[r * scores.Cols + c] : NegativeInfinityMask;
            }
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < scores.Rows; r++)
            {
                for (int c = 0; c <= r && c < scores.Cols; c++)
                {
                    scores.Grad[r * scores.Cols + c] += outTensor.Grad[r * scores.Cols + c];
                }
            }
        };

        return outTensor;
    }

    public static Tensor SoftmaxRows(Tensor x)
    {
        var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        for (int r = 0; r < x.Rows; r++)
        {
            int b = r * x.Cols;
            float max = float.NegativeInfinity;
            for (int c = 0; c < x.Cols; c++)
            {
                max = MathF.Max(max, x.Data[b + c]);
            }

            double sum = 0.0;
            for (int c = 0; c < x.Cols; c++)
            {
                float e = MathF.Exp(x.Data[b + c] - max);
                outTensor.Data[b + c] = e;
                sum += e;
            }

            float inv = (float)(1.0 / sum);
            for (int c = 0; c < x.Cols; c++)
            {
                outTensor.Data[b + c] *= inv;
            }
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < x.Rows; r++)
            {
                int b = r * x.Cols;
                double dot = 0.0;
                for (int c = 0; c < x.Cols; c++)
                {
                    dot += outTensor.Grad[b + c] * (double)outTensor.Data[b + c];
                }

                float d = (float)dot;
                for (int c = 0; c < x.Cols; c++)
                {
                    x.Grad[b + c] += outTensor.Data[b + c] * (outTensor.Grad[b + c] - d);
                }
            }
        };

        return outTensor;
    }

    /// <summary>Tanh approximation of GELU, as used by GPT-2.</summary>
    public static Tensor Gelu(Tensor x)
    {
        const float C = 0.79788456f; // sqrt(2/pi)
        var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x] };
        var tanhCache = new float[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            float v = x.Data[i];
            float t = MathF.Tanh(C * (v + (0.044715f * v * v * v)));
            tanhCache[i] = t;
            outTensor.Data[i] = 0.5f * v * (1.0f + t);
        }

        outTensor.BackwardFn = () =>
        {
            for (int i = 0; i < x.Length; i++)
            {
                float v = x.Data[i];
                float t = tanhCache[i];
                float local = (0.5f * (1.0f + t)) + (0.5f * v * (1.0f - (t * t)) * C * (1.0f + (3.0f * 0.044715f * v * v)));
                x.Grad[i] += outTensor.Grad[i] * local;
            }
        };

        return outTensor;
    }

    public static Tensor LayerNorm(Tensor x, Tensor gain, Tensor bias, float eps = 1e-5f)
    {
        var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x, gain, bias] };
        var normalized = new float[x.Length];
        var invStd = new float[x.Rows];
        int n = x.Cols;

        for (int r = 0; r < x.Rows; r++)
        {
            int b = r * n;
            double mean = 0.0;
            for (int c = 0; c < n; c++)
            {
                mean += x.Data[b + c];
            }

            mean /= n;

            double variance = 0.0;
            for (int c = 0; c < n; c++)
            {
                double d = x.Data[b + c] - mean;
                variance += d * d;
            }

            variance /= n;
            invStd[r] = (float)(1.0 / Math.Sqrt(variance + eps));

            float m = (float)mean;
            for (int c = 0; c < n; c++)
            {
                float xhat = (x.Data[b + c] - m) * invStd[r];
                normalized[b + c] = xhat;
                outTensor.Data[b + c] = (xhat * gain.Data[c]) + bias.Data[c];
            }
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < x.Rows; r++)
            {
                int b = r * n;
                double meanDxhat = 0.0;
                double meanDxhatXhat = 0.0;

                for (int c = 0; c < n; c++)
                {
                    float g = outTensor.Grad[b + c];
                    float dxhat = g * gain.Data[c];
                    meanDxhat += dxhat;
                    meanDxhatXhat += dxhat * (double)normalized[b + c];
                    gain.Grad[c] += g * normalized[b + c];
                    bias.Grad[c] += g;
                }

                float mD = (float)(meanDxhat / n);
                float mDX = (float)(meanDxhatXhat / n);

                for (int c = 0; c < n; c++)
                {
                    float dxhat = outTensor.Grad[b + c] * gain.Data[c];
                    x.Grad[b + c] += invStd[r] * (dxhat - mD - (normalized[b + c] * mDX));
                }
            }
        };

        return outTensor;
    }

    /// <summary>Embedding lookup: gathers the rows of <paramref name="table"/> named by <paramref name="ids"/>.</summary>
    public static Tensor Gather(Tensor table, int[] ids)
    {
        var outTensor = new Tensor(ids.Length, table.Cols) { Parents = [table] };
        for (int r = 0; r < ids.Length; r++)
        {
            Array.Copy(table.Data, ids[r] * table.Cols, outTensor.Data, r * table.Cols, table.Cols);
        }

        outTensor.BackwardFn = () =>
        {
            for (int r = 0; r < ids.Length; r++)
            {
                int src = r * table.Cols;
                int dst = ids[r] * table.Cols;
                for (int c = 0; c < table.Cols; c++)
                {
                    table.Grad[dst + c] += outTensor.Grad[src + c];
                }
            }
        };

        return outTensor;
    }

    /// <summary>Mean softmax cross-entropy over all rows of <paramref name="logits"/>.</summary>
    public static Tensor CrossEntropy(Tensor logits, int[] targets)
    {
        var loss = new Tensor(1, 1) { Parents = [logits] };
        var probs = new float[logits.Length];
        int rows = logits.Rows;
        int vocab = logits.Cols;
        double total = 0.0;

        for (int r = 0; r < rows; r++)
        {
            int b = r * vocab;
            float max = float.NegativeInfinity;
            for (int c = 0; c < vocab; c++)
            {
                max = MathF.Max(max, logits.Data[b + c]);
            }

            double sum = 0.0;
            for (int c = 0; c < vocab; c++)
            {
                float e = MathF.Exp(logits.Data[b + c] - max);
                probs[b + c] = e;
                sum += e;
            }

            float inv = (float)(1.0 / sum);
            for (int c = 0; c < vocab; c++)
            {
                probs[b + c] *= inv;
            }

            total += -Math.Log(Math.Max(probs[b + targets[r]], 1e-12f));
        }

        loss.Data[0] = (float)(total / rows);

        loss.BackwardFn = () =>
        {
            float scale = loss.Grad[0] / rows;
            for (int r = 0; r < rows; r++)
            {
                int b = r * vocab;
                for (int c = 0; c < vocab; c++)
                {
                    float target = c == targets[r] ? 1.0f : 0.0f;
                    logits.Grad[b + c] += (probs[b + c] - target) * scale;
                }
            }
        };

        return loss;
    }
}
