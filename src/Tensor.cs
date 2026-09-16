namespace MiniTransformer;

/// <summary>A 2-D matrix node in the autograd graph.</summary>
public sealed class Tensor
{
    public readonly int Rows;
    public readonly int Cols;
    public readonly float[] Data;
    public readonly float[] Grad;

    internal Tensor[] Parents = [];
    internal Action? BackwardFn;

    public Tensor(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new float[rows * cols];
        Grad = new float[rows * cols];
    }

    public int Length => Data.Length;

    public double this[int r, int c]
    {
        get => Data[r * Cols + c];
        set => Data[r * Cols + c] = (float)value;
    }

    public void ZeroGrad() => Array.Clear(Grad);

    /// <summary>Seeds this (scalar) node with dL/dL = 1 and propagates gradients to all ancestors.</summary>
    public void Backward()
    {
        Grad[0] = 1.0f;
        var order = TopoOrder(this);
        for (int i = order.Count - 1; i >= 0; i--)
        {
            order[i].BackwardFn?.Invoke();
        }
    }

    private static List<Tensor> TopoOrder(Tensor root)
    {
        var order = new List<Tensor>();
        var visited = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<(Tensor Node, int Index)>();

        stack.Push((root, 0));
        visited.Add(root);

        while (stack.Count > 0)
        {
            var (node, index) = stack.Pop();
            if (index < node.Parents.Length)
            {
                stack.Push((node, index + 1));
                var parent = node.Parents[index];
                if (visited.Add(parent))
                {
                    stack.Push((parent, 0));
                }
            }
            else
            {
                order.Add(node);
            }
        }

        return order;
    }
}
