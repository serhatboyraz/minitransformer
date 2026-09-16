using System.Text;

namespace MiniTransformer;

/// <summary>Renders a training curve as an ASCII chart so it can be read straight from the console.</summary>
public static class LossChart
{
    public static string Render(IReadOnlyList<double> values, int width = 76, int height = 18)
    {
        if (values.Count == 0)
        {
            return "(veri yok)";
        }

        int columns = Math.Min(width, values.Count);
        var points = new double[columns];

        // Her sütun, adım aralığının ortalamasıdır; bu aynı zamanda gürültüyü yumuşatır.
        for (int c = 0; c < columns; c++)
        {
            int start = (int)((long)c * values.Count / columns);
            int end = (int)((long)(c + 1) * values.Count / columns);
            if (end <= start)
            {
                end = start + 1;
            }

            double sum = 0.0;
            for (int i = start; i < end; i++)
            {
                sum += values[i];
            }

            points[c] = sum / (end - start);
        }

        double max = points.Max();
        double min = points.Min();
        if (max - min < 1e-9)
        {
            max = min + 1e-9;
        }

        var grid = new char[height][];
        for (int r = 0; r < height; r++)
        {
            grid[r] = new char[columns];
            Array.Fill(grid[r], ' ');
        }

        for (int c = 0; c < columns; c++)
        {
            int row = (int)Math.Round((max - points[c]) / (max - min) * (height - 1));
            grid[row][c] = '*';
        }

        var sb = new StringBuilder();
        for (int r = 0; r < height; r++)
        {
            double label = max - ((max - min) * r / (height - 1));
            sb.Append(label.ToString("F3").PadLeft(7)).Append(" |").Append(grid[r]).AppendLine();
        }

        sb.Append(new string(' ', 7)).Append(" +").Append(new string('-', columns)).AppendLine();
        sb.Append(new string(' ', 9)).Append("1")
          .Append(new string(' ', Math.Max(1, columns - values.Count.ToString().Length - 1)))
          .Append(values.Count)
          .AppendLine();
        sb.Append(new string(' ', 9)).Append("adım ->").AppendLine();

        return sb.ToString();
    }
}
