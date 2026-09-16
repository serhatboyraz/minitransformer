namespace MiniTransformer;

/// <summary>Character-level vocabulary built from a training corpus.</summary>
public sealed class Tokenizer
{
    private readonly char[] _idToChar;
    private readonly Dictionary<char, int> _charToId;

    public Tokenizer(string corpus)
    {
        _idToChar = corpus.Distinct().OrderBy(c => c).ToArray();
        _charToId = _idToChar.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);
    }

    public int VocabSize => _idToChar.Length;

    /// <summary>Sözlüğün kendisi; checkpoint'e yazılıp aynen geri yüklenebilir.</summary>
    public string Vocabulary => new(_idToChar);

    public int[] Encode(string text) =>
        text.Where(_charToId.ContainsKey).Select(c => _charToId[c]).ToArray();

    public string Decode(IEnumerable<int> ids) =>
        new([.. ids.Select(i => _idToChar[i])]);
}
