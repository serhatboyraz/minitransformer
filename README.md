# MiniTransformer

MiniTransformer is a compact, educational GPT-style transformer implemented in C# for learning the core ideas behind autoregressive language models: tokenization, embeddings, multi-head attention, feed-forward layers, training loop, loss optimization, checkpointing, and text generation.

This project is intentionally small and readable so it can be used as a reference implementation for understanding how a minimal decoder-only transformer works in practice.

## Features

- Character-level tokenizer built from the training corpus
- Token and positional embeddings
- Decoder-only transformer blocks with pre-layer normalization
- Multi-head self-attention
- Feed-forward network
- Adam optimizer with gradient clipping
- Training loss tracking with CSV export
- Model checkpoint save/load
- Text generation from a trained model

## Project structure

- `Program.cs` – CLI entry point and training/generation commands
- `src/GptModel.cs` – model and generation logic
- `src/Layers.cs` – transformer blocks, attention, projection, normalization
- `src/Tokenizer.cs` – character-level vocabulary and encoding/decoding
- `src/Checkpoint.cs` – save/load checkpoint support
- `src/AdamOptimizer.cs` – optimizer implementation
- `src/LossChart.cs` – ASCII loss chart renderer
- `data/input.txt` – default training corpus

## Requirements

- .NET 10 SDK (or the version required by your local environment)

## Quick start

Restore dependencies:

```bash
dotnet restore
```

Train a model:

```bash
dotnet run -c Release -- train
```

This creates a checkpoint file named `model.mtf` in the current working directory and saves the loss history to `loss.csv`.

Generate text from a trained model:

```bash
dotnet run -c Release -- generate
```

Or generate from a specific checkpoint:

```bash
dotnet run -c Release -- generate model.mtf "toyota corolla 2021 "
```

Show the training loss chart:

```bash
dotnet run -c Release -- loss
```

## Command reference

```bash
dotnet run -c Release -- train [checkpoint_path]
dotnet run -c Release -- generate [checkpoint_path] [prompt]
dotnet run -c Release -- loss [loss_csv_path]
```

If no command is provided, the app automatically does the following:

- loads `model.mtf` if it exists and generates text
- otherwise trains a new model

## Training configuration

The default training setup is defined in `Program.cs`:

- block size: 64
- embedding size: 96
- heads: 4
- layers: 3
- batch size: 8
- training steps: 1500
- learning rate: 3e-3

## Notes

This project is designed for learning and experimentation rather than production-scale training. The model is intentionally small and trained on a compact text corpus, which makes it suitable for understanding transformer internals and experimenting with architecture changes.

## License

This project is provided as educational source code for learning and experimentation.
