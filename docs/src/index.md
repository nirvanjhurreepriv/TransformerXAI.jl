# TransformerXAI.jl

*A Julia package for basic transformer interpretability experiments.*

TransformerXAI.jl loads a Llama2-compatible model, extracts attention weights
from a specific layer and head, and renders the resulting attention pattern
as an SVG heatmap.

## Features

- Loading a Llama2-compatible model and tokenizer
- Extracting per-layer, per-head attention weights from a forward pass
- Generating attention heatmaps directly from a loaded model
- Generating token-labeled heatmaps from manually supplied attention values

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ConstantConstantin/Llama2.jl")
Pkg.add(url="https://github.com/nirvanjhurreepriv/TransformerXAI.jl")
```

## Quick example

```julia
using TransformerXAI

bot = load_llama_model("models/stories42M.bin", "models/tokenizer.bin")
attention, tokens = extract_att_weights(bot; input_prompt="Once upon a time", desired_layer=6)
```

See the [API Reference](api.md) for full function documentation, or
`examples/BasicExample.jl` in the repository for a runnable end-to-end script.
