# Getting Started

This page explains how to install and run TransformerXAI.jl as a user or as a developer.

## Installation as User

Start Julia:

```julia
julia
```

Enter package mode by pressing:

```julia
]
```

Add Llama2.jl from GitHub:

```julia
add https://github.com/ConstantConstantin/Llama2.jl
```

Add TransformerXAI.jl from GitHub:

```julia
add https://github.com/nirvanjhurreepriv/TransformerXAI.jl
```

Leave package mode by pressing backspace or Ctrl+C, then load the package:

```julia
using TransformerXAI
```

Afterwards, all exported functions of the package can be used if a model and tokenizer are provided. See the [Download a Model](#download-a-model) Section for more information.

### Basic Example
```julia
using TransformerXAI

bot = load_llama_model("models/stories42M.bin", "models/tokenizer.bin")

attention, tokens = extract_att_weights(bot)

heatmap_matrix = generate_attention_heatmap_matrix(
    attention,
    tokens
)
```

For an overview of all available functionality, see the [Functionality Guide](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev/functionality/).

## Installation as Developer

Clone the repository:

```bash
git clone https://github.com/nirvanjhurreepriv/TransformerXAI.jl.git
cd TransformerXAI.jl
```

Start Julia with the project environment:

```bash
julia --project=.
```

Enter package mode inside the Julia shell:

```julia
]
```

Install all recorded dependencies and precompile the project:

```julia
instantiate
precompile
```

### Basic Example

The repository contains a basic runnable example:

```text
examples/basicExample.jl
```
To run the example a model and tokenizer have to be available in the `models` folder. See the [Download a Model](#download-a-model) Section for more information.

Run the example from the project root:

```bash
julia --project=. examples/basicExample.jl
```

The example loads a Llama2-compatible model, extracts attention weights from a selected layer and creates attention heatmap visualizations.

## Download a Model

A basic model and tokenizer can be downloaded in the julia shell:
```julia
using Downloads

mkpath("models")

model_path = joinpath("models", "stories42M.bin")
model_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin"

tokenizer_path = joinpath("models", "tokenizer.bin")
tokenizer_url = "https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin"

Downloads.download(model_url, model_path)
Downloads.download(tokenizer_url, tokenizer_path)
```

## External Dependency: Llama2.jl

TransformerXAI.jl currently uses the external Julia package `Llama2.jl`:

```text
https://github.com/ConstantConstantin/Llama2.jl
```