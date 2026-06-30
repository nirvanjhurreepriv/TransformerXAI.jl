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

Add TransformerXAI.jl from GitHub:

```julia
add https://github.com/nirvanjhurreepriv/TransformerXAI.jl
```

Leave package mode by pressing backspace or Ctrl+C, then load the package:

```julia
using TransformerXAI
```

Afterwards, all exported functions of the package can be used if a model and tokenizer are provided. 

A basic model and tokenizer can be downloaded in the julia shell:
```julia
mkpath("models")

download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", "models/stories42M.bin")
"stories42M.bin"

download("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", "models/tokenizer.bin")
"tokenizer.bin"
```

For an overview of the available functionality, see the [Functionality Guide](functionality.md).

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

Leave the Julia shell:

```julia
exit()
```

### For developers: Running the Basic Example

The repository contains a basic runnable example:

```text
examples/basicExample.jl
```
Run the example from the project root:

```bash
julia --project=. examples/basicExample.jl
```

The example loads a Llama2-compatible model, extracts attention weights from a selected layer and creates attention heatmap visualizations.

## External Dependency: Llama2.jl

TransformerXAI.jl currently uses the external Julia package `Llama2.jl`:

```text
https://github.com/ConstantConstantin/Llama2.jl
```