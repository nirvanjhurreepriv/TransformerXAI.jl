# Transformer XAI

[![CI](https://github.com/nirvanjhurreepriv/TransformerXAI.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nirvanjhurreepriv/TransformerXAI.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nirvanjhurreepriv/TransformerXAI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nirvanjhurreepriv/TransformerXAI.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev)

TransformerXAI.jl is a Julia package for basic transformer interpretability experiments.

The package currently focuses on loading Llama2-compatible models, extracting attention weights and visualizing attention patterns.

## Features

TransformerXAI.jl currently provides:

* loading of a Llama2-compatible model
* extraction of attention weights from a selected layer and head
* generation of a basic attention heatmap matrix
* generation of an extended token-labeled attention heatmap
* calculation of attention rollout and visualization
* a runnable basic example workflow

## Installation

Start Julia, enter package mode and add the package from GitHub:

```julia
]
add https://github.com/ConstantConstantin/Llama2.jl
add https://github.com/nirvanjhurreepriv/TransformerXAI.jl
```

Then exit the package mode and load the package:

```julia
using TransformerXAI
```

For more detailed installation instructions, see the [Getting Started Guide](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev/gettingStarted/).

## Model files

Model and tokenizer files are not included in this repository.

To run the Llama2-based examples, place the required files in the `models/` directory:

```text
models/stories42M.bin
models/tokenizer.bin
```

An example of a model and tokenizer download can be found in the [Getting Started Guide](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev/gettingStarted/).

## Basic Example

A minimal workflow looks like this:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

attention_history, token_strings = extract_att_weights(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer = 1,
    desired_head = 1
)

heatmap = generate_attention_heatmap_matrix(
    attention_history,
    token_strings
)
```

## Documentation

More information is available in the documentation:

* [Getting Started](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev/gettingStarted/)
* [Functionality Guide](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev/functionality/)

## External Dependency

TransformerXAI.jl currently uses the external Julia package `Llama2.jl`:

```text
https://github.com/ConstantConstantin/Llama2.jl
```

## Open TODOs / Currently in progress
* docstrings for rollout visualization tests
* tests for rollout calculation
* merge of attention flow implementation + testing
* more tests for all cases without model files
* matrix visualization of attention rollout
