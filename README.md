# Transformer XAI

[![CI](https://github.com/nirvanjhurreepriv/TransformerXAI.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nirvanjhurreepriv/TransformerXAI.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nirvanjhurreepriv/TransformerXAI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nirvanjhurreepriv/TransformerXAI.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nirvanjhurreepriv.github.io/TransformerXAI.jl/dev)

Transformer XAI is a Julia package for basic transformer interpretability experiments.

## Current functionality
- loading a Llama2 compatible model
- extracting attention weights from a given layer
- generating a basic attention heatmap
- generating an extended token-labeled attention heatmap
- providing a runnable basic example

## External dependency: Llama2.jl
This project uses the Llama2 package:
https://github.com/ConstantConstantin/Llama2.jl

## Start as User
1. Start Julia
```julia
julia
```
2. Enter package mode
```julia
]
```
3. Add the Llama2 dependency
```julia
add https://github.com/ConstantConstantin/Llama2.jl
```
4. Add TransformerXAI.jl
```julia
add https://github.com/nirvanjhurreepriv/TransformerXAI.jl
```
5. Leave package mode and load the package 
```julia
using TransformerXAI
```
To use the functionality you need to have a model and tokenizer installed.
    You can use the following for testing:
```julia
mkpath("models")

download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin", "models/stories42M.bin")
"stories42M.bin"

download("https://raw.githubusercontent.com/karpathy/llama2.c/b4bb47bb7baf0a5fb98a131d80b4e1a84ad72597/tokenizer.bin", "models/tokenizer.bin")
"tokenizer.bin"
```
Afterwards you can use all exported functions (for more information on all available functions take a look at section "Base Functionality Guide")

## Start as developer
1. Clone the repository
```julia
git clone https://github.com/nirvanjhurreepriv/TransformerXAI.jl.git
cd TransformerXAI.jl
```
2. Start Julia with project environment
```julia
julia --project=.
```
3. Enter package mode
Inside the Julia shell:
```julia
]
```
4. Add the Llama2 dependency
```julia
add https://github.com/ConstantConstantin/Llama2.jl
```
5. Install all recorded dependencies and precompile
```julia
instantiate
precompile
```
6. Leave Julia shell
```julia
exit()
```
7. Download the model files (optional, you can download your own model files as well)
```julia
julia --project=. scripts/downloadModel.jl
```

## Running the example
The repository contains a basic example file
```julia
examples/basicExample.jl
```
Before you load the example, make sure the model files exist:
```julia
models/stories42M.bin
models/tokenizer.bin
```
Run the example from the project root:
```julia
julia --project=. examples/basicExample.jl
```

## Base Functionality Guide
### Load a Llama2 model
File:
```julia
src/loadLlamaModel.jl
```
Main function:
```julia
load_llama_model(model_path, tokenizer_path)
```
Example in Julia shell: 
```julia
include("src/loadLlamaModel.jl")

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)
```
The returned object is a ChatBot from Llama2.jl that contains the loaded transformer, tokenizer, current position and last token. 

### Extract attention weights
File:
```julia
src/extractAttWeights.jl
```
Main function:
```julia
extract_att_weights(bot::ChatBot; input_prompt::String="Once upon a time", desired_layer::Int=1, desired_head::Int=1)
```
Example in Julia shell: 
```julia
include("src/extractAttWeights.jl")

bot = load_llama_model(model_path, tokenizer_path)
layer = 1
attention_history, token_strings = extract_att_weights(bot, input_prompt = "In a small village, there was a", desired_layer = 6)
```
The function returns the attention weights for the given layer as a Vector{Float32} as well as a Vector{String} containing the tokens from the input_prompt. This is semantically the same text as the input_prompt, however it is split by the tokenizer of the model. 

### Generate a basic attention heatmap
File:
```julia
src/attentionHeatmapBase.jl
```
Main function:
```julia
generate_attention_heatmap(bot, layer)
```
Example in Julia shell: 
```julia
include("src/attentionHeatmapBase.jl")

bot = load_llama_model(model_path, tokenizer_path)
layer = 1
attention_weights = generate_attention_heatmap(bot, layer)
```
The function plots the attention heatmap.

### Generate an extended token-labeled heatmap
File:
```julia
src/attentionHeatmapExpanded.jl
```
Main function:
```julia
visualize_heatmap(tokens, attention)
```
Example in Julia shell: 
```julia
include("src/attentionHeatmapExpanded.jl")

tokens = ["I", "like", "doing", "my", "homework."]

attention = [
    1.0, 0.0, 0.0, 0.0, 0.0,  
    0.8, 1.0, 0.0, 0.0, 0.0,  
    0.2, 0.9, 1.0, 0.0, 0.0,  
    0.1, 0.1, 0.3, 1.0, 0.0,  
    0.1, 0.2, 0.8, 0.9, 1.0,  
]

heatmap = visualize_heatmap(tokens, attention)
```

The function can be used for manually defined attention patterns and produces an SVG-compatible visualization.
