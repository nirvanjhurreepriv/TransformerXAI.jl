# Functionality Guide

This page gives an overview of the main functionality currently provided by TransformerXAI.jl.

## Current Functionality

TransformerXAI.jl currently supports:

* loading a Llama2-compatible model
* extracting attention weights from a given layer and head
* generating a basic attention heatmap matrix
* generating an extended token-labeled attention heatmap
* calculating attention rollout across multiple layers
* visualizing attention rollout as svg

## Load a Llama2 Model

Source file:

```text
src/loadLlamaModel.jl
```

Main function:

```julia
load_llama_model(model_path, tokenizer_path)
```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)
```

The returned object is a `ChatBot` from `Llama2.jl`. It contains the loaded transformer, tokenizer, current position and last token.

## Extract Attention Weights

Source file:

```text
src/extractAttWeights.jl
```

Main function:

```julia
attention_matrix, tokens = extract_att_weights(
    bot::ChatBot;
    input_prompt::String = "Once upon a time",
    desired_layer::Int = 1,
    desired_head::Int = 1
)
```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer = 1
head = 1

attention_matrix, token_strings = extract_att_weights(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer = layer.
    desired_head = head
)
```

The function returns the attention weights for the selected layer and head, together with the tokenized input prompt.

The returned `token_strings` represent the same semantic text as the input prompt, but split according to the tokenizer of the loaded model.

## Generate a Basic Attention Heatmap Matrix

Source file:

```text
src/attentionHeatmapMatrix.jl
```

Main function:

```julia
generate_attention_heatmap_matrix(
    att_matrix::Matrix{<:Real}, 
    tokens::Vector{String}
)

```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer = 1
head = 1

attention_matrix, token_strings = extract_att_weights(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer = layer.
    desired_head = head
)

heatmap_matrix = generate_attention_heatmap_matrix(
    attention_matrix,
    token_strings
)
```

The function creates a basic attention heatmap matrix from the extracted attention weights and the corresponding tokens.

## Generate an Extended Token-Labeled Heatmap

Source file:

```text
src/attentionHeatmapExpanded.jl
```

Main function:

```julia
visualize_heatmap(
    tokens::Vector{String},
    attention::Array{Float32,2},
)
```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer = 1
head = 1

attention_matrix, token_strings = extract_att_weights(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer = layer.
    desired_head = head
)

heatmap = visualize_heatmap(
    token_strings, 
    attention_matrix    
)
```

The function creates an extended token-labeled heatmap. It can be used with extracted attention weights or with manually defined attention patterns.

The generated visualization is SVG-compatible.


## Calculate Attention Rollout

Source file:

```text
src/extractAttWeights.jl
```

Main function:

```julia
calc_att_rollout(
    bot::ChatBot;
    input_prompt::String = "Once upon a time",
    desired_layer_depth::Int = 1,
    desired_head::Int = 1,
)
```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer_depth = 3
head = 1

rollout_matrix, token_strings = calc_att_rollout(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer_depth = layer_depth,
    desired_head = head,
)
```

The function returns rolled-up attention values across multiple layers.

## Visualize Attention Rollout

Source file:

```text
src/attentionRolloutVisualizer.jl
```

Main function:

```julia
visualize_attention_rollout(
    rollout_matrix::Array{Float32,3},
    output_tokens::Vector{String},
)
```

Example:

```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer_depth = 3
head = 1

rollout_matrix, token_strings = calc_att_rollout(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer_depth = layer_depth,
    desired_head = head,
)

svg = visualize_attention_rollout(rollout_matrix, token_strings)

open("attention_rollout.svg", "w") do file
    write(file, svg)
end
```

The function creates an SVG visualization for the attention rollout. Tokens are shown as dots in a layer grid. Attention strength is represented by line color and thickness.

## Basic workflow example
```julia
using TransformerXAI

model_path = "models/stories42M.bin"
tokenizer_path = "models/tokenizer.bin"

bot = load_llama_model(model_path, tokenizer_path)

layer_depth = 3
head = 1

attention_matrix, token_strings = extract_att_weights(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer = layer.
    desired_head = head
)

heatmap_matrix = generate_attention_heatmap_matrix(
    attention_matrix,
    token_strings
)

heatmap = visualize_heatmap(
    token_strings, 
    attention_matrix    
)

rollout_matrix, token_strings = calc_att_rollout(
    bot;
    input_prompt = "In a small village, there was a",
    desired_layer_depth = layer_depth,
    desired_head = head,
)

svg = visualize_attention_rollout(rollout_matrix, token_strings)

open("attention_rollout.svg", "w") do file
    write(file, svg)
end
```