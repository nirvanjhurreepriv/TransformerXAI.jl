# Transformer XAI
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
]
using TransformerXAI
```

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
insantiate
precompile
```
6. Leave Julia shell
```julia
exit()
```
7. Download the model files (optional, you can download your own model files as well)
```julia
julia --project=. scripts/download_model.jl
```

## Running the example
The repository contains a basic example file
```julia
examples/basic_example.jl
```
Before you load the example, make sure the model files exist:
```julia
models/stories42M.bin
models/tokenizer.bin
```
Run the example from the project root:
```julia
julia --project=. examples/basic_example.jl
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
extract_att_weights_from_layer(bot, layer)
```
Example in Julia shell: 
```julia
include("src/extractAttWeights.jl")

bot = load_llama_model(model_path, tokenizer_path)
layer = 1
attention_weights = extract_att_weights_from_layer(bot, layer)
```
The function returns the attention weights for the given layer.

### Generate a basic attention heatmap
File:
```julia
src/attention_heatmap_base.jl
```
Main function:
```julia
generate_attention_heatmap(bot, layer)
```
Example in Julia shell: 
```julia
include("src/attention_heatmap_base.jl")

bot = load_llama_model(model_path, tokenizer_path)
layer = 1
attention_weights = generate_attention_heatmap(bot, layer)
```
The function plots the attention heatmap.

### Generate an extended token-labeled heatmap
File:
```julia
src/attention_heatmap_expanded.jl
```
Main function:
```julia
visualize_heatmap(tokens,a attention)
```
Example in Julia shell: 
```julia
include("src/attention_heatmap_expanded.jl")

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