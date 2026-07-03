# TransformerXAI.jl

*A Julia package for basic transformer interpretability experiments.*

TransformerXAI.jl provides basic tools for the inspection of attention in Llama2-compatible transformer models. The package can load a Llama2-compatible model, extract attention weights from given layers and heads, calculate the attention rollout for a desired depth and visualize attention heatmaps as well as the attention rollout.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ConstantConstantin/Llama2.jl")
Pkg.add(url="https://github.com/nirvanjhurreepriv/TransformerXAI.jl")
```

*To get a basic example and workflow, take a look into the [Getting started](gettingStarted.md) section and the [Functionality](functionality.md) section.*

See the [API Reference](api.md) for full function documentation, or
`examples/basicExample.jl` in the repository for a runnable end-to-end script.
