module TransformerXAI

# Export the public API so tests can see these functions
export load_llama_model, extract_att_weights_from_layer_llama, AttentionHeatmap

# Include source files
include("loadLlamaModel.jl")
include("extractAttWeights.jl")
include("visualize.jl")

end # module
