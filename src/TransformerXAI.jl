module TransformerXAI

# Export the public API so tests can see these functions
export load_llama_model, extract_att_weights_from_layer_llama, AttentionHeatmap, visualize_heatmap

# Include source files
include("loadLlamaModel.jl")
include("extractAttWeights.jl")
include("attention_heatmap_base.jl")
include("attention_heatmap_expanded.jl")

end # module
