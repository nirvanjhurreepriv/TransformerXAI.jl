module TransformerXAI

include("../examples/basic_example.jl")
# Export the public API so tests can see these functions
export 
    load_llama_model, 
    extract_att_weights, 
    AttentionHeatmap, 
    visualize_heatmap,
    generate_attention_heatmap

# Include source files
include("loadLlamaModel.jl")
include("extractAttWeights.jl")
include("attention_heatmap_base.jl")
include("attention_heatmap_expanded.jl")

end # module
