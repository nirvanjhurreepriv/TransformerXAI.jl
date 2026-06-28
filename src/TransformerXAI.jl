module TransformerXAI


# Export the public API so tests can see these functions
export 
    load_llama_model, 
    extract_att_weights, 
    AttentionHeatmap, 
    visualize_heatmap,
    generate_attention_heatmap_matrix,
    visualize_attention_rollout

# Include source files
include("loadLlamaModel.jl")
include("extractAttWeights.jl")
include("attentionHeatmapExpanded.jl")
include("attentionHeatmapMatrix.jl")
include("attentionRolloutVisualizer.jl")
include("../examples/basicExample.jl")

end # module
