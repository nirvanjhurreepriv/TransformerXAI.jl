module TransformerXAI

# Imports
using LinearAlgebra: dot
using StatsBase: wsample
using Llama2: ChatBot, Transformer, rmsnorm, Sampler, encode, softmax!
using Plots: heatmap

# Export the public API so tests can see these functions
export
    load_llama_model,
    extract_att_weights,
    AttentionHeatmap,
    visualize_heatmap,
    generate_attention_heatmap_matrix,
    calc_att_rollout,
    visualize_attention_rollout,
    attention_flow

# Include source files
include("loadLlamaModel.jl")
include("extractAttWeights.jl")
include("attentionHeatmapExpanded.jl")
include("attentionHeatmapMatrix.jl")
include("attentionRolloutVisualizer.jl")
include("attentionFlow.jl")

end # end module
