using Plots

function generate_attention_heatmap(bot, layer)
    weights = extract_att_weights_from_layer_llama(bot, layer)

    heatmap(
        weights, 
        title="Attention Output Weights $layer", 
        xlabel="Dimension", 
        ylabel="Head"
    )
end