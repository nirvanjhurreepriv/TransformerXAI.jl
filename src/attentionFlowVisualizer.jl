"""
    attention_flow_heatmap(bot::ChatBot; input_prompt::String, source_pos::Int)

Computes the attention flow from a source token to all tokens at every layer,
producing a matrix that can be visualized as a heatmap across layers.

# Arguments
- `bot::ChatBot` : A base Llama2 ChatBot containing tokenizer and model

# Keyword Arguments
- `input_prompt::String` : The input text for analysis
- `source_pos::Int` : Position of the source token at the top layer

# Returns
- `Matrix{Float32}` : Flow matrix of shape `(n_layers+1) × n_tokens`, where
  entry `[l, j]` is the maximum flow from the source token to token `j` at layer `l-1`
- `Vector{String}` : The input token strings
"""
function attention_flow_heatmap(
    bot::ChatBot;
    input_prompt::String,
    source_pos::Int
)

    attention_history, tokens =
        get_att_matrix(bot; input_prompt=input_prompt)

    n_tokens = length(tokens)
    n_layers = bot.transformer.config.n_layers

    heat = zeros(Float32, n_layers + 1, n_tokens)

    for layer in 0:n_layers
        heat[layer + 1, :], _ =
            attention_flow(
                bot;
                input_prompt=input_prompt,
                source_pos=source_pos,
                target_layer=layer
            )
    end

    return heat, tokens
end



"""
    generate_attention_flow_heatmap(flow_matrix::AbstractMatrix{<:Real}, tokens::Vector{String})

Creates a heatmap visualization of attention flow across all layers.

Rows correspond to layers (L0 at top = input embeddings, higher rows = deeper layers),
columns correspond to input tokens. Color intensity reflects flow strength.

# Arguments
- `flow_matrix::AbstractMatrix{<:Real}` : Flow matrix of shape `(n_layers+1) × n_tokens`,
  as returned by `attention_flow_heatmap`
- `tokens::Vector{String}` : Token strings used as x-axis labels

# Returns
- `Plots.Plot` : A heatmap plot that can be displayed or saved with `savefig`

# Example
    heat, tokens = attention_flow_heatmap(bot; input_prompt="Once upon a time", source_pos=4)
    p = generate_attention_flow_heatmap(heat, tokens)
    savefig(p, "attention_flow.svg")
"""
function generate_attention_flow_heatmap(
    flow_matrix::AbstractMatrix{<:Real},
    tokens::Vector{String}
)

    n_layers, n_tokens = size(flow_matrix)

    heatmap(
        1:n_tokens,
        1:n_layers,
        flow_matrix,

        xlabel="Token",
        ylabel="Layer",

        xticks=(1:n_tokens, tokens),
        yticks=(1:n_layers, ["L$(i-1)" for i in 1:n_layers]),

        yflip=true,

        color=:YlOrRd,

        clim=(0, maximum(flow_matrix)),

        title="Attention Flow"
    )
end