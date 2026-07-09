using TransformerXAI
using Plots

project_root = normpath(pkgdir(TransformerXAI))
model_path = joinpath(project_root, "models", "stories42M.bin")
tokenizer_path = joinpath(project_root, "models", "tokenizer.bin")
bot = load_llama_model(model_path, tokenizer_path)

prompt = "The author talked to Sara about her book"

# Get tokens from the first call
heat, tokens = attention_flow_heatmap(
    bot;
    input_prompt=prompt,
    source_pos=1  # temporary, we need n_tokens first
)

n_tokens = length(tokens)

# Redo with correct source_pos
heat, tokens = attention_flow_heatmap(
    bot;
    input_prompt=prompt,
    source_pos=n_tokens
)

heatmap_layers = generate_attention_flow_heatmap(heat, tokens)
savefig(heatmap_layers, "attention_flow_layers.svg")

# Pairwise token-to-token flow at input layer
pairwise_flow = zeros(Float32, n_tokens, n_tokens)

for src in 1:n_tokens
    pairwise_flow[src, :], _ = attention_flow(
        bot;
        input_prompt=prompt,
        source_pos=src,
        target_layer=0
    )
end

token2token_heatmap = generate_attention_heatmap_matrix(pairwise_flow, tokens)
savefig(token2token_heatmap, "token2token_attention_flow.svg")