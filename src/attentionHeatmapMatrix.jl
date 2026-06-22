using Plots

"""
The function creates a basic heatmap matrix

# Arguments: 
    att_vector: a one dimensional vector containing the attention weights (can be created with the extract_att_weights() function)
    tokens: a one dimensional vector containing strings (the word given by the input prompt in the attention weight extraction)

# Returns: 
    A heatmap with given tokens and weights

# Example: 
    bot = load_llama_model("models/stories24M.bin", "models/tokenizer.bin")

    attention_history, token_strings = extract_att_weights(
        bot,
        input_prompt = "In a small village, there was a",
        desired_layer = 2
    )

    m = generate_attention_heatmap_matrix(attention_history, token_strings)
"""
function generate_attention_heatmap_matrix(att_vector, tokens)
    n = length(tokens)

    if length(att_vector) != length(tokens)^2
        throw(ArgumentError("input size mismatch"))
    end

    # att_vector is 1D -> need it to be 2D
    matrix = reshape(att_vector, length(tokens), length(tokens))'

    heatmap(
        1:n,
        1:n,
        matrix,
        title = "Attention Heatmap",
        xlabel = "Attended Token",
        ylabel = "Query Token",
        xticks = (1:n, tokens),
        yticks = (1:n, tokens),
        yflip = true,
    )
end