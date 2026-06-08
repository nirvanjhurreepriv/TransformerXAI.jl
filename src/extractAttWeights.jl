"""
Function returns attention weights from the selected layer as Matrix{Float32}.\n
The Matrix has shape (n_heads * headsize, dim).\n
Returns a string if arguments are invalid, describing the error.\n
"""
function extract_att_weights_from_layer_llama(bot::ChatBot, desiredLayer::Int)
    # According to the documentation of Llama2.jl, the type ChatBot has a field of type TransformerWeights.
    # In there, there's a lot of different fields, some of which I will describe quickly. All of the fields are (briefly!) described here: https://kleincode.github.io/Llama2.jl/stable/#Llama2.TransformerWeights
    #
    # Source used for writing description: https://www.billparker.ai/2024/10/transformer-attention-simple-guide-to-q.html
    # wk::Array{Float32, 3}: This is the key matrix. If I understood it correctly, here, each word's (more precise: token's) features are represented as a vector. So, this should be the matrix, where all the vectors in the high-dimensional embedding space live. (As in "aunt" - "woman" + "man" \approx "uncle") 
    # wq::Array{Float32, 3}: This is the query matrix. This.. also seems to be about the meaning of the word, but in the sense of "which words can relate to this" not exactly "what do the words mean semantically". I think.
    # wv::Array{Float32, 3}: This is the value matrix. This get's generated via dot product of Query and Key, so it's the similarity between a given query and the other words. This should (if I understood it correctly) therefore contain the info on how much each word in the given context relates to the query-word. F.e. (example from source): "The cat sat on the mat", query "cat". One of the words with the highest value would be "sat", because it relates to "cat" the most. (This is the thing from the images where there's the same text on both sides and colored lines connect one word with all the others, intensity showing the values for the query. Like they appear here: https://jalammar.github.io/illustrated-transformer/)
    
    # According to https://kleincode.github.io/Llama2.jl/stable/#Llama2.TransformerWeights, wo::Array{Float32, 3} are the output weights for the attention layers. So those will be used here.
    # Specifically: "wo::Array{T, 3} where T<:Real: Output weights for each attention layer. Shape: (nheads * headsize, dim, n_layers)"

    # Check if bot is valid
    if !(bot isa ChatBot)
        return "Argument given for 'bot' was not of type ChatBot."
    end

    # Check if desiredLayer is a valid number
    if (desiredLayer >= bot.transformer.config.n_layers || desiredLayer < 0)
        return "Desired Layer does not exist. Choose one between 0 and $(bot.transformer.config.n_layers-1)."
    end

    # Returns the desired layer from the matrix of attention output weights. 
    return bot.transformer.weights.wo[:,:,desiredLayer]
end
