# This file contains two functions from the Llama2.jl project which have been slightly modified, to allow the extraction of attention.

using LinearAlgebra: dot
using StatsBase: wsample
using Llama2: Config, ChatBot, Transformer, rmsnorm, Tokenizer, encode, softmax!



"""
This is mostly a copy of the "forward!" function from https://github.com/ConstantConstantin/Llama2.jl/blob/main/src/forward.jl. Modified to allow accessing the attention which the original function does not permit.

Perform a single forward pass through the transformer.-

Compute logits for the next token prediction given the -current `token` at
position `pos` in the sequence. Updates the internal KV-cache in `transformer.state`
with keys and values from this forward pass.

# Arguments
- `transformer::Transformer`: The model (modified in-place via KV-cache updates).
- `token::Int32`: Current input token index (must be in range `1:vocab_size`).
- `pos::Int32`: Position in the sequence (1-indexed, must be ≤ `seq_len`).
- #### Added: get_att::Bool=false : If `true`, returns the collected attention at the end.

# Returns
- `Vector{Float32}`: Logits over the vocabulary for next token prediction (length = `vocab_size`).
"""
function forward_changed!(transformer::Transformer, token::Int32, pos::Int32, get_att::Bool = false)

    config = transformer.config
    weights = transformer.weights
    state = transformer.state

    dim = config.dim
    kv_dim = div(dim * config.n_kv_heads, config.n_heads)
    kv_mul = div(config.n_heads, config.n_kv_heads)
    hidden_dim = config.hidden_dim
    head_size = div(dim, config.n_heads)

    # assigning input token embedding to x
    x = weights.token_embedding_table[token, :]
    
    #### Addition: Defining "att" outside the for-loops allows returning it afterwards
    #### It naturally needs additional dimensions to hold the additional information
    att = zeros(Float32, pos, config.n_heads, config.n_layers)

    for l in 1:config.n_layers

        xb = rmsnorm(x, weights.rms_att_weight[l, :])

        k = @view state.key_cache[l, pos, :]
        v = @view state.value_cache[l, pos, :]
        # matmul to get q, k, v

        q = weights.wq[l, :, :] * xb
        k .= weights.wk[l, :, :] * xb
        v .= weights.wv[l, :, :] * xb

        for i in 1:2:dim

            head_dim = (i - 1) % head_size
            val = (pos - 1)  / (10000.0f0^(head_dim / head_size))
            fcr = cos(val)
            fci = sin(val)
            
            for vi in 1:(1 + (i <= kv_dim))
                if vi == 1
                    vec = @view q[i:(i + 1)]
                else
                    vec = @view k[i:(i + 1)]
                end
                v0 = vec[1]
                v1 = vec[2]
                vec[1] = v0 * fcr - v1 * fci
                vec[2] = v0 * fci + v1 * fcr
            end

        end

        xb .= 0

        for h in 1:config.n_heads # multi-head attention

            q_head = @view q[((h - 1) * head_size + 1):(h  * head_size)]
            #### Removed local definition of "att"
            ####att = Vector{Float32}(undef, pos)
            attention = @view att[:,h,l]

            for t in 1:pos

                k = @view state.key_cache[l, t, (div(h - 1, kv_mul) * head_size + 1):((div(h - 1, kv_mul) + 1) * head_size)]
                
                #### Added iterators
                attention[t] = dot(q_head, k)/sqrt(head_size)

            end

            #### Added iterators
            softmax!(attention)

            xb_head = @view xb[((h - 1) * head_size + 1):(h * head_size)]

            for t in 1:pos

                v = @view state.value_cache[l, t, (div(h - 1, kv_mul) * head_size + 1):((div(h - 1, kv_mul) + 1) * head_size)]
                
                #### Added iterators    
                xb_head .+= att[t,h,l] * v

            end

        end

        x .+= weights.wo[l, :, :] * xb

        xb = rmsnorm(x, weights.rms_ffn_weight[l, :])

        hb = weights.w1[l, :, :] * xb
        hb2 = weights.w3[l, :, :] * xb

        for i in 1:hidden_dim
            hb[i] = hb[i] * hb2[i] / (1.0f0 + exp(-hb[i]))
        end

        x .+= weights.w2[l, :, :] * hb

    end
    
    x .= rmsnorm(x, weights.rms_final_weight) # final rmsnorm
    
    # classifier into logits
    state.logits .= weights.wcls * x
    
    #### Adding returnvalue "att" if requested
    if get_att
        return state.logits, att
    end
    return state.logits

end






"""
This is mostly a copy of the "talktollm" function from https://github.com/ConstantConstantin/Llama2.jl/blob/main/src/talk.jl. Modified to allow accessing the attention which the original function does not permit.

# Arguments
- #### Changed: `bot::ChatBot`: struct defined in Llama2.jl, contains the model

# Keyword Arguments
- #### Changed: `prompt::String=""`: Initial text to condition generation (default: empty, starts with BOS token).
- `max_tokens::Int=255`: Maximum number of tokens to generate.
- #### Removed: ()`vocabpath::String`: Path to tokenizer binary (default: "data/tokenizer.bin").)
- `verbose::Bool=false`: If `true`, print tokens as they are generated.
- `temperature::Float32=0.0f0`: Sampling temperature (0 = greedy, higher = more random).
- `topp::Float32=1.1f0`: Nucleus sampling threshold (≤0 or ≥1 disables nucleus sampling).
- #### Added: get_att::Bool=false : If `true`, returns the collected attention at the end.

# Returns
- `String`: The generated text.
- #### Added: If get_att, also returns an object containing the attention used during the text creation.
"""
function talktollm_changed(bot::ChatBot; prompt::String = "", max_tokens::Int=255, verbose::Bool = false, temperature::Union{Nothing, Float32} = nothing, topp::Float32 = 1.1f0, get_att::Bool = false)

    #### Not using the modelpath anymore, since this is cumbersome if analysing a specific bot.
    transformer = bot.transformer
    tok = bot.tokenizer

    if temperature !== nothing
        sampler = Sampler(transformer.config.vocab_size, temperature, topp, 1234)
    end

    input_tokens = encode(tok, prompt)

    result = Vector{Int32}()
    sizehint!(result, max_tokens)

    if isempty(input_tokens)
        input_tokens = [2] # default for empty prompt
    else
        push!(result, input_tokens[1])
        verbose &&  print(tok.vocab[input_tokens[1]])
    end

    token = input_tokens[1]
    n_input_tokens = length(input_tokens)

    #### Added returnable attention object.
    #### Meaning of the dimensions are: (current position in the sequence (->pos), #Heads, #Layers, max_tokens)
    if get_att
        attention_history = zeros(Float32, max_tokens, transformer.config.n_heads, transformer.config.n_layers, max_tokens)
    end

    for pos in 1:max_tokens
        #### Changing behavior depending on get_att
        
        if get_att
            logits, attention_history[1:pos,:,:,pos] = forward_changed!(transformer, Int32(token), Int32(pos), get_att)
        else
            logits = forward_changed!(transformer, Int32(token), Int32(pos), get_att)
        end
        ####logits = forward_changed!(transformer, Int32(token), Int32(pos))

        if pos < n_input_tokens
            next = input_tokens[pos + 1]
        else
            if temperature === nothing
                softmax!(logits)
                next = wsample(logits)
            else
                next = sampler(logits)
            end
        end

        next == 2 && break
        push!(result, next)
        verbose && print(tok.vocab[next])
        token = next
    end

    verbose && println()
    
    #### Changing behavior depending on get_att
    if get_att
        return string(broadcast(x -> tok.vocab[x], result)...), attention_history;
    end

    return string(broadcast(x -> tok.vocab[x], result)...)
end


"""
    extract_att_weights(bot::ChatBot; input_prompt::String="Once upon a time", desired_layer::Int=1, desired_head::Int=1)

Calls a modified version of talktollm from Llama2.jl to collect attention weights.
Calling this function does not generate new text, so the text inside output_tokens_strings will be semantically the same as input_prompt.

# Arguments: 
- `bot::ChatBot` : Model type from Llama2.jl
- `input_prompt::String` : Text for which attention should be generated
- `desired_layer::Int` : The layer from which the attention should be returned
- `desired_layer::Int` : The attentionhead from which the attention should be returned

# Returns: 
- `attention_vector::Vector{Float32}` : contains the generated attention values (after softmax)
- `output_tokens_strings::Vector{String}` : contains the individual tokens from input_prompt as separate strings.
"""
function extract_att_weights(bot::ChatBot; input_prompt::String="Once upon a time", desired_layer::Int=1, desired_head::Int=1)
    # Safety Checks
    (desired_layer > 0) || throw(ArgumentError("desired_layer needs to be larger than 0"))
    (desired_head > 0) || throw(ArgumentError("desired_head needs to be larger than 0"))

    tok = bot.tokenizer
    input_tokens = encode(tok, input_prompt)
    length_of_input = length(input_tokens)

    (length_of_input > 0) || throw(ArgumentError("input_prompt must contain more than $length_of_input tokens in the form of a string."))
    

    # Get the model to "look at" the prompt to get the attention history.
        #output_prompt, attention_history = talktollm_changed(bot.transformer, prompt=input_prompt, max_tokens=length_of_input, get_att=true);
        # We don't need output_prompt
    _, attention_history = talktollm_changed(bot, prompt=input_prompt, max_tokens=length_of_input, get_att=true);

    # Split input_prompt into separate strings which contain one token each, for visualisation purposes
    output_tokens_strings = Vector{String}();
    # Restructure the attention matrix into a Vector, since that is what is what the visualize_heatmap function takes as a parameter.
    # Also: Bring output_tokens_strings into the correct format.
    attention_vector = Vector{Float32}();
    for i in 1:length_of_input
        append!(attention_vector, attention_history[:,desired_head,desired_layer,i]);
        push!(output_tokens_strings,tok.vocab[input_tokens[i]])
    end

    return attention_vector, output_tokens_strings
end
