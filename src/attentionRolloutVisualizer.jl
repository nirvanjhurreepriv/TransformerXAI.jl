"""
    visualize_attention_rollout(rollout_matrix::Array{Float32,3}, output_tokens::Vector{String})

Create a simple SVG visualization for the rollout returned by `calc_att_rollout`.
Tokens are shown as dots in a layer grid. Attention strength is encoded by line
color and thickness.

Returns storable SVG-code.
"""
function visualize_attention_rollout(rollout_matrix::Array{Float32,3}, output_tokens::Vector{String})
    token_count = length(output_tokens)
    matrix_rows, matrix_cols, layer_count = size(rollout_matrix)

    if token_count == 0
        throw(ArgumentError("output_tokens must not be empty"))
    end

    if matrix_rows != token_count || matrix_cols != token_count
        throw(ArgumentError("rollout_matrix must have one row and column per output token"))
    end

    # visual parameters
    start_x = 120
    start_y = 50
    layer_gap = 130
    token_gap = 45
    width = start_x + layer_count * layer_gap + 80
    height = start_y + (token_count - 1) * token_gap + 50

    # Used to normalize line thickness mapping
    largest_attention = maximum(abs, rollout_matrix)

    #avoid zero division
    if largest_attention == 0
        largest_attention = 1.0f0
    end

    io = IOBuffer()

    #built svg:
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")

    #draw connections between dots (tokens)
    
    #iterate through all pairings of tokens in adjacent layers
    for layer in axes(rollout_matrix, 3)
        for start in axes(rollout_matrix, 1)
            for target in axes(rollout_matrix, 2)
                attention_strength = rollout_matrix[start, target, layer]

                #skip connection with no attention
                if attention_strength == 0
                    continue
                end
                
                #normalize thickness
                scaled_strength = abs(attention_strength) / largest_attention
                
                #calculate color and thickness
                line_red = round(Int, 255 * scaled_strength)
                line_blue = round(Int, 255 * (1 - scaled_strength))
                line_width = 0.5 + 4 * scaled_strength

                #calculate connection positioning based on visualisation parameters
                from_x = start_x + (layer - 1) * layer_gap
                from_y = start_y + (start - 1) * token_gap
                to_x = start_x + layer * layer_gap
                to_y = start_y + (target - 1) * token_gap

                #draw connection
                println(io, """<line x1="$from_x" y1="$from_y" x2="$to_x" y2="$to_y" stroke="rgb($line_red,0,$line_blue)" stroke-width="$line_width" stroke-opacity="0.7"/>""")
            end
        end
    end

    #draw dots (tokens) for first layer with labels
    for (token_index, token) in enumerate(output_tokens)
        y = start_y + (token_index - 1) * token_gap

        #ensure token labels dont break SVG
        safe_token = replace(token, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
                             '"' => "&quot;", '\'' => "&apos;")

        # add token to visualization
        println(io, """<text x="$(start_x - 15)" y="$y" text-anchor="end" dominant-baseline="middle" font-family="sans-serif">$safe_token</text>""")

        # add initial dot
        println(io, """<circle cx="$start_x" cy="$y" r="4" fill="black"/>""")
    end
    
    #draw dots (tokens) for deeper layers
    for layer in axes(rollout_matrix, 3)
        x = start_x + layer * layer_gap

        # add new dots
        for token_index in eachindex(output_tokens)
            y = start_y + (token_index - 1) * token_gap
            println(io, """<circle cx="$x" cy="$y" r="4" fill="black"/>""")
        end
    end

    #export svg
    print(io, "</svg>")
    return String(take!(io))
end
