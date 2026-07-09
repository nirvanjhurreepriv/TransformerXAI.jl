"""
The data needed to display an attention-flow heatmap.
"""
struct AttentionHeatmap{T<:Real}
    tokens::Vector{String}
    attention::Matrix{T}
end

"""
    visualize_heatmap(tokens::Vector{String}, attention::Array{Float32,2})

Create an attention-flow heatmap for `tokens`.

`attention` must contain one value for every token pair. The values are ordered
by referencing token first and reference token second. For two tokens, the
order is `[1->1, 1->2, 2->1, 2->2]`.

# Arguments
- `tokens::Vector{String}` : A vector containing the input tokens
- `attention::Array{Float32,2}` :  A matrix conatining the attention for given input tokens

# Returns
The function returns an attention heatmap that can be plotted afterwards
"""
function visualize_heatmap(
    tokens::Vector{String},
    attention::Array{Float32,2},
)
    number_of_tokens = length(tokens)
    expected_values = number_of_tokens * number_of_tokens

    if number_of_tokens == 0
        throw(ArgumentError("tokens must not be empty"))
    end

    if length(attention) != expected_values
        throw(ArgumentError(
            "attention must contain $expected_values values for $number_of_tokens tokens"
        ))
    end

    return AttentionHeatmap(copy(tokens), collect(attention'))
end

# Replace characters that have a special meaning in SVG.
function escape_svg(text::String)
    return replace(text, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
                   '"' => "&quot;", '\'' => "&apos;")
end

"""
    Base.show(io::IO, ::MIME"image/svg+xml", heatmap::AttentionHeatmap)

This function displays an extended Attention Heatmap
    
# Arguments
- `io::IO` : The output to where the svg is written
- `::MIME"image/svg+xml"` : Heatmap type specification
- `heatmap::AttentionHeatmap` : The attention heatmap that was created with the `visualize_heatmap` function

# Returns
The function writes a svg in the output stream which is then visible in the folder structure.

"""
# Display the heatmap as an SVG image in notebooks and other Julia displays.
function Base.show(io::IO, ::MIME"image/svg+xml", heatmap::AttentionHeatmap)
    number_of_tokens = length(heatmap.tokens)
    width = 700
    row_height = 45
    height = number_of_tokens * row_height + 40
    left_line_x = 220
    right_line_x = width - 220
    largest_attention = maximum(abs, heatmap.attention)

    if largest_attention == 0
        largest_attention = 1.0
    end

    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")

    for referencing_index in eachindex(heatmap.tokens)
        for reference_index in eachindex(heatmap.tokens)
            attention_strength = heatmap.attention[referencing_index, reference_index]

            # Do not draw a line when there is no attention.
            if attention_strength == 0
                continue
            end

            scaled_strength = abs(attention_strength) / largest_attention
            red = round(Int, 255 * scaled_strength)
            blue = round(Int, 255 * (1 - scaled_strength))
            line_width = 0.5 + 5 * scaled_strength
            reference_y = 20 + (reference_index - 0.5) * row_height
            referencing_y = 20 + (referencing_index - 0.5) * row_height

            println(io, """<line x1="$left_line_x" y1="$reference_y" x2="$right_line_x" y2="$referencing_y" stroke="rgb($red,0,$blue)" stroke-width="$line_width" stroke-opacity="0.7"/>""")
        end
    end

    # Draw the token labels after the lines so they stay readable.
    for (index, token) in enumerate(heatmap.tokens)
        y = 20 + (index - 0.5) * row_height
        safe_token = escape_svg(token)
        println(io, """<text x="$(left_line_x - 10)" y="$y" text-anchor="end" dominant-baseline="middle" font-family="sans-serif">$safe_token</text>""")
        println(io, """<text x="$(right_line_x + 10)" y="$y" text-anchor="start" dominant-baseline="middle" font-family="sans-serif">$safe_token</text>""")
    end

    print(io, "</svg>")
end

"""
    Base.show(io::IO, heatmap::AttentionHeatmap)

The function shows a textual presentation of the svg image

# Arguments
- `io::IO` : The output to where the svg is written
- `heatmap::AttentionHeatmap` : The attention heatmap that was created with the `visualize_heatmap` function

# Returns
The function gives a short description of the heatmap svg 

"""
function Base.show(io::IO, heatmap::AttentionHeatmap)
    print(io, "AttentionHeatmap with $(length(heatmap.tokens)) tokens")
end
