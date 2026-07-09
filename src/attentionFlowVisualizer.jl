"""
    visualize_attention_flow(flow_values::Vector{Float32}, output_tokens::Vector{String}; source_pos::Int=1)

Create a simple SVG visualization for the flow values returned by `attention_flow`.
Tokens are shown as dots, with the selected source token connected to every token
that has a non-zero attention flow. Flow strength is encoded by line color and
thickness.

Returns storable SVG-code.
"""
function visualize_attention_flow(flow_values::Vector{Float32}, output_tokens::Vector{String}; source_pos::Int=1)
    token_count = length(output_tokens)

    if token_count == 0
        throw(ArgumentError("output_tokens must not be empty"))
    end

    if length(flow_values) != token_count
        throw(ArgumentError("flow_values must contain one value per output token"))
    end

    if source_pos < 1 || source_pos > token_count
        throw(ArgumentError("source_pos must be in 1:$token_count, got $source_pos"))
    end

    start_x = 120
    target_x = 330
    start_y = 50
    token_gap = 45
    width = target_x + 170
    height = start_y + (token_count - 1) * token_gap + 50

    largest_flow = maximum(abs, flow_values)
    if largest_flow == 0
        largest_flow = 1.0f0
    end

    io = IOBuffer()

    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height">""")
    println(io, """<rect width="100%" height="100%" fill="white"/>""")

    source_y = start_y + (source_pos - 1) * token_gap

    for (token_index, flow_strength) in enumerate(flow_values)
        if flow_strength == 0
            continue
        end

        scaled_strength = abs(flow_strength) / largest_flow
        line_red = round(Int, 255 * scaled_strength)
        line_blue = round(Int, 255 * (1 - scaled_strength))
        line_width = 0.5 + 4 * scaled_strength
        target_y = start_y + (token_index - 1) * token_gap

        println(io, """<line x1="$start_x" y1="$source_y" x2="$target_x" y2="$target_y" stroke="rgb($line_red,0,$line_blue)" stroke-width="$line_width" stroke-opacity="0.7"/>""")
    end

    for (token_index, token) in enumerate(output_tokens)
        y = start_y + (token_index - 1) * token_gap
        safe_token = replace(token, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
                             '"' => "&quot;", '\'' => "&apos;")

        println(io, """<text x="$(start_x - 15)" y="$y" text-anchor="end" dominant-baseline="middle" font-family="sans-serif">$safe_token</text>""")
        println(io, """<circle cx="$start_x" cy="$y" r="4" fill="black"/>""")
        println(io, """<circle cx="$target_x" cy="$y" r="4" fill="black"/>""")
        println(io, """<text x="$(target_x + 15)" y="$y" text-anchor="start" dominant-baseline="middle" font-family="sans-serif">$safe_token</text>""")
    end

    print(io, "</svg>")
    return String(take!(io))
end
