
#=
Basic usage
To use this example you need to have a model installed (see README for more information)
=#

export run_basic_example

function run_basic_example(; output_path=joinpath(pwd(), "examples"))
    # create a Chatbot with loadLlama.jl
    project_root = normpath(pkgdir(TransformerXAI))
    model_path = joinpath(project_root, "models", "stories42M.bin")
    tokenizer_path = joinpath(project_root, "models", "tokenizer.bin")

    bot = load_llama_model(model_path, tokenizer_path)

    # Get attention weights for given layer (not needed, if you only want to visualize them)
    layer = 1

    attention_weights = extract_att_weights_from_layer_llama(bot, layer)
    println("Extracted attention weights size: ", size(attention_weights))

    # Plot attention heatmap for given layer
    plot = generate_attention_heatmap(bot, layer)


    #=
    Usage of extended heat map plot
    =#
    tokens = ["I", "like", "doing", "my", "homework."]

    # Each row describes how much one token attends to all tokens.
    # Strong examples are "like" -> "I", "doing" -> "like",
    # and "homework." -> "doing" or "my".
    attention = [
        1.0, 0.0, 0.0, 0.0, 0.0,  # I
        0.8, 1.0, 0.0, 0.0, 0.0,  # like
        0.2, 0.9, 1.0, 0.0, 0.0,  # doing
        0.1, 0.1, 0.3, 1.0, 0.0,  # my
        0.1, 0.2, 0.8, 0.9, 1.0,  # homework.
    ]

    heatmap = visualize_heatmap(tokens, attention)

    # A terminal cannot display SVG images, so save the heatmap to a file.
    mkpath(output_path)
    output_file = joinpath(output_path, "attentionHeatmap.svg")
    open(output_file, "w") do file
        show(file, MIME("image/svg+xml"), heatmap)
    end

    println("Heatmap saved to: $output_file")
    return output_file
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_basic_example()
end
