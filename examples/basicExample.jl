using Plots
using TransformerXAI
#=
Basic usage
To use this example you need to have a model installed (see README for more information)
=#

export run_basic_example

function run_basic_example(; output_path=joinpath(pwd(), "examples"))
    # create a Chatbot with loadLlama.jl
    project_root = normpath(pkgdir(TransformerXAI))
    model_path = joinpath(project_root, "models", "stories42M.bin")
    print("model path: ", model_path)
    tokenizer_path = joinpath(project_root, "models", "tokenizer.bin")

    bot = load_llama_model(model_path, tokenizer_path)

    # Get attention weights for given layer (not needed, if you only want to visualize them)
    layer = 1

    attention_weights, tokens = extract_att_weights(bot; desired_layer=layer)
    println("Extracted attention weights size: ", size(attention_weights))
    
    #=
    Usage of matrix heatmap plot
    =#
    heatmap_matrix = generate_attention_heatmap_matrix(attention_weights, tokens)
    
    mkpath(output_path)

    output_file_matrix = joinpath(output_path, "attentionHeatmapMatrix.svg")
    Plots.savefig(heatmap_matrix, output_file_matrix)

    println("Heatmap Matrix saved to: $output_file_matrix")

    #=
    Usage of extended heat map plot
    =#
    heatmap = visualize_heatmap(tokens, attention_weights)

    # A terminal cannot display SVG images, so save the heatmap to a file.
    output_file_heatmap = joinpath(output_path, "attentionHeatmap.svg")
    open(output_file_heatmap, "w") do file
        show(file, MIME("image/svg+xml"), heatmap)
    end

    println("Heatmap saved to: $output_file")

    #=
    Usage of Attention Rollout
    =#
    rollout_matrix, output_tokens = calc_att_rollout(bot)

    rollout_visualization = visualize_attention_rollout(rollout_matrix, output_tokens)
    
    output_file_rollout = joinpath(output_path, "attentionRollout.svg")
    open(output_file_rollout, "w") do file
        show(file, MIME("image/svg+xml"), rollout_visualization)
    end

    #=
    Usage of Attention Flow
    =#
    attention_flow_vals, flow_tokens = attention_flow(bot)

    print("Attention Flow: ", attention_flow)
    print("Attention Flow Tokens: ", flow_tokens)
    
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_basic_example()
end
