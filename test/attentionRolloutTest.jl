@testset "extractAttWeights.jl - Rollout" begin
    model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
    vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

    if isfile(model_path) && isfile(vocab_path) 
        @testset "Check if returns are correct" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            desired_layer_depth = 1

            att_matrix, tokens = TransformerXAI.calc_att_rollout(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer_depth=desired_layer_depth
            )

            exp_shape = zeros(Float32,length(tokens),length(tokens),min(desired_layer_depth,bot.transformer.config.n_layers))

            @test att_matrix isa Array{Float32,3}
            @test all(isfinite, att_matrix)
            @test !isempty(tokens)
            @test tokens isa Vector{String}
            @test size(exp_shape) == size(att_matrix)
        end

        # returned tokens correct?
        @testset "Check tokens == tokeinzer output" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_matrix, tokens = TransformerXAI.calc_att_rollout(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer_depth=1
            )

            exp_tokens = [bot.tokenizer.vocab[token] for token in input_tokens]
            
            @test tokens == exp_tokens
        end

        # can we build heatmap with extract attention returns? 
        """
        @testset "Check if heatmap can be built" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_matrix, tokens = TransformerXAI.extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            heatmap = TransformerXAI.visualize_heatmap(tokens, att_matrix)
        
            @test heatmap isa TransformerXAI.AttentionHeatmap
            @test size(heatmap.attention) == (length(tokens), length(tokens))
        end
        """

        # wrong inputs should not be possible
        @testset "Throws errors with wrong input" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            @test_throws ArgumentError TransformerXAI.calc_att_rollout(
                bot;
                desired_head=0
            )
            @test_throws ArgumentError TransformerXAI.calc_att_rollout(
                bot;
                desired_layer_depth=0
            )
            # Tests if the function accepts a desired head above the number of heads (8) the bot has by default
            @test_throws ArgumentError TransformerXAI.calc_att_rollout(
                bot;
                desired_head=9
            )
            # Tests if the function accepts a desired layer depth above the depth of 8 the bot has by default
            @test_throws ArgumentError TransformerXAI.calc_att_rollout(
                bot;
                desired_layer_depth=9
            )
        end

    end

end
