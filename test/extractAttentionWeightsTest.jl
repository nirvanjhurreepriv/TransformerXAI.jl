@testset "extractAttWeights.jl" begin
    model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
    vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

    if isfile(model_path) && isfile(vocab_path) 
        @testset "Check if returns are correct" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            att_matrix, tokens = extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            exp_shape = zeros(Float32,length(tokens),length(tokens))

            @test att_matrix isa Array{Float32,2}
            @test all(isfinite, att_matrix)
            @test !isempty(tokens)
            @test tokens isa Vector{String}
            @test size(exp_shape) == size(att_matrix)
        end

        # returned tokens correct?
        @testset "Check tokens == tokeinzer output" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_matrix, tokens = extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            exp_tokens = [bot.tokenizer.vocab[token] for token in input_tokens]
            
            @test tokens == exp_tokens
        end

        # can we build heatmap with extract attention returns? 
        @testset "Check if heatmap can be built" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_matrix, tokens = extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            heatmap = visualize_heatmap(tokens, att_matrix)
        
            @test heatmap isa AttentionHeatmap
            @test size(heatmap.attention) == (length(tokens), length(tokens))
        end

        # wrong inputs should not be possible
        @testset "Throws errors with wrong input" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            @test_throws ArgumentError extract_att_weights(
                bot;
                desired_head=0
            )
            @test_throws ArgumentError extract_att_weights(
                bot;
                desired_layer=0
            )
            # Tests if the function accepts a desired head above the number of heads (8) the bot has by default
            @test_throws ArgumentError extract_att_weights(
                bot;
                desired_head=9
            )
            # Tests if the function accepts a desired layer above the depth of 8 the bot has by default
            @test_throws ArgumentError extract_att_weights(
                bot;
                desired_layer=9
            )
        end

    else    
        # bare minimum tests for test run without model and tokenizer
        
        # basic case -> everything should be fine
        @testset "Returns are correct - without model" begin
            tokens = ["I", "really", "like", "dogs"]
            attention_history = zeros(Float32, 4, 2, 2, 4)
            
            exp = Float32[
                1.0 0.2 0.3 0.4;
                0.0 0.8 0.1 0.2;
                0.0 0.0 0.7 0.3;
                0.0 0.0 0.0 0.9
            ]
            attention_history[:, 1, 1, :] = exp

            att_matrix, r_tokens = TransformerXAI._extract_att_weights_from_history(
                attention_history,
                tokens;
                desired_head=1,
                desired_layer=1
            )

            exp_shape = zeros(Float32, length(tokens), length(tokens))

            @test att_matrix isa Array{Float32,2}
            @test all(isfinite, att_matrix)
            @test !isempty(r_tokens)
            @test r_tokens isa Vector{String}
            @test size(exp_shape) == size(att_matrix)
            @test att_matrix == exp
        end

        # wrong inputs should not be possible
        @testset "Throws errors with wrong input - without model" begin
            tokens = ["I", "really", "like", "dogs"]
            attention_history = zeros(Float32, 4, 2, 2, 4)
            
            @test_throws ArgumentError TransformerXAI._extract_att_weights_from_history(
                attention_history, 
                tokens;
                desired_head=0
            )
            @test_throws ArgumentError TransformerXAI._extract_att_weights_from_history(
                attention_history, 
                tokens;
                desired_layer=0
            )

            # Tests if the function accepts a desired head above the number of heads (2) the bot has by default
            @test_throws ArgumentError TransformerXAI._extract_att_weights_from_history(
                attention_history, 
                tokens;
                desired_head=3
            )

            # Tests if the function accepts a desired layer above the depth of 2 the bot has by default
            @test_throws ArgumentError TransformerXAI._extract_att_weights_from_history(
                attention_history, 
                tokens;
                desired_layer=3
            )
        end

        # returned tokens correct?
        @testset "Check tokens == tokeinzer output - without model" begin
            tokens = ["I", "really", "like", "dogs"]
            attention_history = zeros(Float32, 4, 2, 2, 4)

            att_matrix, r_tokens = TransformerXAI._extract_att_weights_from_history(
                attention_history,
                tokens;
                desired_head=1,
                desired_layer=1
            )
            
            @test tokens == r_tokens
        end

        # can we build heatmap with extract attention returns? 
        @testset "Check if heatmap can be built - without model" begin
            tokens = ["I", "really", "like", "dogs"]
            attention_history = zeros(Float32, 4, 2, 2, 4)
            
            exp = Float32[
                1.0 0.2 0.3 0.4;
                0.0 0.8 0.1 0.2;
                0.0 0.0 0.7 0.3;
                0.0 0.0 0.0 0.9
            ]
            attention_history[:, 1, 1, :] = exp

            att_matrix, r_tokens = TransformerXAI._extract_att_weights_from_history(
                attention_history,
                tokens;
                desired_head=1,
                desired_layer=1
            )

            heatmap = visualize_heatmap(r_tokens, att_matrix)
        
            @test heatmap isa AttentionHeatmap
            @test size(heatmap.attention) == (length(r_tokens), length(r_tokens))
        end
    end
end
