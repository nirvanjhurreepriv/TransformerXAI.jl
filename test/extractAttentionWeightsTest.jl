@testset "extractAttWeights.jl" begin
    model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
    vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

    if isfile(model_path) && isfile(vocab_path) 
        @testset "Check if returns are correct" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            att_vector, tokens = TransformerXAI.extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            @test att_vector isa Vector{Float32}
            @test all(isfinite, att_vector)
            @test !isempty(tokens)
            @test tokens isa Vector{String}
        end

        # returned tokens correct?
        @testset "Check tokens == tokeinzer output" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_vector, tokens = TransformerXAI.extract_att_weights(
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
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"
            input_tokens = Llama2.encode(bot.tokenizer, prompt)

            att_vector, tokens = TransformerXAI.extract_att_weights(
                bot;
                input_prompt=prompt,
                desired_head=1,
                desired_layer=1
            )

            heatmap = TransformerXAI.visualize_heatmap(tokens, att_vector)
        
            @test heatmap isa TransformerXAI.AttentionHeatmap
            @test size(heatmap.attention) == (length(tokens), length(tokens))
        end

        # wrong inputs should not be possible
        @testset "Throws errors with wrong input" begin
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            @test_throws ArgumentError TransformerXAI.extract_att_weights(
                bot;
                desired_head=0
            )
            @test_throws ArgumentError TransformerXAI.extract_att_weights(
                bot;
                desired_layer=0
            )
        end

    end

end