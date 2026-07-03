@testset "talktollm_changed" begin
    model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
    vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

    if isfile(model_path) && isfile(vocab_path)
        # check get_att false -> same behaviour as llama2 function
        @testset "Check get_att == false" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            out = TransformerXAI.talktollm_changed(
                bot;
                prompt=prompt,
                max_tokens=3,
                get_att=false
            )

            @test out isa String
        end

        # check get_att true 
        @testset "Check get_att == true" begin
            bot = load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time in a small town"

            out, att_history = TransformerXAI.talktollm_changed(
                bot;
                prompt=prompt,
                max_tokens=3,
                get_att=true
            )

            @test out isa String
            @test att_history isa Array{Float32, 4}
            @test all(isfinite, att_history)
        end
    end

end