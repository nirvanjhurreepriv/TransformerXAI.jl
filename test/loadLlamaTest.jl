@testset "loadLlamaModel.jl" begin
    # load llama model check missing model path
    @testset "Check Missin Model Path in LoadLlama Fucntion" begin
        @test_throws ArgumentError load_llama_model("models/nothing.bin", "models/tokenizer.bin")
    end

    # load llama model check missing vocab path
    @testset "Check Missin Vocabulary Path in LoadLlama Fucntion" begin
        @test_throws ArgumentError load_llama_model("models/stories42M.bin", "models/nothing.bin")
    end

    # load llama model check if output is actually a bot
    @testset "Check if Bot is returned" begin
        model_path = joinpath(@__DIR__, "..", "models", "stories42M.bin")
        vocabulary_path = joinpath(@__DIR__, "..", "models", "tokenizer.bin")

        # test is only available, if models are available. otherwise it has to be skipped
        if isfile(model_path) && isfile(vocabulary_path)
            bot = load_llama_model(model_path, vocabulary_path)
            @test bot isa Llama2.ChatBot
        end
    end
end