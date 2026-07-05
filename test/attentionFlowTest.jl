@testset "attentionFlow.jl" begin

    @testset "bfs_path - finds path in linear graph" begin
        cap = zeros(Float32, 3, 3)
        cap[1, 2] = 1.0f0
        cap[2, 3] = 1.0f0
        path = TransformerXAI.bfs_path(cap, 1, 3)
        @test path == [1, 2, 3]
    end

    @testset "bfs_path - returns nothing when no path exists" begin
        cap = zeros(Float32, 3, 3)
        path = TransformerXAI.bfs_path(cap, 1, 3)
        @test path === nothing
    end

    @testset "bfs_path - direct edge" begin
        cap = zeros(Float32, 2, 2)
        cap[1, 2] = 0.5f0
        path = TransformerXAI.bfs_path(cap, 1, 2)
        @test path == [1, 2]
    end

    @testset "max_flow - single edge" begin
        cap = zeros(Float32, 2, 2)
        cap[1, 2] = 0.75f0
        flow = TransformerXAI.max_flow(cap, 1, 2)
        @test isapprox(flow, 0.75f0, atol=1e-5)
    end

    @testset "max_flow - no path returns zero" begin
        cap = zeros(Float32, 2, 2)
        flow = TransformerXAI.max_flow(cap, 1, 2)
        @test flow == 0.0f0
    end

    @testset "max_flow - two parallel paths" begin
        # 1→2→4 (bottleneck 0.3) and 1→3→4 (bottleneck 0.5), total = 0.8
        cap = zeros(Float32, 4, 4)
        cap[1, 2] = 0.3f0
        cap[2, 4] = 0.3f0
        cap[1, 3] = 0.5f0
        cap[3, 4] = 0.5f0
        flow = TransformerXAI.max_flow(cap, 1, 4)
        @test isapprox(flow, 0.8f0, atol=1e-5)
    end

    @testset "adjusted_attention - shape and row sums" begin
        n_tokens = 4
        n_heads  = 2
        n_layers = 3
        # Identity attention: each token attends only to itself
        att = zeros(Float32, n_tokens, n_heads, n_layers, n_tokens)
        for h in 1:n_heads, l in 1:n_layers, i in 1:n_tokens
            att[i, h, l, i] = 1.0f0
        end

        A = TransformerXAI.adjusted_attention(att, 1, n_tokens)

        @test A isa Matrix{Float32}
        @test size(A) == (n_tokens, n_tokens)
        @test all(isfinite, A)
        @test all(A .>= 0.0f0)
        # Each row must sum to 1 after re-normalization
        @test all(isapprox.(sum(A, dims=2), 1.0f0, atol=1e-5))
    end

    @testset "adjusted_attention - identity input gives identity output" begin
        n_tokens = 3
        n_heads  = 1
        n_layers = 1
        att = zeros(Float32, n_tokens, n_heads, n_layers, n_tokens)
        for i in 1:n_tokens
            att[i, 1, 1, i] = 1.0f0
        end

        A = TransformerXAI.adjusted_attention(att, 1, n_tokens)

        # 0.5*I + 0.5*I = I, rows already sum to 1
        expected = zeros(Float32, n_tokens, n_tokens)
        for i in 1:n_tokens; expected[i, i] = 1.0f0; end
        @test isapprox(A, expected, atol=1e-5)
    end

    @testset "build_capacity_graph - output shape and value bounds" begin
        n_tokens = 3
        n_heads  = 2
        n_layers = 2
        att = zeros(Float32, n_tokens, n_heads, n_layers, n_tokens)
        for h in 1:n_heads, l in 1:n_layers, i in 1:n_tokens
            att[i, h, l, i] = 1.0f0
        end

        cap = TransformerXAI.build_capacity_graph(att, n_tokens, n_layers)

        @test size(cap) == ((n_layers + 1) * n_tokens, (n_layers + 1) * n_tokens)
        @test all(cap .>= 0.0f0)
        @test all(cap .<= 1.0f0)
        @test all(isfinite, cap)
    end

    @testset "attention_flow - rejects invalid source_pos" begin
        model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
        vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

        if isfile(model_path) && isfile(vocab_path)
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            @test_throws ArgumentError TransformerXAI.attention_flow(bot; input_prompt="Hello world", source_pos=0)
            @test_throws ArgumentError TransformerXAI.attention_flow(bot; input_prompt="Hello world", source_pos=999)
        end
    end

    @testset "attention_flow - integration with real model" begin
        model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))
        vocab_path = normpath(joinpath(@__DIR__, "..", "models", "tokenizer.bin"))

        if isfile(model_path) && isfile(vocab_path)
            bot = TransformerXAI.load_llama_model(model_path, vocab_path)
            prompt = "Once upon a time"

            flow_values, tokens = TransformerXAI.attention_flow(bot; input_prompt=prompt, source_pos=1)

            @test flow_values isa Vector{Float32}
            @test tokens isa Vector{String}
            @test length(flow_values) == length(tokens)
            @test all(isfinite, flow_values)
            @test all(flow_values .>= 0.0f0)
        end
    end

end
