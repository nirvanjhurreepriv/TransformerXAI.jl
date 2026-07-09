@testset "attentionRolloutVisualizer.jl" begin
    @testset "rollout visualizer returns valid SVG" begin
        tokens = ["I", "test"]
        rollout = ones(Float32, 2, 2, 1)

        svg = visualize_attention_rollout(rollout, tokens)

        @test startswith(svg, "<svg")
        @test occursin("</svg>", svg)
        @test occursin("<rect", svg)
        @test occursin("<line", svg)
        @test occursin("<circle", svg)
        @test occursin("<text", svg)
        @test occursin("I", svg)
        @test occursin("test", svg)
    end

    @testset "rollout data contains 0 without breaking" begin
        tokens = ["I", "test"]
        rollout = Float32[
            1.0 0.0
            0.0 1.0
        ]
        rollout = reshape(rollout, 2, 2, 1)

        svg = visualize_attention_rollout(rollout, tokens)

        @test occursin("<svg", svg)
        @test occursin("</svg>", svg)
        @test length(collect(eachmatch(r"<line", svg))) == 2
    end

    @testset "SVG layer count matches attention layers" begin
        tokens = ["A", "B", "C"]
        rollout = ones(Float32, 3, 3, 2)

        svg = visualize_attention_rollout(rollout, tokens)

        @test length(collect(eachmatch(r"<line", svg))) == 18
        @test length(collect(eachmatch(r"<circle", svg))) == 9
    end

    @testset "Thickest SVG line encodes strongest attention" begin
        tokens = ["A", "B"]
        rollout = zeros(Float32, 2, 2, 1)
        rollout[1, 1, 1] = 0.25f0
        rollout[2, 2, 1] = 1.0f0

        svg = visualize_attention_rollout(rollout, tokens)

        @test occursin("stroke-width=\"1.5\"", svg)
        @test occursin("stroke-width=\"4.5\"", svg)
        @test occursin("stroke=\"rgb(255,0,0)\"", svg)
    end

    @testset "rollout attention of 0 draws no line" begin
        tokens = ["A", "B"]
        rollout = zeros(Float32, 2, 2, 1)
        rollout[1, 2, 1] = 1.0f0

        svg = visualize_attention_rollout(rollout, tokens)

        @test length(collect(eachmatch(r"<line", svg))) == 1
        @test occursin("x1=\"120\" y1=\"50\" x2=\"250\" y2=\"95\"", svg)
    end

    @testset "visualizer rejects invalid input" begin
        @test_throws ArgumentError visualize_attention_rollout(zeros(Float32, 1, 1, 1), String[])
        @test_throws ArgumentError visualize_attention_rollout(zeros(Float32, 2, 2, 1), ["only one"])
    end
end
