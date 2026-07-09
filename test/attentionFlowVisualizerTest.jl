@testset "attentionFlowVisualizer.jl" begin
    @testset "flow visualizer returns valid SVG" begin
        tokens = ["I", "test"]
        flow = Float32[1.0, 0.5]

        svg = visualize_attention_flow(flow, tokens)

        @test startswith(svg, "<svg")
        @test occursin("</svg>", svg)
        @test occursin("<rect", svg)
        @test occursin("<line", svg)
        @test occursin("<circle", svg)
        @test occursin("<text", svg)
        @test occursin("I", svg)
        @test occursin("test", svg)
    end

    @testset "flow data contains 0 without breaking" begin
        tokens = ["A", "B", "C"]
        flow = Float32[1.0, 0.0, 0.25]

        svg = visualize_attention_flow(flow, tokens)

        @test occursin("<svg", svg)
        @test occursin("</svg>", svg)
        @test length(collect(eachmatch(r"<line", svg))) == 2
    end

    @testset "thickest SVG line encodes strongest flow" begin
        tokens = ["A", "B"]
        flow = Float32[0.25, 1.0]

        svg = visualize_attention_flow(flow, tokens)

        @test occursin("stroke-width=\"1.5\"", svg)
        @test occursin("stroke-width=\"4.5\"", svg)
        @test occursin("stroke=\"rgb(255,0,0)\"", svg)
    end

    @testset "source_pos changes source coordinate" begin
        tokens = ["A", "B"]
        flow = Float32[0.0, 1.0]

        svg = visualize_attention_flow(flow, tokens; source_pos=2)

        @test length(collect(eachmatch(r"<line", svg))) == 1
        @test occursin("x1=\"120\" y1=\"95\" x2=\"330\" y2=\"95\"", svg)
    end

    @testset "flow attention of 0 draws no line" begin
        tokens = ["A", "B"]
        flow = Float32[0.0, 0.0]

        svg = visualize_attention_flow(flow, tokens)

        @test length(collect(eachmatch(r"<line", svg))) == 0
    end

    @testset "visualizer rejects invalid input" begin
        @test_throws ArgumentError visualize_attention_flow(Float32[], String[])
        @test_throws ArgumentError visualize_attention_flow(Float32[1.0, 0.5], ["only one"])
        @test_throws ArgumentError visualize_attention_flow(Float32[1.0], ["A"]; source_pos=0)
        @test_throws MethodError visualize_attention_flow([1.0], ["A"])
    end
end
