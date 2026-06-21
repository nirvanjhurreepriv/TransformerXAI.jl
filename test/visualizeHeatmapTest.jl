@testset "attentionHeatmapExpanded.jl" begin
        # visuealize heatmap exp normal case -> should just return an attention heatmap 
        @testset "Normal Case, check if output is correct" begin
            tokens = ["I", "like", "doing", "my", "homework."]
            attention = [
                1.0, 0.0, 0.0, 0.0, 0.0,  # I
                0.8, 1.0, 0.0, 0.0, 0.0,  # like
                0.2, 0.9, 1.0, 0.0, 0.0,  # doing
                0.1, 0.1, 0.3, 1.0, 0.0,  # my
                0.1, 0.2, 0.8, 0.9, 1.0,  # homework.
            ]
            out = visualize_heatmap(tokens, attention)

            @test out isa AttentionHeatmap 
            @test out.tokens == tokens 
            @test size(out.attention) == (5, 5)
        end

        # visualize heatmap exp check empty input tokens
        @testset "Check empty input tokens" begin
            tokens = String[]
            attention = [
                1.0, 0.0, 0.0, 0.0, 0.0,  # I
                0.8, 1.0, 0.0, 0.0, 0.0,  # like
                0.2, 0.9, 1.0, 0.0, 0.0,  # doing
                0.1, 0.1, 0.3, 1.0, 0.0,  # my
                0.1, 0.2, 0.8, 0.9, 1.0,  # homework.
            ]

            @test_throws ArgumentError visualize_heatmap(tokens, attention)
        end

        # visualize heatmap exp check wrong input attention size
        @testset "Check wrong and empty attention size" begin
            tokens = ["I", "like", "doing", "my", "homework."]
            attention1 = Float64[]
            attention2 = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]

            @test_throws ArgumentError visualize_heatmap(tokens, attention1)
            @test_throws ArgumentError visualize_heatmap(tokens, attention2)
        end

        # visualize heatmap exp check svg not empty
        @testset "Check svg output" begin
            tokens = ["I", "like", "doing", "my", "homework."]
            attention = [
                1.0, 0.0, 0.0, 0.0, 0.0,  # I
                0.8, 1.0, 0.0, 0.0, 0.0,  # like
                0.2, 0.9, 1.0, 0.0, 0.0,  # doing
                0.1, 0.1, 0.3, 1.0, 0.0,  # my
                0.1, 0.2, 0.8, 0.9, 1.0,  # homework.
            ]
            out = visualize_heatmap(tokens, attention)
            
            # instead of creating a file, we save all information in a string
            out_img = sprint(show, MIME("image/svg+xml"), out)

            @test occursin("<svg", out_img)
            @test occursin("</svg>", out_img)
            @test occursin("<rect", out_img)
            @test occursin("<line", out_img)
            @test occursin("<text", out_img)
            @test occursin("I", out_img)
            @test occursin("like", out_img)
            @test occursin("doing", out_img)
            @test occursin("my", out_img)
            @test occursin("homework", out_img)
        end

        @testset "Check svg: only one line per attention value" begin
            tokens = ["My", "test"]
            attention = [
                1.0, 0.0,
                0.0, 1.0
            ]

            out = visualize_heatmap(tokens, attention)
            out_img = sprint(show, MIME("image/svg+xml"), out)

            lines = length(collect(eachmatch(r"<line", out_img)))
            
            @test lines == 2
        end

        # visualize heatmap exp escape_svg function fine?
        @testset "Check svg: transformes svg important tokens correct" begin
            @test TransformerXAI.escape_svg("&") == "&amp;"
            @test TransformerXAI.escape_svg("<") == "&lt;"
            @test TransformerXAI.escape_svg(">") == "&gt;"
            @test TransformerXAI.escape_svg("\"") == "&quot;"
            @test TransformerXAI.escape_svg("'") == "&apos;"
        end
    end