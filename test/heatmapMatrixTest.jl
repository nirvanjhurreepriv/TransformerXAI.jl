@testset "attentionHeatmapMatrix.jl" begin
    # base case -> everything should be fine
    @testset "Check normal case" begin
        tokens = ["I", "like"]
        attention = [
            1.0 0.0
            0.0 1.0
        ]

        plot = generate_attention_heatmap_matrix(attention, tokens)

        @test plot !== nothing
    end

    # check if wrong input size and shape are handled correctly
    @testset "Check wrong input sizes" begin
        tokens = ["I", "like", "dogs"]
        wrong_att_size = [
            1.0 0.0
            0.0 1.0
        ]
        wrong_att_shape = reshape([1, 2, 3, 4], 1, 4)

        @test_throws ArgumentError generate_attention_heatmap_matrix(wrong_att_size, tokens)
        @test_throws ArgumentError generate_attention_heatmap_matrix(wrong_att_shape, tokens)
    end
end