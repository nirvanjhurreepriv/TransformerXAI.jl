using Test
using Llama2
using TransformerXAI

@testset "TransformerXAI.jl" begin
    @testset "Module Structure" begin
        @test TransformerXAI isa Module
    end
    
    @testset "Visualization Type Exists" begin
        # Just test that the type is defined (doesn't require function calls)
        @test isdefined(TransformerXAI, :AttentionHeatmap)
    end

    @testset "Edge Cases" begin
        # Type mismatch correctly triggers MethodError via strict dispatch
        @test_throws MethodError extract_att_weights(nothing)
        @test_throws MethodError extract_att_weights("not a bot")
    end

    include("loadLlamaTest.jl")
    include("forwardTestChanged.jl")
    include("talkTestChanged.jl")
    include("extractAttentionWeightsTest.jl")
    include("visualizeHeatmapTest.jl")
end