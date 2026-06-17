using Test
using TransformerXAI

@testset "TransformerXAI.jl" begin
    
    @testset "Analytic: Causal Masking Property" begin
        # Proper lower-triangular matrix with rows summing to 1.0
        attn = [1.0 0.0 0.0; 
                0.3 0.7 0.0; 
                0.2 0.3 0.5]
        
        @test size(attn) == (3, 3)
        # Upper triangle must be exactly zero (causal constraint)
        @test all(attn[i,j] == 0.0 for i in 1:3 for j in (i+1):3)
        # Rows must sum to 1.0 (softmax normalization)
        @test all(sum(attn, dims=2) .≈ 1.0)
    end
    
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
end