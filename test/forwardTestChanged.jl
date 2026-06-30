@testset "forward_changed!" begin
    model_path = normpath(joinpath(@__DIR__, "..", "models", "stories42M.bin"))

    if isfile(model_path)
        # if get_att == false -> same behaviour as standard forward! func
        @testset "Check get_att == false same behaviour as Llama2.forward!" begin
            transformer = Llama2.Transformer(model_path)
            transformer_changed = Llama2.Transformer(model_path)

            token = Int32(2)
            pos = Int32(1)

            logits = Llama2.forward!(transformer, token, pos)
            logits_changed = forward_changed!(transformer_changed, token, pos)

            @test logits_changed isa Vector{Float32}
            @test logits_changed == transformer_changed.state.logits
            @test isapprox(logits, logits_changed)
            @test all(isfinite, logits_changed)
        end

        # check if logits and attention is returned when get_att == true
        @testset "Check correct return" begin
            transformer = Llama2.Transformer(model_path)

            token = Int32(2)
            pos = Int32(1)

            # set get_att == true
            logits, attention = forward_changed!(transformer, token, pos, true)

            @test logits isa Vector{Float32}
            @test attention isa Array{Float32, 3}
            @test logits == transformer.state.logits
            @test all(isfinite, logits)
            @test all(isfinite, attention)
        end
    end
end