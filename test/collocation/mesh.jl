@testset "mesh placement" begin
    @testset "uniform" begin
        taus = ExaModelsCollocation._get_taus(GaussRadau(), 3)
        mesh = ExaModelsCollocation._get_mesh(collect(range(0.0, 5.0; length = 21)), taus)

        @test propertynames(mesh) == (:nodes, :h, :t)
        @test mesh.h isa Vector{Float64}
        @test mesh.t isa Matrix{Float64}
        @test length(mesh.nodes) == 21
        @test length(mesh.h) == 20
        @test all(mesh.h .≈ 0.25)
        @test size(mesh.t) == (20, 3)
        @test mesh.t[1, :] ≈ mesh.nodes[1] .+ mesh.h[1] .* taus
        @test mesh.t[end, end] ≈ 5.0
    end

    @testset "nonuniform" begin
        nodes = [0.0, 0.1, 1.0, 1.05, 3.0]
        taus = ExaModelsCollocation._get_taus(GaussLegendre(), 2)
        mesh = ExaModelsCollocation._get_mesh(nodes, taus)

        @test mesh.h ≈ diff(nodes)
        @test size(mesh.t) == (4, 2)
        for i in 1:4, j in 1:2
            @test mesh.t[i, j] ≈ nodes[i] + mesh.h[i] * taus[j]
            @test nodes[i] <= mesh.t[i, j] <= nodes[i + 1]
        end
    end

    @testset "one name each" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 6), 3; adaptive = true)
        @test core.mesh.h isa ExaModels.Parameter
        @test core.mesh.t isa ExaModels.Parameter
        @test core.mesh.t[1, 1] isa ExaModels.ParameterNode
        @test size(ExaModelsCollocation._tval(core.mesh)) == (5, 3)
        @test ExaModelsCollocation._hval(core.mesh) ≈ fill(0.2, 5)
    end
end
