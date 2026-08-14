@testset "collocation points" begin
    @testset "$(nameof(typeof(r))), K = $K" for r in ROOTS, K in DEGREES
        r isa GaussLobatto && K < 2 && continue

        taus = ExaModelsCollocation._get_taus(r, K)
        @test length(taus) == K
        @test eltype(taus) <: AbstractFloat
        @test issorted(taus)
        @test allunique(taus)
        @test all(0 .<= taus .<= 1)

        @test (last(taus) ≈ 1) == (r isa GaussRadau || r isa GaussLobatto)
        @test (first(taus) ≈ 0) == (r isa GaussLobatto)
    end

    @testset "K = 1 special cases" begin
        @test ExaModelsCollocation._get_taus(GaussRadau(), 1) ≈ [1.0]
        @test ExaModelsCollocation._get_taus(GaussLegendre(), 1) ≈ [0.5]
        @test_throws ArgumentError Collocation([0.0, 1.0], 1; roots = GaussLobatto())
    end
end
