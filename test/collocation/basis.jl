struct NotAPolynomial <: ExaModelsCollocation.AbstractPolynomial end

@testset "BasisWeights build paths" begin
    @testset "$(nameof(typeof(b))), $(nameof(typeof(r))), K = $K" for
            b in BASES, r in ROOTS, K in DEGREES

        r isa GaussLobatto && K < 2 && continue
        r isa GaussLobatto && b isa StateForm && continue

        taus = ExaModelsCollocation._get_taus(r, K)
        w = ExaModelsCollocation._get_weights(ExaModelsCollocation.Lagrange(), b, taus)

        @test w isa ExaModelsCollocation.BasisWeights
        @test w.taus === taus
        @test all(isconcretetype, fieldtypes(typeof(w)))

        n = b isa StateForm ? K + 1 : K
        @test size(w.A) == (n, K)
        @test length(w.b) == n
        @test all(isfinite, w.A)
        @test all(isfinite, w.b)
    end

    @testset "mode resolution" begin
        nodes = range(0.0, 1.0; length = 5)

        @test_throws ArgumentError Collocation(nodes, 3; roots = GaussLobatto(), basis = StateForm())
        tag = Collocation(nodes, 3; roots = GaussLobatto(), basis = DerivativeForm())
        @test tag.mode.basis isa DerivativeForm

        @test_throws ArgumentError Collocation(nodes, 3; polynomial = NotAPolynomial())

        @test_throws ArgumentError CollocationExaCore([0.0], 3)
        @test_throws ArgumentError CollocationExaCore(nodes, 0)
        @test_throws ArgumentError CollocationExaCore([0.0, 2.0, 1.0], 3)
        @test_throws ArgumentError CollocationExaCore([0.0, 0.5, 0.5, 1.0], 3)
    end
end
