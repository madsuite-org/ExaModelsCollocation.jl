@testset "Lagrange basis operators" begin
    pval(p, nodes) = [p(x) for x in nodes]

    @testset "$(nameof(typeof(r))), K = $K" for r in ROOTS, K in DEGREES
        r isa GaussLobatto && K < 2 && continue

        taus = ExaModelsCollocation._get_taus(r, K)
        anchored = ExaModelsCollocation._add_tau0(taus)
        @test length(anchored) == K + 1
        @test anchored[1] == 0

        for nodes in (anchored, taus)
            n = length(nodes)
            allunique(nodes) || continue

            D = ExaModelsCollocation.delljk(nodes, nodes)
            b = ExaModelsCollocation.ell1j(nodes) 
            O = ExaModelsCollocation.Omegajk(nodes, nodes)
            o = ExaModelsCollocation.Omega1j(nodes)

            @test size(D) == (n, n)
            @test size(O) == (n, n)
            @test length(b) == n
            @test length(o) == n

            for m in 0:(n - 1)
                p = x -> x^m
                dp = x -> m == 0 ? zero(x) : m * x^(m - 1)
                ip = x -> x^(m + 1) / (m + 1)
                v = pval(p, nodes)

                @test permutedims(D) * v ≈ pval(dp, nodes) atol = 1e-9
                @test b' * v ≈ p(1.0) atol = 1e-10
                @test permutedims(O) * v ≈ pval(ip, nodes) atol = 1e-10
                @test o' * v ≈ ip(1.0) atol = 1e-10
            end

            @test sum(D; dims = 1) ≈ zeros(1, n) atol = 1e-9
            @test sum(b) ≈ 1
            @test vec(sum(O; dims = 1)) ≈ nodes atol = 1e-10
            @test sum(o) ≈ 1 atol = 1e-10

            last(nodes) ≈ 1 && @test b ≈ [i == n ? 1.0 : 0.0 for i in 1:n] atol = 1e-10
        end
    end

    @testset "delljk only differentiates at its own nodes" begin
        taus = ExaModelsCollocation._get_taus(GaussRadau(), 3)
        @test_throws ArgumentError ExaModelsCollocation.delljk(taus, [0.5])
    end
end
