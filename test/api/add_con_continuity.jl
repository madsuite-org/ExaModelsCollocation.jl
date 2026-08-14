@testset "continuity ties every slot the collocation calls cover" begin
    # The slots come off the recorded collocation calls, for either basis, so those calls
    # come first and a slot left uncollocated is an error rather than a silent gap.
    for b in BASES
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3; basis = b)
        @add_var_collocation(core, z, 1:2)

        @testset "nothing collocated yet" begin
            @test_throws ArgumentError add_con_continuity(core, z)
        end

        @add_con_collocation(core, coll, z[1], -z[1])

        @testset "z[1] collocated, z[2] not" begin
            @test_throws ArgumentError add_con_continuity(core, z)
        end

        @add_con_collocation(core, coll2, z[2], -z[2])
        @add_con_continuity(core, cont, z)
        @test :cont in propertynames(core)
        @test core.cont isa ExaModels.Constraint
    end

    @testset "no slot may be covered twice" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
        @add_var_collocation(core, z, 1:1)
        @add_con_collocation(core, c1, z[1], -z[1])
        @add_con_collocation(core, c2, z[1], -2 * z[1])
        @test_throws ArgumentError add_con_continuity(core, z)
    end

    @testset "a non-collocation variable is refused" begin
        core = CollocationExaCore(range(0.0, 1.0; length = 5), 3)
        core, w = ExaModels.add_var(core, 1:2)
        @test_throws ArgumentError add_con_continuity(core, w)
    end
end

@testset "a point at tau = 1 collapses the junction row" begin
    # Radau and Lobatto make continuity z[i,K] = z[i+1,0], two nonzeros and no b-sum, where
    # Legendre keeps the sum at K + 2. The row count is the same either way.
    N, K = 5, 3

    function junction_nnzj(basis, roots)
        core = CollocationExaCore(range(0.0, TF; length = N + 1), K; basis, roots)
        @add_var_collocation(core, z)
        @add_con_collocation(core, coll, z[], -z)

        nnzj = core.nnzj
        @add_con_continuity(core, cont, z)
        @test core.ncon == N * K + (N - 1)
        return (core.nnzj - nnzj) ÷ (N - 1)
    end

    @testset "$(nameof(typeof(b)))" for b in BASES
        @test junction_nnzj(b, GaussRadau()) == 2
        @test junction_nnzj(b, GaussLegendre()) == K + 2
    end
    @test junction_nnzj(DerivativeForm(), GaussLobatto()) == 2
end
