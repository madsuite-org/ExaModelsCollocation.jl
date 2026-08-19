@testset "continuity ties every slot the collocation calls cover" begin
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

    @testset "every mesh counts as a slot" begin
        both() = CollocationExaCore([range(0.0, tf; length = 5) for tf in (1.0, 4.0)], 3)

        core = both()
        @add_var_collocation(core, z, 1:2)
        @add_con_collocation(core, cz, z[v,m], -z[v,m] for v in 1:2, m in 1:1)
        @test_throws ArgumentError add_con_continuity(core, z)

        core2 = both()
        @add_var_collocation(core2, z2, 1:2)
        @add_con_collocation(core2, cz2, z2[v,m], -z2[v,m] for v in 1:2, m in 1:2)
        @add_con_collocation(core2, cz3, z2[v,m], -2 * z2[v,m] for v in 1:2, m in 2:2)
        @test_throws ArgumentError add_con_continuity(core2, z2)
    end
end

@testset "a block pinned to one mesh is that mesh's core" begin
    N, K = 8, 3
    tfs = (1.0, 4.0)

    function decay(basis, core, kw...)
        @add_var_collocation(core, w, 1:2; kw...)
        @add_con_collocation(core, cw, w[v], -2 * t * w[v] for v in 1:2)
        @add_con_continuity(core, contw, w)
        ExaModels.@add_con(core, icw, w[v, 1, 0] - v for v in 1:2)
        r = madnlp(ExaModels.ExaModel(core); print_level = MadNLP.ERROR, tol = 1e-12)
        return ExaModels.solution(r, w)
    end

    @testset "$(nameof(typeof(b)))" for b in BASES
        pinned = decay(b, CollocationExaCore(
            [range(0.0, tf; length = N + 1) for tf in tfs], K; basis = b,
        ), :mesh => 2)
        on2 = decay(b, CollocationExaCore(range(0.0, tfs[2]; length = N + 1), K; basis = b))
        on1 = decay(b, CollocationExaCore(range(0.0, tfs[1]; length = N + 1), K; basis = b))

        @test pinned ≈ on2 rtol = 1e-9
        @test !isapprox(pinned, on1; rtol = 1e-3)
    end
end

@testset "h in a junction row moves with the mesh" begin
    N, K = 5, 3
    b, r = DerivativeForm(), GaussLegendre()

    decay(nodes; kw...) = begin
        core = CollocationExaCore(nodes, K; basis = b, roots = r, kw...)
        @add_var_collocation(core, z, 1:2)
        @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:2)
        @add_con_continuity(core, cont, z)
        ExaModels.ExaModel(core)
    end

    @testset "under a scale" begin
        t0, tnom, ts = 2.0, 3.0, 4.5
        nom = collect(range(t0, t0 + tnom; length = N + 1))
        stretched = t0 .+ (nom .- t0) .* (ts / tnom)

        m = decay(nom; unknown_horizon = true)
        x = collect(range(0.1, 2.0; length = m.meta.nvar - 1))
        @test ExaModels.NLPModels.cons(m, [ts; x]) ≈ ExaModels.NLPModels.cons(decay(stretched), x)
    end

    @testset "on two meshes" begin
        tfs = (1.0, 4.0)
        fam = CollocationExaCore(
            [range(0.0, tf; length = N + 1) for tf in tfs], K; basis = b, roots = r,
        )
        @add_var_collocation(fam, zf, 1:2)
        @add_con_collocation(fam, cf, zf[v,m], -2 * t * zf[v,m] for v in 1:2, m in 1:2)
        @add_con_continuity(fam, contf, zf)
        ExaModels.@add_con(fam, icf, zf[v,m,1,0] - v for v in 1:2, m in 1:2)
        rf = madnlp(ExaModels.ExaModel(fam); print_level = MadNLP.ERROR, tol = 1e-12)
        sol = ExaModels.solution(rf, zf)

        for (j, tf) in enumerate(tfs)
            single = CollocationExaCore(range(0.0, tf; length = N + 1), K; basis = b, roots = r)
            @add_var_collocation(single, z1, 1:2)
            @add_con_collocation(single, c1, z1[v], -2 * t * z1[v] for v in 1:2)
            @add_con_continuity(single, cont1, z1)
            ExaModels.@add_con(single, ic1, z1[v,1,0] - v for v in 1:2)
            r1 = madnlp(ExaModels.ExaModel(single); print_level = MadNLP.ERROR, tol = 1e-12)
            @test sol[:, j, :, :] ≈ ExaModels.solution(r1, z1) rtol = 1e-9
        end
    end
end

@testset "a point at tau = 1 collapses the junction row" begin
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
