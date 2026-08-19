@testset "CollocationExaCore construction" begin
    nodes = range(0.0, 5.0; length = 21)
    K = 3

    @testset "mesh geometry" begin
        core = CollocationExaCore(nodes, K)
        @test core.N == 20
        @test core.K == 3
        @test length(core.mesh.h) == 20
        @test size(core.mesh.t) == (20, K)
        @test core.mesh.t[1, end] <= core.nodes[2]
        @test all(core.mesh.h .≈ 0.25)
        @test length(core.weights.taus) == K
        @test core.adaptive == false

        @test core.nodes === core.mesh.nodes
        @test core.weights === core.mode.weights
        @test core.roots === core.mode.roots
        @test fieldnames(ExaModelsCollocation.CollocationTag) ==
            (:mode, :mesh, :block, :resid)
    end

    @testset "it is an ExaCore" begin
        core = CollocationExaCore(nodes, K)
        @test core isa ExaModels.ExaCore
        @test core isa ExaModelsCollocation.CollocationExaCore

        core, x = ExaModels.add_var(core, 1:3; name = Val(:x))
        core, p = ExaModels.add_par(core, 1:2; value = [1.0, 2.0])
        ExaModels.@add_var(core, w, 1:4)
        ExaModels.@add_con(core, g, x[i] - 1.0 for i in 1:3)
        @test core.nvar == 7 && core.npar == 2 && core.ncon == 3
        @test core.x === x && core.w === w && core.g === g

        m = ExaModels.ExaModel(core)
        @test m isa ExaModelsCollocation.CollocationExaModel
        @test m.N == 20 && m.K == 3
        @test m.mesh === core.mesh
        @test m.basis isa StateForm
    end

    @testset "a registered handle takes precedence over a derived name" begin
        core = CollocationExaCore(nodes, K)
        core, Kvar = ExaModels.add_var(core, 1:3; name = Val(:K))
        core, Nvar = ExaModels.add_var(core, 1:2; name = Val(:N))

        @test core.K === Kvar
        @test core.N === Nvar
        @test ExaModels.ExaModel(core).K === Kvar
        @test core.mesh isa ExaModelsCollocation.CollocationMesh
        @test core.nodes === core.mesh.nodes
        @test core.weights === core.mode.weights
    end

    @testset "derived cores do not share block and resid" begin
        base = CollocationExaCore(nodes, K)
        b1, z1 = add_var_collocation(base, 1:1)
        b2, z2 = add_var_collocation(base, 1:1)

        @test isempty(base.block)
        @test length(b1.block) == 1 && b1.block[1] === z1
        @test length(b2.block) == 1 && b2.block[1] === z2

        b1, _ = add_con_collocation(
            b1, (z1[v] => -z1[v, i, k] for (v, i, k, t) in [(v,) for v in 1:1]),
        )
        @test isempty(base.resid) && isempty(b2.resid)
        @test length(b1.resid) == 1

        b1, _ = add_con_continuity(b1, z1)
        @test b1.ncon == b1.N * K + (b1.N - 1)
    end

    @testset "both entry points agree" begin
        c1 = CollocationExaCore(nodes, K; roots = GaussLegendre())
        c2 = ExaModels.ExaCore(tag = Collocation(nodes, K; roots = GaussLegendre()))
        @test typeof(c1) === typeof(c2)
        @test c1.N == c2.N && c1.K == c2.K && c1.roots === c2.roots

        plain = ExaModels.ExaCore()
        plain, p = ExaModels.add_var(plain, 1:4; name = Val(:p))
        attached = ExaModels.ExaCore(plain; tag = Collocation(nodes, K))
        @test attached.nvar == 4
        @test attached.p === p
        @test attached.N == 20

        @test CollocationExaCore(nodes, K; concrete = Val(true)) isa CollocationExaCore
    end

    @testset "it prints" begin
        one = CollocationExaCore(nodes, K)
        @add_var_collocation(one, z, 1:2)
        s = sprint(show, one)
        @test occursin("N = 20 intervals, K = 3 degree\n", s)
        @test occursin("horizon [0.0, 5.0]", s)
        @test occursin("z[1:2, i, k=0:3]", s)
        @test sprint(show, ExaModels.ExaModel(one)) != s

        fam = CollocationExaCore([range(0.0, tf; length = 21) for tf in (1.0, 5.0)], K; unknown_horizon = true)
        f = sprint(show, fam)
        @test occursin("2 meshes, unknown horizon", f)
        @test occursin("horizon [0.0, 1.0] [0.0, 5.0]", f)

        big = CollocationExaCore([range(0.0, Float64(j); length = 21) for j in 1:9], K)
        @test occursin("…", sprint(show, big))
    end

    @testset "unknown property" begin
        core = CollocationExaCore(nodes, K)
        @test_throws Exception core.nope
    end

    @testset "$(something(b, "CPU")), adaptive = $ad" for b in BACKENDS, ad in (false, true)
        N, Nz, K = 5, 2, 3
        build(backend) = begin
            core = CollocationExaCore(
                range(0.0, 1.0; length = N + 1), K; backend = backend, adaptive = ad,
            )
            @add_var_collocation(core, z, 1:Nz)
            @add_con_collocation(core, coll, z[v], -2 * t * z[v] for v in 1:Nz)
            @add_con_continuity(core, cont, z)
            ExaModels.ExaModel(core)
        end

        m, ref = build(b), build(nothing)
        @test m isa ExaModelsCollocation.CollocationExaModel
        @test m.meta.ncon == Nz * N * K + Nz * (N - 1)

        xs = collect(range(0.1, 2.0; length = ref.meta.nvar))
        x = copyto!(similar(m.meta.x0), xs)
        c = similar(m.meta.x0, m.meta.ncon)
        j = similar(m.meta.x0, m.meta.nnzj)
        ExaModels.NLPModels.cons!(m, x, c)
        ExaModels.NLPModels.jac_coord!(m, x, j)

        cref, jref = zeros(ref.meta.ncon), zeros(ref.meta.nnzj)
        ExaModels.NLPModels.cons!(ref, xs, cref)
        ExaModels.NLPModels.jac_coord!(ref, xs, jref)

        @test Array(c) ≈ cref
        @test Array(j) ≈ jref
    end
end
